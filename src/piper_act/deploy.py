"""通用部署流程：真机运行 ACT 策略。

每轮：机械臂在起始位姿 → 按提示摆好场景 → 回车开跑 → 动作停稳（spec.settle_s 秒内指令几乎不变）或超时即停
→ 回车沿刚才走过的路径退回起始位姿（夹爪也倒放）→ 下一轮。

安全措施：每帧关节指令相对上一帧最多变化 --max-step rad（限速），并按 URDF 关节限位截断。
⚠️ 首次运行用低 --speed / 小 --max-step；手放在急停旁，Ctrl+C = 停在当前位置。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from .arm import PiperX, acquire_can_lock, clip_target, rewind
from .camera import open_camera
from .config import APPROACH_SPEED, FPS
from .task import TaskSpec

_SETTLE_GRIP = 0.002   # m，夹爪指令变化小于它算停稳


def load_policy(path: str, device: str, n_action_steps: int):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(path)
    assert n_action_steps <= policy.config.chunk_size, f"--n-action-steps 不能大于 chunk_size={policy.config.chunk_size}"
    # 每执行 n_action_steps 帧重新看一次图像预测新的动作块；越小越闭环，推理越频繁
    policy.config.n_action_steps = n_action_steps
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def infer(policy, pre, post, device: str, task: str, image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """一帧观测（RGB 图像 HxWx3 uint8，状态向量）→ 目标向量（关节 [+ 夹爪]）。"""
    import torch
    from lerobot.utils.control_utils import predict_action

    obs = {"observation.images.wrist": image, "observation.state": state.astype(np.float32)}
    action = predict_action(obs, policy, torch.device(device), pre, post, use_amp=False,
                            task=task, robot_type="piper_x")
    return action.squeeze(0).numpy().astype(float)  # 0.4.4 的 predict_action 不去 batch 维（与其文档不符）


def run(arm: PiperX, cam, policy, pre, post, spec: TaskSpec, args) -> np.ndarray:
    """跑一轮策略，返回实际下发的指令轨迹（用于原路退回）。"""
    policy.reset()
    x_cmd = arm.read_state()
    sent = [x_cmd]
    window = round(spec.settle_s * FPS)
    recent = deque(maxlen=window)
    thr = np.array([args.settle] * 6 + [_SETTLE_GRIP] * (arm.dof - 6))
    t = time.monotonic()
    t_end = t + args.max_seconds
    try:
        while True:
            if time.monotonic() > t_end:
                print(f"⏱ {args.max_seconds:.0f} 秒没停稳，停在当前位置")
                break
            a = infer(policy, pre, post, args.device, spec.task, cam.read_latest(), arm.read_state())
            x_new = a.copy()
            x_new[:6] = x_cmd[:6] + np.clip(a[:6] - x_cmd[:6], -args.max_step, args.max_step)  # 只限关节速度
            x_cmd = clip_target(x_new)
            arm.command(x_cmd, args.speed)
            sent.append(x_cmd)
            recent.append(x_cmd)
            if len(recent) == window and (np.ptp(np.array(recent), axis=0) < thr).all():
                print(f"✓ 已停稳。状态 {arm.read_state().round(3)}")
                break
            # 推理换动作块那一帧会慢；落后就从现在重新计时，不连发补帧
            t = max(t + 1 / FPS, time.monotonic())
            time.sleep(max(0.0, t - time.monotonic()))
    except KeyboardInterrupt:
        print("\n⏹ 已中止")
    arm.hold()
    return np.array(sent)


def main(spec: TaskSpec) -> None:
    import torch

    ap = argparse.ArgumentParser(description=f"真机运行 {spec.name} 策略")
    ap.add_argument("--policy", required=True, help="checkpoint 的 pretrained_model 目录")
    ap.add_argument("--start-pose", default=str(spec.start_pose))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-action-steps", type=int, default=20)
    ap.add_argument("--speed", type=int, default=50, help="move_j 速度百分比，与采集时 --speed 一致")
    ap.add_argument("--max-step", type=float, default=0.03, help="每帧每关节最大变化 (rad)，25Hz 下 0.03 ≈ 43°/s")
    ap.add_argument("--settle", type=float, default=0.005, help=f"{spec.settle_s} 秒内关节指令变化都小于它 (rad) 算停稳")
    ap.add_argument("--max-seconds", type=float, default=30)
    ap.add_argument("--channel", default="can0")
    args = ap.parse_args()

    start = np.array(json.loads(Path(args.start_pose).read_text()))
    policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps)
    dim = policy.config.output_features["action"].shape[0]
    if dim != len(spec.names) or len(start) != dim:
        sys.exit(f"❌ 维度不一致：策略 {dim} 维、起始位姿 {len(start)} 维、任务 {spec.name} 要 {len(spec.names)} 维")
    print(f"策略已加载（{args.device}）")

    lock = acquire_can_lock(args.channel)
    cam = open_camera()
    arm = PiperX(channel=args.channel, gripper=spec.gripper, grip_force=spec.grip_force)
    arm.connect()
    arm.enable()
    arm.wait_feedback()
    try:
        input(f"确认工作区无人，回车：慢速走到起始位姿 {start.round(3)}…")
        arm.goto(start)
        while True:
            if input(f"\n{spec.setup_prompt}，回车开跑（q 退出；运行中 Ctrl+C 停在原地）：").strip().lower() == "q":
                break
            sent = run(arm, cam, policy, pre, post, spec, args)
            input("回车：沿刚才的路径慢速退回起始位姿（确认路径无遮挡）…")
            rewind(arm, sent, APPROACH_SPEED)
            arm.goto(start)
    finally:
        cam.disconnect()
        lock.unlink(missing_ok=True)
        input("\n回车：机械臂慢速回零位并失能（确认周围无人）…")
        arm.home()
        arm.disable()
        arm.disconnect()
