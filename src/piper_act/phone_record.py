"""真机 iPhone 遥操作采集：手机控制指尖位置和夹爪开度，逆运动学（官方 MuJoCo 模型）算关节角下发真机，同时录手腕相机。

手机操作与仿真 sim_pick_place 相同（见 piper_act/sim/phone.py）：点一下 B1 开始/暂停跟随（第一次开始跟随才开始录制），
A3 滑块开合夹爪，B2 保存（再原地录 spec.hold_s 秒让策略学会停），B3 丢弃，B8 退出。每条结束后机械臂自动慢速回起始位姿。
起始位姿与部署共用 tasks/<任务>/start_pose.json；还没有时用 DEFAULT_START 并存进去。数据格式与拖拽示教相同。

安全措施：每帧关节指令最多变化 --max-step rad；指尖不低于 --z-min；目标够不到时原地不动；Ctrl+C = 停在当前位置。
⚠️ 首次运行用小 --scale、小 --max-step，手放在急停旁。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .arm import PiperX, acquire_can_lock, clip_target
from .camera import open_camera
from .config import FPS
from .record import open_dataset
from .sim.env import GRIP_OPEN, PickPlaceEnv, yaw_of
from .sim.phone import PhoneTeleop, connect_iphone
from .task import TaskSpec

_UNREACHABLE = 0.003  # m，逆运动学误差超过它算够不到（太宽的话在边界上每帧偏一点，会越攒越偏）
# 默认起始位姿：指尖 (0.30, 0, 0.12) m、夹爪竖直朝下、张开。夹爪朝下时指尖最高只到约 16 cm，
# 仿真的起始点 (0.38, 0, 0.16) 在边角上（J4 卡 89° 限位，往上往前动不了）；这里往前 18 / 后 22 / 左右 24 / 上 7 cm 都能走
DEFAULT_START = [0.0, 1.5387, -1.2202, 1.3397, 0.0, 0.0, GRIP_OPEN]


def _wait_follow_off(teleop: PhoneTeleop) -> None:
    """B1 是开关：上一条结束时还开着的话，先等它关掉，免得新一条一开始就跟随、录制。"""
    teleop.update(1 / FPS)
    if teleop.enabled:
        print("先在手机上点「跟随 开/关」暂停跟随…")
    while teleop.enabled:
        teleop.update(1 / FPS)
        time.sleep(1 / FPS)


def episode(arm: PiperX, cam, teleop: PhoneTeleop, ik, x0: np.ndarray, yaw: float,
            ds, spec: TaskSpec, args) -> str:
    """遥操作一条示范。返回 "saved" / "discarded" / "quit"。"""
    _wait_follow_off(teleop)
    last_ok = ik.fk(x0[:6])[0]   # 上一个够得到的目标
    teleop.reset(last_ok, x0[6])
    print(f"{spec.setup_prompt}。站到机械臂后方，手机顶边指向机械臂前方，点「跟随 开/关」开始")
    x_cmd = x0.copy()
    recording, warned, frames = False, False, 0

    def step(target: np.ndarray) -> None:
        """录一帧（录制中）并下发限速后的目标。"""
        nonlocal x_cmd, frames
        x_new = target.copy()
        x_new[:6] = x_cmd[:6] + np.clip(target[:6] - x_cmd[:6], -args.max_step, args.max_step)
        x_new = clip_target(x_new)
        if ik.fk(x_new[:6])[0][2] < args.z_min - 0.005:   # 限速后的中间姿态会碰桌面：关节不动，只动夹爪
            x_new[:6] = x_cmd[:6]
        x_cmd = x_new
        if recording:
            ds.add_frame({"observation.images.wrist": cam.read_latest(),
                          "observation.state": arm.read_state().astype(np.float32),
                          "action": x_cmd.astype(np.float32), "task": spec.task})
            frames += 1
        arm.command(x_cmd, args.speed)

    t = time.monotonic()
    try:
        while True:
            was_following = teleop.enabled
            try:
                pressed = teleop.update(1 / FPS)
            except RuntimeError as e:   # 手机断开：App 重开后 AR 坐标系会变，之前的标定不能再用
                arm.hold()
                ds.clear_episode_buffer()
                print(f"\n⚠ {e}\n  机械臂已停在原地，本条丢弃。重新运行程序（会重新标定）")
                return "quit"
            if teleop.enabled != was_following:
                print(f"  跟随：{'开' if teleop.enabled else '关'}")
            if teleop.enabled and not recording:
                recording = True
                print("● 开始录制")
            pos = teleop.pos.copy()
            if pos[2] < args.z_min:
                pos[2] = args.z_min
                if teleop.enabled:
                    teleop.hold_at(pos)   # 手机压到桌面以下也不攒距离，往上一抬就跟着起来
            q, err = ik.solve(pos, yaw, x_cmd[:6], iters=30)   # 从实际下发的姿态接着解，关节变化连续
            if err >= _UNREACHABLE:                            # 手机移得快时 30 次可能不够，多算几次再下结论
                q, err = ik.solve(pos, yaw, q, iters=100)
            if err < _UNREACHABLE:
                warned, last_ok = False, pos
            else:
                q = x_cmd[:6]
                if teleop.enabled:
                    teleop.hold_at(last_ok)   # 目标顶在边界上，不再往外跑（不用实际位置：限速时它偏离目标，会越攒越偏）
                if not warned:
                    print("⚠ 到边界了（再往外够不到），机械臂停在边界上")
                    warned = True
            step(np.append(q, teleop.grip))

            if "b8" in pressed:
                ds.clear_episode_buffer()
                return "quit"
            if "b3" in pressed:
                ds.clear_episode_buffer()
                print("✗ 丢弃本条")
                return "discarded"
            if "b2" in pressed:
                if not recording:
                    print("还不能保存：还没开始录制")
                else:
                    print(f"原地再录 {spec.hold_s:.0f} 秒…")
                    hold = x_cmd.copy()
                    for _ in range(round(spec.hold_s * FPS)):
                        step(hold)
                        t += 1 / FPS
                        time.sleep(max(0.0, t - time.monotonic()))
                    ds.save_episode()
                    print(f"✓ 已保存第 {ds.meta.total_episodes} 条（{frames} 帧，{frames / FPS:.1f}s）")
                    return "saved"
            t += 1 / FPS
            time.sleep(max(0.0, t - time.monotonic()))
    except KeyboardInterrupt:
        arm.hold()
        ds.clear_episode_buffer()
        print("\n⏹ 已中止，本条丢弃")
        return "quit"


def main(spec: TaskSpec) -> None:
    ap = argparse.ArgumentParser(description=f"{spec.name}：iPhone 遥操作真机采集")
    ap.add_argument("--root", default=str(spec.dataset))
    ap.add_argument("--episodes", type=int, default=50, help="数据集总条数到这个数就结束")
    ap.add_argument("--scale", type=float, default=0.5, help="手机移动 1 m → 指尖移动多少 m")
    ap.add_argument("--speed", type=int, default=50, help="move_j 速度百分比")
    ap.add_argument("--max-step", type=float, default=0.03, help="每帧每关节最大变化 (rad)，25Hz 下 0.03 ≈ 43°/s")
    ap.add_argument("--z-min", type=float, default=0.01, help="指尖最低高度 (m，相对机械臂底座底面)")
    ap.add_argument("--channel", default="can0")
    args = ap.parse_args()
    assert spec.gripper, "手机遥操作采集需要带夹爪的任务"

    teleop = PhoneTeleop(connect_iphone(), scale=args.scale)   # 先连手机：连不上就直接退出
    ik = PickPlaceEnv(render=False).ik      # 只用仿真的运动学模型
    path = spec.start_pose
    if path.exists():
        x0 = np.array(json.loads(path.read_text()), float)
    else:
        x0 = np.array(DEFAULT_START)
        path.write_text(json.dumps(x0.round(4).tolist()))
        print(f"✓ 起始位姿（默认）已保存到 {path}，部署也用它")
    R = ik.fk(x0[:6])[1]
    if len(x0) != 7 or R[2, 2] > -0.98:     # 遥操作时夹爪保持竖直朝下；起始就歪的话第一次跟随会突然转过去
        sys.exit(f"❌ {path} 的起始位姿不是 7 维或夹爪没有竖直朝下（偏 {np.degrees(np.arccos(-R[2, 2])):.0f}°）。"
                 "删掉它重新运行，会改用默认起始位姿")
    yaw = yaw_of(R)                         # 夹爪转角保持起始值

    lock = acquire_can_lock(args.channel)
    ds = open_dataset(Path(args.root), spec.names)
    cam = open_camera()
    arm = PiperX(channel=args.channel, gripper=True, grip_force=spec.grip_force)
    arm.connect()
    arm.enable()
    arm.wait_feedback()
    try:
        input(f"起始位姿 {x0.round(3)}。确认工作区无人，回车：慢速走过去…")
        while ds.meta.total_episodes < args.episodes:
            arm.goto(x0)
            print(f"\n=== 第 {ds.meta.total_episodes + 1}/{args.episodes} 条 ===")
            if episode(arm, cam, teleop, ik, x0, yaw, ds, spec, args) == "quit":
                break
    finally:
        ds.finalize()
        cam.disconnect()
        lock.unlink(missing_ok=True)
        input("\n回车：机械臂慢速回零位并失能（确认周围无人）…")
        arm.home()
        arm.disable()
        arm.disconnect()
    print(f"数据集: {args.root}（{ds.meta.total_episodes} 条）")
