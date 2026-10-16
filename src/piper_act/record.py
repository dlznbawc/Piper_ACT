"""通用采集流程：拖拽示教（不录画面）→ 机械臂自己回放同一条轨迹并录制 → LeRobot 数据集。

为什么分两遍：拖拽时人手会进手腕相机画面；回放那一遍画面里没有人手。
数据：25Hz，手腕图像 640x480 + 状态 + 动作（6 维关节，带夹爪时 7 维）。

所有示范都从同一个起始位姿出发（tasks/<任务>/start_pose.json；第一次运行时拖拽设定）。每条示范：
  1. 按提示摆好场景
  2. 托住机械臂 → 回车进入拖拽 → 按提示完成示教（带夹爪的任务：手动开合夹爪）→ 回车结束
  3. 回车：机械臂沿示教轨迹倒着回起始位姿（不录；夹爪也倒放，抓放任务会顺手把物体搬回原处），
     再按示教节奏正向回放并录制，到终点后再原地录 hold_s 秒（让策略学会停住）
  4. s 保存 / r 丢弃、重新示教 / q 丢弃、退出。选完机械臂都倒放回起始位姿

⚠️ 回放时机械臂会自己动：确认工作区无人，手放在急停旁。回放中 Ctrl+C = 停在当前位置、丢弃本条。
同一个 --root 再运行 = 接着录。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .arm import PiperX, acquire_can_lock, clip_target, play, rewind
from .camera import open_camera
from .config import FPS, IMG_H, IMG_W
from .task import TaskSpec

_MOVING_JOINT = 1e-3   # rad/帧，大于它算在动
_MOVING_GRIP = 5e-4    # m/帧


def open_dataset(root: Path, names: list[str]):
    """新建 LeRobot 数据集，或打开已有的接着录。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = root / "meta/info.json"
    if root.exists() and not (root / "meta/tasks.parquet").exists():
        # 上次一条都没保存就退出，只留下空壳（本地读不全时 LeRobot 会转去 HF Hub 找而报错）
        # 只删确认是 0 条的 LeRobot 空壳；别的目录（如误写成 --root datasets）一律不动
        if not info.exists() or json.loads(info.read_text())["total_episodes"] > 0:
            sys.exit(f"❌ {root} 不是空的 LeRobot 数据集、也没法续录，先检查")
        shutil.rmtree(root)
    if root.exists():
        ds = LeRobotDataset(f"local/{root.name}", root=root)
        if ds.meta.features["action"]["names"] != names:
            sys.exit(f"❌ {root} 的维度 {ds.meta.features['action']['names']} 与本任务 {names} 不一致")
        ds.start_image_writer(num_threads=4)
        print(f"接着录：已有 {ds.meta.total_episodes} 条")
        return ds
    features = {
        "observation.images.wrist": {"dtype": "video", "shape": (IMG_H, IMG_W, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    return LeRobotDataset.create(repo_id=f"local/{root.name}", fps=FPS, features=features, root=root,
                                 robot_type="piper_x", use_videos=True, image_writer_threads=4)


def get_start_pose(arm: PiperX, path: Path) -> np.ndarray:
    """读起始位姿并慢速走过去；文件不存在就拖拽设定一个并保存。"""
    if path.exists():
        x = np.array(json.loads(path.read_text()))
        if len(x) != arm.dof:
            sys.exit(f"❌ {path} 有 {len(x)} 维，本任务要 {arm.dof} 维。删掉它重新设定")
        input(f"起始位姿 {x.round(3)}（{path.name}）。确认工作区无人，回车：慢速走过去…")
        arm.goto(x)
        return x
    print(f"还没有起始位姿（{path}），现在拖拽设定。起始位姿下手腕相机要能看到整个工作区域"
          + ("，夹爪张开" if arm.gripper else "") + "。")
    input("托住机械臂，回车进入拖拽…")
    arm.drag_mode()
    x = None
    try:
        while x is None:
            input("拖到起始位姿、停稳后回车记录…")
            x = arm.read_leader_state()
            if x is None:
                print("没读到拖拽中的状态（拖拽模式下没动过时不发帧），稍微动一下再回车")
    finally:   # Ctrl+C 中断也要切回 Follower，否则机械臂一直停在示教模式，下次 enable 超时
        arm.exit_drag_mode()
    x = clip_target(x)
    path.write_text(json.dumps(x.round(4).tolist()))
    print(f"✓ 起始位姿已保存到 {path}：{x.round(3)}")
    return x


def teach(arm: PiperX, spec: TaskSpec) -> np.ndarray | None:
    """拖拽示教：按 FPS 记录 Leader 状态，回车结束。

    去掉首尾没动的帧，返回 x[N, dof]；没动过返回 None。
    """
    input(f"{spec.setup_prompt}，托住机械臂，回车进入拖拽…")
    arm.drag_mode()
    print(f"● 示教中：{spec.teach_prompt}，完成后回车")
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    xs = []
    try:
        t = time.monotonic()
        while not stop.is_set():
            x = arm.read_leader_state()
            if x is not None:
                xs.append(x)
            t += 1 / FPS
            time.sleep(max(0.0, t - time.monotonic()))
    finally:
        arm.exit_drag_mode()
    if len(xs) < 2:
        return None
    x = np.array(xs)
    thr = np.array([_MOVING_JOINT] * 6 + [_MOVING_GRIP] * (arm.dof - 6))
    moving = np.flatnonzero((np.abs(np.diff(x, axis=0)) > thr).any(1))
    if len(moving) == 0:
        return None
    a, b = moving[0], moving[-1] + 2
    print(f"示教轨迹 {b - a} 帧（{(b - a) / FPS:.1f}s）")
    # 截断到限位：动作标签 = 实际下发的指令
    return np.array([clip_target(xk) for xk in x[a:b]])


def replay(arm: PiperX, cam, ds, spec: TaskSpec, traj: np.ndarray, speed: int, hold_frames: int) -> bool:
    """先倒放回起点（不录），再正向回放 + 终点停留，逐帧录制。

    每帧先记观测、再发这一步的动作。Ctrl+C 或出错 → 停在当前位置、丢弃本条，返回 False。
    """
    input("确认工作区无人，回车：机械臂先倒放回起点，再正向回放并录制…")
    states = []

    def record(xk: np.ndarray) -> None:
        state = arm.read_state()
        if state is None:
            raise RuntimeError("读不到关节/夹爪反馈")
        state = state.astype(np.float32)
        states.append(state)
        ds.add_frame({"observation.images.wrist": cam.read_latest(), "observation.state": state,
                      "action": xk.astype(np.float32), "task": spec.task})

    try:
        rewind(arm, traj, speed)
        time.sleep(0.5)  # 等夹爪动作到位
        print("● 回放录制中…（Ctrl+C 立即停止）")
        play(arm, np.vstack([traj, np.repeat(traj[-1:], hold_frames, axis=0)]), speed, on_step=record)
    except (KeyboardInterrupt, RuntimeError, TimeoutError) as e:
        arm.hold()
        ds.clear_episode_buffer()
        print(f"\n⏹ {str(e) or '已中止'}。机械臂停在当前位置，本条丢弃")
        return False
    n = min(len(states), len(traj))
    lag = np.abs(np.array(states)[:n, :6] - traj[:n, :6]).max(0)
    print(f"录了 {len(states)} 帧。各关节 实际-指令 最大偏差 (rad): {lag.round(3)}"
          "（持续偏大 = 跟不上示教速度，可调高 --speed）")
    return True


def session(spec: TaskSpec, arm: PiperX, cam, ds, episodes: int, speed: int, start_pose: Path,
            make_traj=teach) -> None:
    """采集主循环（硬件已连好）。make_traj(arm, spec) → 要录的轨迹，默认拖拽示教。"""
    start = get_start_pose(arm, start_pose)
    c = ""
    while c != "q" and ds.meta.total_episodes < episodes:
        print(f"\n=== 第 {ds.meta.total_episodes + 1}/{episodes} 条 ===")
        traj = make_traj(arm, spec)
        if traj is None:
            print("没录到动作，重来")
            arm.goto(start)
            continue
        done = replay(arm, cam, ds, spec, traj, speed, round(spec.hold_s * FPS))
        opts = ("s", "r", "q") if done else ("r", "q")
        c = ""
        while c not in opts:
            c = input(("[s]保存  " if done else "") + "[r]丢弃、重新示教  [q]丢弃、退出"
                      "（选完机械臂都会先倒放回起始位姿）：").strip().lower()
        if c == "s":
            ds.save_episode()
            print(f"✓ 已保存，共 {ds.meta.total_episodes} 条")
        elif done:
            ds.clear_episode_buffer()
        rewind(arm, traj, speed)
        arm.goto(start)  # 消除示教起点与起始位姿的微小差别，每条都从同一姿态开始


def main(spec: TaskSpec, make_traj=teach) -> None:
    ap = argparse.ArgumentParser(description=f"采集 {spec.name} 示范")
    ap.add_argument("--root", default=str(spec.dataset))
    ap.add_argument("--start-pose", default=str(spec.start_pose))
    ap.add_argument("--episodes", type=int, default=50, help="数据集总条数到这个数就结束")
    ap.add_argument("--speed", type=int, default=50, help="回放速度百分比；跟不上示教轨迹就调高")
    ap.add_argument("--channel", default="can0")
    args = ap.parse_args()

    lock = acquire_can_lock(args.channel)
    ds = open_dataset(Path(args.root), spec.names)
    cam = open_camera()
    arm = PiperX(channel=args.channel, gripper=spec.gripper, grip_force=spec.grip_force)
    arm.connect()
    arm.enable()
    arm.wait_feedback()
    try:
        session(spec, arm, cam, ds, args.episodes, args.speed, Path(args.start_pose), make_traj)
    finally:
        ds.finalize()
        cam.disconnect()
        lock.unlink(missing_ok=True)
        input("\n回车：机械臂慢速回零位并失能（确认周围无人）…")
        arm.home()
        arm.disable()
        arm.disconnect()
    print(f"数据集: {args.root}（{ds.meta.total_episodes} 条）")
