#!/usr/bin/env python3
"""仿真采集：iPhone（HEBI Mobile I/O）遥操作 MuJoCo 里的 PiPER X 抓放方块 → LeRobot 数据集。

每条示范：方块随机出现 → 点一下 B1 开始跟随、移动手机控制机械臂（第一次开始跟随才开始录制）、A3 滑块开合夹爪
→ 把方块放到蓝色区域、松开夹爪、抬起 → 窗口显示 SUCCESS 后按 B2 保存（没成功不能保存）；B3 丢弃重来；B8 退出。
数据与真机 pick_place 同格式：25Hz，手腕相机 640x480 + 7 维状态 + 7 维动作（6 关节 + 夹爪开度 m）。

在仓库根目录运行（同一个 --root 再运行 = 接着录）：
    python3 tasks/sim_pick_place/record.py
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# 手腕相机离屏渲染用 EGL，窗口用 GLFW：两者都用 GLFW 时程序退出会在 glfw.terminate 崩溃
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np
from task import DATASET, TASK

from piper_act.config import FPS, JOINT_NAMES
from piper_act.record import open_dataset
from piper_act.sim.env import GRIP_OPEN, VIS_GROUP, PickPlaceEnv
from piper_act.sim.phone import PhoneTeleop, connect_iphone

NAMES = list(JOINT_NAMES) + ["gripper"]


def open_viewer(env: PickPlaceEnv):
    """打开 MuJoCo 窗口（显示相机支架和外壳），视角从机械臂后方往前看：屏幕上的前后左右 = 手机的前后左右。"""
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
    with viewer.lock():
        viewer.opt.geomgroup[VIS_GROUP] = 1
        # 方位角 0 = 朝 +x 看（MuJoCo 默认 90 是从右侧看）；偏 20° 避免机械臂挡住方块
        viewer.cam.azimuth, viewer.cam.elevation = 20.0, -50.0
        viewer.cam.lookat[:], viewer.cam.distance = [0.32, 0.05, 0.03], 1.1
    return viewer


def show(viewer, image: np.ndarray, lines: list[tuple[str, str]]) -> None:
    """MuJoCo 窗口：右上角叠加手腕相机画面（缩小一半），左上角显示状态（MuJoCo 字体只支持英文）。"""
    small = image[::2, ::2]
    vp = viewer.viewport
    h, w = small.shape[:2]
    if vp is not None and vp.width > w + 10 and vp.height > h + 10:   # 窗口刚打开时可能还没有尺寸
        viewer.set_images((mujoco.MjrRect(vp.width - w - 10, vp.height - h - 10, w, h), small))
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      "\n".join(a for a, _ in lines), "\n".join(b for _, b in lines)))
    viewer.sync()


def episode(env: PickPlaceEnv, teleop: PhoneTeleop, ds, seed: int, viewer=None) -> str:
    """遥操作一条示范。返回 "saved" / "discarded" / "quit"。"""
    obs = env.reset(seed)
    q = env.q_home.copy()
    pos, R = env.ik.fk(q)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))   # 夹爪转角保持起始值
    teleop.reset(pos, GRIP_OPEN)
    recording, was_success, frames = False, False, 0
    t = time.monotonic()
    while True:
        was_following = teleop.enabled
        pressed = teleop.update(1 / FPS)
        if teleop.enabled != was_following:
            print(f"  跟随：{'开' if teleop.enabled else '关'}（指尖目标 {np.round(teleop.pos, 3)}）")
        if teleop.enabled and not recording:
            recording = True
            print("● 开始录制")
        q, err = env.ik.solve(teleop.pos, yaw, q, iters=30)
        action = np.append(q, teleop.grip).astype(np.float32)
        if recording:
            ds.add_frame({"observation.images.wrist": obs["image"], "observation.state": obs["state"],
                          "action": action, "task": TASK})
            frames += 1
        obs = env.step(action)
        ok = env.success()
        if ok and not was_success:
            print("✓ 方块已放到蓝色区域：按 B2 保存")
        was_success = ok

        if viewer is not None:
            if not viewer.is_running():
                pressed.add("b8")
            else:
                show(viewer, obs["image"], [
                    ("Episode", f"{ds.meta.total_episodes + 1}  seed {seed}"),
                    ("Recording", f"{frames / FPS:.1f} s" if recording else "no (tap B1 to start)"),
                    ("B1 follow", "ON" if teleop.enabled else "off (tap B1)"),
                    ("Gripper", f"{teleop.grip * 1000:.0f} mm"),
                    ("Target", "UNREACHABLE" if err > 0.01 else "ok"),
                    ("Status", "SUCCESS - press B2 to save" if ok else "-"),
                    ("Buttons", "B2 save  B3 discard  B8 quit"),
                ])
        if "b8" in pressed:
            ds.clear_episode_buffer()
            return "quit"
        if "b3" in pressed:
            ds.clear_episode_buffer()
            print("✗ 丢弃本条")
            return "discarded"
        if "b2" in pressed:
            if recording and ok:
                ds.save_episode()
                print(f"✓ 已保存第 {ds.meta.total_episodes} 条（{frames} 帧，{frames / FPS:.1f}s）")
                return "saved"
            print("还不能保存：" + ("还没开始录制" if not recording else "方块没放到蓝色区域 / 夹爪没松开"))
        t += 1 / FPS
        time.sleep(max(0.0, t - time.monotonic()))


def main() -> None:
    ap = argparse.ArgumentParser(description="仿真抓放：iPhone 遥操作采集")
    ap.add_argument("--root", default=str(DATASET))
    ap.add_argument("--episodes", type=int, default=50, help="数据集总条数到这个数就结束")
    ap.add_argument("--scale", type=float, default=1.0, help="手机移动 1 m → 指尖移动多少 m")
    ap.add_argument("--seed", type=int, default=None, help="方块位置的随机种子起点（默认接着已有条数）")
    args = ap.parse_args()

    teleop = PhoneTeleop(connect_iphone(), scale=args.scale)   # 先连手机：连不上就直接退出，不留半截环境
    ds = open_dataset(Path(args.root), NAMES)
    seed = ds.meta.total_episodes if args.seed is None else args.seed
    env = PickPlaceEnv()
    viewer = open_viewer(env)
    try:
        while ds.meta.total_episodes < args.episodes:
            print(f"\n=== 第 {ds.meta.total_episodes + 1}/{args.episodes} 条（种子 {seed}）===")
            if episode(env, teleop, ds, seed, viewer) == "quit":
                break
            seed += 1
    finally:
        ds.finalize()
        viewer.close()
        env.close()
    print(f"数据集: {args.root}（{ds.meta.total_episodes} 条）")


if __name__ == "__main__":
    main()
