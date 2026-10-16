#!/usr/bin/env python3
"""iPhone 遥操作真机测试（不录数据）：调通方向和平顺度用。在仓库根目录运行：
    python3 tasks/pick_place/phone_teleop_test.py

1. 连手机、启动标定（屏幕朝上平放，点一下「跟随 开/关」）
2. 方向标定（机械臂不动）：站在机械臂后方，手机往前推一段、往上抬一段，按实际动作定方向
3. 方向校验（真机不动）：点「跟随 开/关」后移动手机，终端和窗口实时显示程序理解的前/左/上位移，窗口里的机械臂模型
   预览跟随效果；对了点「保存」继续，不对点「丢弃重来」重新标定
4. 机械臂跟随：终端实时对照显示 目标位移 / 机械臂实际位移（cm）。A3 开合夹爪，「退出」或 Ctrl+C 结束

MuJoCo 窗口（从机械臂右后方看）：灰色 = 虚拟机械臂（程序算出、发给真机的目标姿态），蓝色 = 真机实际姿态；
黄球 = 手机目标点，红/绿/蓝箭头 = 程序理解的前/左/上（开始跟随时对齐，之后跟着手机姿态转）；青色小点 = 目标最近的轨迹；白球 = 真机指尖。
程序一开始就只读连上真机（不使能）；方向校验时灰色机械臂预览跟随、蓝色真机不动，点「保存」后真机才使能并跟随。

平顺：控制频率 --hz（默认 50）；手机位置和姿态先过 One Euro 滤波（慢时强滤波消抖、快时弱滤波不拖延迟；
--min-cutoff 越小静止越稳，--beta 越大快速移动时越跟手）；关节限速 --max-vel；move_j 速度 --speed 100
（每帧目标很近，速度开满只是跟得更紧，不会更快）。--mode js 改用 SDK 的跟随模式 move_js：不做轨迹规划、响应最快，
SDK 标注风险高，确认 j 模式仍抖再试。
姿态：夹爪跟着手机转（开始跟随那一刻为起点，像夹爪和手机绑在一起）；腕关节 ±89°，转不过去时位置优先、姿态停在能到的最大角度。
--no-rot：夹爪始终竖直朝下（和采集脚本一致）。
记录：每次跟随都把逐帧数据存到 outputs/teleop_logs/<时间>.npz（手机目标、滤波后目标、指令、真机关节、时间），用来分析抖动来源。
安全：指尖不低于 --z-min；够不到时停在边界；机械臂失能（如碰撞保护）立即停止；手机断开时机械臂停住、等手机回来重新标定方向。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque

import mujoco
import numpy as np
from lerobot.utils.rotation import Rotation
from task import SPEC

from piper_act.arm import PiperX, acquire_can_lock, clip_target
from piper_act.phone_record import DEFAULT_START
from piper_act.sim.env import PickPlaceEnv, yaw_of
from piper_act.sim.phone import PhoneTeleop, calibrate_directions, connect_iphone

UNREACHABLE = 0.003   # m


ROT_BETA = 2.0       # 姿态 One Euro 的 beta（每 rad/s）


class OneEuro:
    """One Euro 滤波（向量）：慢时截止频率 = min_cutoff，强滤波消抖；快时截止频率随速度升高，跟随不拖延迟。"""

    def __init__(self, x, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset(x)

    def reset(self, x) -> None:
        self.x, self.dx = np.array(x, float), np.zeros(len(x))

    @staticmethod
    def _alpha(fc: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * fc * dt))

    def __call__(self, x, dt: float) -> np.ndarray:
        self.dx += self._alpha(self.d_cutoff, dt) * ((np.asarray(x) - self.x) / dt - self.dx)
        fc = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        self.x = self.x + self._alpha(fc, dt) * (np.asarray(x) - self.x)
        return self.x.copy()


class Orient:
    """夹爪姿态目标：开始跟随那一刻的夹爪姿态 R0，叠加手机此后转过的角度（teleop.axes()，先过 One Euro 滤波）。
    像夹爪和手机绑在一起：手机左转 30°，夹爪绕竖直轴左转 30°；手机前端低头，夹爪跟着同方向倾斜。"""

    def __init__(self, R: np.ndarray, args):
        self.R0, self.R = R.copy(), R.copy()
        self.filt = OneEuro(np.zeros(3), args.min_cutoff, ROT_BETA)

    def start(self, R: np.ndarray) -> None:   # 开始跟随时：以当前夹爪姿态为起点
        self.R0, self.R = R.copy(), R.copy()
        self.filt.reset(np.zeros(3))

    def update(self, turn: np.ndarray, dt: float) -> np.ndarray:
        rv = self.filt(Rotation.from_matrix(turn).as_rotvec(), dt)
        self.R = Rotation.from_rotvec(rv).as_matrix() @ self.R0
        return self.R


def solve(ik, pos, yaw, q0, R, args):
    """位置 + 姿态（R=None 时夹爪竖直朝下）。每帧从上一帧的结果接着解、只解一轮固定次数：
    耗时稳定（不拖慢控制节奏），也不会在几种解法之间来回跳（会让指令抖）。姿态权重低：够不到的姿态自然退让、位置优先。"""
    return ik.solve(pos, yaw, q0, iters=20, rot=R, wrot=0.6 if R is None else 0.3)


def fmt(d: np.ndarray) -> str:   # 要短：超过终端宽度会折行，\r 就没法原地刷新
    return f"前{d[0] * 100:+.1f} 左{d[1] * 100:+.1f} 上{d[2] * 100:+.1f}"


def fmt_en(d: np.ndarray) -> str:   # MuJoCo 窗口的字体只支持英文
    return f"F {d[0] * 100:+5.1f}  L {d[1] * 100:+5.1f}  U {d[2] * 100:+5.1f} cm"


class Viz:
    """MuJoCo 窗口：虚拟机械臂（灰，程序算出的目标姿态）+ 真机实际姿态（蓝）+ 手机目标点（黄球 + 前/左/上箭头）
    + 目标轨迹（青）+ 真机指尖（白球）。"""

    REAL_RGBA = (0.25, 0.55, 1.0, 0.4)

    AXES = ((np.array([1.0, 0, 0]), (1, 0.15, 0.15, 1)),    # 前：红
            (np.array([0, 1.0, 0]), (0.15, 0.9, 0.15, 1)),   # 左：绿
            (np.array([0, 0, 1.0]), (0.2, 0.4, 1, 1)))       # 上：蓝

    def __init__(self, env: PickPlaceEnv):
        import mujoco.viewer

        self.m, self.d = env.model, mujoco.MjData(env.model)
        self.d_real = mujoco.MjData(env.model)   # 真机姿态，画成蓝色叠加
        self._pert = mujoco.MjvPerturb()
        self.qadr, self.grip_adr = env.qadr, env.grip_adr
        for name in ("cube", "place_mark"):   # 仿真里的方块和放置区，真机测试用不到
            b = self.m.body(name).id
            self.m.geom_rgba[self.m.geom_bodyid == b, 3] = 0
        shown = (self.m.geom_rgba[:, 3] > 0) & (self.m.geom_bodyid != 0)
        self.m.geom_rgba[shown, 3] = 0.45   # 机械臂半透明，黄球和箭头不会被夹爪挡住
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
        with self.viewer.lock():
            self.viewer.cam.azimuth, self.viewer.cam.elevation = 30.0, -40.0   # 从机械臂右后方斜着往下看
            self.viewer.cam.lookat[:], self.viewer.cam.distance = [0.33, 0.0, 0.08], 0.9
        self.trail: deque = deque(maxlen=150)
        self._t = 0.0

    def _geom(self, kind, size, pos, rgba):
        scn = self.viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        return g

    def show(self, q: np.ndarray, grip: float, target: np.ndarray, actual: np.ndarray | None,
             lines: list[tuple[str, str]], x_real: np.ndarray | None = None, axes: np.ndarray | None = None) -> None:
        """更新画面（最多 15 帧/秒）：q/grip = 虚拟机械臂，x_real = 真机关节 + 夹爪，
        axes = 黄球上三个箭头的方向（列 = 前/左/上，跟着手机转）。窗口关掉了就什么都不做。"""
        now = time.monotonic()
        if now - self._t < 1 / 15 or not self.viewer.is_running():   # 15 帧/秒：绘图每次要几毫秒，别挤占控制循环
            return
        self._t = now
        self.trail.append(target.copy())
        with self.viewer.lock():
            self.d.qpos[self.qadr] = q
            self.d.qpos[self.grip_adr], self.d.qpos[self.grip_adr + 1] = grip / 2, -grip / 2
            mujoco.mj_forward(self.m, self.d)
            self.viewer.user_scn.ngeom = 0
            if x_real is not None:
                self._add_real(x_real)
            for p in list(self.trail)[:-1]:
                self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, (0.2, 0.9, 0.9, 0.6))
            self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], target, (1, 0.85, 0.1, 1))
            for i, (axis, rgba) in enumerate(self.AXES):
                if axes is not None:
                    axis = axes[:, i]
                g = self._geom(mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], [0, 0, 0], rgba)
                if g is not None:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, target, target + 0.10 * axis)
            if actual is not None:
                self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.01, 0, 0], actual, (1, 1, 1, 0.9))
        self.viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               "\n".join(a for a, _ in lines), "\n".join(b for _, b in lines)))
        self.viewer.sync()

    def _add_real(self, x: np.ndarray) -> None:
        """真机姿态的机械臂几何体复制进 user_scn，染成蓝色。"""
        d = self.d_real
        d.qpos[self.qadr] = x[:6]
        d.qpos[self.grip_adr], d.qpos[self.grip_adr + 1] = x[6] / 2, -x[6] / 2
        mujoco.mj_forward(self.m, d)
        scn = self.viewer.user_scn
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(self.m, d, self.viewer.opt, self._pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            if scn.geoms[i].rgba[3] > 0:   # 隐藏的（方块、放置区）保持隐藏
                scn.geoms[i].rgba[:] = self.REAL_RGBA

    def close(self) -> None:
        self.viewer.close()


def setup_directions(teleop: PhoneTeleop, viz: Viz, x0: np.ndarray, center: np.ndarray, ik, yaw: float,
                     arm: PiperX, args) -> None:
    """方向标定 + 校验（真机不动），校验不对就重新标定。

    校验时窗口里的虚拟机械臂按黄球做逆运动学跟着动（预览跟随效果），蓝色真机停在原地。
    """
    while True:
        viz.trail.clear()
        viz.show(x0[:6], x0[6], center, None, [("Step", "direction calibration (real arm does not move)")],
                 arm.read_state())
        teleop.set_frame(calibrate_directions(teleop.phone))
        print("\n方向校验（机械臂不动）：点「跟随 开/关」，往前/后/左/右/上/下移动手机，看下面数字和窗口里黄球的方向对不对。\n"
              "  对了点「保存」继续；不对点「丢弃重来」重新标定")
        teleop.reset(center, teleop.grip)
        viz.trail.clear()
        q = x0[:6].copy()
        orient = Orient(ik.fk(q)[1], args)
        while True:
            was = teleop.enabled
            pressed = teleop.update(0.02)
            if teleop.enabled and not was:
                orient.start(ik.fk(q)[1])
            d = (teleop.pos - center) / teleop.scale
            print(f"\r  跟随{'开' if teleop.enabled else '关'}  手机 {fmt(d)} cm    ", end="", flush=True)
            R = orient.update(teleop.axes(), 0.02) if args.rot and teleop.enabled else (orient.R if args.rot else None)
            q_new, err = solve(ik, teleop.pos, yaw, q, R, args)
            if err < UNREACHABLE:
                q = q_new
            viz.show(q, teleop.grip, teleop.pos, None, [
                ("Step", "direction check: preview only, REAL ARM DOES NOT MOVE"),
                ("Follow", "ON" if teleop.enabled else "off (tap B1)"),
                ("Phone moved", fmt_en(d)),
                ("Real arm", "not moving (enabled after B2)"),
                ("Colors", "grey = virtual arm  blue = real arm  yellow = phone"),
                ("Arrows", "red = forward  green = left  blue = up (turn with phone)"),
                ("Next", "B2 ok -> real arm   B3 redo calibration"),
            ], arm.read_state(), teleop.axes())
            if "b2" in pressed:
                print()
                return
            if "b3" in pressed:
                print("\n重新标定")
                break


def send(arm: PiperX, x: np.ndarray, args) -> None:
    if args.mode == "js":
        arm._robot.move_js(x[:6].tolist())
        arm._effector.move_gripper_m(float(x[6]), arm.grip_force)
    else:
        arm.command(x, args.speed)


def follow(arm: PiperX, teleop: PhoneTeleop, ik, x0: np.ndarray, yaw: float, viz: Viz, args) -> str:
    """跟随循环。返回 "quit" / "disabled"。"""
    dt = 1.0 / args.hz
    max_step = args.max_vel * dt
    p0 = ik.fk(x0[:6])[0]
    x_cmd = x0.copy()
    teleop.reset(p0, x0[6])
    target, last_ok = p0.copy(), p0.copy()
    pos_filt = OneEuro(p0, args.min_cutoff, args.beta)
    orient = Orient(ik.fk(x0[:6])[1], args)
    viz.trail.clear()
    log: dict[str, list] = {k: [] for k in ("t", "phone", "target", "cmd", "real", "enabled")}
    try:
        warned, n, t_rate, hz = False, 0, time.monotonic(), 0.0
        print("\n真机跟随：点「跟随 开/关」开始。A3 开合夹爪，「退出」或 Ctrl+C 结束\n"
              "  窗口：灰 = 虚拟机械臂（发给真机的目标）  蓝 = 真机实际  黄球 = 手机  白球 = 真机指尖")
        t = time.monotonic()
        while True:
            was = teleop.enabled
            try:
                pressed = teleop.update(dt)
            except RuntimeError as e:
                arm.hold()
                print(f"\n⚠ {e}\n  机械臂停在原地，等手机回来…")
                while True:
                    try:
                        teleop.update(dt)
                        break
                    except RuntimeError:
                        pass
                print("✓ 手机回来了。App 重开过的话手机坐标会变，重新标定方向")
                x_cmd = np.append(arm.read_joints(), teleop.grip)
                p = ik.fk(x_cmd[:6])[0]
                setup_directions(teleop, viz, x_cmd, p, ik, yaw, arm, args)
                orient.start(ik.fk(x_cmd[:6])[1])
                teleop.reset(p, teleop.grip)
                target, last_ok = p.copy(), p.copy()
                pos_filt.reset(p)
                t = time.monotonic()
                continue
            if "b8" in pressed:
                print()
                return "quit"
            if teleop.enabled != was:
                print(f"\n  跟随：{'开' if teleop.enabled else '关'}")
                if teleop.enabled:
                    orient.start(ik.fk(x_cmd[:6])[1])   # 夹爪从当前姿态开始跟着手机转

            goal = teleop.pos.copy()
            if goal[2] < args.z_min:
                goal[2] = args.z_min
                if teleop.enabled:
                    teleop.hold_at(goal)
            target = pos_filt(goal, dt)                       # One Euro：滤掉手抖和手机定位噪声
            R = orient.update(teleop.axes(), dt) if args.rot and teleop.enabled else (orient.R if args.rot else None)
            q, err = solve(ik, target, yaw, x_cmd[:6], R, args)
            if err < UNREACHABLE:
                last_ok, warned = target.copy(), False
            else:
                q = x_cmd[:6]
                if teleop.enabled:
                    teleop.hold_at(last_ok)
                target = last_ok.copy()
                pos_filt.reset(last_ok)
                if not warned:
                    print("\n⚠ 到边界了（再往外够不到），机械臂停在边界上")
                    warned = True
            x_new = np.append(x_cmd[:6] + np.clip(q - x_cmd[:6], -max_step, max_step), teleop.grip)
            x_new = clip_target(x_new)
            if ik.fk(x_new[:6])[0][2] < args.z_min - 0.005:   # 限速后的中间姿态会碰桌面：关节不动
                x_new[:6] = x_cmd[:6]
            x_cmd = x_new
            send(arm, x_cmd, args)
            x_real = arm.read_state()
            actual = ik.fk(x_real[:6])[0]
            lag = np.degrees(np.abs(x_cmd[:6] - x_real[:6]).max())
            for k, v in (("t", time.monotonic()), ("phone", teleop.pos.copy()), ("target", target.copy()),
                         ("cmd", x_cmd.copy()), ("real", x_real.copy()), ("enabled", teleop.enabled)):
                log[k].append(v)
            viz.show(x_cmd[:6], x_cmd[6], teleop.pos, actual, [
                ("Step", "REAL ARM FOLLOWS PHONE"),
                ("Follow", "ON" if teleop.enabled else "off (tap B1)"),
                ("Phone (yellow)", fmt_en(teleop.pos - p0)),
                ("Real tip (white)", fmt_en(actual - p0)),
                ("Real arm", f"enabled  lag {lag:4.1f} deg  {hz:4.1f} Hz"),
                ("Gripper", f"cmd {x_cmd[6] * 1000:3.0f} mm  real {x_real[6] * 1000:3.0f} mm"),
                ("Boundary", "REACHED" if warned else "-"),
                ("Colors", "grey = virtual (command)  blue = real arm"),
            ], x_real, teleop.axes())

            n += 1
            now = time.monotonic()
            if now - t_rate >= 0.25:
                if not all(arm._robot.get_joints_enable_status_list()):
                    print("\n❌ 机械臂失能了（多半是碰撞保护触发）。已停止发指令")
                    return "disabled"
                hz, n, t_rate = n / (now - t_rate), 0, now
                print(f"\r  手机 {fmt(teleop.pos - p0)} | 真机 {fmt(actual - p0)}   ", end="", flush=True)
            t = max(t + dt, time.monotonic() - dt)   # 落后太多不连发补帧
            time.sleep(max(0.0, t - time.monotonic()))

    finally:
        if log["t"]:
            out = SPEC.dir.parents[1] / "outputs" / "teleop_logs" / time.strftime("%Y%m%d_%H%M%S.npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **{k: np.array(v) for k, v in log.items()}, hz=args.hz, min_cutoff=args.min_cutoff,
                     beta=args.beta, speed=args.speed, mode=args.mode)
            print(f"\n运行记录：{out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="iPhone 遥操作真机测试（不录数据）")
    ap.add_argument("--scale", type=float, default=0.5, help="手机移动 1 m → 指尖移动多少 m")
    ap.add_argument("--hz", type=float, default=50, help="控制频率")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="One Euro 静止时截止频率 (Hz)：越小静止越稳、越迟钝")
    ap.add_argument("--beta", type=float, default=20.0, help="One Euro 速度系数（每 m/s）：越大快速移动时越跟手")
    ap.add_argument("--max-vel", type=float, default=0.75, help="每个关节最大速度 (rad/s)")
    ap.add_argument("--speed", type=int, default=100, help="move_j 速度百分比")
    ap.add_argument("--mode", choices=["j", "js"], default="j", help="j = move_j（默认）；js = move_js 跟随模式（风险高）")
    ap.add_argument("--z-min", type=float, default=0.01, help="指尖最低高度 (m，相对机械臂底座底面)")
    ap.add_argument("--no-rot", dest="rot", action="store_false", help="夹爪不跟手机转，始终竖直朝下")
    ap.add_argument("--channel", default="can0")
    args = ap.parse_args()

    teleop = PhoneTeleop(connect_iphone(), scale=args.scale)
    env = PickPlaceEnv(render=False)
    ik = env.ik
    x0 = np.array(json.loads(SPEC.start_pose.read_text()) if SPEC.start_pose.exists() else DEFAULT_START, float)
    yaw = yaw_of(ik.fk(x0[:6])[1])
    viz = Viz(env)

    lock = acquire_can_lock(args.channel)
    arm = PiperX(channel=args.channel, gripper=True, grip_force=SPEC.grip_force)
    arm.connect()
    arm.wait_feedback()   # 先只读：窗口里显示真机姿态，方向校验通过后才使能
    result, enabled = "quit", False
    try:
        setup_directions(teleop, viz, x0, ik.fk(x0[:6])[0], ik, yaw, arm, args)
        arm.enable()
        enabled = True
        input(f"\n起始位姿 {x0.round(3)}。确认工作区无人，回车：慢速走过去…")
        arm.goto(x0)
        if args.mode == "js":
            input("⚠ js 跟随模式：不做轨迹规划、响应最快，SDK 标注风险高。手放在急停旁，回车继续…")
        result = follow(arm, teleop, ik, x0, yaw, viz, args)
    except KeyboardInterrupt:
        print()
    finally:
        lock.unlink(missing_ok=True)
        if enabled and result != "disabled":
            arm.hold()
            input("\n回车：机械臂慢速回收起姿态并失能（确认周围无人）…")
            arm.home()
            arm.disable()
        arm.disconnect()
        viz.close()


if __name__ == "__main__":
    main()
