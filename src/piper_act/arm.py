"""PiPER X 真机接口 —— 封装官方 pyAgxArm（CAN）。

⚠️ 真机运动有碰撞/夹伤风险：确认工作区无人、急停可及，首次上电先低速小范围测试。

状态 / 目标统一用一个向量：前 6 维关节角（rad，joint1~joint6），带夹爪时第 7 维是夹爪开度（m）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import APPROACH_SPEED, FPS, JOINT_LIMITS_RAD, REACHED

_LO, _HI = np.array(JOINT_LIMITS_RAD).T
# 收起姿态 = 失能后自然停住的姿态（实测失能后停在 [0, -0.04, 0.05, 0.66, 0.02, 0]）。回到这里再失能，关节松开时几乎不动；
# 回全零位再失能，手腕（J4）会甩 38° 撞到机身。J2/J3 实际停在软件限位外 2~3°，只能给 0，失能时会轻微下沉
_HOME_Q = np.array([0.0, 0.0, 0.0, 0.62, 0.0, 0.0])


def clip_target(x) -> np.ndarray:
    """关节按 URDF 限位截断；夹爪开度（若有）不小于 0。"""
    x = np.array(x, dtype=float)
    x[:6] = np.clip(x[:6], _LO, _HI)
    x[6:] = np.maximum(x[6:], 0.0)
    return x


def acquire_can_lock(channel: str) -> Path:
    """单实例锁：同一个 CAN 口同时只能一个进程控制。进程已退出的旧锁自动接管。"""
    lock = Path(f"/tmp/piperx_sync_{channel}.lock")
    pid = lock.read_text().strip() if lock.exists() else ""
    if pid.isdigit() and Path(f"/proc/{pid}").exists():
        sys.exit(f"❌ 进程 {pid} 正在占用 {channel}，先关掉它")
    lock.write_text(str(os.getpid()))
    return lock


class PiperX:
    def __init__(self, channel: str = "can0", gripper: bool = False, grip_force: float = 1.0,
                 interface: str = "socketcan"):
        from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

        cfg = create_agx_arm_config(robot=ArmModel.PIPER_X, firmeware_version=PiperFW.DEFAULT,
                                    interface=interface, channel=channel)
        self._robot = AgxArmFactory.create_arm(cfg)
        self._effector = None
        self.channel = channel
        self.gripper = gripper
        self.grip_force = grip_force   # N，回放和部署都用这个固定夹持力
        self.dof = 7 if gripper else 6

    # ---------- 生命周期 ----------
    def connect(self) -> None:
        self._robot.connect()
        if self.gripper:
            self._effector = self._robot.init_effector(self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        print(f"[PiperX] 已连接 {self.channel}{'（带夹爪）' if self.gripper else ''}")

    def disconnect(self) -> None:
        self._robot.disconnect()

    def enable(self) -> None:
        for _ in range(200):
            if self._robot.enable():
                return
            time.sleep(0.01)
        raise RuntimeError("enable 超时")

    def disable(self) -> None:
        """失能 = 关节立刻松开。零位也不是自稳姿态（实测失能后手腕等会下垂、撞到机身），必须先托住。"""
        input("失能后关节会松开下垂：用手托住机械臂，回车失能…")
        for _ in range(200):
            if self._robot.disable():
                return
            time.sleep(0.01)
        raise RuntimeError("disable 超时")

    def wait_feedback(self, timeout: float = 5.0) -> None:
        """等第一帧有效反馈（CAN 反馈异步到达）。"""
        t0 = time.monotonic()
        while self.read_state() is None:
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("等待关节角/夹爪反馈超时：确认机械臂已上电、can0 已 up")
            time.sleep(0.2)

    # ---------- 读取 ----------
    def read_joints(self) -> Optional[np.ndarray]:
        resp = self._robot.get_joint_angles()
        return np.array(resp.msg, dtype=float) if resp is not None else None

    def read_state(self) -> Optional[np.ndarray]:
        """正常（非拖拽）模式下的实际状态：关节角 [+ 夹爪开度，读 0x2A8 反馈]。"""
        q = self.read_joints()
        if q is None or not self.gripper:
            return q
        g = self._effector.get_gripper_status()
        return np.append(q, g.msg.value) if g is not None else None

    def read_leader_state(self) -> Optional[np.ndarray]:
        """拖拽示教中的实时状态：关节角（CAN 0x155~0x157）[+ 夹爪开度（0x159）]。

        拖拽模式下普通反馈（get_joint_angles、夹爪 0x2A8）停在旧值或几乎停播，必须读这两个通道；
        没动时不发帧，开始拖之前读不到（返回 None）。
        """
        resp = self._robot.get_leader_joint_angles()
        if resp is None:
            return None
        q = np.array(resp.msg, dtype=float)
        if not self.gripper:
            return q
        g = self._effector.get_gripper_ctrl_states()
        return np.append(q, g.msg.value) if g is not None else None

    # ---------- 运动 ----------
    def command(self, x, speed: int) -> None:
        """下发目标 x（关节 [+ 夹爪]），不等待。"""
        x = clip_target(x)
        self._robot.set_speed_percent(int(speed))
        self._robot.set_motion_mode(self._robot.OPTIONS.MOTION_MODE.J)
        self._robot.move_j(x[:6].tolist())
        if self.gripper and len(x) > 6:
            self._effector.move_gripper_m(float(x[6]), self.grip_force)

    def at(self, x) -> bool:
        """关节是否到位。不看夹爪：夹住物体时夹爪到不了目标开度。"""
        return np.abs(self.read_joints() - clip_target(x)[:6]).max() < REACHED

    def goto(self, x, speed: int = APPROACH_SPEED, timeout: float = 20.0) -> None:
        """慢速直接走到 x，关节到位才返回。"""
        if self.at(x):
            self.command(x, speed)   # 关节已到位也要下发一次，夹爪才会动
            return
        t0 = time.monotonic()
        while not self.at(x):
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"{timeout:.0f} 秒内没到目标：目标 {clip_target(x)[:6].round(3)}，"
                                   f"实际 {self.read_joints().round(3)}")
            self.command(x, speed)   # 一直重发：刚使能后的第一条运动指令会被机械臂忽略（实测）
            time.sleep(0.05)
        time.sleep(0.3)

    def hold(self) -> None:
        """停在当前位置（关节目标设为当前实际姿态；夹爪不动）。"""
        self.command(self.read_joints(), APPROACH_SPEED)

    def home(self, timeout: float = 15.0) -> None:
        """慢速回收起姿态（_HOME_Q），关节实际到位才返回，之后才能安全失能。

        不看 get_arm_status 的 motion_status：实测没动也会报"已到达"，按它判断会在半空失能、整条臂掉下来。
        """
        t0 = time.monotonic()
        while not self.at(_HOME_Q):
            if time.monotonic() - t0 > timeout:
                input(f"⚠ 没回到零位（实际 {self.read_joints().round(3)}），失能后会掉下来：托住机械臂，回车…")
                return
            self.command(_HOME_Q, APPROACH_SPEED)   # 一直重发，理由同 goto
            time.sleep(0.05)

    # ---------- 拖拽示教 ----------
    def drag_mode(self) -> None:
        """进入拖拽示教（零力矩）：设为 Leader-Follower 协议中的示教输入臂（CAN 0x470 = 0xFA）。

        ⚠️ 进入后关节不再锁位，受重力会下垂！进入前必须托住机械臂（尤其 J2/J3）。
        （MIT kp=kd=0、CAN 0x150 示教指令在真机上都实测无效，只有这个能释放扭矩。）
        """
        self._robot._set_leader_follower_config(linkage_config=0xFA)

    def exit_drag_mode(self) -> None:
        """退出拖拽：切回 Follower（0xFC）重新锁位，并立刻把目标设为当前姿态，减少回弹。"""
        x = self.read_leader_state()
        if x is None:
            x = self.read_joints()
        self._robot._set_leader_follower_config(linkage_config=0xFC)
        if x is not None:
            self.command(x, APPROACH_SPEED)


def play(arm: PiperX, traj: np.ndarray, speed: int, on_step: Callable[[np.ndarray], None] | None = None) -> None:
    """按 FPS 逐帧下发轨迹 traj[N, dof]；on_step(xk) 在下发每帧之前调用（录制用）。

    不在起点时先慢速过去；结尾最多再等 2 秒让机械臂追上最后一个目标。
    """
    def send(xk: np.ndarray) -> None:
        if on_step:
            on_step(xk)
        arm.command(xk, speed)

    arm.goto(traj[0])
    t = time.monotonic()
    for xk in traj:
        send(xk)
        t += 1 / FPS
        time.sleep(max(0.0, t - time.monotonic()))
    for _ in range(2 * FPS):
        if arm.at(traj[-1]):
            break
        send(traj[-1])
        t += 1 / FPS
        time.sleep(max(0.0, t - time.monotonic()))


def rewind(arm: PiperX, traj: np.ndarray, speed: int) -> None:
    """沿轨迹倒着走回起点（夹爪也倒放）：从轨迹上离当前关节姿态最近的点开始倒，中途停下也能原路退回。"""
    k = int(np.argmin(np.abs(traj[:, :6] - arm.read_joints()).max(1)))
    play(arm, traj[k::-1], speed)
