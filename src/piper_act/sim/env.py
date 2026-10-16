"""MuJoCo 抓放仿真：PiPER X（带夹爪）+ 50mm 红色方块（位置随机）+ 蓝色放置区 + 手腕相机。

观测：{"image": (480, 640, 3) uint8 RGB, "state": (7,) 6 关节角 rad + 夹爪开度 m}
动作：(7,) 6 关节目标角 + 夹爪开度目标。夹爪开度 = 两指间距（0~0.1 m），与真机夹爪读数同义。

手腕相机按官方 agx_arm_sim 的 D435 支架安装位置建模；真机相机/支架不同时改 CAM_CHAIN / CAM_FOVY。
支架网格 realsense_mid_stand.stl 来自 agx_arm_sim（仓库未声明许可证，公开分发前确认）。
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..config import FPS, IMG_H, IMG_W

SCENE = Path(__file__).resolve().parents[3] / "assets/piper_x/mjcf/piper_x_gripper.xml"
CUBE_HALF = 0.025                              # 50mm 方块
CUBE_X, CUBE_Y = (0.34, 0.40), (-0.07, 0.07)   # 方块随机范围（m），起始画面里方块完整可见
PLACE = np.array([0.28, 0.22])                 # 放置区中心（固定）
PLACE_TOL = 0.02                               # 成功：方块中心离放置区 < 2cm
HOME_TCP = np.array([0.38, 0.0, 0.16])         # 起始指尖位置（再高 J4 会卡 ±89° 限位）
GRIP_OPEN = 0.1                                # 夹爪全开（两指间距 m）

# 官方 URDF 的相机安装链：gripper_base → 支架 → d435_camera_link → camera_link（xyz, URDF rpy）
CAM_CHAIN = [
    ((-0.034, -0.001, 0.022), (0, 0, 0)),
    ((-0.031, -0.028, 0.0018), (0, -1.22, 0)),
    ((0.0106, 0.0175, 0.0125), (0, 0, 0)),
]
CAM_FOVY = 42.0                                # D435 彩色约 69°×42°
VIS_GROUP = 5                                  # 相机支架/外壳所在显示层（窗口里打开，手腕相机画面不渲染）


def _quat(rpy) -> list[float]:
    """URDF rpy（固定轴 XYZ）→ MuJoCo wxyz。"""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(rpy, float), "XYZ")
    return q.tolist()


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    cube = spec.worldbody.add_body(name="cube", pos=[0.36, 0, CUBE_HALF])
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  rgba=[0.9, 0.2, 0.2, 1], mass=0.05, friction=[1.5, 0.01, 0.001])
    mark = spec.worldbody.add_body(name="place_mark", pos=[*PLACE, 0.001])
    mark.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.001],
                  rgba=[0.2, 0.3, 0.9, 1], contype=0, conaffinity=0)
    body = spec.body("gripper_base")
    chain = []
    for i, (xyz, rpy) in enumerate(CAM_CHAIN):
        body = body.add_body(name=f"cam_chain{i}", pos=list(xyz), quat=_quat(rpy))
        chain.append(body)
    # 支架 + D435 外壳：只给人看。放在 VIS_GROUP 显示层：窗口里显示，手腕相机画面里不渲染（否则相机在外壳里拍出一片黑）；
    # 无质量无碰撞，不影响物理。D435 网格面数超 MuJoCo 上限，用官方碰撞盒尺寸的长方体代替
    spec.add_mesh(name="camera_stand", file="realsense_mid_stand.stl",
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    show = dict(group=VIS_GROUP, contype=0, conaffinity=0, mass=0)
    chain[0].add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="camera_stand",
                      quat=_quat((-1.57, 0, 1.57)), rgba=[1.0, 0.55, 0.1, 1], **show)
    chain[2].add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012525, 0.045, 0.0125], pos=[0, -0.0175, 0],
                      rgba=[0.25, 0.25, 0.3, 1], **show)
    # realsense camera_link: x 前 / y 左 / z 上；MuJoCo 相机看 -z、y 朝上。彩色传感器在 y+15mm
    R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    qc = np.zeros(4)
    mujoco.mju_mat2Quat(qc, R.flatten())
    body.add_camera(name="wrist", pos=[0, 0.015, 0], quat=qc.tolist(), fovy=CAM_FOVY)
    model = spec.compile()
    model.site_rgba[model.site("tcp").id, 3] = 0  # 绿色 TCP 标记不能出现在训练图像里
    return model


def yaw_of(R: np.ndarray) -> float:
    """TCP 姿态矩阵 → 夹爪绕竖直轴的转角（TCP x 轴在水平面的方向）。"""
    return float(np.arctan2(R[1, 0], R[0, 0]))


class IK:
    """末端逆运动学：指尖位置 + 夹爪竖直朝下 [+ 绕竖直轴转角 yaw]。阻尼最小二乘。"""

    def __init__(self, model: mujoco.MjModel, qadr: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        self.m, self.d = model, mujoco.MjData(model)
        self.qadr, self.lo, self.hi = qadr, lo, hi
        self.tcp = model.site("tcp").id
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def solve(self, pos, yaw: float | None, q_init, iters: int = 100, wrot: float = 0.6,
              rot: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """返回 (关节角, 位置误差 m)。yaw=None 时只约束夹爪朝下、转角随意。
        rot（3x3 TCP 姿态矩阵）给了就按完整姿态求解，忽略"朝下"和 yaw。"""
        q = np.array(q_init, float)
        zt = np.array([0.0, 0.0, -1.0])
        if rot is not None:
            xt, yt, zt = np.asarray(rot, float).T
            yaw = 0.0   # 下面按三根轴都约束
        elif yaw is not None:
            xt = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            yt = np.cross(zt, xt)
        for _ in range(iters):
            p, R = self.fk(q)
            perr = pos - p
            rerr = np.cross(R[:, 2], zt)
            if yaw is not None:
                rerr = 0.5 * (rerr + np.cross(R[:, 0], xt) + np.cross(R[:, 1], yt))
            if np.linalg.norm(perr) < 1e-4 and np.linalg.norm(rerr) < 1e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.tcp)
            J = np.vstack([self.jp[:, self.qadr], wrot * self.jr[:, self.qadr]])
            e = np.concatenate([perr, wrot * rerr])
            # 顶在限位上还要往外走的关节锁住、让其余关节补偿；否则这一步被截断，位置越解越偏，
            # 在限位附近连"待在原地"都解不出来（实测起始位姿 J4 卡 89° 时误差 14 mm）
            free = np.ones(len(q), bool)
            for _ in range(len(q)):
                Jf = J * free
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 0.06**2 * np.eye(6), e)
                out = free & (((q <= self.lo + 1e-6) & (dq < 0)) | ((q >= self.hi - 1e-6) & (dq > 0)))
                if not out.any():
                    break
                free &= ~out
            q = np.clip(q + 0.5 * dq, self.lo, self.hi)
        return q, float(np.linalg.norm(pos - self.fk(q)[0]))


class PickPlaceEnv:
    def __init__(self, render: bool = True) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W) if render else None
        self.n_sub = round(1.0 / FPS / self.model.opt.timestep)
        m = self.model
        arm = [m.joint(f"joint{i}").id for i in range(1, 7)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in arm])
        lo = np.array([m.jnt_range[j][0] for j in arm])
        hi = np.array([m.jnt_range[j][1] for j in arm])
        self.ik = IK(m, self.qadr, lo, hi)
        self.grip_adr = m.jnt_qposadr[m.joint("gripper_joint1").id]
        self.cube_adr = m.jnt_qposadr[m.body("cube").jntadr[0]]
        self.q_home, _ = self.ik.solve(HOME_TCP, None, np.zeros(6), iters=800)

    def reset(self, seed: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = self.q_home
        d.qpos[self.grip_adr], d.qpos[self.grip_adr + 1] = GRIP_OPEN / 2, -GRIP_OPEN / 2
        d.qpos[self.cube_adr:self.cube_adr + 7] = [rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y), CUBE_HALF, 1, 0, 0, 0]
        d.ctrl[:6] = self.q_home
        d.ctrl[6], d.ctrl[7] = GRIP_OPEN / 2, -GRIP_OPEN / 2
        mujoco.mj_forward(self.model, d)
        return self.obs()

    def step(self, action) -> dict:
        a = np.asarray(action, float)
        self.data.ctrl[:6] = a[:6]
        self.data.ctrl[6], self.data.ctrl[7] = a[6] / 2, -a[6] / 2
        mujoco.mj_step(self.model, self.data, self.n_sub)
        return self.obs()

    def obs(self) -> dict:
        state = np.append(self.data.qpos[self.qadr], 2 * self.data.qpos[self.grip_adr]).astype(np.float32)
        out = {"state": state}
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="wrist")
            out["image"] = self.renderer.render().copy()
        return out

    def tcp_pose(self) -> tuple[np.ndarray, float]:
        """当前指尖位置与夹爪转角（实际状态）。"""
        sid = self.model.site("tcp").id
        return self.data.site_xpos[sid].copy(), yaw_of(self.data.site_xmat[sid].reshape(3, 3))

    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_adr:self.cube_adr + 3].copy()

    def success(self) -> bool:
        """方块在放置区内、落在桌面上、夹爪已松开。"""
        c = self.cube_pos()
        return bool(np.linalg.norm(c[:2] - PLACE) < PLACE_TOL and c[2] < CUBE_HALF + 0.005
                    and 2 * self.data.qpos[self.grip_adr] > 0.06)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
