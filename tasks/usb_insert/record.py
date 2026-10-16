#!/usr/bin/env python3
"""采集示范 → datasets/usb_insert/。在仓库根目录运行：python3 tasks/usb_insert/record.py

不示教插入，示教的是插好后的位姿：拖着机械臂把插头插进母头 → 脚本生成拔出轨迹
（竖直上拔 LIFT → 随机中间点 → 起始位姿）→ 倒过来就是插入轨迹，交给通用流程（piper_act/record.py）：
机械臂先沿它倒着走 = 执行拔出（不录），再正向真实插入并录制。保存后同样再拔出、回起始位姿，插头一直夹在手里。

数据多样性靠每条挪母头位置；随机中间点只让接近路线不完全相同。
"""
import numpy as np
from task import SPEC

from piper_act.arm import clip_target
from piper_act.config import FPS
from piper_act.record import main
from piper_act.sim.env import PickPlaceEnv

LIFT = 0.04            # m，竖直上拔距离（USB-A 插入深度约 12 mm）
LIFT_SPEED = 0.01      # m/s，慢 = 准静态，倒过来插入时不会一下子顶进去
VIA_XY = 0.03          # m，中间点在上拔终点基础上水平随机 ±VIA_XY
VIA_Z = (0.02, 0.05)   # m，再随机升高这么多
JOINT_SPEED = 0.5      # rad/s，关节插值段最快关节的速度

IK = PickPlaceEnv(render=False).ik   # 只用仿真的运动学模型
rng = np.random.default_rng()


def _line(q0, q1) -> list:
    n = max(2, round(np.abs(q1 - q0).max() / JOINT_SPEED * FPS))
    return list(np.linspace(q0, q1, n)[1:])


def unplug_traj(q_in, q_start) -> np.ndarray:
    """拔出关节轨迹 [N, 6]：q_in 竖直上拔 LIFT（姿态不变）→ 随机中间点 → q_start。"""
    p0, R0 = IK.fk(q_in)
    qs = [np.asarray(q_in, float)]
    for s in np.linspace(0, LIFT, round(LIFT / LIFT_SPEED * FPS) + 1)[1:]:
        q, err = IK.solve(p0 + [0, 0, s], None, qs[-1], rot=R0)
        if err > 1e-3:
            raise ValueError(f"上拔 {s * 100:.1f} cm 处解不出（误差 {err * 1000:.1f} mm）")
        qs.append(q)
    via = p0 + [*rng.uniform(-VIA_XY, VIA_XY, 2), LIFT + rng.uniform(*VIA_Z)]
    q_via, err = IK.solve(via, None, qs[-1], rot=R0)
    if err > 1e-3:
        raise ValueError(f"随机中间点 {via.round(3)} 解不出")
    return np.array(qs + _line(qs[-1], q_via) + _line(q_via, q_start))


def make_traj(arm, spec) -> np.ndarray:
    """拖到插好的位姿 → 返回插入轨迹（拔出轨迹倒过来）。"""
    start = arm.read_joints()   # 每条开始时机械臂停在起始位姿
    while True:
        input(f"{spec.setup_prompt}，托住机械臂，回车进入拖拽…")
        arm.drag_mode()
        x = None
        try:
            while x is None:
                input(f"{spec.teach_prompt}后回车…")
                x = arm.read_leader_state()
        finally:
            arm.exit_drag_mode()
        try:
            traj = unplug_traj(x, start)[::-1]
            break
        except ValueError as e:
            print(f"⚠ {e}：换个插入姿态，重新拖")
    print(f"插入轨迹 {len(traj)} 帧（{len(traj) / FPS:.1f}s）")
    return np.array([clip_target(xk) for xk in traj])


if __name__ == "__main__":
    main(SPEC, make_traj)
