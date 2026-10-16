"""各任务共用的常量（硬件相关）。任务自己的设定放在 tasks/<任务>/task.py。"""
from __future__ import annotations

import math

FPS = 25                        # 控制 / 录制频率（LeRobot 要求整数 fps）
IMG_H, IMG_W = 480, 640         # 奥比中光 DaBai 彩色流

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
# PiPER X 关节限位（rad），取自官方 URDF agilexrobotics/agx_arm_urdf piper_x/
JOINT_LIMITS_RAD = (
    (-2.6179938, 2.6179938),    # J1 ≈ ±150°
    (0.0, math.pi),             # J2 ≈ 0~180°
    (-2.9670597, 0.0),          # J3 ≈ -170°~0°
    (-1.553343, 1.553343),      # J4 ≈ ±89°
    (-1.553343, 1.553343),      # J5 ≈ ±89°
    (-2.0943951, 2.0943951),    # J6 ≈ ±120°
)

APPROACH_SPEED = 20             # 回起始位姿、回零位等非示教运动的速度（%）
REACHED = 0.02                  # rad，各关节与目标差都小于它算到位
