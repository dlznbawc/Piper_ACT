#!/usr/bin/env python3
"""测手机坐标轴：手机分别往前、往左、往上移动一次，看手机报告的位移落在哪个轴上、正负号如何。

用来确定"手机移动方向 → 机械臂移动方向"的对应关系（phone.py 里的 AXES）。在仓库根目录运行：
    python3 tasks/sim_pick_place/check_axes.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from piper_act.sim.phone import connect_iphone

MOVES = [("前（远离你）", "x"), ("左", "y"), ("上（抬高）", "z")]


def main() -> None:
    phone = connect_iphone()
    group = phone._group
    state: dict[int, int] = {}

    def read() -> np.ndarray:
        """读一次手机，更新按键状态（包里有才更新），返回标定坐标系下的手机位置。"""
        f = group.get_next_feedback()
        io = f[0].io
        for ch in (1, 2):
            if io.b.has_int(ch):
                state[ch] = io.b.get_int(ch)
        p = np.array(f[0].ar_position)
        return phone._calib_rot_inv.apply(p)

    def wait_b2() -> np.ndarray:
        """等「保存」(B2) 点一下，返回那一刻的位置。"""
        while state.get(2, 0):   # 先等松开
            read()
        while not state.get(2, 0):
            p = read()
        return p

    print("\n全程保持：屏幕朝上、顶边对着机械臂正前方，只平移、不要转动手机。每一步用手机上的「保存」(B2) 确认。")
    result = {}
    for name, axis in MOVES:
        print(f"\n【{name}】手机放在舒服的位置，点一下「保存」…", flush=True)
        p0 = wait_b2()
        print(f"   现在把手机往【{name}】平移约 20 厘米，停住后再点一下「保存」…", flush=True)
        p1 = wait_b2()
        d = p1 - p0
        k = int(np.argmax(np.abs(d)))
        result[axis] = {"phone_axis": k, "sign": int(np.sign(d[k])), "delta_cm": np.round(d * 100, 1).tolist()}
        print(f"   手机报告的位移（厘米）：x={d[0]*100:.1f}  y={d[1]*100:.1f}  z={d[2]*100:.1f}"
              f" → 主要在手机第 {k} 轴（{'xyz'[k]}），符号 {'+' if d[k] > 0 else '-'}", flush=True)
    print("\n结果：", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
