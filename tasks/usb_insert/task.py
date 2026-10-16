"""USB 插入：夹爪一直夹着 USB 公头，插进母头。夹爪不受控（一直夹紧），状态/动作只有 6 维关节。"""
from pathlib import Path

from piper_act.task import TaskSpec

SPEC = TaskSpec(
    name="usb_insert",
    dir=Path(__file__).resolve().parent,
    task="insert the USB plug into the socket",
    gripper=False,
    setup_prompt="挪好母头位置",
    teach_prompt="拖着机械臂把夹着的插头插进母头、插到底，停稳",
    hold_s=2.0,
    settle_s=1.0,
)
