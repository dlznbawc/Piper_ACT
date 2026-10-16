"""抓取放置：手腕相机看到物体，机械臂抓起物体放到目标位置。"""
from pathlib import Path

from piper_act.task import TaskSpec

SPEC = TaskSpec(
    name="pick_place",
    dir=Path(__file__).resolve().parent,
    task="pick up the object and place it at the target",   # 按实际物体改，如 "pick up the red cube ..."
    gripper=True,
    setup_prompt="物体放到要抓的位置",
    teach_prompt="拖着机械臂完成抓取和放置，手动开合夹爪；放完后把机械臂抬离物体、停稳",
    grip_force=1.0,   # N，夹不稳就调大
    hold_s=3.0,       # 比 charge_align 长：中途夹爪开合时会短暂停顿，别被当成结束
    settle_s=2.0,
)
