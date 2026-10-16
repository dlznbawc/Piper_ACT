"""充电口对准：手腕相机看到充电座，机械臂运动到充电座正前方的对准位姿并停住（不含插入）。"""
from pathlib import Path

from piper_act.task import TaskSpec

SPEC = TaskSpec(
    name="charge_align",
    dir=Path(__file__).resolve().parent,
    task="move to the pre-insertion pose in front of the charging socket",
    gripper=False,
    setup_prompt="充电座放到新位置（起始位姿下手腕相机要能看到它）",
    teach_prompt="把机械臂拖到充电座正前方的对准位姿、停稳",
    hold_s=2.0,
    settle_s=1.0,
)
