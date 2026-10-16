"""TaskSpec：一个任务的全部设定。每个 tasks/<任务>/task.py 填一份，采集和部署共用。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import JOINT_NAMES


@dataclass(frozen=True)
class TaskSpec:
    name: str               # 任务名 = 数据集名（datasets/<name>）= 训练输出名（outputs/train/act_<name>）
    dir: Path               # 任务文件夹 tasks/<name>/
    task: str               # 写进数据集的任务描述
    gripper: bool           # True：状态/动作 7 维（6 关节 + 夹爪开度 m）
    setup_prompt: str       # 每条示范 / 每轮部署前，提示人摆好场景
    teach_prompt: str       # 拖拽示教时的提示
    grip_force: float = 1.0  # N，回放和部署时的夹持力
    hold_s: float = 2.0     # 采集：到终点后原地继续录制的秒数（让策略学会停）
    settle_s: float = 1.0   # 部署：指令持续这么久几乎不变就算停稳、本轮结束

    @property
    def start_pose(self) -> Path:
        return self.dir / "start_pose.json"

    @property
    def dataset(self) -> Path:
        return self.dir.parents[1] / "datasets" / self.name

    @property
    def names(self) -> list[str]:
        return list(JOINT_NAMES) + (["gripper"] if self.gripper else [])
