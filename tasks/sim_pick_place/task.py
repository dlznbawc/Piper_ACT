"""仿真抓放：MuJoCo 里用 iPhone 遥操作 PiPER X，把红色方块放到蓝色区域。"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

NAME = "sim_pick_place"
TASK = "pick up the red cube and place it on the blue square"
DATASET = REPO / "datasets" / NAME
