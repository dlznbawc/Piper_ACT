#!/usr/bin/env python3
"""真机运行策略。流程见 piper_act/deploy.py；在仓库根目录运行：
    python3 tasks/pick_place/deploy.py --policy outputs/train/act_pick_place/checkpoints/last/pretrained_model
"""
from task import SPEC

from piper_act.deploy import main

main(SPEC)
