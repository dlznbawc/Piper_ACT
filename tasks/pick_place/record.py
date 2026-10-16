#!/usr/bin/env python3
"""采集示范 → datasets/pick_place/。流程见 piper_act/record.py；在仓库根目录运行：python3 tasks/pick_place/record.py"""
from task import SPEC

from piper_act.record import main

main(SPEC)
