#!/usr/bin/env python3
"""iPhone 遥操作真机采集 → datasets/pick_place/。流程见 piper_act/phone_record.py；在仓库根目录运行：python3 tasks/pick_place/record_phone.py"""
from task import SPEC

from piper_act.phone_record import main

main(SPEC)
