"""手腕相机：奥比中光 DaBai 彩色流（UVC）。取帧用 LeRobot 的 OpenCVCamera（后台线程持续取帧）。"""
from __future__ import annotations

import sys
from pathlib import Path

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from .config import IMG_H, IMG_W


def find_camera() -> str:
    """DaBai 会注册多个 /dev/video*，取 index 0 的那个（彩色流）。"""
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        if "DaBai" in (d / "name").read_text() and (d / "index").read_text().strip() == "0":
            return f"/dev/{d.name}"
    sys.exit("❌ 没找到奥比中光 DaBai 相机，检查 USB 连接")


def open_camera(dev: str | None = None) -> OpenCVCamera:
    """连接相机（默认自动找 DaBai），返回已在后台取帧的 OpenCVCamera。

    取最新一帧：cam.read_latest() → RGB, HxWx3 uint8；超过 0.5 秒没出新画面抛 TimeoutError。
    用完 cam.disconnect()。分辨率设不成 IMG_W x IMG_H 时 connect 会报错。
    """
    cfg = OpenCVCameraConfig(index_or_path=dev or find_camera(), width=IMG_W, height=IMG_H,
                             backend=Cv2Backends.V4L2)
    cam = OpenCVCamera(cfg)
    cam.connect()
    return cam
