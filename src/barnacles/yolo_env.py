"""Import Ultralytics with project-local settings and analytics disabled."""
from __future__ import annotations

import os
from pathlib import Path

from .config import find_root


def yolo_class():
    root = find_root(Path(__file__).resolve().parent)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".ultralytics"))
    from ultralytics import YOLO, settings
    if settings.get("sync"):
        settings.update({"sync": False})  # no usage analytics
    return YOLO


def device_string() -> str:
    import torch
    return "0" if torch.cuda.is_available() else "cpu"
