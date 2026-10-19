"""Unicode-safe image I/O and hashing (Pillow only; no cv2 path handling)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB. Alpha is dropped, never used as a channel."""
    with Image.open(Path(path)) as im:
        return np.array(im.convert("RGB"))


def load_rgba(path) -> np.ndarray:
    with Image.open(Path(path)) as im:
        return np.array(im.convert("RGBA"))


def image_size(path) -> tuple[int, int]:
    with Image.open(Path(path)) as im:
        return im.width, im.height


def save_image(array: np.ndarray, path, quality: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(array)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=quality)
    else:
        im.save(path)
    return path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
