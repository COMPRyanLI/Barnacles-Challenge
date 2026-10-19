"""Experiment configuration and ROI loading."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .geometry import validate_polygon


def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError(f"no pyproject.toml above {start}")


def load_config(path, root: Path | None = None) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(root).resolve() if root else find_root(path.parent)
    cfg["root"] = root
    cfg["config_path"] = path
    cfg["paths"] = {k: (root / v) for k, v in cfg["paths"].items()}
    raw = cfg["paths"]["raw_dir"]
    cfg["source_paths"] = {}
    for sid, s in cfg["sources"].items():
        entry = {"image": raw / s["image"], "role": s["role"]}
        if s.get("mask"):
            entry["mask"] = raw / s["mask"]
        cfg["source_paths"][sid] = entry
    for name, fold in cfg["folds"].items():
        for key in ("dev", "heldout"):
            if cfg["sources"].get(fold[key], {}).get("role") != "labeled":
                raise ValueError(f"fold {name} {key} source {fold[key]!r} must be a labeled source")
        if fold["dev"] == fold["heldout"]:
            raise ValueError(f"fold {name}: development and held-out sources must differ")
    return cfg


def load_rois(path) -> dict:
    """image_id -> validated polygon (list of [x, y]) or None for whole-image regions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for sid, r in data["rois"].items():
        out[sid] = None if r.get("whole_image") else validate_polygon(r["polygon"]).tolist()
    return out
