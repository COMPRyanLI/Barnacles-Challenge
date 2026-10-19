import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from barnacles import train as T

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolo11n.pt"


def make_dataset(tmp_path, label_text="0 0.5 0.5 0.05 0.05\n", n=2):
    d = tmp_path / "数据集"
    for split in ("train", "val"):
        (d / "images" / split).mkdir(parents=True)
        (d / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = np.full((640, 640, 3), 60, np.uint8)
            for k in range(12):
                x, y = 40 + 50 * k, 40 + 45 * k
                img[y:y + 30, x:x + 30] = 230
                img[y + 10:y + 20, x + 10:x + 20] = 30
            Image.fromarray(img).save(d / "images" / split / f"t{i}.png")
            lines = label_text if label_text is not None else "".join(
                f"0 {(40 + 50 * k + 15) / 640:.6f} {(40 + 45 * k + 15) / 640:.6f} {30 / 640:.6f} {30 / 640:.6f}\n"
                for k in range(12))
            (d / "labels" / split / f"t{i}.txt").write_text(lines, encoding="utf-8")
    y = d / "dataset.yaml"
    y.write_text(yaml.safe_dump({"path": str(d), "train": "images/train", "val": "images/val",
                                 "names": {0: "barnacle_target"}}, allow_unicode=True), encoding="utf-8")
    return y


def test_unknown_training_argument_rejected():
    with pytest.raises(ValueError, match="unknown"):
        T.check_train_args({"epochs": 1, "not_an_ultralytics_arg": 3})


def test_known_training_arguments_accepted():
    T.check_train_args({"epochs": 1, "lr0": 1e-4, "mosaic": 0.0, "max_det": 2000})


@pytest.mark.parametrize("text,msg", [
    ("1 0.5 0.5 0.1 0.1\n", "class"),
    ("0 0.5 0.5 0.0 0.1\n", "size"),
    ("0 1.5 0.5 0.1 0.1\n", "range"),
    ("0 nan 0.5 0.1 0.1\n", "finite"),
    ("0 0.5 0.5 0.1\n", "5 values"),
])
def test_invalid_labels_fail_clearly(tmp_path, text, msg):
    with pytest.raises(ValueError, match=msg):
        T.check_labels(make_dataset(tmp_path, text))


def test_valid_labels_counted(tmp_path):
    info = T.check_labels(make_dataset(tmp_path, None))
    assert info["train"]["instances"] == 24 and info["val"]["images"] == 2


def test_missing_pretrained_weights_fail_before_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="pretrained"):
        T.train_fold(make_dataset(tmp_path, None), {"epochs": 1}, tmp_path / "run", tmp_path / "missing.pt")


def test_existing_run_dir_is_never_overwritten(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(FileExistsError):  # must not depend on the git-ignored weights file
        T.train_fold(make_dataset(tmp_path, None), {"epochs": 1}, run, tmp_path / "no_weights.pt")


@pytest.mark.training
@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights/yolo11n.pt not downloaded")
def test_one_epoch_training_round_trip(tmp_path):
    y = make_dataset(tmp_path, None)
    args = {"epochs": 1, "imgsz": 640, "batch": 2, "nbs": 2, "workers": 0, "mosaic": 0.0, "plots": False,
            "seed": 42, "deterministic": True, "amp": False, "val": True, "optimizer": "AdamW", "lr0": 1e-4}
    # fp32: under AMP the GradScaler skips the first few calibration steps, so one step would not update
    rec = T.train_fold(y, args, tmp_path / "runs" / "smoke", WEIGHTS, init_checkpoint=True)
    assert rec["optimizer_steps"] >= 1
    assert rec["losses_finite"] and rec["grad_checks"]["finite_steps"] >= 1
    assert rec["grad_checks"]["applied_updates"] >= 1
    assert rec["best_epoch"] == 1 and rec["best_epoch_crosscheck_ok"]
    assert rec["parameters_changed"]
    assert Path(rec["checkpoints"]["last"]["path"]).exists() and Path(rec["checkpoints"]["best"]["path"]).exists()
    assert rec["model_names"] == {0: "barnacle_target"}
    assert rec["reload_ok"]
    assert all(math.isfinite(v) for v in rec["final_losses"].values())
