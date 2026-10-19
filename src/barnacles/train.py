"""Thin Ultralytics training wrapper that records everything needed to audit a run.

Every run starts from the original COCO-pretrained checkpoint, refuses to overwrite an
existing run directory and writes run_record.json next to Ultralytics' outputs.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import platform
import random
import shutil
import statistics
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .imaging import sha256_file

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
NO_AUGMENT = {"fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
              "perspective": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "mosaic": 0.0, "mixup": 0.0,
              "copy_paste": 0.0}


# ---- validation ----------------------------------------------------------------------

def check_train_args(args: dict) -> dict:
    from .yolo_env import yolo_class
    yolo_class()
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT
    unknown = sorted(set(args) - set(DEFAULT_CFG_DICT))
    if unknown:
        raise ValueError(f"unknown training arguments for ultralytics {ultralytics.__version__}: {unknown}")
    return args


def check_labels(dataset_yaml) -> dict:
    """Validate every YOLO label file referenced by a dataset.yaml; raise ValueError on the first problem."""
    d = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))
    root, out = Path(d["path"]), {}
    for split in ("train", "val"):
        img_dir = root / d[split]
        lab_dir = root / str(d[split]).replace("images", "labels", 1)
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        n_inst = 0
        for img in images:
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                raise ValueError(f"missing label file for {img.name}")
            for ln, line in enumerate(lab.read_text(encoding="utf-8").splitlines(), 1):
                vals = line.split()
                where = f"{lab.name}:{ln}"
                if len(vals) != 5:
                    raise ValueError(f"{where}: expected 5 values (class cx cy w h), got {len(vals)}")
                c, *xywh = (float(v) for v in vals)
                if not all(math.isfinite(v) for v in (c, *xywh)):
                    raise ValueError(f"{where}: non-finite value")
                if c != 0:
                    raise ValueError(f"{where}: class {c:g} but this is a one-class dataset (class 0)")
                if xywh[2] <= 0 or xywh[3] <= 0:
                    raise ValueError(f"{where}: non-positive box size")
                if not all(0.0 <= v <= 1.0 for v in xywh):
                    raise ValueError(f"{where}: normalised coordinate out of range [0, 1]")
                n_inst += 1
        if not images:
            raise ValueError(f"no images in {img_dir}")
        out[split] = {"images": len(images), "instances": n_inst}
    return out


# ---- environment and records ------------------------------------------------------

def env_record() -> dict:
    import cv2
    import PIL
    import torch
    import torchvision
    import ultralytics
    rec = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
           "torch": torch.__version__, "torchvision": torchvision.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
           "ultralytics": ultralytics.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
           "pillow": PIL.__version__, "torch_num_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec |= {"gpu": torch.cuda.get_device_name(0), "gpu_memory_gib": round(p.total_memory / 2 ** 30, 2)}
    return rec


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_weights(cfg) -> dict:
    """Record (first call) or verify (later calls) the pretrained checkpoint hash."""
    path = cfg["paths"]["pretrained_weights"]
    if not path.exists():
        raise FileNotFoundError(f"pretrained weights not found at {path}; download {WEIGHTS_URL} "
                                "(training from scratch is not a substitute)")
    rec_path = cfg["paths"]["manifests_dir"] / "weights.json"
    sha = sha256_file(path)
    if rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec["sha256"] != sha:
            raise ValueError(f"pretrained weights changed: {sha} != recorded {rec['sha256']}")
        return rec
    rec = {"file": path.relative_to(cfg["root"]).as_posix(), "source_url": WEIGHTS_URL, "sha256": sha,
           "bytes": path.stat().st_size, "description": "COCO-pretrained YOLO11n detection checkpoint (official Ultralytics asset)"}
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def read_results_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _load_ckpt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _ckpt_info(path: Path) -> dict:
    """Ultralytics strips final checkpoints (epoch=-1, no best_fitness); the embedded results table
    still records how many epochs had completed when the file was written."""
    ck = _load_ckpt(path)
    res = ck.get("train_results") or {}
    n = len(res.get("epoch", [])) if isinstance(res, dict) else None
    metrics = ck.get("train_metrics") or {}
    return {"path": str(path), "sha256": sha256_file(path), "epochs_in_results": n,
            "fitness_at_save": None if metrics.get("fitness") is None else float(metrics["fitness"])}


def _weights(ck: dict):
    return ck.get("ema") or ck.get("model")


# ---- training -----------------------------------------------------------------------------

def train_fold(dataset_yaml, train_args: dict, run_dir, pretrained, init_checkpoint: bool = False,
               extra: dict | None = None) -> dict:
    """Fine-tune the pretrained detector on one dataset.yaml; return and save the run record."""
    import torch
    pretrained, run_dir = Path(pretrained), Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; evaluated runs are never overwritten")
    if not pretrained.exists():
        raise FileNotFoundError(f"pretrained weights not found: {pretrained} (refusing to train from scratch)")
    check_train_args(train_args)
    labels = check_labels(dataset_yaml)
    from .yolo_env import device_string, yolo_class
    YOLO = yolo_class()
    from ultralytics.utils import LOGGER

    seed_everything(int(train_args.get("seed", 42)))
    device = device_string()
    args = {**train_args, "data": str(dataset_yaml), "project": str(run_dir.parent), "name": run_dir.name,
            "exist_ok": False, "device": device}
    if device == "cpu":
        args["amp"] = False
    state = {"steps": 0, "finite_steps": 0, "nonfinite_steps": 0, "first_grad_norm": None, "trainer": {},
             "best_written_epoch": None, "epoch_fitness": [], "amp_skipped_steps": 0}

    def on_model_save(trainer):
        # mirrors BaseTrainer.save_model: best.pt is (re)written when best_fitness == fitness
        fit = None if trainer.fitness is None else float(trainer.fitness)
        state["epoch_fitness"].append([trainer.epoch + 1, fit])
        if trainer.best_fitness is not None and trainer.best_fitness == trainer.fitness:
            state["best_written_epoch"] = trainer.epoch + 1

    def on_train_start(trainer):
        orig = trainer.optimizer_step

        def counted_step():
            # grads are still loss-scaled here; GradScaler skips the update when they are inf/NaN
            grads = [p.grad.detach().float().norm() for p in trainer.model.parameters() if p.grad is not None]
            norm = float(torch.stack(grads).norm()) if grads else float("nan")
            state["finite_steps" if math.isfinite(norm) and norm > 0 else "nonfinite_steps"] += 1
            if state["first_grad_norm"] is None:
                state["first_grad_norm"] = norm
            before = trainer.scaler.get_scale() if trainer.scaler.is_enabled() else None
            state["steps"] += 1
            orig()
            if before is not None and trainer.scaler.get_scale() < before:
                state["amp_skipped_steps"] += 1

        trainer.optimizer_step = counted_step
        nb = len(trainer.train_loader)
        state["trainer"] = {"batch_size": trainer.batch_size, "batches_per_epoch": nb,
                            "accumulate": trainer.accumulate, "warmup_iterations": trainer._get_warmup_iterations(nb),
                            "train_images": len(trainer.train_loader.dataset), "amp": bool(trainer.amp)}
        if init_checkpoint:
            init = Path(trainer.wdir) / "init_untrained.pt"
            init.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": -1, "best_fitness": None, "model": None, "ema": deepcopy(trainer.ema.ema).half(),
                        "updates": 0, "train_args": vars(trainer.args)}, init)

    model = YOLO(str(pretrained))
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_model_save", on_model_save)
    cap = _LogCapture()
    LOGGER.addHandler(cap)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.train(**args)
    finally:
        LOGGER.removeHandler(cap)
    elapsed = time.perf_counter() - t0
    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    if save_dir.resolve() != run_dir.resolve():
        raise RuntimeError(f"ultralytics wrote to {save_dir}, expected {run_dir}")
    rows = read_results_csv(save_dir / "results.csv")
    loss_keys = [k for k in (rows[-1] if rows else {}) if "loss" in k]
    losses_finite = bool(rows) and all(math.isfinite(r[k]) for r in rows for k in loss_keys)
    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
    ck_best, ck_last = _ckpt_info(best), _ckpt_info(last)

    # parameters moved away from the pretrained values? (first backbone convolution)
    pre_sd = _weights(_load_ckpt(pretrained)).float().state_dict()
    last_sd = _weights(_load_ckpt(last)).float().state_dict()
    key = next(k for k in pre_sd if k.endswith("conv.weight"))
    changed = not torch.allclose(pre_sd[key], last_sd[key])

    reloaded = YOLO(str(last))
    val_dir = Path(yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))["path"]) / "images" / "val"
    sample = sorted(val_dir.iterdir())[0]
    pred = reloaded.predict(str(sample), imgsz=args.get("imgsz", 640), device=device, verbose=False, max_det=2000)
    reload_ok = all(int(c) == 0 for c in pred[0].boxes.cls.tolist())

    crit = getattr(unwrap(trainer.model), "criterion", None)
    nondeterministic = sorted({str(w.message)[:300] for w in caught if "determinist" in str(w.message).lower()})
    record = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir), "dataset_yaml": str(dataset_yaml), "label_check": labels,
        "pretrained": {"path": str(pretrained), "sha256": sha256_file(pretrained)},
        "requested_args": train_args, "resolved_args": {k: (str(v) if isinstance(v, Path) else v)
                                                        for k, v in vars(trainer.args).items()},
        "device": device, "trainer": state["trainer"], "optimizer_steps": state["steps"],
        "grad_checks": {"finite_steps": state["finite_steps"], "nonfinite_or_zero_steps": state["nonfinite_steps"],
                        "first_grad_norm_scaled": state["first_grad_norm"],
                        "amp_skipped_steps": state["amp_skipped_steps"],
                        "applied_updates": state["steps"] - state["amp_skipped_steps"]},
        "epochs_completed": len(rows), "elapsed_s": elapsed, "losses_finite": losses_finite,
        "final_losses": {k: rows[-1][k] for k in loss_keys} if rows else {},
        "fitness_definition": "ultralytics 8.4.182 DetMetrics fitness = mAP50-95 on the val split (weights [0,0,0,1])",
        "best_epoch": state["best_written_epoch"], "epoch_fitness": state["epoch_fitness"],
        "best_epoch_crosscheck_ok": state["best_written_epoch"] is not None and ck_best["fitness_at_save"] is not None
        and abs(dict(state["epoch_fitness"])[state["best_written_epoch"]] - ck_best["fitness_at_save"]) < 1e-4,
        "checkpoints": {"best": ck_best, "last": ck_last},
        "criterion": {"class": type(crit).__name__ if crit is not None else None,
                      "box_gain": trainer.args.box, "cls_gain": trainer.args.cls, "dfl_gain": trainer.args.dfl},
        "parameters_changed": bool(changed), "model_names": dict(reloaded.names), "reload_ok": reload_ok,
        "warnings": sorted(set(cap.messages))[:50], "nondeterministic_ops": nondeterministic,
        "environment": env_record(), **(extra or {}),
    }
    if init_checkpoint:
        record["init_checkpoint"] = str(Path(trainer.wdir) / "init_untrained.pt")
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return record


def unwrap(model):
    return getattr(model, "module", model)


# ---- commands ------------------------------------------------------------------------------

def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _subset_dataset(cfg, name: str, train_tiles: list[Path], val_tiles: list[Path]) -> Path:
    out = cfg["paths"]["processed_dir"] / name
    if out.exists():
        shutil.rmtree(out)
    for split, tiles in (("train", train_tiles), ("val", val_tiles)):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for t in tiles:
            shutil.copy(t, out / "images" / split / t.name)
            shutil.copy(t.parent.parent.parent / "labels" / t.parent.name / f"{t.stem}.txt",
                        out / "labels" / split / f"{t.stem}.txt")
    y = out / "dataset.yaml"
    y.write_text(yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val",
                                 "names": dict(enumerate(cfg["class_names"]))}, allow_unicode=True), encoding="utf-8")
    return y


def _fold_manifest(cfg, fold: str) -> dict:
    p = cfg["paths"]["manifests_dir"] / f"fold_{fold}" / "split_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) | {"_sha256": sha256_file(p), "_path": str(p)}


def smoke_train(cfg) -> dict:
    verify_weights(cfg)
    m = _fold_manifest(cfg, "A")
    base = cfg["paths"]["processed_dir"] / "fold_A"
    tr = [base / t["image_file"] for t in m["tiles"] if t["split"] == "train"][:cfg["smoke"]["max_tiles"]]
    va = [base / t["image_file"] for t in m["tiles"] if t["split"] == "val"][:2]
    y = _subset_dataset(cfg, "smoke", tr, va)
    # fp32 so the single smoke update is observable (AMP skips its first calibration steps);
    # the AMP path itself is exercised and its skipped steps recorded in every real run
    args = {**cfg["train"], "epochs": cfg["smoke"]["epochs"], "amp": False}
    run = cfg["paths"]["runs_dir"] / cfg["experiment_id"] / "diagnostics" / f"smoke_{_stamp()}"
    rec = train_fold(y, args, run, cfg["paths"]["pretrained_weights"],
                     extra={"purpose": "smoke test only; never used for evaluation", "provisional": m["provisional"]})
    ok = rec["losses_finite"] and rec["optimizer_steps"] >= 1 and rec["reload_ok"] and rec["parameters_changed"] \
        and rec["model_names"] == {0: cfg["class_names"][0]}
    print(f"smoke: {'PASS' if ok else 'FAIL'} steps={rec['optimizer_steps']} losses_finite={rec['losses_finite']} "
          f"grad finite steps={rec['grad_checks']['finite_steps']} reload_ok={rec['reload_ok']} -> {run}")
    return rec


def _tile_f1(checkpoint, tiles: list[Path], cfg, conf=0.25) -> dict:
    """Detection quality on training tiles (used only for the overfit diagnostic)."""
    from .geometry import yolo_to_box
    from .imaging import load_rgb
    from .metrics import match_boxes, prf
    from .predict import YoloDetector
    det = YoloDetector(checkpoint, cfg["inference"])
    tp = fp = fn = 0
    per = []
    for t in tiles:
        lab = t.parent.parent.parent / "labels" / t.parent.name / f"{t.stem}.txt"
        gt = np.array([yolo_to_box(list(map(float, ln.split()))[1:], 640)
                       for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]).reshape(-1, 4)
        d = det([load_rgb(t)])[0]
        d = d[d[:, 4] >= conf]
        r = match_boxes(d[:, :4], d[:, 4], gt, 0.5)
        tp, fp, fn = tp + r.tp, fp + r.fp, fn + r.fn
        per.append({"tile": t.name, "gt": len(gt), "pred": int(len(d)), "tp": r.tp})
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn), "tiles": per}


def overfit_check(cfg) -> dict:
    verify_weights(cfg)
    m = _fold_manifest(cfg, "A")
    base = cfg["paths"]["processed_dir"] / "fold_A"
    nonempty = sorted((t for t in m["tiles"] if t["split"] == "train" and t["object_ids"]),
                      key=lambda t: t["image_file"])[:cfg["overfit"]["max_tiles"]]
    tiles = [base / t["image_file"] for t in nonempty]
    y = _subset_dataset(cfg, "overfit", tiles, tiles)
    n = len(tiles)
    args = {**cfg["train"], **NO_AUGMENT, "batch": n, "nbs": n, "epochs": cfg["overfit"]["max_updates"],
            "patience": 0, "val": False, "plots": False}
    run = cfg["paths"]["runs_dir"] / cfg["experiment_id"] / "diagnostics" / f"overfit_{_stamp()}"
    rec = train_fold(y, args, run, cfg["paths"]["pretrained_weights"], init_checkpoint=True,
                     extra={"purpose": "overfit diagnostic on training tiles; never used for evaluation"})
    before = _tile_f1(rec["init_checkpoint"], tiles, cfg)
    after = _tile_f1(rec["checkpoints"]["last"]["path"], tiles, cfg)
    rule = "PASS if training-tile F1@IoU0.5 (conf 0.25) after <= max_updates updates exceeds the untrained-head F1 by >= 0.30"
    passed = rec["optimizer_steps"] <= cfg["overfit"]["max_updates"] and after["f1"] - before["f1"] >= 0.30
    out = {"rule": rule, "passed": passed, "optimizer_steps": rec["optimizer_steps"], "before": before,
           "after": after, "run_dir": str(run), "lr0": args["lr0"], "tiles": [t.name for t in tiles]}
    (run / "overfit_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"overfit: {'PASS' if passed else 'FAIL'} after {rec['optimizer_steps']} updates: "
          f"F1 {before['f1']:.3f} -> {after['f1']:.3f} (P {after['precision']:.3f} R {after['recall']:.3f}) -> {run}")
    return out


def train_outer_fold(cfg, fold: str) -> dict:
    weights = verify_weights(cfg)
    m = _fold_manifest(cfg, fold)
    y = Path(cfg["root"]) / m["dataset_yaml"]
    run = cfg["paths"]["runs_dir"] / cfg["experiment_id"] / f"fold_{fold}"
    rec = train_fold(y, dict(cfg["train"]), run, cfg["paths"]["pretrained_weights"], extra={
        "fold": fold, "dev_source": m["dev_source"], "heldout_source": m["heldout_source"],
        "split_manifest": m["_path"], "split_manifest_sha256": m["_sha256"],
        "annotation_sha256": m["annotation_sha256"], "review_status": m["review_status"],
        "provisional": m["provisional"], "pretrained_sha256": weights["sha256"],
        "initialisation": "original COCO-pretrained yolo11n.pt (not another fold's weights)",
        "config_sha256": sha256_file(cfg["config_path"]),
        "train_objects": len(m["train_object_ids"]), "train_tiles": m["counts"]["train"]["tiles"]})
    print(f"fold {fold}: {rec['epochs_completed']} epochs, {rec['optimizer_steps']} optimizer steps, best epoch "
          f"{rec['best_epoch']} (inner-val mAP50-95 {rec['checkpoints']['best']['fitness_at_save']:.4f}), "
          f"{rec['elapsed_s']:.0f}s -> {run}")
    return rec


def train_final(cfg) -> dict:
    """Fixed-epoch refit on both photographs after the outer folds are frozen (plan section 15)."""
    weights = verify_weights(cfg)
    exp = cfg["paths"]["runs_dir"] / cfg["experiment_id"]
    best_epochs, thresholds = [], []
    for fold in cfg["folds"]:
        ev = exp / f"fold_{fold}" / "evaluation" / "metrics.json"
        if not ev.exists():
            raise SystemExit(f"fold {fold} has no frozen held-out evaluation ({ev}); run evaluate first")
        rec = json.loads((exp / f"fold_{fold}" / "run_record.json").read_text(encoding="utf-8"))
        sel = json.loads((exp / f"fold_{fold}" / "threshold_selection.json").read_text(encoding="utf-8"))
        best_epochs.append(rec["best_epoch"])
        thresholds.append(sel["selected_confidence"])
    epochs = int(math.floor(statistics.median(best_epochs) + 0.5))
    conf = float(statistics.median(thresholds))
    from .dataset import build_final
    build_final(cfg, allow_provisional=True)
    y = cfg["paths"]["processed_dir"] / "final" / "dataset.yaml"
    args = {**cfg["train"], "epochs": epochs, "patience": 0, "val": False}
    run = exp / cfg["final"]["run_name"]
    rec = train_fold(y, args, run, cfg["paths"]["pretrained_weights"], extra={
        "purpose": "final train-all demonstration model; img1/img2 scores from it are NOT test results",
        "fixed_epochs_rule": "round-half-up median of the two outer-fold best epochs",
        "outer_fold_best_epochs": best_epochs, "pretrained_sha256": weights["sha256"],
        "validation_note": "val split = training tiles; any val metric is resubstitution only; no early stopping"})
    if rec["epochs_completed"] != epochs or rec["checkpoints"]["last"]["epochs_in_results"] != epochs:
        raise RuntimeError(f"final last.pt reflects {rec['checkpoints']['last']['epochs_in_results']} epochs, "
                           f"expected {epochs}")
    deploy = {"checkpoint": rec["checkpoints"]["last"]["path"], "checkpoint_sha256": rec["checkpoints"]["last"]["sha256"],
              "confidence": conf, "confidence_rule": "median of the two inner-validation selected thresholds "
              "(provisional heuristic; refitting changes score calibration)", "outer_fold_thresholds": thresholds,
              "epochs": epochs, "inference": cfg["inference"], "tiles": cfg["tiles"]}
    (run / "deployment.json").write_text(json.dumps(deploy, indent=2) + "\n", encoding="utf-8")
    print(f"final: {epochs} epochs ({rec['optimizer_steps']} steps), deployment confidence {conf} -> {run}")
    return rec
