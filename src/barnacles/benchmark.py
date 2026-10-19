"""Complete-image latency: disk load -> region -> tiles -> inference -> restore -> merge -> count.

File writing is excluded from the primary latency and timed separately. Model loading is
reported on its own. Warm-up runs precede timed runs; CUDA is synchronised around timed work.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from .imaging import save_image


def _stats(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"median_ms": float(np.median(a) * 1e3), "p90_ms": float(np.percentile(a, 90) * 1e3),
            "mean_ms": float(a.mean() * 1e3), "min_ms": float(a.min() * 1e3), "max_ms": float(a.max() * 1e3),
            "n": int(len(a))}


def _sync(device: str) -> None:
    if device != "cpu":
        import torch
        torch.cuda.synchronize()


def benchmark_region(image_path, roi, checkpoint, infer_cfg: dict, tile_cfg: dict, warmup: int = 3, runs: int = 10,
                     device: str | None = None, single_pass: bool = False, detector=None) -> dict:
    import torch
    from .dataset import make_region
    from .imaging import load_rgb
    from .predict import YoloDetector, predict_region, predict_single_pass
    t0 = time.perf_counter()
    det = detector if detector is not None else YoloDetector(checkpoint, infer_cfg, device=device,
                                                             imgsz=tile_cfg["size"])
    load_s = time.perf_counter() - t0

    def once():
        if single_pass:
            img = load_rgb(image_path)
            region = make_region(Path(image_path).stem, roi, img.shape[1], img.shape[0])
            return predict_single_pass(img, region, det, infer_cfg, tile_cfg["pad_value"])
        return predict_region(image_path, roi, det, tile_cfg, infer_cfg)

    cold_start = time.perf_counter()
    once()
    _sync(det.device)
    cold_s = time.perf_counter() - cold_start
    for _ in range(max(warmup - 1, 0)):
        once()
    if det.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    totals, stages, res = [], {}, None
    for _ in range(runs):
        _sync(det.device)
        s = time.perf_counter()
        res = once()
        _sync(det.device)
        totals.append(time.perf_counter() - s)
        for k, v in res.timings.items():
            stages.setdefault(k, []).append(v)
    # output generation, timed separately from the latency above
    with tempfile.TemporaryDirectory() as tmp:
        s = time.perf_counter()
        text = json.dumps(res.to_json(conf=infer_cfg["candidate_conf"]))
        (Path(tmp) / "p.json").write_text(text, encoding="utf-8")
        json_s = time.perf_counter() - s
        from .review import draw_boxes
        s = time.perf_counter()
        img = load_rgb(image_path)
        save_image(draw_boxes(img, res.boxes, (0, 255, 0)), Path(tmp) / "o.jpg")
        overlay_s = time.perf_counter() - s
    return {
        "image": str(image_path), "region_crop_rect": res.region.get("crop_rect"),
        "count_at_conf": res.count(infer_cfg["candidate_conf"]), "confidence": infer_cfg["candidate_conf"],
        "mode": "single resized ROI pass" if single_pass else "tiled",
        "device": det.device, "precision": "fp16" if infer_cfg.get("half") else "fp32",
        "torch_threads": torch.get_num_threads(), "tiles": res.tile_count, "inference_batch": infer_cfg["batch"],
        "tile": tile_cfg, "model_load_s": load_s, "cold_first_run_s": cold_s, "warmup_runs": warmup,
        "latency": _stats(totals), "stages": {k: _stats(v) for k, v in stages.items()},
        "peak_cuda_memory_mib": (torch.cuda.max_memory_allocated() / 2 ** 20) if det.device != "cpu" else None,
        "output_generation_s": {"predictions_json": json_s, "overlay_jpg": overlay_s},
        "boundary": "latency = load photo from disk + ROI preparation + all tiles + restore + merge + count; "
                    "excludes model load and file writing",
    }


def benchmark_folds(cfg, fold: str) -> dict:
    import torch
    from .train import env_record
    exp = cfg["paths"]["runs_dir"] / cfg["experiment_id"]
    folds = list(cfg["folds"]) if fold == "all" else [fold]
    b = cfg["benchmark"]
    out = {"environment": env_record(), "protocol": f"{b['warmup_runs']} warm-up + {b['timed_runs']} timed "
           "complete-image runs per setting (time.perf_counter, CUDA synchronised)", "folds": {}}
    for f in folds:
        run = exp / f"fold_{f}"
        rec = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
        sel = json.loads((run / "threshold_selection.json").read_text(encoding="utf-8"))
        held = cfg["folds"][f]["heldout"]
        ann = json.loads((cfg["paths"]["annotations_dir"] / f"{held}.json").read_text(encoding="utf-8"))
        image = cfg["source_paths"][held]["image"]
        infer = {**cfg["inference"], "candidate_conf": sel["selected_confidence"]}
        ck = rec["checkpoints"]["best"]["path"]
        res = {"selected_confidence": sel["selected_confidence"], "checkpoint_sha256": rec["checkpoints"]["best"]["sha256"],
               "training_duration_s": rec["elapsed_s"]}
        devices = (["0"] if torch.cuda.is_available() else []) + ["cpu"]
        for dev in devices:
            key = "gpu" if dev != "cpu" else "cpu"
            res[f"{key}_tiled"] = benchmark_region(image, ann["roi_polygon"], ck, infer, cfg["tiles"],
                                                   b["warmup_runs"], b["timed_runs"], device=dev)
            res[f"{key}_single_resized_roi"] = benchmark_region(image, ann["roi_polygon"], ck, infer, cfg["tiles"],
                                                                b["warmup_runs"], b["timed_runs"], device=dev,
                                                                single_pass=True)
            print(f"fold {f} {key}: tiled median {res[f'{key}_tiled']['latency']['median_ms']:.0f} ms "
                  f"(p90 {res[f'{key}_tiled']['latency']['p90_ms']:.0f}), single resized pass median "
                  f"{res[f'{key}_single_resized_roi']['latency']['median_ms']:.0f} ms, "
                  f"{res[f'{key}_tiled']['tiles']} tiles")
        out["folds"][f] = res
        (run / "benchmark.json").write_text(json.dumps(out | {"folds": {f: res}}, indent=2) + "\n", encoding="utf-8")
    return out
