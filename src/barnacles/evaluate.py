"""Inner-validation threshold selection, held-out full-region evaluation and inference ablations.

Held-out scoring happens once per fold on the merged full-region predictions; tile counts are
never summed. Every output carries the label review status (provisional unless human-verified).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .dataset import compute_full_region, compute_inner_split
from .geometry import box_centers, distance_to_polygon_edges, points_in_polygon, rasterize_polygon, tile_starts
from .imaging import load_rgb, save_image, sha256_file
from .metrics import count_errors, match_boxes, match_centers, prf
from .review import draw_boxes, draw_polygon

SELECTION_RULE = ("max instance F1 at box IoU >= 0.5 on the inner-validation region over the fixed grid; "
                  "ties -> higher recall, then higher confidence")


# ---- pure helpers -------------------------------------------------------------------

def choose_threshold(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (round(r["f1"], 12), round(r["recall"], 12), r["confidence"]))


def score_predictions(pred_boxes, scores, ref_boxes, conf: float, iou: float, center_factor: float) -> dict:
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    ref_boxes = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    sel = np.asarray(scores) >= conf
    pb, ps = pred_boxes[sel], np.asarray(scores)[sel]
    m = match_boxes(pb, ps, ref_boxes, iou)
    c = match_centers(pb, ps, ref_boxes, center_factor)
    return {"confidence": conf, **count_errors(len(ref_boxes), int(sel.sum())), "tp": m.tp, "fp": m.fp, "fn": m.fn,
            **prf(m.tp, m.fp, m.fn),
            "center_match": {"tolerance": f"{center_factor} * sqrt(ref_w * ref_h)", "tp": c.tp, "fp": c.fp,
                             "fn": c.fn, **prf(c.tp, c.fp, c.fn)}}


def seam_band(points, crop_rect, tile: int, stride: int, band: float) -> np.ndarray:
    """True for points within `band` px of an interior tile boundary (region borders excluded)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    x0, y0, x1, y1 = crop_rect
    near = np.zeros(len(pts), bool)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        edges = set()
        for s in tile_starts(hi - lo, tile, stride):
            if s > 0:
                edges.add(lo + s)
            if lo + s + tile < hi:
                edges.add(lo + s + tile)
        for e in edges:
            near |= np.abs(pts[:, axis] - e) < band
    return near


def size_bin(boxes, bins: dict) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    s = np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    return np.where(s < bins["small_lt"], "small", np.where(s >= bins["large_ge"], "large", "medium"))


def recall_by_group(ref_to_pred, groups) -> dict:
    out = {}
    for gname in sorted(set(np.asarray(groups).tolist())):
        idx = np.asarray(groups) == gname
        n, tp = int(idx.sum()), int((np.asarray(ref_to_pred)[idx] >= 0).sum())
        out[gname] = {"n": n, "tp": tp, "recall": tp / n if n else 0.0}
    return out


def precision_by_group(pred_to_ref, groups) -> dict:
    out = {}
    for gname in sorted(set(np.asarray(groups).tolist())):
        idx = np.asarray(groups) == gname
        n, tp = int(idx.sum()), int((np.asarray(pred_to_ref)[idx] >= 0).sum())
        out[gname] = {"n_pred": n, "tp": tp, "fp": n - tp, "precision": tp / n if n else 0.0}
    return out


# ---- shared I/O ---------------------------------------------------------------------------

def _exp(cfg) -> Path:
    return cfg["paths"]["runs_dir"] / cfg["experiment_id"]


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _annotation(cfg, sid: str) -> dict:
    path = cfg["paths"]["annotations_dir"] / f"{sid}.json"
    ann = _read(path)
    ann["_sha256"] = sha256_file(path)
    return ann


def _fold_context(cfg, fold: str) -> dict:
    run = _exp(cfg) / f"fold_{fold}"
    rec = _read(run / "run_record.json")
    manifest = _read(cfg["paths"]["manifests_dir"] / f"fold_{fold}" / "split_manifest.json")
    ckpt = Path(rec["checkpoints"]["best"]["path"])
    if sha256_file(ckpt) != rec["checkpoints"]["best"]["sha256"]:
        raise ValueError(f"{ckpt} changed since training")
    return {"run": run, "record": rec, "manifest": manifest, "checkpoint": ckpt}


def _detector(cfg, checkpoint, imgsz: int = 640):
    from .predict import YoloDetector
    return YoloDetector(checkpoint, cfg["inference"], imgsz=imgsz)


def _boxes(ann: dict, ids) -> np.ndarray:
    by_id = {o["id"]: o["bbox_xyxy"] for o in ann["objects"]}
    return np.array([by_id[i] for i in ids], dtype=float).reshape(-1, 4)


# ---- threshold selection -------------------------------------------------------------------

def select_threshold(cfg, fold: str) -> dict:
    from .predict import predict_image
    ctx = _fold_context(cfg, fold)
    if (ctx["run"] / "evaluation" / "metrics.json").exists():
        raise SystemExit(f"fold {fold} is already evaluated; its threshold is frozen")
    m = ctx["manifest"]
    ann = _annotation(cfg, m["dev_source"])
    if ann["_sha256"] != m["annotation_sha256"]:
        raise ValueError("development annotation changed since the split was built")
    split = compute_inner_split(ann, cfg["inner_split"])
    val = split["val"]
    if val.support_sha256() != m["regions"]["val"]["support_sha256"]:
        raise ValueError("inner validation geometry differs from the frozen split manifest")
    image = load_rgb(cfg["source_paths"][m["dev_source"]]["image"])
    res = predict_image(image, val, _detector(cfg, ctx["checkpoint"]), cfg["tiles"], cfg["inference"])
    refs = _boxes(ann, val.ref_ids)
    ev = cfg["evaluation"]
    rows = [score_predictions(res.boxes[res.in_region], res.scores[res.in_region], refs, c, ev["match_iou"],
                              ev["center_tolerance_factor"]) for c in cfg["inference"]["threshold_grid"]]
    chosen = choose_threshold(rows)
    out = {"fold": fold, "dev_source": m["dev_source"], "checkpoint": str(ctx["checkpoint"]),
           "checkpoint_sha256": ctx["record"]["checkpoints"]["best"]["sha256"], "rule": SELECTION_RULE,
           "grid": cfg["inference"]["threshold_grid"], "local_nms_iou": cfg["inference"]["local_nms_iou"],
           "global_nms_iou": cfg["inference"]["global_nms_iou"], "candidate_conf": cfg["inference"]["candidate_conf"],
           "n_val_references": len(refs), "val_region": val.summary(), "tiles": res.tile_count,
           "saturated_tiles": res.saturated_tiles, "warnings": res.warnings, "provisional": m["provisional"],
           "review_status": m["review_status"], "rows": rows, "selected_confidence": chosen["confidence"],
           "selected": chosen, "note": "inner validation is one region of the development photo; limited independence"}
    _write(ctx["run"] / "threshold_selection.json", out)
    with open(ctx["run"] / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence", "gt", "pred", "tp", "fp", "fn", "precision", "recall", "f1"])
        for r in rows:
            w.writerow([r["confidence"], r["gt_count"], r["pred_count"], r["tp"], r["fp"], r["fn"],
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}"])
    print(f"fold {fold}: selected confidence {chosen['confidence']} on inner validation "
          f"(F1 {chosen['f1']:.3f}, P {chosen['precision']:.3f}, R {chosen['recall']:.3f}, "
          f"{len(refs)} references){' [PROVISIONAL labels]' if m['provisional'] else ''}")
    return out


# ---- held-out evaluation -------------------------------------------------------------------

def _overlays(image, region, res, sel_idx, refs, match, out_dir: Path, bins, cfg) -> list[str]:
    """Full-region TP/FP/FN overlay plus zoomed dense/small/dark/seam windows."""
    x0, y0, x1, y1 = region.crop_rect
    crop = image[y0:y1, x0:x1].copy()
    crop[~region.support] = (crop[~region.support] * 0.4).astype(np.uint8)  # dim unscored pixels
    pb = res.boxes[sel_idx]
    tp = pb[match.pred_to_ref >= 0]
    fp = pb[match.pred_to_ref < 0]
    fn = refs[match.ref_to_pred < 0]
    vis = draw_boxes(crop, tp, (0, 220, 0), 2, origin=(x0, y0))
    vis = draw_boxes(vis, fp, (255, 0, 0), 2, origin=(x0, y0))
    vis = draw_boxes(vis, fn, (255, 230, 0), 2, origin=(x0, y0))
    vis = draw_polygon(vis, region.roi, (0, 255, 255), 3, origin=(x0, y0))
    files = [save_image(vis, out_dir / "overlay_full.jpg", quality=88)]
    # zoom windows chosen from label geometry and pixel brightness only
    win = 320
    centers = box_centers(refs)
    small = size_bin(refs, bins) == "small"
    seam = seam_band(centers, region.crop_rect, cfg["tiles"]["size"], cfg["tiles"]["stride"], 32)
    gray = image.mean(axis=2)
    best = {"dense": (-1, None), "small": (-1, None), "dark": (1e9, None), "seam": (-1, None)}
    for wy in range(y0, max(y1 - win, y0) + 1, 80):
        for wx in range(x0, max(x1 - win, x0) + 1, 80):
            inside = (centers[:, 0] >= wx) & (centers[:, 0] < wx + win) & (centers[:, 1] >= wy) & (centers[:, 1] < wy + win)
            if inside.sum() < 5 or not region.support[wy - y0:wy - y0 + win, wx - x0:wx - x0 + win].all():
                continue
            for key, score in (("dense", inside.sum()), ("small", (inside & small).sum()), ("seam", (inside & seam).sum())):
                if score > best[key][0]:
                    best[key] = (score, (wx, wy))
            dark = gray[wy:wy + win, wx:wx + win].mean()
            if dark < best["dark"][0]:
                best["dark"] = (dark, (wx, wy))
    for key, (_, pos) in best.items():
        if pos is None:
            continue
        wx, wy = pos
        z = vis[wy - y0:wy - y0 + win, wx - x0:wx - x0 + win]
        z = np.kron(z, np.ones((2, 2, 1), dtype=np.uint8))
        files.append(save_image(z, out_dir / f"zoom_{key}_x{wx}_y{wy}.png"))
    return [str(f) for f in files]


def evaluate_fold(cfg, fold: str) -> dict:
    from .predict import predict_image
    ctx = _fold_context(cfg, fold)
    out_dir = ctx["run"] / "evaluation"
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} exists; evaluated results are never overwritten")
    sel = _read(ctx["run"] / "threshold_selection.json")
    if sel["checkpoint_sha256"] != ctx["record"]["checkpoints"]["best"]["sha256"]:
        raise ValueError("threshold was selected for a different checkpoint")
    m = ctx["manifest"]
    held = m["heldout_source"]
    ann = _annotation(cfg, held)
    region = compute_full_region(ann, name="heldout_roi")
    refs = _boxes(ann, region.ref_ids)
    image = load_rgb(cfg["source_paths"][held]["image"])
    det = _detector(cfg, ctx["checkpoint"])
    res = predict_image(image, region, det, cfg["tiles"], cfg["inference"])
    ev, conf = cfg["evaluation"], sel["selected_confidence"]
    in_r = np.where(res.in_region)[0]
    pb, ps = res.boxes[in_r], res.scores[in_r]
    primary = score_predictions(pb, ps, refs, conf, ev["match_iou"], ev["center_tolerance_factor"])
    fixed = score_predictions(pb, ps, refs, cfg["inference"]["fixed_sensitivity_conf"], ev["match_iou"],
                              ev["center_tolerance_factor"])
    sel_idx = in_r[ps >= conf]
    match = match_boxes(res.boxes[sel_idx], res.scores[sel_idx], refs, ev["match_iou"])

    # diagnostics (reference-based groups; definitions fixed before evaluation)
    bins = m["size_bins"]
    rc, pc = box_centers(refs), box_centers(res.boxes[sel_idx])
    band = ev["seam_band_px"]
    r_seam = seam_band(rc, region.crop_rect, cfg["tiles"]["size"], cfg["tiles"]["stride"], band)
    p_seam = seam_band(pc, region.crop_rect, cfg["tiles"]["size"], cfg["tiles"]["stride"], band)
    r_size, p_size = size_bin(refs, bins), size_bin(res.boxes[sel_idx], bins)
    roi_d_ref = distance_to_polygon_edges(rc, region.roi)
    roi_d_pred = distance_to_polygon_edges(pc, region.roi)
    corners = np.stack([refs[:, [0, 1]], refs[:, [2, 1]], refs[:, [2, 3]], refs[:, [0, 3]]], 1)
    crossing = ~np.array([points_in_polygon(c, region.roi).all() for c in corners]) if len(refs) else np.zeros(0, bool)
    eroded = score_predictions(res.boxes[sel_idx][roi_d_pred >= band], res.scores[sel_idx][roi_d_pred >= band],
                               refs[roi_d_ref >= band], 0.0, ev["match_iou"], ev["center_tolerance_factor"])
    roi_pixels = int(rasterize_polygon(region.roi, image.shape[1], image.shape[0]).sum())
    metrics = {
        "fold": fold, "experiment_id": cfg["experiment_id"], "train_source": m["dev_source"], "heldout_source": held,
        "annotation_version": ann.get("annotation_version"), "annotation_sha256": ann["_sha256"],
        "review_status": ann["review_status"], "provisional": ann["review_status"] != "human_verified",
        "scored_region": "inner-frame ROI polygon of the held-out photo intersected with reviewed regions",
        "reviewed_area_fraction": float(region.support.sum()) / roi_pixels,
        "checkpoint": str(ctx["checkpoint"]), "checkpoint_sha256": sel["checkpoint_sha256"],
        "selected_epoch": ctx["record"]["best_epoch"], "seed": ctx["record"]["requested_args"].get("seed"),
        "confidence": conf, "confidence_source": "inner-validation selection (development photo only)",
        "local_nms_iou": cfg["inference"]["local_nms_iou"], "global_nms_iou": cfg["inference"]["global_nms_iou"],
        "primary": primary, "fixed_confidence_sensitivity": fixed,
        "merge": {"raw_tile_detections_at_conf": res.raw_count_at(conf),
                  "after_global_nms_at_conf": int((res.scores >= conf).sum()),
                  "counted_in_region": int(len(sel_idx)),
                  "seam_edge_detections_counted": res.seam_edge_detections(conf)},
        "seam_diagnostic": {"band_px": band, "definition": f"centre within {band}px of an interior tile boundary "
                            "(tile 640/stride 512, so the 128px overlaps lie inside or next to the band)",
                            "recall": recall_by_group(match.ref_to_pred, np.where(r_seam, "seam", "interior")),
                            "precision": precision_by_group(match.pred_to_ref, np.where(p_seam, "seam", "interior")),
                            "count": {"seam_refs": int(r_seam.sum()), "seam_preds": int(p_seam.sum()),
                                      "interior_refs": int((~r_seam).sum()), "interior_preds": int((~p_seam).sum())}},
        "size_diagnostic": {"bins": bins, "recall": recall_by_group(match.ref_to_pred, r_size),
                            "precision": precision_by_group(match.pred_to_ref, p_size)},
        "boundary": {"references_crossing_roi_edge": int(crossing.sum()),
                     "eroded_interior_band_px": band, "eroded_interior_scores": eroded},
        "heldout_confidence_curve": {
            "note": "DIAGNOSTIC ONLY - computed after selection, never used to choose the threshold",
            "rows": [{k: r[k] for k in ("confidence", "pred_count", "signed_count_error", "absolute_percentage_error",
                                        "precision", "recall", "f1")}
                     for r in (score_predictions(pb, ps, refs, c, ev["match_iou"], ev["center_tolerance_factor"])
                               for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))]},
        "tiles": res.tile_count, "saturated_tiles": res.saturated_tiles, "warnings": res.warnings,
        "tile_size": cfg["tiles"]["size"], "tile_stride": cfg["tiles"]["stride"],
        "device": det.device, "precision_mode": "fp16" if cfg["inference"].get("half") else "fp32",
        "timings_s": res.timings,
    }
    files = _overlays(image, region, res, sel_idx, refs, match, out_dir, bins, cfg)
    metrics["overlays"] = files
    preds = res.to_json(conf=cfg["inference"]["candidate_conf"], meta={"image_id": held, "fold": fold,
                                                                        "checkpoint_sha256": sel["checkpoint_sha256"]})
    preds["selected_confidence"] = conf
    preds["count_at_selected_confidence"] = res.count(conf)
    _write(out_dir / "predictions.json", preds)
    with open(out_dir / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "x1", "y1", "x2", "y2", "confidence", "matched_index", "size_bin", "seam_band"])
        for i, b in enumerate(res.boxes[sel_idx]):
            w.writerow(["prediction", *np.round(b, 2), f"{res.scores[sel_idx][i]:.4f}", match.pred_to_ref[i],
                        p_size[i], bool(p_seam[i])])
        ref_ids = region.ref_ids
        for j, b in enumerate(refs):
            w.writerow([f"reference:{ref_ids[j]}", *np.round(b, 2), "", match.ref_to_pred[j], r_size[j], bool(r_seam[j])])
    _write(out_dir / "metrics.json", metrics)
    p = primary
    print(f"fold {fold} ({m['dev_source']} -> {held}){' [PROVISIONAL labels]' if metrics['provisional'] else ''}: "
          f"conf {conf}: N={p['gt_count']} P={p['pred_count']} err={p['signed_count_error']:+d} "
          f"({p['absolute_percentage_error']:.1f}%) TP/FP/FN={p['tp']}/{p['fp']}/{p['fn']} "
          f"P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f}")
    return metrics


def inner_val_overlay(cfg, fold: str, out_path: Path, win=(540, 640)) -> dict:
    """Error overlay on the inner-validation region (development photo only) at the selected threshold.

    Shows the window with the most false positives; green = matched, red = unmatched prediction,
    yellow = missed reference.
    """
    from .predict import predict_image
    ctx = _fold_context(cfg, fold)
    m = ctx["manifest"]
    ann = _annotation(cfg, m["dev_source"])
    val = compute_inner_split(ann, cfg["inner_split"])["val"]
    conf = _read(ctx["run"] / "threshold_selection.json")["selected_confidence"]
    image = load_rgb(cfg["source_paths"][m["dev_source"]]["image"])
    res = predict_image(image, val, _detector(cfg, ctx["checkpoint"]), cfg["tiles"], cfg["inference"])
    keep = res.select(conf)
    refs = _boxes(ann, val.ref_ids)
    match = match_boxes(res.boxes[keep], res.scores[keep], refs, cfg["evaluation"]["match_iou"])
    fp = res.boxes[keep][match.pred_to_ref < 0]
    vis = draw_boxes(image, res.boxes[keep][match.pred_to_ref >= 0], (0, 220, 0), 2)
    vis = draw_boxes(vis, fp, (255, 0, 0), 2)
    vis = draw_boxes(vis, refs[match.ref_to_pred < 0], (255, 230, 0), 2)
    x0, y0, x1, y1 = val.crop_rect
    w, h = min(win[0], x1 - x0), min(win[1], y1 - y0)
    fc = box_centers(fp)
    best, pos = -1, (x0, y0)
    for wy in range(y0, y1 - h + 1, 40):
        for wx in range(x0, x1 - w + 1, 40):
            n = int(((fc[:, 0] >= wx) & (fc[:, 0] < wx + w) & (fc[:, 1] >= wy) & (fc[:, 1] < wy + h)).sum())
            if n > best:
                best, pos = n, (wx, wy)
    save_image(vis[pos[1]:pos[1] + h, pos[0]:pos[0] + w], out_path)
    return {"fold": fold, "source": m["dev_source"], "confidence": conf, "tp": match.tp, "fp": match.fp,
            "fn": match.fn, "window_xy": list(pos), "window_fp": best}


# ---- ablations -----------------------------------------------------------------------------

def run_ablations(cfg, fold: str) -> dict:
    """Same checkpoint and threshold: overlap tiling vs non-overlapping tiles vs one resized pass."""
    from .predict import predict_image, predict_single_pass
    ctx = _fold_context(cfg, fold)
    metrics = _read(ctx["run"] / "evaluation" / "metrics.json")
    conf = metrics["confidence"]
    ann = _annotation(cfg, metrics["heldout_source"])
    region = compute_full_region(ann, name="heldout_roi")
    refs = _boxes(ann, region.ref_ids)
    image = load_rgb(cfg["source_paths"][metrics["heldout_source"]]["image"])
    ev = cfg["evaluation"]
    out = {"fold": fold, "confidence": conf, "provisional": metrics["provisional"],
           "note": "inference-only diagnostics with the same trained checkpoint; not separately optimised models"}
    det = _detector(cfg, ctx["checkpoint"])
    variants = {"overlap_640_512": {**cfg["tiles"]},
                "no_overlap_640": {**cfg["tiles"], "stride": cfg["tiles"]["size"], "pad_last": True}}
    for name, tcfg in variants.items():
        res = predict_image(image, region, det, tcfg, cfg["inference"])
        s = score_predictions(res.boxes[res.in_region], res.scores[res.in_region], refs, conf, ev["match_iou"],
                              ev["center_tolerance_factor"])
        out[name] = {"tiles": res.tile_count, "timings_s": res.timings, "seam_edge_detections": res.seam_edge_detections(conf),
                     **{k: s[k] for k in ("gt_count", "pred_count", "signed_count_error", "absolute_percentage_error",
                                          "tp", "fp", "fn", "precision", "recall", "f1")}}
    single = _detector(cfg, ctx["checkpoint"], imgsz=cfg["tiles"]["size"])
    res = predict_single_pass(image, region, single, cfg["inference"])
    s = score_predictions(res.boxes[res.in_region], res.scores[res.in_region], refs, conf, ev["match_iou"],
                          ev["center_tolerance_factor"])
    out["single_resized_roi_640"] = {"tiles": 1, "timings_s": res.timings, "region_wh": res.settings["region_wh"],
                                     **{k: s[k] for k in ("gt_count", "pred_count", "signed_count_error",
                                                          "absolute_percentage_error", "tp", "fp", "fn",
                                                          "precision", "recall", "f1")}}
    _write(ctx["run"] / "evaluation" / "ablations.json", out)
    for k in ("overlap_640_512", "no_overlap_640", "single_resized_roi_640"):
        v = out[k]
        print(f"fold {fold} {k}: tiles={v['tiles']} P={v['pred_count']} err={v['signed_count_error']:+d} "
              f"F1={v['f1']:.3f} R={v['recall']:.3f}")
    return out
