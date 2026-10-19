"""Full-image tiled inference: one path shared by evaluation, prediction and benchmarking.

photo + region -> grey-masked native-resolution tiles -> detector -> drop padding centres
-> restore original coordinates -> global one-class NMS -> centre-in-region count.
The detector is injected (YoloDetector in production, fakes in tests).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import Region, iter_region_tiles, make_region
from .geometry import box_centers
from .imaging import load_rgb, sha256_file


def merge_detections(boxes, scores, iou: float) -> np.ndarray:
    """Greedy one-class NMS (torchvision) in descending score with a stable index tie-break.

    Returns kept indices in ascending order. Suppresses boxes with IoU > iou.
    """
    import torch
    from torchvision.ops import nms
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranks = torch.arange(len(order), 0, -1, dtype=torch.float64)  # strictly decreasing = stable order
    keep = nms(torch.as_tensor(boxes[order], dtype=torch.float64), ranks, iou).numpy()
    return np.sort(order[keep])


@dataclass
class PredictionResult:
    boxes: np.ndarray            # merged detections, original pixels, conf >= candidate_conf
    scores: np.ndarray
    classes: np.ndarray
    in_region: np.ndarray        # centre inside the scored region
    edge_touch: np.ndarray       # box touched an interior tile edge in its source tile
    raw_count: int               # tile detections (padding centres dropped) before global merging
    raw_scores: np.ndarray
    tile_count: int
    tile_raw_counts: list
    saturated_tiles: list
    warnings: list
    timings: dict
    region: dict
    settings: dict = field(default_factory=dict)

    def select(self, conf: float) -> np.ndarray:
        return (self.scores >= conf) & self.in_region

    def count(self, conf: float) -> int:
        return int(self.select(conf).sum())

    def raw_count_at(self, conf: float) -> int:
        return int((self.raw_scores >= conf).sum())

    def seam_edge_detections(self, conf: float) -> int:
        return int((self.select(conf) & self.edge_touch).sum())

    def to_json(self, conf: float, meta: dict | None = None) -> dict:
        keep = np.where(self.scores >= conf)[0]
        dets = [{"bbox_xyxy": [round(float(v), 3) for v in self.boxes[i]],
                 "center": [round(float(v), 3) for v in box_centers(self.boxes[i:i + 1])[0]],
                 "confidence": round(float(self.scores[i]), 5), "class_id": int(self.classes[i]),
                 "class_name": "barnacle_target", "in_region": bool(self.in_region[i])} for i in keep]
        return {**(meta or {}), "confidence_threshold": conf, "count": self.count(conf),
                "count_rule": "box centre inside or on the region polygon (and inside reviewed support)",
                "n_detections_at_threshold": len(dets),
                "merge": {"raw_tile_detections": self.raw_count_at(conf), "after_global_nms": int(len(keep))},
                "tiles": self.tile_count, "saturated_tiles": self.saturated_tiles, "warnings": self.warnings,
                "timings_s": self.timings, "region": self.region, "settings": self.settings,
                "score_note": "confidence scores are model scores, not calibrated probabilities",
                "detections": dets}


def predict_image(image: np.ndarray, region: Region, detector, tile_cfg: dict, infer_cfg: dict) -> PredictionResult:
    t0 = time.perf_counter()
    tiles = list(iter_region_tiles(image, region, tile_cfg))
    t1 = time.perf_counter()
    cx0, cy0, cx1, cy1 = region.crop_rect
    boxes, scores, classes, edges, raw_counts, saturated = [], [], [], [], [], []
    infer_time = 0.0
    batch = infer_cfg.get("batch", 8)
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        s = time.perf_counter()
        dets = detector([t for t, _, _ in chunk])
        infer_time += time.perf_counter() - s
        for k, ((_, (tx, ty), (vw, vh)), d) in enumerate(zip(chunk, dets)):
            d = np.asarray(d, dtype=float).reshape(-1, 6)
            if len(d) >= infer_cfg["max_det"]:
                saturated.append({"tile_index": i + k, "origin": [tx, ty], "n": int(len(d))})
            d = d[d[:, 4] >= infer_cfg["candidate_conf"]]
            c = box_centers(d[:, :4])
            d = d[(c[:, 0] < vw) & (c[:, 1] < vh)]           # centre in padding -> drop
            raw_counts.append(int(len(d)))
            touch = np.zeros(len(d), bool)
            if tx > cx0:
                touch |= d[:, 0] <= 1
            if ty > cy0:
                touch |= d[:, 1] <= 1
            if tx + vw < cx1:
                touch |= d[:, 2] >= vw - 1
            if ty + vh < cy1:
                touch |= d[:, 3] >= vh - 1
            boxes.append(d[:, :4] + np.array([tx, ty, tx, ty], dtype=float))
            scores.append(d[:, 4])
            classes.append(d[:, 5])
            edges.append(touch)
    t2 = time.perf_counter()
    boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4))
    scores = np.concatenate(scores) if scores else np.zeros(0)
    classes = np.concatenate(classes) if classes else np.zeros(0)
    edges = np.concatenate(edges) if edges else np.zeros(0, bool)
    keep = merge_detections(boxes, scores, infer_cfg["global_nms_iou"])
    in_region = region.contains(box_centers(boxes[keep])) if len(keep) else np.zeros(0, bool)
    t3 = time.perf_counter()
    return PredictionResult(
        boxes=boxes[keep], scores=scores[keep], classes=classes[keep].astype(int), in_region=in_region,
        edge_touch=edges[keep], raw_count=int(len(boxes)), raw_scores=scores, tile_count=len(tiles),
        tile_raw_counts=raw_counts, saturated_tiles=saturated, warnings=list(getattr(detector, "warnings", [])),
        timings={"prepare_tiles": t1 - t0, "tile_inference": infer_time,
                 "restore_merge_count": (t3 - t2) + (t2 - t1 - infer_time)},
        region={"image_id": region.source_id, "name": region.name, "polygon": [list(map(float, p)) for p in region.roi],
                "crop_rect": list(map(int, region.crop_rect)), "support_pixels": int(region.support.sum())},
        settings={"tile": dict(tile_cfg), **{k: infer_cfg[k] for k in ("candidate_conf", "local_nms_iou",
                                                                        "global_nms_iou", "max_det") if k in infer_cfg}})


def predict_single_pass(image: np.ndarray, region: Region, detector, infer_cfg: dict, pad_value: int = 114):
    """Diagnostic baseline: the whole grey-masked region in one pass; the detector (built with
    imgsz=640) letterbox-resizes it, so small targets shrink. Not a separately optimised model."""
    from .dataset import region_image
    t0 = time.perf_counter()
    img = region_image(image, region, pad_value)
    t1 = time.perf_counter()
    d = np.asarray(detector([img])[0], dtype=float).reshape(-1, 6)
    t2 = time.perf_counter()
    d = d[d[:, 4] >= infer_cfg["candidate_conf"]]
    x0, y0 = region.crop_rect[:2]
    boxes = d[:, :4] + np.array([x0, y0, x0, y0], dtype=float)
    keep = merge_detections(boxes, d[:, 4], infer_cfg["global_nms_iou"])
    in_region = region.contains(box_centers(boxes[keep])) if len(keep) else np.zeros(0, bool)
    t3 = time.perf_counter()
    return PredictionResult(
        boxes=boxes[keep], scores=d[keep, 4], classes=d[keep, 5].astype(int), in_region=in_region,
        edge_touch=np.zeros(len(keep), bool), raw_count=int(len(d)), raw_scores=d[:, 4], tile_count=1,
        tile_raw_counts=[int(len(d))], saturated_tiles=[] if len(d) < infer_cfg["max_det"] else [{"n": int(len(d))}],
        warnings=list(getattr(detector, "warnings", [])),
        timings={"prepare_tiles": t1 - t0, "tile_inference": t2 - t1, "restore_merge_count": t3 - t2},
        region={"image_id": region.source_id, "name": region.name, "crop_rect": list(map(int, region.crop_rect))},
        settings={"mode": "single resized pass", "region_wh": [img.shape[1], img.shape[0]]})


def predict_region(image_path, roi, detector, tile_cfg: dict, infer_cfg: dict, image_id: str | None = None):
    """Photograph + ROI polygon (None = whole image) -> PredictionResult. Never reads mask files."""
    t0 = time.perf_counter()
    image = load_rgb(image_path)
    region = make_region(image_id or Path(image_path).stem, roi, image.shape[1], image.shape[0],
                         name="whole_image" if roi is None else "roi")
    t1 = time.perf_counter()
    res = predict_image(image, region, detector, tile_cfg, infer_cfg)
    res.timings = {"load_and_region": t1 - t0, **res.timings}
    return res


def check_source_role(image_path, source_hashes: dict) -> None:
    """Reject annotation masks and blue-overlay reference images; only photographs are model inputs."""
    path = Path(image_path)
    banned = {sha: name for name, sha in source_hashes["files"].items()
              if Path(name).name.startswith(("mask", "masked_"))}
    if path.name.startswith(("mask", "masked_")):
        raise ValueError(f"{path.name} is an annotation mask/overlay, not an original photograph")
    sha = sha256_file(path)
    if sha in banned:
        raise ValueError(f"{path.name} has the same content as {banned[sha]}; only original photographs are accepted")


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class YoloDetector:
    """Ultralytics YOLO wrapper returning per-tile [x1, y1, x2, y2, conf, cls] arrays in tile pixels.

    Tiles are RGB; Ultralytics expects BGR numpy arrays, so channels are reversed here only.
    """

    def __init__(self, checkpoint, infer_cfg: dict, device: str | None = None, imgsz: int = 640):
        from .yolo_env import device_string, yolo_class
        self.model = yolo_class()(str(checkpoint))
        self.device = device or device_string()
        self.kw = {"imgsz": imgsz, "conf": infer_cfg["candidate_conf"], "iou": infer_cfg["local_nms_iou"],
                   "max_det": infer_cfg["max_det"], "device": self.device, "verbose": False}
        if infer_cfg.get("half"):
            self.kw["half"] = True  # passing half=False only triggers a deprecation warning in 8.4.x
        self.warnings: list[str] = []
        names = getattr(self.model, "names", {})
        if len(names) != 1:
            raise ValueError(f"expected a single-class checkpoint, got classes {names}")

    def __call__(self, tiles):
        import torch
        from ultralytics.utils import LOGGER
        cap = _WarningCapture()
        LOGGER.addHandler(cap)
        try:
            results = self.model.predict([np.ascontiguousarray(t[..., ::-1]) for t in tiles], **self.kw)
            if self.device != "cpu":
                torch.cuda.synchronize()
        finally:
            LOGGER.removeHandler(cap)
        self.warnings += cap.messages
        out = []
        for r in results:
            b = r.boxes
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                       b.cls.cpu().numpy()[:, None]], 1) if len(b) else np.zeros((0, 6)))
        return out


def predict_cli(cfg, image, roi_file, whole_image: bool, checkpoint, conf, out_dir, detector=None) -> dict:
    """Demonstration prediction on one photograph: JSON + overlay. No reference labels are used."""
    from .geometry import validate_polygon
    from .imaging import save_image
    from .review import draw_boxes, draw_polygon
    image = Path(image)
    check_source_role(image, json.loads((cfg["paths"]["manifests_dir"] / "source_hashes.json").read_text(encoding="utf-8")))
    image_id = image.stem
    roi = None
    if not whole_image:
        rois = json.loads(Path(roi_file).read_text(encoding="utf-8"))["rois"]
        if image_id not in rois or not rois[image_id].get("polygon"):
            raise ValueError(f"no ROI polygon for {image_id} in {roi_file}; use --whole-image for close-ups")
        roi = validate_polygon(rois[image_id]["polygon"]).tolist()
    ck_sha = sha256_file(checkpoint) if checkpoint else None
    if conf is None:
        dpath = Path(checkpoint).parent.parent / "deployment.json"
        if not dpath.exists():
            raise SystemExit("--conf is required for a checkpoint without deployment.json")
        deploy = json.loads(dpath.read_text(encoding="utf-8"))
        if deploy["checkpoint_sha256"] != ck_sha:
            raise ValueError("deployment.json describes a different checkpoint")
        conf = deploy["confidence"]
    det = detector if detector is not None else YoloDetector(checkpoint, cfg["inference"])
    res = predict_region(image, roi, det, cfg["tiles"], cfg["inference"], image_id=image_id)
    out = Path(out_dir) if out_dir else cfg["paths"]["reports_dir"] / "predictions" / image_id
    doc = res.to_json(conf, meta={
        "image": image.name, "image_id": image_id, "image_sha256": sha256_file(image),
        "checkpoint": str(checkpoint) if checkpoint else None, "checkpoint_sha256": ck_sha,
        "status": "demonstration: no reference labels exist for this image; the count is not an accuracy measurement",
        "region_note": "whole image (close-up; no frame, no physical area implied)" if roi is None
        else "inner-frame ROI polygon from rois.json (AI-proposed, not human-verified)"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    img = load_rgb(image)
    keep = res.in_region & (res.scores >= conf)
    review = res.in_region & (res.scores < conf)   # lower-score candidates a person may want to check
    vis = draw_boxes(img, res.boxes[review], (255, 150, 0), 1)
    vis = draw_boxes(vis, res.boxes[keep], (0, 230, 0), 2)
    vis = draw_polygon(vis, res.region["polygon"], (0, 255, 255), 3)
    save_image(vis, out / "overlay.jpg", quality=90)
    print(f"{image_id}: {doc['count']} detections at confidence >= {conf} in {res.region['name']} "
          f"({int(review.sum())} lower-score candidates shown in orange) -> {out}  [demonstration]")
    return doc
