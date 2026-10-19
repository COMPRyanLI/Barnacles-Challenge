"""Spatial splits, region support masks and YOLO tile export.

A Region is the set of original-image pixels a split may use: inside the ROI
polygon, inside a reviewed polygon, inside the split's x-range, and outside
masked boxes of excluded objects. Pixels outside the support are painted
grey (pad value) before tiling, so no split ever sees another split's pixels.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .geometry import (box_centers, box_to_yolo, clip_box, extract_tile, points_in_polygon, polygon_bbox,
                       rasterize_polygon, tile_grid, to_local)
from .imaging import load_rgb, save_image, sha256_file


@dataclass
class Region:
    source_id: str
    name: str
    roi: list
    reviewed: list
    x_min: float | None = None       # inclusive bound on x
    x_max: float | None = None       # exclusive bound on x
    masked_boxes: list = field(default_factory=list)
    crop_rect: tuple = (0, 0, 0, 0)
    support: np.ndarray | None = None
    objects: list = field(default_factory=list)   # dicts: id, bbox_xyxy, train_box
    ref_ids: list = field(default_factory=list)   # objects whose original centre is in the region

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        ok = points_in_polygon(pts, self.roi)
        ok &= np.any([points_in_polygon(pts, r) for r in self.reviewed], axis=0) if self.reviewed \
            else np.zeros(len(pts), bool)
        if self.x_min is not None:
            ok &= pts[:, 0] >= self.x_min
        if self.x_max is not None:
            ok &= pts[:, 0] < self.x_max
        for x1, y1, x2, y2 in self.masked_boxes:
            ok &= ~((pts[:, 0] >= x1) & (pts[:, 0] < x2) & (pts[:, 1] >= y1) & (pts[:, 1] < y2))
        return ok

    def support_sha256(self) -> str:
        h = hashlib.sha256(json.dumps(list(map(int, self.crop_rect))).encode())
        h.update(np.packbits(self.support).tobytes())
        return h.hexdigest()

    def summary(self) -> dict:
        return {"source_id": self.source_id, "name": self.name, "crop_rect": list(map(int, self.crop_rect)),
                "x_min": self.x_min, "x_max": self.x_max, "masked_boxes": self.masked_boxes,
                "support_pixels": int(self.support.sum()), "support_sha256": self.support_sha256(),
                "n_label_objects": len(self.objects), "n_reference_objects": len(self.ref_ids)}


def _support(region: Region, width: int, height: int) -> None:
    """Rasterise region.contains over pixel centres and shrink crop_rect to the support."""
    x0, y0, x1, y1 = polygon_bbox(region.roi, width, height)
    if region.x_min is not None:
        x0 = max(x0, math.floor(region.x_min))
    if region.x_max is not None:
        x1 = min(x1, math.ceil(region.x_max))
    sup = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), bool)
    if sup.size:
        sup = rasterize_polygon(region.roi, x1 - x0, y1 - y0, origin=(x0, y0))
        rev = np.zeros_like(sup)
        for r in region.reviewed:
            rev |= rasterize_polygon(r, x1 - x0, y1 - y0, origin=(x0, y0))
        sup &= rev
        cx = np.arange(x0, x1) + 0.5
        if region.x_min is not None:
            sup &= (cx >= region.x_min)[None, :]
        if region.x_max is not None:
            sup &= (cx < region.x_max)[None, :]
        cy = np.arange(y0, y1) + 0.5
        for bx1, by1, bx2, by2 in region.masked_boxes:
            sup &= ~(((cy >= by1) & (cy < by2))[:, None] & ((cx >= bx1) & (cx < bx2))[None, :])
    if not sup.any():
        region.crop_rect, region.support = (x0, y0, x0, y0), np.zeros((0, 0), bool)
        return
    ys, xs = np.nonzero(sup)
    region.crop_rect = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
    region.support = sup[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _assign_objects(region: Region, objects: list, skip: set) -> None:
    """Label every object with supported pixels (clipped box); reference = centre inside region."""
    cx0, cy0, cx1, cy1 = region.crop_rect
    region.objects, region.ref_ids = [], []
    for o in objects:
        if o["id"] in skip:
            continue
        x1, y1, x2, y2 = o["bbox_xyxy"]
        px0, py0 = max(math.floor(x1), cx0), max(math.floor(y1), cy0)
        px1, py1 = min(math.ceil(x2), cx1), min(math.ceil(y2), cy1)
        if px1 <= px0 or py1 <= py0:
            continue
        sub = region.support[py0 - cy0:py1 - cy0, px0 - cx0:px1 - cx0]
        if not sub.any():
            continue
        ys, xs = np.nonzero(sub)
        box = [max(x1, px0 + float(xs.min())), max(y1, py0 + float(ys.min())),
               min(x2, px0 + float(xs.max()) + 1), min(y2, py0 + float(ys.max()) + 1)]
        region.objects.append({"id": o["id"], "bbox_xyxy": list(o["bbox_xyxy"]), "train_box": box})
        if region.contains(box_centers([o["bbox_xyxy"]]))[0]:
            region.ref_ids.append(o["id"])


def _base_region(ann: dict, name: str, **kw) -> Region:
    return Region(ann["image_id"], name, ann["roi_polygon"], ann.get("reviewed_region_polygons") or [], **kw)


def compute_full_region(ann: dict, name: str = "all") -> Region:
    region = _base_region(ann, name)
    _support(region, ann["width"], ann["height"])
    _assign_objects(region, ann["objects"], skip=set())
    return region


def make_region(image_id: str, roi, width: int, height: int, name: str = "roi") -> Region:
    """Inference region for a photograph: an ROI polygon, or the whole image when roi is None."""
    poly = roi if roi is not None else [[0, 0], [width, 0], [width, height], [0, height]]
    region = Region(image_id, name, poly, [poly])
    _support(region, width, height)
    return region


def compute_inner_split(ann: dict, cfg: dict) -> dict:
    """Left/right split of the ROI bounding rectangle with an unused vertical gap.

    Objects whose boxes touch the gap are excluded from both sets and their boxes are
    masked in both; a retained object losing more than the allowed fraction of its box
    to masking is excluded too (iterated to a fixed point).
    """
    rx0, _, rx1, _ = polygon_bbox(ann["roi_polygon"])
    split_x = rx0 + cfg["train_fraction"] * (rx1 - rx0)
    gl, gr = split_x - cfg["gap_px"] / 2, split_x + cfg["gap_px"] / 2
    boxes = {o["id"]: o["bbox_xyxy"] for o in ann["objects"]}
    excluded = {oid for oid, b in boxes.items() if b[0] < gr and b[2] > gl}
    limit = cfg["max_masked_fraction_of_retained"]

    def masked_fraction(b, masks):
        # sum of overlaps (an upper bound on the union when masks overlap) -> conservative exclusion
        tot = 0.0
        for m in masks:
            c = clip_box(b, m)
            if c:
                tot += (c[2] - c[0]) * (c[3] - c[1])
        return min(tot / ((b[2] - b[0]) * (b[3] - b[1])), 1.0)

    while True:
        masks = [boxes[i] for i in sorted(excluded)]
        newly = {oid for oid, b in boxes.items() if oid not in excluded and masked_fraction(b, masks) > limit}
        if not newly:
            break
        excluded |= newly
    masks = [list(boxes[i]) for i in sorted(excluded)]
    partial = {oid: masked_fraction(b, masks) for oid, b in boxes.items()
               if oid not in excluded and masked_fraction(b, masks) > 0}
    out = {"geometry": {"roi_bbox_x": [rx0, rx1], "train_fraction": cfg["train_fraction"], "split_x": split_x,
                        "gap": [gl, gr], "gap_px": cfg["gap_px"], "max_masked_fraction_of_retained": limit},
           "excluded_ids": sorted(excluded), "partially_masked": partial, "masked_boxes": masks}
    for name, kw in (("train", {"x_max": gl}), ("val", {"x_min": gr})):
        region = _base_region(ann, name, masked_boxes=masks, **kw)
        _support(region, ann["width"], ann["height"])
        _assign_objects(region, ann["objects"], skip=excluded)
        if not region.objects or not region.ref_ids:
            raise ValueError(f"{name} split has no reviewed targets; choose another partition from annotation geometry")
        out[name] = region
    return out


# ---- tiles -----------------------------------------------------------------------

def region_image(image: np.ndarray, region: Region, pad_value: int = 114) -> np.ndarray:
    x0, y0, x1, y1 = region.crop_rect
    out = image[y0:y1, x0:x1].copy()
    out[~region.support] = pad_value
    return out


def iter_region_tiles(image: np.ndarray, region: Region, tile_cfg: dict):
    """Yield (tile, (tx, ty) original origin, (valid_w, valid_h)) for tiles with any support."""
    size, pad = tile_cfg["size"], tile_cfg["pad_value"]
    x0, y0, x1, y1 = region.crop_rect
    split_img = region_image(image, region, pad)
    for tx, ty in tile_grid(x1 - x0, y1 - y0, size, tile_cfg["stride"], tile_cfg.get("pad_last", False)):
        if not region.support[ty:ty + size, tx:tx + size].any():
            continue
        tile, valid = extract_tile(split_img, tx, ty, size, pad)
        yield tile, (x0 + tx, y0 + ty), valid


def export_tiles(image: np.ndarray, region: Region, out_dir: Path, split: str, tile_cfg: dict,
                 thin_px: float = 4.0) -> list[dict]:
    out_dir = Path(out_dir)
    size = tile_cfg["size"]
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    records = []
    for tile, (tx, ty), (vw, vh) in iter_region_tiles(image, region, tile_cfg):
        rect = (tx, ty, tx + vw, ty + vh)
        lines, ids, thin = [], [], []
        for o in region.objects:
            c = clip_box(o["train_box"], rect)
            if c is None:
                continue
            local = to_local(c, (tx, ty))
            lines.append("0 " + " ".join(f"{v:.8f}" for v in box_to_yolo(local, size)))
            ids.append(o["id"])
            if min(c[2] - c[0], c[3] - c[1]) < thin_px:
                thin.append(o["id"])
        stem = f"{region.source_id}_{split}_x{tx}_y{ty}"
        img_rel, lab_rel = f"images/{split}/{stem}.png", f"labels/{split}/{stem}.txt"
        save_image(tile, out_dir / img_rel)
        (out_dir / lab_rel).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        records.append({"source_id": region.source_id, "split": split, "image_file": img_rel, "label_file": lab_rel,
                        "crop_rect": list(map(int, region.crop_rect)), "tile_origin_original": [tx, ty],
                        "tile_offset_in_crop": [tx - region.crop_rect[0], ty - region.crop_rect[1]],
                        "valid_wh": [vw, vh], "padding_wh": [size - vw, size - vh],
                        "object_ids": ids, "thin_fragments": thin})
    return records


def assert_no_heldout(records, heldout: str) -> None:
    bad = [r for r in records if r["source_id"] == heldout]
    assert not bad, f"held-out source {heldout} appears in {len(bad)} training/selection tiles"


def assert_disjoint(a: Region, b: Region) -> None:
    ids = {o["id"] for o in a.objects} & {o["id"] for o in b.objects}
    assert not ids, f"object ids shared between {a.name} and {b.name}: {sorted(ids)[:5]}"
    ax0, ay0, ax1, ay1 = a.crop_rect
    bx0, by0, bx1, by1 = b.crop_rect
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if ix1 > ix0 and iy1 > iy0:
        sa = a.support[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
        sb = b.support[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
        assert not (sa & sb).any(), f"{a.name} and {b.name} share original pixels"


def size_bins(regions: list[Region]) -> dict:
    """Small/medium/large thresholds on sqrt(box area) from training labels (fixed before testing)."""
    s = np.array([math.sqrt((o["bbox_xyxy"][2] - o["bbox_xyxy"][0]) * (o["bbox_xyxy"][3] - o["bbox_xyxy"][1]))
                  for r in regions for o in r.objects])
    t1, t2 = np.percentile(s, [100 / 3, 200 / 3])
    return {"metric": "sqrt(box_area) px", "small_lt": float(t1), "large_ge": float(t2), "source": "training labels"}


# ---- commands ----------------------------------------------------------------------

def load_annotation(cfg, source_id: str, allow_provisional: bool) -> dict:
    path = cfg["paths"]["annotations_dir"] / f"{source_id}.json"
    ann = json.loads(path.read_text(encoding="utf-8"))
    ann["_file"], ann["_sha256"] = path, sha256_file(path)
    if ann["review_status"] != "human_verified" and not allow_provisional:
        raise SystemExit(f"{source_id} labels are '{ann['review_status']}', not human-verified. "
                         "Re-run with --allow-provisional-labels; all outputs will be marked PROVISIONAL.")
    return ann


def _write_yaml(path: Path, out_dir: Path, names) -> None:
    path.write_text(yaml.safe_dump({"path": str(out_dir), "train": "images/train", "val": "images/val",
                                    "names": dict(enumerate(names))}, allow_unicode=True), encoding="utf-8")


def _fresh(out_dir: Path, processed_root: Path) -> None:
    out_dir = out_dir.resolve()
    assert processed_root.resolve() in out_dir.parents, f"refusing to clear {out_dir}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def build_fold(cfg, fold: str, allow_provisional: bool) -> dict:
    f = cfg["folds"][fold]
    dev, heldout = f["dev"], f["heldout"]
    ann = load_annotation(cfg, dev, allow_provisional)
    split = compute_inner_split(ann, cfg["inner_split"])
    assert_disjoint(split["train"], split["val"])
    image = load_rgb(cfg["source_paths"][dev]["image"])
    out_dir = cfg["paths"]["processed_dir"] / f"fold_{fold}"
    _fresh(out_dir, cfg["paths"]["processed_dir"])
    thin = cfg["labels"]["thin_clip_px"]
    records = []
    for name in ("train", "val"):
        records += export_tiles(image, split[name], out_dir, name, cfg["tiles"], thin)
    assert_no_heldout(records, heldout)
    _write_yaml(out_dir / "dataset.yaml", out_dir, cfg["class_names"])
    provisional = ann["review_status"] != "human_verified"
    manifest = {
        "fold": fold, "dev_source": dev, "heldout_source": heldout,
        "annotation_file": ann["_file"].relative_to(cfg["root"]).as_posix(), "annotation_sha256": ann["_sha256"],
        "annotation_version": ann.get("annotation_version"), "review_status": ann["review_status"],
        "provisional": provisional, "image_sha256": ann["image_sha256"],
        "geometry": split["geometry"], "excluded_ids": split["excluded_ids"],
        "partially_masked": split["partially_masked"],
        "regions": {n: split[n].summary() for n in ("train", "val")},
        "train_object_ids": [o["id"] for o in split["train"].objects],
        "val_object_ids": [o["id"] for o in split["val"].objects],
        "val_reference_ids": split["val"].ref_ids,
        "size_bins": size_bins([split["train"]]),
        "tiles": records,
        "counts": {n: {"tiles": sum(r["split"] == n for r in records),
                       "empty_tiles": sum(r["split"] == n and not r["object_ids"] for r in records),
                       "label_instances": sum(len(r["object_ids"]) for r in records if r["split"] == n),
                       "thin_fragments": sum(len(r["thin_fragments"]) for r in records if r["split"] == n)}
                   for n in ("train", "val")},
        "checks": {"train_val_pixels_disjoint": True, "train_val_ids_disjoint": True, "heldout_absent": True},
        "dataset_yaml": (out_dir / "dataset.yaml").relative_to(cfg["root"]).as_posix(),
    }
    mdir = cfg["paths"]["manifests_dir"] / f"fold_{fold}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    c = manifest["counts"]
    print(f"fold {fold}: dev={dev} heldout={heldout} {'PROVISIONAL labels' if provisional else 'verified labels'}; "
          f"train {c['train']['tiles']} tiles/{len(manifest['train_object_ids'])} objects, "
          f"val {c['val']['tiles']} tiles/{len(manifest['val_object_ids'])} objects "
          f"({len(split['val'].ref_ids)} reference centres); excluded {len(split['excluded_ids'])} gap objects")
    return manifest


def build_final(cfg, allow_provisional: bool) -> dict:
    """Train-all dataset: full reviewed ROIs of both labeled photos. 'val' repeats the training
    tiles only because the trainer needs a path; any metrics on it are resubstitution."""
    out_dir = cfg["paths"]["processed_dir"] / "final"
    _fresh(out_dir, cfg["paths"]["processed_dir"])
    records, summary = [], {}
    for sid in sorted({f["dev"] for f in cfg["folds"].values()}):
        ann = load_annotation(cfg, sid, allow_provisional)
        region = compute_full_region(ann)
        image = load_rgb(cfg["source_paths"][sid]["image"])
        recs = export_tiles(image, region, out_dir, "train", cfg["tiles"], cfg["labels"]["thin_clip_px"])
        records += recs
        summary[sid] = {"annotation_sha256": ann["_sha256"], "review_status": ann["review_status"],
                        "region": region.summary(), "tiles": len(recs)}
    shutil.copytree(out_dir / "images" / "train", out_dir / "images" / "val")
    shutil.copytree(out_dir / "labels" / "train", out_dir / "labels" / "val")
    _write_yaml(out_dir / "dataset.yaml", out_dir, cfg["class_names"])
    manifest = {"purpose": "final train-all demonstration model; val split = training tiles (resubstitution only)",
                "sources": summary, "tiles": records,
                "provisional": any(s["review_status"] != "human_verified" for s in summary.values())}
    mdir = cfg["paths"]["manifests_dir"] / "final"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"final: {len(records)} tiles from {sorted(summary)}")
    return manifest


def build_all(cfg, allow_provisional: bool) -> None:
    for fold in cfg["folds"]:
        build_fold(cfg, fold, allow_provisional)
    build_final(cfg, allow_provisional)
