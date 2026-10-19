"""Contour-mask parsing, explicit label corrections and annotation validation.

The supplied masks are blue contour strokes on a transparent background. A
connected component is a *candidate*, not a verified barnacle: touching
outlines merge and broken outlines fragment. Corrections are explicit,
replayable operations with reasons; nothing is silently discarded.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import validate_polygon
from .imaging import load_rgba, image_size, sha256_file

DEFAULT_FLAGS = {"tiny_area_px": 10, "thin_aspect_ratio": 3.0, "large_side_factor": 2.5, "min_hole_area_px": 5}


@dataclass
class Candidate:
    id: str
    area: int
    bbox_xyxy: list
    holes: int
    flags: list = field(default_factory=list)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return [(x1 + x2) / 2, (y1 + y2) / 2]


@dataclass
class CandidateSet:
    image_id: str
    image_path: str
    mask_path: str
    image_sha256: str
    mask_sha256: str
    width: int
    height: int
    candidates: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d["warning"] = ("Connected contour components are candidates, NOT verified barnacle counts. "
                        "Touching outlines can merge; broken outlines can fragment.")
        return d


def stroke_mask(rgba: np.ndarray) -> np.ndarray:
    """Annotation foreground = alpha > 0 (background is transparent white, so RGB thresholds fail)."""
    return rgba[..., 3] > 0


def _count_holes(component: np.ndarray, min_area: int) -> int:
    """Enclosed background regions of at least min_area pixels (4-connected background).

    Thick strokes in the supplied masks contain 1-2 px pinholes; those are not ring interiors.
    """
    padded = np.pad(component, 1).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats((1 - padded).astype(np.uint8), connectivity=4)
    # label 0 is the component itself, label 1 the outer background (touches the padding)
    return int((stats[2:, cv2.CC_STAT_AREA] >= min_area).sum()) if n > 2 else 0


def propose_ring_split(component: np.ndarray, origin=(0, 0), min_hole_area: int = 5) -> list[list[float]]:
    """Proposed per-ring boxes for a component whose outline encloses several interiors.

    Each stroke pixel is assigned to its nearest enclosed interior; a ring's box is the
    bounding box of its interior plus assigned stroke pixels. This is a proposal that a
    reviewer accepts explicitly through a `split` correction, never an automatic edit.
    """
    padded = np.pad(component, 1).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((1 - padded).astype(np.uint8), connectivity=4)
    holes = [k for k in range(2, n) if stats[k, cv2.CC_STAT_AREA] >= min_hole_area]
    ox, oy = origin[0] - 1, origin[1] - 1  # undo padding

    def box(mask):
        ys, xs = np.nonzero(mask)
        return [float(xs.min() + ox), float(ys.min() + oy), float(xs.max() + 1 + ox), float(ys.max() + 1 + oy)]

    if len(holes) <= 1:
        return [box(padded > 0)]
    dist = np.stack([cv2.distanceTransform((labels != h).astype(np.uint8), cv2.DIST_L2, 5) for h in holes])
    owner = dist.argmin(axis=0)
    stroke = padded > 0
    return [box((labels == h) | (stroke & (owner == i))) for i, h in enumerate(holes)]


def candidates_from_mask(rgba: np.ndarray, image_id: str, flag_config: dict | None = None) -> list[Candidate]:
    cfg = {**DEFAULT_FLAGS, **(flag_config or {})}
    fg = stroke_mask(rgba).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    raw = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        holes = _count_holes(labels[y:y + h, x:x + w] == k, cfg["min_hole_area_px"])
        raw.append((y, x, w, h, area, holes))
    raw.sort()  # stable raster order of the box corner -> stable IDs
    sides = np.array([max(r[2], r[3]) for r in raw]) if raw else np.zeros(0)
    median_side = float(np.median(sides)) if len(sides) else 0.0
    out = []
    for i, (y, x, w, h, area, holes) in enumerate(raw, start=1):
        flags = []
        if area < cfg["tiny_area_px"]:
            flags.append("tiny")
        elif holes == 0:
            flags.append("open_contour")
        if holes >= 2:
            flags.append("multi_ring")
        if max(w, h) / max(min(w, h), 1) > cfg["thin_aspect_ratio"]:
            flags.append("thin")
        if median_side and max(w, h) > cfg["large_side_factor"] * median_side:
            flags.append("large")
        out.append(Candidate(f"{image_id}_c{i:06d}", area, [float(x), float(y), float(x + w), float(y + h)],
                             holes, flags))
    return out


def extract_candidates(image_path, mask_path, image_id: str | None = None,
                       flag_config: dict | None = None) -> CandidateSet:
    image_path, mask_path = Path(image_path), Path(mask_path)
    iw, ih = image_size(image_path)
    rgba = load_rgba(mask_path)
    if rgba.shape[:2] != (ih, iw):
        raise ValueError(f"image/mask dimensions differ: {iw}x{ih} vs {rgba.shape[1]}x{rgba.shape[0]}")
    image_id = image_id or image_path.stem
    return CandidateSet(image_id, str(image_path), str(mask_path), sha256_file(image_path),
                        sha256_file(mask_path), iw, ih, candidates_from_mask(rgba, image_id, flag_config))


# ---- corrections ---------------------------------------------------------------

def _check_box(box, where) -> list:
    b = [float(v) for v in box]
    if len(b) != 4 or not all(math.isfinite(v) for v in b):
        raise ValueError(f"{where}: box must have 4 finite values, got {box}")
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError(f"{where}: box has non-positive area: {box}")
    return b


def _num(candidate_id: str) -> str:
    return candidate_id.rsplit("_c", 1)[1]


def apply_corrections(candidates: list[Candidate], corrections: dict) -> list[dict]:
    """Deterministically turn candidates into annotation objects.

    Untouched candidates become objects with status 'unreviewed_candidate'.
    Every operation must name a reason; a candidate may appear in at most one operation.
    """
    by_id = {c.id: c for c in candidates}
    prefix = corrections.get("image_id") or (candidates[0].id.rsplit("_c", 1)[0] if candidates else "img")
    reviewer = corrections.get("reviewer", {})
    used: dict[str, int] = {}
    replaced: dict[str, dict] = {}   # candidate id -> object replacing it (None = deleted)
    extra: list[dict] = []
    n_added = 0

    def claim(cid, k):
        if cid not in by_id:
            raise ValueError(f"operation {k}: unknown candidate id {cid!r}")
        if cid in used:
            raise ValueError(f"candidate {cid} is used by more than one operation ({used[cid]} and {k})")
        used[cid] = k

    def make(oid, box, origins, op, k):
        return {"id": oid, "class_id": 0, "bbox_xyxy": box, "origin_candidate_ids": origins,
                "review_status": "ai_reviewed" if reviewer.get("type") == "ai_visual_inspection"
                else "human_reviewed" if reviewer.get("type") == "human" else "reviewed",
                "correction": {"op": op["op"], "reason": op["reason"], "operation_index": k}}

    for k, op in enumerate(corrections.get("operations", [])):
        kind = op.get("op")
        if not op.get("reason"):
            raise ValueError(f"operation {k} ({kind}): a reason is required")
        if kind in ("keep", "delete_artifact"):
            for cid in op.get("ids", []):
                claim(cid, k)
                c = by_id[cid]
                replaced[cid] = None if kind == "delete_artifact" else \
                    make(f"{prefix}_{_num(cid)}", list(c.bbox_xyxy), [cid], op, k)
        elif kind == "merge":
            ids = op.get("ids", [])
            if len(ids) < 2:
                raise ValueError(f"operation {k}: merge needs at least two candidate ids")
            for cid in ids:
                claim(cid, k)
            boxes = np.array([by_id[c].bbox_xyxy for c in ids])
            box = [float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())]
            replaced[ids[0]] = make(f"{prefix}_m{_num(ids[0])}", box, list(ids), op, k)
            for cid in ids[1:]:
                replaced[cid] = None
        elif kind == "split":
            cid, boxes = op.get("id"), op.get("boxes", [])
            if len(boxes) < 2:
                raise ValueError(f"operation {k}: split needs at least two boxes")
            claim(cid, k)
            objs = [make(f"{prefix}_{_num(cid)}_s{j}", _check_box(b, f"operation {k}"), [cid], op, k)
                    for j, b in enumerate(boxes, start=1)]
            replaced[cid] = objs
        elif kind == "replace_box":
            cid = op.get("id")
            claim(cid, k)
            replaced[cid] = make(f"{prefix}_{_num(cid)}", _check_box(op.get("box"), f"operation {k}"), [cid], op, k)
        elif kind == "add_missing":
            n_added += 1
            extra.append(make(f"{prefix}_a{n_added:04d}", _check_box(op.get("box"), f"operation {k}"), [], op, k))
        else:
            raise ValueError(f"operation {k}: unknown operation {kind!r}")

    objects = []
    for c in candidates:
        if c.id in replaced:
            r = replaced[c.id]
            if r is None:
                continue
            objects.extend(r if isinstance(r, list) else [r])
        else:
            objects.append({"id": f"{prefix}_{_num(c.id)}", "class_id": 0, "bbox_xyxy": list(c.bbox_xyxy),
                            "origin_candidate_ids": [c.id], "review_status": "unreviewed_candidate",
                            "correction": None, "flags": list(c.flags)})
    objects.extend(extra)
    ids = [o["id"] for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("corrections produced duplicate object ids")
    return objects


# ---- validation ------------------------------------------------------------

@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_annotations(ann: dict, image_metadata: dict) -> ValidationReport:
    rep = ValidationReport()
    W, H = int(image_metadata["width"]), int(image_metadata["height"])
    if (ann.get("width"), ann.get("height")) != (W, H):
        rep.errors.append(f"size mismatch: annotation {ann.get('width')}x{ann.get('height')} vs image {W}x{H}")
    if image_metadata.get("sha256") and ann.get("image_sha256") != image_metadata["sha256"]:
        rep.errors.append("image sha256 does not match the annotated image")
    for name, poly in [("roi", ann.get("roi_polygon"))] + \
            [(f"reviewed region {i}", p) for i, p in enumerate(ann.get("reviewed_region_polygons") or [])]:
        try:
            validate_polygon(poly)
        except (ValueError, TypeError) as e:
            rep.errors.append(f"{name} polygon invalid: {e}")
    if ann.get("review_status") != "pending" and not ann.get("review_provenance"):
        rep.errors.append("review provenance missing for a non-pending review status")
    seen = set()
    for o in ann.get("objects", []):
        oid = o.get("id")
        if oid in seen:
            rep.errors.append(f"duplicate object id {oid}")
        seen.add(oid)
        b = o.get("bbox_xyxy", [])
        if len(b) != 4 or not all(math.isfinite(float(v)) for v in b):
            rep.errors.append(f"{oid}: box values must be 4 finite numbers")
            continue
        if b[2] <= b[0] or b[3] <= b[1]:
            rep.errors.append(f"{oid}: box has non-positive area")
        if b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H:
            rep.errors.append(f"{oid}: box outside image bounds")
        if o.get("class_id") != 0:
            rep.errors.append(f"{oid}: class_id must be 0")
    return rep
