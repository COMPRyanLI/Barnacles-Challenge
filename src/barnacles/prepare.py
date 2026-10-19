"""Label-stage orchestration: source hashes, candidates, review renderings, corrected annotations."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import review
from .annotations import (Candidate, CandidateSet, apply_corrections, extract_candidates, propose_ring_split,
                          stroke_mask, validate_annotations)
from .config import load_rois
from .imaging import load_rgb, load_rgba, save_image, sha256_file

STATUS_BY_REVIEWER = {"ai_visual_inspection": "ai_visual_inspection_provisional", "human": "human_verified"}


def rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assemble_annotation(cs: CandidateSet, corrections: dict | None, roi, version: str) -> dict:
    """Build the per-image annotation document (plan section 5.3) from candidates + corrections."""
    if corrections is not None and corrections.get("image_id") != cs.image_id:
        raise ValueError(f"corrections are for image {corrections.get('image_id')!r}, not {cs.image_id!r}")
    corr = corrections or {"image_id": cs.image_id, "operations": []}
    objects = apply_corrections(cs.candidates, corr)
    reviewer = corr.get("reviewer")
    status = STATUS_BY_REVIEWER.get(reviewer.get("type"), "pending") if reviewer else "pending"
    provenance = None
    if reviewer:
        provenance = {**reviewer, "reviewer_type": reviewer.get("type"),
                      "human_verified": reviewer.get("type") == "human",
                      "n_operations": len(corr.get("operations", [])),
                      "notes": corr.get("notes")}
    ops = Counter(o["op"] for o in corr.get("operations", []))
    return {
        "schema_version": 1,
        "annotation_version": version,
        "image_id": cs.image_id,
        "image_path": cs.image_path,
        "image_sha256": cs.image_sha256,
        "mask_sha256": cs.mask_sha256,
        "width": cs.width,
        "height": cs.height,
        "roi_polygon": [list(map(float, p)) for p in roi],
        "reviewed_region_polygons": corr.get("reviewed_region_polygons", []) if reviewer else [],
        "review_status": status,
        "review_provenance": provenance,
        "summary": {"n_candidates": len(cs.candidates), "n_objects": len(objects), "operations": dict(ops),
                    "object_status": dict(Counter(o["review_status"] for o in objects))},
        "objects": objects,
    }


def candidate_set_from_dict(d: dict) -> CandidateSet:
    return CandidateSet(d["image_id"], d["image_path"], d["mask_path"], d["image_sha256"], d["mask_sha256"],
                        d["width"], d["height"], [Candidate(**c) for c in d["candidates"]])


# ---- commands ----------------------------------------------------------------

def hash_sources(cfg) -> dict:
    root = cfg["root"]
    files = sorted(p for p in cfg["paths"]["raw_dir"].glob("*.png") if not p.name.startswith("._"))
    data = {"description": "SHA-256 of the supplied PNGs (raw files are never modified).",
            "files": {rel(p, root): sha256_file(p) for p in files}}
    write_json(cfg["paths"]["manifests_dir"] / "source_hashes.json", data)
    return data


def _component(labels: np.ndarray, bbox) -> np.ndarray:
    """The candidate's own component inside its box (the largest label there)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = labels[y1:y2, x1:x2]
    vals, counts = np.unique(crop[crop > 0], return_counts=True)
    return crop == vals[counts.argmax()]


def labeled_sources(cfg):
    return [sid for sid, s in cfg["source_paths"].items() if s["role"] == "labeled"]


def prepare_labels(cfg) -> dict:
    hashes = hash_sources(cfg)
    out = {}
    for sid in labeled_sources(cfg):
        s = cfg["source_paths"][sid]
        cs = extract_candidates(s["image"], s["mask"], image_id=sid, flag_config=cfg["labels"])
        cs.image_path, cs.mask_path = rel(s["image"], cfg["root"]), rel(s["mask"], cfg["root"])
        write_json(cfg["paths"]["annotations_dir"] / "candidates" / f"{sid}_candidates.json", cs.to_dict())
        flags = Counter(f for c in cs.candidates for f in c.flags)
        out[sid] = {"n_candidates": len(cs.candidates), "flags": dict(flags),
                    "unflagged": sum(1 for c in cs.candidates if not c.flags)}
        print(f"{sid}: {len(cs.candidates)} candidate components (NOT a verified count); flags {dict(flags)}")
    print(f"hashed {len(hashes['files'])} source PNGs")
    return out


def render_label_review(cfg) -> None:
    rois = load_rois(cfg["paths"]["annotations_dir"] / "rois.json")
    out_root = cfg["paths"]["reports_dir"] / "label_review"
    for sid in labeled_sources(cfg):
        s = cfg["source_paths"][sid]
        cs = candidate_set_from_dict(read_json(cfg["paths"]["annotations_dir"] / "candidates" / f"{sid}_candidates.json"))
        img, stroke = load_rgb(s["image"]), stroke_mask(load_rgba(s["mask"]))
        d = out_root / sid
        overlay = review.draw_polygon(review.tint_strokes(img, stroke), rois[sid], thickness=4)
        flagged = [c for c in cs.candidates if c.flags]
        for c in flagged:
            overlay = review.draw_boxes(overlay, [c.bbox_xyxy], review.FLAG_COLORS[c.flags[0]], 2)
        save_image(overlay, d / "overlay_full.jpg")
        order = sorted(flagged, key=lambda c: (c.flags[0], c.id))
        for start in range(0, len(order), 80):
            sheet, index = review.contact_sheet(img, stroke, order[start:start + 80], panel=96, cols=10)
            n = start // 80 + 1
            save_image(sheet, d / f"flagged_sheet_{n:02d}.png")
            write_json(d / f"flagged_sheet_{n:02d}.json", index)
        # detail sheets: split proposals for multi-ring outlines, neighbours for open contours
        _, labels = cv2.connectedComponents(stroke.astype(np.uint8), connectivity=8)
        boxes = np.array([c.bbox_xyxy for c in cs.candidates])
        proposals, extra = {}, {}
        for c in flagged:
            if "multi_ring" in c.flags:
                proposals[c.id] = propose_ring_split(_component(labels, c.bbox_xyxy), c.bbox_xyxy[:2],
                                                     cfg["labels"]["min_hole_area_px"])
                extra[c.id] = [(b, (0, 255, 0)) for b in proposals[c.id]]
            elif "open_contour" in c.flags:
                x1, y1, x2, y2 = c.bbox_xyxy
                near = (boxes[:, 0] < x2 + 25) & (boxes[:, 2] > x1 - 25) & (boxes[:, 1] < y2 + 25) & (boxes[:, 3] > y1 - 25)
                extra[c.id] = [(b.tolist(), (0, 255, 255)) for b in boxes[near] if b.tolist() != c.bbox_xyxy]
        write_json(d / "split_proposals.json", proposals)
        detail = [c for c in order if c.id in extra]
        if detail:
            sheet, index = review.contact_sheet(img, stroke, detail, panel=160, cols=6, margin=18, extra=extra)
            save_image(sheet, d / "detail_sheet.png")
            write_json(d / "detail_sheet.json", index)
        for t in review.review_tiles(img, stroke, rois[sid]):
            x, y = t["origin"]
            save_image(t["image"], d / "roi_tiles" / f"tile_x{x}_y{y}.jpg", quality=92)
        print(f"{sid}: review renderings in {rel(d, cfg['root'])} ({len(flagged)} flagged candidates)")
    for sid, s in cfg["source_paths"].items():
        if s["role"] == "unlabeled_demo" and rois.get(sid):
            img = review.draw_polygon(load_rgb(s["image"]), rois[sid], thickness=4)
            save_image(img, out_root / f"{sid}_roi.jpg")


def build_annotations(cfg) -> dict:
    """validate-data: apply corrections, validate, and write the per-image annotation files."""
    rois = load_rois(cfg["paths"]["annotations_dir"] / "rois.json")
    hashes = read_json(cfg["paths"]["manifests_dir"] / "source_hashes.json")["files"]
    results = {}
    for sid in labeled_sources(cfg):
        cdir = cfg["paths"]["annotations_dir"]
        cs = candidate_set_from_dict(read_json(cdir / "candidates" / f"{sid}_candidates.json"))
        corr_path = cdir / "corrections" / f"{sid}_corrections.json"
        corr = read_json(corr_path) if corr_path.exists() else None
        ann = assemble_annotation(cs, corr, rois[sid], cfg["labels"]["annotation_version"])
        if corr is not None:
            ann["review_provenance"]["corrections_file"] = rel(corr_path, cfg["root"])
            ann["review_provenance"]["corrections_sha256"] = sha256_file(corr_path)
        meta = {"width": cs.width, "height": cs.height, "sha256": hashes[cs.image_path]}
        rep = validate_annotations(ann, meta)
        if sha256_file(cfg["root"] / cs.image_path) != hashes[cs.image_path]:
            rep.errors.append("raw image changed since hashing")
        write_json(cdir / f"{sid}.json", ann)
        results[sid] = {"ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings,
                        "review_status": ann["review_status"], **ann["summary"]}
        print(f"{sid}: {'valid' if rep.ok else 'INVALID'}; {ann['summary']['n_objects']} objects "
              f"from {ann['summary']['n_candidates']} candidates; review_status={ann['review_status']}")
        for e in rep.errors:
            print("  ERROR", e)
    write_json(cfg["paths"]["manifests_dir"] / "annotation_validation.json", results)
    if not all(r["ok"] for r in results.values()):
        raise SystemExit("annotation validation failed")
    return results
