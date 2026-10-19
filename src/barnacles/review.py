"""Review renderings: overlays, numbered contact sheets and gridded ROI tiles.

All functions take and return arrays; callers decide where files go.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import polygon_bbox

FLAG_COLORS = {"tiny": (255, 140, 0), "open_contour": (255, 230, 0), "multi_ring": (255, 0, 0),
               "thin": (0, 200, 255), "large": (180, 0, 255)}


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=1, origin=(0, 0)) -> np.ndarray:
    out = np.ascontiguousarray(image.copy())
    ox, oy = origin
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x1 - ox, y1 - oy), (x2 - 1 - ox, y2 - 1 - oy), color, thickness)
    return out


def draw_polygon(image, poly, color=(255, 255, 0), thickness=2, origin=(0, 0)) -> np.ndarray:
    out = np.ascontiguousarray(image.copy())
    pts = (np.asarray(poly, dtype=float) - np.asarray(origin, dtype=float)).round().astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, color, thickness)
    return out


def tint_strokes(image, stroke, color=(255, 0, 255), alpha=0.6) -> np.ndarray:
    out = image.copy()
    out[stroke] = (alpha * np.array(color) + (1 - alpha) * out[stroke]).astype(np.uint8)
    return out


def _label(img, text, org, scale=0.4):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y), (x + tw + 2, y + th + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 1, y + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def contact_sheet(image, stroke, items, panel=96, cols=8, margin=10, extra=None):
    """Numbered zoom panels for candidates (objects with .id/.bbox_xyxy/.flags).

    extra: optional {candidate_id: [(box, color), ...]} drawn on that candidate's panel
    (e.g. split proposals or neighbouring candidates). Returns (sheet, index) where index
    maps panel numbers to candidate IDs.
    """
    extra = extra or {}
    H, W = image.shape[:2]
    rows = max(1, -(-len(items) // cols))
    sheet = np.full((rows * panel, cols * panel, 3), 255, np.uint8)
    index = []
    for n, c in enumerate(items):
        x1, y1, x2, y2 = c.bbox_xyxy
        side = max(x2 - x1, y2 - y1, 24) + 2 * margin
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        sx1, sy1 = sx0 + int(round(side)), sy0 + int(round(side))
        crop = np.zeros((sy1 - sy0, sx1 - sx0, 3), np.uint8)
        ax0, ay0, ax1, ay1 = max(sx0, 0), max(sy0, 0), min(sx1, W), min(sy1, H)
        if ax1 > ax0 and ay1 > ay0:
            part = tint_strokes(image[ay0:ay1, ax0:ax1], stroke[ay0:ay1, ax0:ax1], alpha=0.45)
            crop[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = part
        scale = panel / crop.shape[0]
        big = cv2.resize(crop, (panel, panel), interpolation=cv2.INTER_NEAREST)
        bx = [(x1 - sx0) * scale, (y1 - sy0) * scale, (x2 - sx0) * scale, (y2 - sy0) * scale]
        color = FLAG_COLORS.get(c.flags[0], (0, 255, 255)) if c.flags else (0, 255, 0)
        for box, col in extra.get(c.id, []):
            big = draw_boxes(big, [[(box[0] - sx0) * scale, (box[1] - sy0) * scale,
                                    (box[2] - sx0) * scale, (box[3] - sy0) * scale]], color=col)
        big = draw_boxes(big, [bx], color=color)
        _label(big, str(n + 1), (1, 1))
        r, k = divmod(n, cols)
        sheet[r * panel:(r + 1) * panel, k * panel:(k + 1) * panel] = big
        cv2.rectangle(sheet, (k * panel, r * panel), ((k + 1) * panel - 1, (r + 1) * panel - 1), (255, 255, 255), 1)
        entry = {"panel": n + 1, "candidate_id": c.id, "bbox_xyxy": list(c.bbox_xyxy), "flags": list(c.flags)}
        if c.id in extra:
            entry["extra_boxes"] = [list(b) for b, _ in extra[c.id]]
        index.append(entry)
    return sheet, index


def review_tiles(image, stroke, roi, size=640, grid=50):
    """Native-resolution tiles over the ROI bounding box with strokes, ROI outline and a labelled
    coordinate grid (original-image pixels) for locating targets by eye."""
    H, W = image.shape[:2]
    x0, y0, x1, y1 = polygon_bbox(roi, W, H)
    tiles = []
    for ty in range(y0, y1, size):
        for tx in range(x0, x1, size):
            ex, ey = min(tx + size, W), min(ty + size, H)
            t = tint_strokes(image[ty:ey, tx:ex], stroke[ty:ey, tx:ex], alpha=0.55)
            t = draw_polygon(t, roi, origin=(tx, ty))
            for gx in range((tx // grid + 1) * grid, ex, grid):
                cv2.line(t, (gx - tx, 0), (gx - tx, 6), (255, 255, 0), 1)
                if gx % (2 * grid) == 0:
                    _label(t, str(gx), (gx - tx + 1, 0), 0.33)
            for gy in range((ty // grid + 1) * grid, ey, grid):
                cv2.line(t, (0, gy - ty), (6, gy - ty), (255, 255, 0), 1)
                if gy % (2 * grid) == 0:
                    _label(t, str(gy), (0, gy - ty + 1), 0.33)
            tiles.append({"origin": (tx, ty), "image": t})
    return tiles
