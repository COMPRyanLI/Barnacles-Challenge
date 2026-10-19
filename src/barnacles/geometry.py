"""Pure geometry: polygons, pixel support, tiles and box transforms.

Conventions: original-image pixel coordinates, origin top-left, x rightward.
Boxes are float [x1, y1, x2, y2] with x2/y2 exclusive. Pixel (i, j) covers
[i, i+1) x [j, j+1); a pixel belongs to a polygon when its centre does.
A point on a polygon boundary (within TOL) counts as inside.
"""
from __future__ import annotations

import math

import numpy as np

TOL = 1e-6


# ---- polygons -----------------------------------------------------------

def as_polygon(poly) -> np.ndarray:
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"polygon must be an (N, 2) array, got shape {arr.shape}")
    return arr


def polygon_area(poly) -> float:
    p = as_polygon(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, q1, q2) -> bool:
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
        return True

    def on_seg(a, b, c):  # c collinear with a-b and within its extent
        return (abs(_orient(a, b, c)) <= TOL and min(a[0], b[0]) - TOL <= c[0] <= max(a[0], b[0]) + TOL
                and min(a[1], b[1]) - TOL <= c[1] <= max(a[1], b[1]) + TOL)

    return on_seg(q1, q2, p1) or on_seg(q1, q2, p2) or on_seg(p1, p2, q1) or on_seg(p1, p2, q2)


def validate_polygon(poly) -> np.ndarray:
    """Return the polygon as an (N, 2) float array or raise ValueError."""
    p = as_polygon(poly)
    if len(p) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    if not np.isfinite(p).all():
        raise ValueError("polygon has non-finite coordinates")
    if abs(polygon_area(p)) <= TOL:
        raise ValueError("polygon has zero area")
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex
            if _segments_intersect(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]):
                raise ValueError(f"polygon is self-intersecting (edges {i} and {j})")
    return p


def points_in_polygon(points, poly, tol: float = TOL) -> np.ndarray:
    """Boolean mask of points inside or on the boundary of a simple polygon."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    p = as_polygon(poly)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    a, b = p, np.roll(p, -1, axis=0)
    px, py = pts[:, 0:1], pts[:, 1:2]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    # distance to each edge segment -> boundary test
    ex, ey = bx - ax, by - ay
    seg_len2 = np.maximum(ex * ex + ey * ey, 1e-300)
    t = np.clip(((px - ax) * ex + (py - ay) * ey) / seg_len2, 0.0, 1.0)
    dist2 = (px - (ax + t * ex)) ** 2 + (py - (ay + t * ey)) ** 2
    on_boundary = (dist2 <= tol * tol).any(axis=1)
    # even-odd ray casting to +x
    crosses = (ay > py) != (by > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = ax + (py - ay) * ex / np.where(ey == 0, np.inf, ey)
    inside = (crosses & (px < x_int)).sum(axis=1) % 2 == 1
    return inside | on_boundary


def distance_to_polygon_edges(points, poly) -> np.ndarray:
    """Euclidean distance from each point to the nearest polygon edge (inside or outside)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    p = as_polygon(poly)
    a, b = p, np.roll(p, -1, axis=0)
    px, py = pts[:, 0:1], pts[:, 1:2]
    ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    t = np.clip(((px - a[:, 0]) * ex + (py - a[:, 1]) * ey) / np.maximum(ex * ex + ey * ey, 1e-300), 0, 1)
    d2 = (px - (a[:, 0] + t * ex)) ** 2 + (py - (a[:, 1] + t * ey)) ** 2
    return np.sqrt(d2.min(axis=1)) if len(pts) else np.zeros(0)


def polygon_bbox(poly, width: int | None = None, height: int | None = None) -> tuple[int, int, int, int]:
    """Integer pixel rectangle (x0, y0, x1, y1), x1/y1 exclusive, optionally clipped to the image."""
    p = as_polygon(poly)
    x0, y0 = math.floor(p[:, 0].min()), math.floor(p[:, 1].min())
    x1, y1 = math.ceil(p[:, 0].max()), math.ceil(p[:, 1].max())
    if width is not None:
        x0, x1 = max(x0, 0), min(x1, width)
    if height is not None:
        y0, y1 = max(y0, 0), min(y1, height)
    return x0, y0, x1, y1


def rasterize_polygon(poly, width: int, height: int, origin=(0, 0)) -> np.ndarray:
    """(height, width) mask of pixels whose centres lie in the polygon; pixel (0,0) is at `origin`."""
    p = as_polygon(poly)
    ox, oy = origin
    mask = np.zeros((height, width), dtype=bool)
    bx0, by0, bx1, by1 = polygon_bbox(p)
    x0, x1 = max(bx0 - ox, 0), min(bx1 - ox, width)
    y0, y1 = max(by0 - oy, 0), min(by1 - oy, height)
    if x0 >= x1 or y0 >= y1:
        return mask
    # Scanline parity: for each row of pixel centres, count edge crossings to the right of each
    # centre (the same ray-casting rule and x-intersection formula as points_in_polygon).
    ys = np.arange(height) + oy + 0.5
    a, b = p, np.roll(p, -1, axis=0)
    ax, ay, ex, ey = a[:, 0], a[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    py = ys[:, None]
    crosses = (ay > py) != (b[:, 1] > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = ax + (py - ay) * ex / np.where(ey == 0, np.inf, ey)
    k = np.clip(np.ceil(x_int - ox - 0.5), 0, width)       # centres with cx < x_int are i < k
    diff = np.zeros((height, width + 1), dtype=np.int32)
    rows = np.nonzero(crosses)
    np.add.at(diff, (rows[0], np.zeros(len(rows[0]), int)), 1)
    np.add.at(diff, (rows[0], k[rows].astype(int)), -1)
    mask = (np.cumsum(diff[:, :width], axis=1) % 2 == 1)
    # boundary rule: centres within TOL of an edge count as inside; only centres inside an edge's
    # (TOL-expanded) bounding box can qualify, so test just those exactly
    for (sx, sy), (tx, ty) in zip(a, b):
        cx0 = max(math.ceil(min(sx, tx) - TOL - ox - 0.5), 0)
        cx1 = min(math.floor(max(sx, tx) + TOL - ox - 0.5), width - 1)
        cy0 = max(math.ceil(min(sy, ty) - TOL - oy - 0.5), 0)
        cy1 = min(math.floor(max(sy, ty) + TOL - oy - 0.5), height - 1)
        if cx1 < cx0 or cy1 < cy0:
            continue
        yy, xx = np.mgrid[cy0:cy1 + 1, cx0:cx1 + 1]
        pts = np.stack([xx.ravel() + ox + 0.5, yy.ravel() + oy + 0.5], 1)
        on = points_in_polygon(pts, [[sx, sy], [tx, ty], [sx, sy]]).reshape(yy.shape)  # degenerate = segment test
        mask[cy0:cy1 + 1, cx0:cx1 + 1] |= on
    return mask


# ---- tiles -------------------------------------------------------------

def tile_starts(length: int, tile: int, stride: int, pad_last: bool = False) -> list[int]:
    """Tile origins along one axis. Default: stride steps plus a final tile ending at the edge.
    pad_last=True (non-overlap ablation): stride steps only; the last tile runs into padding."""
    if pad_last:
        return list(range(0, max(length, 1), stride))
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last = max(length - tile, 0)
    if last not in starts:
        starts.append(last)
    return starts


def tile_grid(width: int, height: int, tile: int, stride: int, pad_last: bool = False) -> list[tuple[int, int]]:
    """Tile origins (x0, y0), row-major."""
    return [(x, y) for y in tile_starts(height, tile, stride, pad_last)
            for x in tile_starts(width, tile, stride, pad_last)]


def extract_tile(image: np.ndarray, x0: int, y0: int, size: int, pad_value: int = 114):
    """Copy a size x size tile; regions past the image edge are padded, never rescaled.

    Returns (tile, (valid_width, valid_height)).
    """
    part = image[y0:y0 + size, x0:x0 + size]
    vh, vw = part.shape[:2]
    tile = np.full((size, size) + image.shape[2:], pad_value, dtype=image.dtype)
    tile[:vh, :vw] = part
    return tile, (vw, vh)


# ---- boxes -------------------------------------------------------------

def clip_box(box, rect):
    """Intersect a box with rect (x0, y0, x1, y1); None when the intersection has no area."""
    x1, y1 = max(box[0], rect[0]), max(box[1], rect[1])
    x2, y2 = min(box[2], rect[2]), min(box[3], rect[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def to_local(box, crop_origin, tile_origin=(0, 0)) -> np.ndarray:
    dx, dy = crop_origin[0] + tile_origin[0], crop_origin[1] + tile_origin[1]
    return np.asarray(box, dtype=float) - np.array([dx, dy, dx, dy], dtype=float)


def to_original(box, crop_origin, tile_origin=(0, 0)) -> np.ndarray:
    dx, dy = crop_origin[0] + tile_origin[0], crop_origin[1] + tile_origin[1]
    return np.asarray(box, dtype=float) + np.array([dx, dy, dx, dy], dtype=float)


def box_to_yolo(box, size: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size)


def yolo_to_box(line, size: int) -> np.ndarray:
    cx, cy, w, h = (float(v) * size for v in line)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def box_centers(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def box_iou(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
