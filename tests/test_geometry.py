import numpy as np
import pytest

from barnacles import geometry as g

SQUARE = [[10.0, 10.0], [110.0, 10.0], [110.0, 110.0], [10.0, 110.0]]


# ---- polygons -----------------------------------------------------------

def test_points_inside_outside_and_on_boundary():
    pts = np.array([[50, 50], [5, 50], [10, 50], [110, 110], [110 + 1e-7, 60], [110 + 1e-3, 60]])
    inside = g.points_in_polygon(pts, SQUARE)
    assert inside.tolist() == [True, False, True, True, True, False]


def test_points_in_rotated_polygon_differs_from_its_bounding_box():
    diamond = [[50, 0], [100, 50], [50, 100], [0, 50]]
    corner_of_bbox = np.array([[5.0, 5.0]])
    assert not g.points_in_polygon(corner_of_bbox, diamond)[0]
    x0, y0, x1, y1 = g.polygon_bbox(diamond)
    assert x0 <= 5 < x1 and y0 <= 5 < y1


def test_points_in_polygon_empty_input():
    assert g.points_in_polygon(np.zeros((0, 2)), SQUARE).shape == (0,)


@pytest.mark.parametrize("poly", [
    [[0, 0], [10, 0]],                                  # too few vertices
    [[0, 0], [10, 0], [20, 0]],                         # zero area
    [[0, 0], [10, 10], [10, 0], [0, 10]],               # bow-tie self-intersection
    [[0, 0], [10, 0], [float("nan"), 10]],              # non-finite
])
def test_invalid_polygons_rejected(poly):
    with pytest.raises(ValueError):
        g.validate_polygon(poly)


def test_validate_polygon_accepts_convex_quad_in_either_orientation():
    g.validate_polygon(SQUARE)
    g.validate_polygon(SQUARE[::-1])


def test_rasterize_polygon_uses_pixel_centres():
    mask = g.rasterize_polygon([[2, 1], [6, 1], [6, 4], [2, 4]], width=8, height=6)
    assert mask.shape == (6, 8)
    assert mask.sum() == 4 * 3
    assert mask[1, 2] and mask[3, 5] and not mask[4, 2] and not mask[1, 6]


def test_rasterize_polygon_with_origin_offset():
    full = g.rasterize_polygon(SQUARE, width=200, height=200)
    part = g.rasterize_polygon(SQUARE, width=50, height=40, origin=(90, 100))
    assert np.array_equal(part, full[100:140, 90:140])


@pytest.mark.parametrize("seed", range(6))
def test_rasterize_matches_point_test_on_random_polygons(seed):
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
    rad = rng.uniform(15, 40, 7)
    poly = np.stack([50 + rad * np.cos(ang), 45 + rad * np.sin(ang)], 1)
    if seed % 2:  # half-pixel vertices put pixel centres exactly on edges
        poly = np.round(poly) + 0.5
    W, H, origin = 97, 91, (3, -2)
    mask = g.rasterize_polygon(poly, W, H, origin=origin)
    ys, xs = np.mgrid[0:H, 0:W]
    centres = np.stack([xs.ravel() + origin[0] + 0.5, ys.ravel() + origin[1] + 0.5], 1)
    assert np.array_equal(mask, g.points_in_polygon(centres, poly).reshape(H, W))


def test_rasterize_counts_centres_on_horizontal_and_vertical_edges():
    poly = [[2.5, 1.5], [6.5, 1.5], [6.5, 4.5], [2.5, 4.5]]  # edges pass exactly through pixel centres
    mask = g.rasterize_polygon(poly, 9, 7)
    assert mask.sum() == 5 * 4 and mask[1, 2] and mask[4, 6] and not mask[0, 2]


def test_polygon_bbox_is_integer_and_clipped():
    assert g.polygon_bbox([[-5.5, 2.2], [30.1, 2.2], [30.1, 9.7]], width=20, height=50) == (0, 2, 20, 10)


# ---- tiles -------------------------------------------------------------

@pytest.mark.parametrize("length,expected", [
    (500, [0]),
    (640, [0]),
    (641, [0, 1]),
    (1152, [0, 512]),
    (1153, [0, 512, 513]),
    (1417, [0, 512, 777]),
])
def test_tile_starts(length, expected):
    assert g.tile_starts(length, 640, 512) == expected


def test_non_overlapping_ablation_tiles_pad_the_last_tile():
    assert g.tile_starts(1430, 640, 640, pad_last=True) == [0, 640, 1280]
    assert g.tile_starts(1280, 640, 640, pad_last=True) == [0, 640]
    tiles = g.tile_grid(1430, 700, 640, 640, pad_last=True)
    covered = np.zeros((700, 1430), int)
    for x0, y0 in tiles:
        covered[y0:y0 + 640, x0:x0 + 640] += 1
    assert (covered == 1).all()


@pytest.mark.parametrize("width,height", [(500, 500), (1417, 1433), (1280, 1152), (641, 3), (2001, 640)])
def test_tile_grid_covers_every_pixel_without_duplicates(width, height):
    tiles = g.tile_grid(width, height, 640, 512)
    assert len(set(tiles)) == len(tiles)
    covered = np.zeros((height, width), dtype=bool)
    for x0, y0 in tiles:
        covered[y0:y0 + 640, x0:x0 + 640] = True
    assert covered.all()
    for x0, y0 in tiles:
        assert 0 <= x0 and 0 <= y0
        assert x0 + 640 <= max(width, 640) and y0 + 640 <= max(height, 640)


def test_extract_tile_pads_small_images_without_rescaling():
    img = np.arange(500 * 500 * 3, dtype=np.uint32).reshape(500, 500, 3).astype(np.uint8)
    tile, valid = g.extract_tile(img, 0, 0, 640, pad_value=114)
    assert tile.shape == (640, 640, 3)
    assert valid == (500, 500)
    assert np.array_equal(tile[:500, :500], img)
    assert (tile[500:] == 114).all() and (tile[:, 500:] == 114).all()


# ---- boxes -------------------------------------------------------------

def test_clip_box_border_crossing_and_outside():
    assert g.clip_box([-5, 10, 20, 30], (0, 0, 100, 100)) == (0, 10, 20, 30)
    assert g.clip_box([90, 95, 120, 130], (0, 0, 100, 100)) == (90, 95, 100, 100)
    assert g.clip_box([100, 10, 120, 30], (0, 0, 100, 100)) is None
    assert g.clip_box([10, 10, 20, 30], (20, 0, 100, 100)) is None


def test_original_to_tile_to_yolo_round_trip():
    rng = np.random.default_rng(0)
    crop_origin = (1123, 463)
    tile_origin = (512, 777)
    for _ in range(200):
        x1, y1 = rng.uniform(0, 600, 2)
        w, h = rng.uniform(1, 40, 2)
        orig = np.array([x1, y1, x1 + w, y1 + h]) + np.r_[crop_origin, crop_origin] + np.r_[tile_origin, tile_origin]
        tile_box = g.to_local(orig, crop_origin, tile_origin)
        line = g.box_to_yolo(tile_box, 640)
        back = g.to_original(g.yolo_to_box(line, 640), crop_origin, tile_origin)
        assert np.abs(back - orig).max() < 1e-4


def test_box_to_yolo_values():
    cx, cy, w, h = g.box_to_yolo([0, 0, 64, 32], 640)
    assert (cx, cy, w, h) == pytest.approx((0.05, 0.025, 0.1, 0.05))


def test_box_iou_matrix():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    iou = g.box_iou(a, b)
    assert iou.shape == (2, 3)
    assert iou[0].tolist() == pytest.approx([1.0, 50 / 150, 0.0])
    assert g.box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_box_centers():
    assert g.box_centers(np.array([[0, 0, 10, 20]])).tolist() == [[5.0, 10.0]]
