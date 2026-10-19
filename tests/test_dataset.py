import numpy as np
import pytest
from PIL import Image

from barnacles import dataset as D
from barnacles.geometry import rasterize_polygon, yolo_to_box

ROI = [[100, 100], [1100, 100], [1100, 900], [100, 900]]
SPLIT = {"train_fraction": 0.65, "gap_px": 64, "max_masked_fraction_of_retained": 0.5}
TILES = {"size": 640, "stride": 512, "pad_value": 114}
# ROI bbox x 100..1100 -> split line 750, gap [718, 782)


def obj(oid, box):
    return {"id": oid, "class_id": 0, "bbox_xyxy": [float(v) for v in box], "origin_candidate_ids": [],
            "review_status": "ai_reviewed"}


OBJECTS = [
    obj("left", [200, 200, 230, 230]),
    obj("right", [900, 200, 930, 230]),
    obj("cross", [770, 400, 800, 430]),
    obj("inside_gap", [730, 500, 760, 530]),
    obj("neighbor", [795, 400, 815, 430]),
    obj("cross2", [775, 600, 803, 630]),
    obj("victim", [785, 600, 805, 630]),
    obj("edge", [90, 300, 120, 330]),
    obj("edge_out", [80, 400, 103, 420]),
]


def ann(objects=OBJECTS, reviewed=None):
    return {"image_id": "s", "width": 1300, "height": 1000, "roi_polygon": ROI,
            "reviewed_region_polygons": [reviewed or ROI], "review_status": "ai_visual_inspection_provisional",
            "objects": objects}


def full_support(region, W=1300, H=1000):
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = region.crop_rect
    m[y0:y1, x0:x1] = region.support
    return m


# ---- inner split --------------------------------------------------------------

def test_inner_split_is_disjoint_in_pixels_and_object_ids():
    s = D.compute_inner_split(ann(), SPLIT)
    tr, va = s["train"], s["val"]
    assert not (full_support(tr) & full_support(va)).any()
    tr_ids, va_ids = {o["id"] for o in tr.objects}, {o["id"] for o in va.objects}
    assert not tr_ids & va_ids
    assert "left" in tr_ids and "right" in va_ids
    assert s["geometry"]["split_x"] == 750 and s["geometry"]["gap"] == [718.0, 782.0]
    assert full_support(tr)[:, 718:].sum() == 0 and full_support(va)[:, :782].sum() == 0


def test_gap_crossing_objects_are_excluded_and_masked_in_both_sets():
    s = D.compute_inner_split(ann(), SPLIT)
    assert {"cross", "inside_gap", "cross2"} <= set(s["excluded_ids"])
    sup = full_support(s["val"])
    assert not sup[400:430, 782:800].any()  # visible part of 'cross' masked on the val side


def test_masking_cascade_excludes_mostly_erased_targets_and_records_partial_ones():
    s = D.compute_inner_split(ann(), SPLIT)
    assert "victim" in s["excluded_ids"]
    assert "neighbor" in {o["id"] for o in s["val"].objects}
    assert "neighbor" in s["partially_masked"]
    assert 0 < s["partially_masked"]["neighbor"] <= 0.5


def test_boundary_targets_are_clipped_and_reference_uses_original_centre():
    tr = D.compute_inner_split(ann(), SPLIT)["train"]
    by_id = {o["id"]: o for o in tr.objects}
    assert by_id["edge"]["train_box"] == [100.0, 300.0, 120.0, 330.0]
    assert by_id["edge_out"]["train_box"] == [100.0, 400.0, 103.0, 420.0]
    assert "edge" in tr.ref_ids and "edge_out" not in tr.ref_ids


def test_region_contains_agrees_with_support_raster():
    va = D.compute_inner_split(ann(), SPLIT)["val"]
    x0, y0, x1, y1 = va.crop_rect
    ys, xs = np.mgrid[y0:y1, x0:x1]
    centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
    assert np.array_equal(va.contains(centres).reshape(va.support.shape), va.support)


def test_split_without_reviewed_validation_content_fails():
    left_only = [[100, 100], [600, 100], [600, 900], [100, 900]]
    with pytest.raises(ValueError, match="val"):
        D.compute_inner_split(ann(reviewed=left_only), SPLIT)


def test_unreviewed_objects_never_become_labels():
    left_only = [[100, 100], [600, 100], [600, 900], [100, 900]]
    region = D.compute_full_region(ann(reviewed=left_only))
    ids = {o["id"] for o in region.objects}
    assert "left" in ids and "right" not in ids


def test_full_region_reference_is_centre_in_roi():
    region = D.compute_full_region(ann())
    assert set(region.ref_ids) == {o["id"] for o in OBJECTS} - {"edge_out"}
    assert np.array_equal(full_support(region), rasterize_polygon(ROI, 1300, 1000))


# ---- tiles --------------------------------------------------------------------

def halves_image():
    img = np.zeros((1000, 1300, 3), np.uint8)
    img[:, :750] = 10
    img[:, 750:] = 200
    return img


def read_labels(path):
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else [list(map(float, ln.split())) for ln in text.splitlines()]


def test_exported_labels_round_trip_to_original_coordinates(tmp_path):
    s = D.compute_inner_split(ann(), SPLIT)
    out = tmp_path / "数据 tiles"
    records = D.export_tiles(halves_image(), s["train"], out, "train", TILES, thin_px=4)
    assert records
    boxes = {o["id"]: o["train_box"] for o in s["train"].objects}
    for r in records:
        labels = read_labels(out / r["label_file"])
        assert len(labels) == len(r["object_ids"])
        tx, ty = r["tile_origin_original"]
        for line, oid in zip(labels, r["object_ids"]):
            assert line[0] == 0
            back = yolo_to_box(line[1:], 640) + np.array([tx, ty, tx, ty])
            b = boxes[oid]
            expect = [max(b[0], tx), max(b[1], ty), min(b[2], tx + r["valid_wh"][0]), min(b[3], ty + r["valid_wh"][1])]
            assert np.abs(back - expect).max() < 1e-4
        img = np.array(Image.open(out / r["image_file"]))
        assert img.shape == (640, 640, 3)


def test_thin_clipped_fragments_are_flagged_not_dropped(tmp_path):
    s = D.compute_inner_split(ann(), SPLIT)
    records = D.export_tiles(halves_image(), s["train"], tmp_path, "train", TILES, thin_px=4)
    flagged = {oid for r in records for oid in r["thin_fragments"]}
    labelled = {oid for r in records for oid in r["object_ids"]}
    assert "edge_out" in flagged and "edge_out" in labelled


def test_background_tiles_get_empty_label_files(tmp_path):
    region = D.compute_full_region(ann(objects=[obj("one", [200, 200, 220, 220])]))
    records = D.export_tiles(halves_image(), region, tmp_path, "all", TILES, thin_px=4)
    empty = [r for r in records if not r["object_ids"]]
    assert empty
    for r in empty:
        assert (tmp_path / r["label_file"]).read_text(encoding="utf-8") == ""


def test_val_tiles_never_contain_train_pixels_and_vice_versa(tmp_path):
    s = D.compute_inner_split(ann(), SPLIT)
    img = halves_image()
    for name, forbidden in (("train", 200), ("val", 10)):
        for r in D.export_tiles(img, s[name], tmp_path, name, TILES, thin_px=4):
            tile = np.array(Image.open(tmp_path / r["image_file"]))
            assert not (tile == forbidden).all(axis=2).any(), (name, r["image_file"])


def test_small_region_is_padded_without_rescaling(tmp_path):
    small_roi = [[100, 100], [400, 100], [400, 300], [100, 300]]
    a = ann(objects=[obj("one", [200, 200, 220, 220])])
    a["roi_polygon"], a["reviewed_region_polygons"] = small_roi, [small_roi]
    region = D.compute_full_region(a)
    records = D.export_tiles(halves_image(), region, tmp_path, "all", TILES, thin_px=4)
    assert len(records) == 1 and records[0]["valid_wh"] == [300, 200]
    tile = np.array(Image.open(tmp_path / records[0]["image_file"]))
    assert (tile[:200, :300] == 10).all() and (tile[200:] == 114).all() and (tile[:, 300:] == 114).all()


def test_heldout_source_in_training_manifest_is_rejected():
    with pytest.raises(AssertionError, match="held-out"):
        D.assert_no_heldout([{"source_id": "img1"}, {"source_id": "img2"}], heldout="img2")
