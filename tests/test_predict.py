import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from barnacles import predict as P
from barnacles.dataset import make_region
from barnacles.geometry import box_centers, points_in_polygon, tile_starts
from barnacles.metrics import match_boxes

ROOT = Path(__file__).resolve().parents[1]

TILES = {"size": 640, "stride": 512, "pad_value": 114}
INFER = {"candidate_conf": 0.05, "local_nms_iou": 0.5, "global_nms_iou": 0.5, "max_det": 2000, "batch": 4}


class PixelDetector:
    """Fake detector: every white connected blob in a tile is one detection.

    Confidence encodes blob size so truncated seam fragments score lower, like a real model.
    """

    def __init__(self, max_det=2000):
        self.max_det = max_det
        self.calls = 0

    def __call__(self, tiles):
        self.calls += 1
        out = []
        for t in tiles:
            white = (t == 255).all(axis=2).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
            dets = [[x, y, x + w, y + h, min(1.0, 0.3 + a / 100), 0] for x, y, w, h, a in st[1:]]
            out.append(np.array(dets[:self.max_det], dtype=float).reshape(-1, 6))
        return out


def squares_image(width, height, boxes, bg=60):
    img = np.full((height, width, 3), bg, np.uint8)
    for x1, y1, x2, y2 in boxes:
        img[int(y1):int(y2), int(x1):int(x2)] = 255
    return img


# ---- merging ------------------------------------------------------------------

def test_merge_keeps_highest_score_of_duplicates():
    boxes = np.array([[0, 0, 10, 10], [0.5, 0, 10.5, 10], [30, 0, 40, 10]], dtype=float)
    keep = P.merge_detections(boxes, np.array([0.6, 0.9, 0.5]), iou=0.5)
    assert keep.tolist() == [1, 2]


def test_merge_keeps_two_nearby_non_overlapping_objects():
    boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=float)
    assert sorted(P.merge_detections(boxes, np.array([0.9, 0.8]), 0.5).tolist()) == [0, 1]


def test_merge_empty():
    assert P.merge_detections(np.zeros((0, 4)), np.zeros(0), 0.5).tolist() == []


def test_merge_confidence_ties_are_deterministic():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    for _ in range(5):
        assert P.merge_detections(boxes, np.array([0.5, 0.5, 0.5]), 0.5).tolist() == [0]


def test_threshold_then_merge_equals_merge_then_threshold():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 300, (400, 2))
    wh = rng.uniform(8, 30, (400, 2))
    boxes = np.concatenate([xy, xy + wh], 1)
    scores = rng.uniform(0.05, 1, 400)
    keep_all = P.merge_detections(boxes, scores, 0.5)
    for t in (0.1, 0.25, 0.5):
        a = set(keep_all[scores[keep_all] >= t].tolist())
        sel = np.where(scores >= t)[0]
        b = set(sel[P.merge_detections(boxes[sel], scores[sel], 0.5)].tolist())
        assert a == b


# ---- full region path ---------------------------------------------------------------

SEAM_ROI = [[0, 0], [1152, 0], [1152, 640], [0, 640]]  # tiles start at x = 0 and 512


def test_object_seen_in_two_tiles_counts_once():
    box = [560, 300, 580, 320]  # inside the 512..640 overlap: fully visible in both tiles
    img = squares_image(1152, 640, [box])
    res = P.predict_image(img, make_region("s", SEAM_ROI, 1152, 640), PixelDetector(), TILES, INFER)
    assert res.raw_count == 2
    assert res.count(0.25) == 1
    assert res.seam_edge_detections(0.25) == 0
    assert res.boxes[res.select(0.25)].tolist() == [box]


def test_heavily_truncated_seam_fragment_is_a_known_nms_failure():
    # x 630..660 straddles tile 0's right edge (640): tile 0 sees a 10 px fragment (IoU 1/3 with the
    # full box seen by tile 1). Global IoU-NMS at 0.5 cannot merge it: documented limitation.
    img = squares_image(1152, 640, [[630, 300, 660, 330]])
    res = P.predict_image(img, make_region("s", SEAM_ROI, 1152, 640), PixelDetector(), TILES, INFER)
    assert res.count(0.25) == 2
    assert res.seam_edge_detections(0.25) == 1  # the fragment touching tile 0's interior edge


def test_predictions_with_centres_in_padding_are_dropped():
    img = squares_image(500, 500, [[100, 100, 120, 120]])
    det = PixelDetector()

    def det_with_ghost(tiles):
        out = det(tiles)
        return [np.vstack([o, [[560, 560, 580, 580, 0.99, 0]]]) for o in out]

    res = P.predict_image(img, make_region("s", None, 500, 500), det_with_ghost, TILES, INFER)
    assert res.count(0.25) == 1


def test_outside_roi_predictions_do_not_count():
    roi = [[200, 200], [800, 150], [850, 600], [180, 640]]
    inside, outside = [400, 400, 420, 420], [50, 50, 70, 70]
    img = squares_image(1000, 700, [inside, outside])
    res = P.predict_image(img, make_region("s", roi, 1000, 700), PixelDetector(), TILES, INFER)
    assert res.count(0.25) == 1


def dense_layout():
    # ROI crop starts at (100, 100); squares at crop offsets 4 + 20k never straddle a tile edge
    roi = [[100, 100], [1400, 100], [1400, 1400], [100, 1400]]
    starts = 100 + 4 + 20 * np.arange(65)
    boxes = [[x, y, x + 8, y + 8] for y in starts for x in starts]
    return roi, np.array(boxes, dtype=float)


def test_dense_region_counts_thousands_without_hidden_caps():
    roi, boxes = dense_layout()
    img = squares_image(1500, 1500, boxes)
    res = P.predict_image(img, make_region("s", roi, 1500, 1500), PixelDetector(), TILES, INFER)
    assert len(boxes) == 4225 > 1501
    assert res.count(0.25) == 4225
    assert not res.saturated_tiles


def test_tile_detection_cap_saturation_is_reported():
    roi, boxes = dense_layout()
    img = squares_image(1500, 1500, boxes)
    res = P.predict_image(img, make_region("s", roi, 1500, 1500), PixelDetector(max_det=100), TILES,
                          {**INFER, "max_det": 100})
    assert res.saturated_tiles


def test_integration_counts_and_metrics_match_known_layout():
    roi = [[150, 120], [1250, 160], [1220, 1180], [130, 1150]]   # slightly rotated quad
    region = make_region("s", roi, 1500, 1300)
    cx0, cy0, cx1, cy1 = region.crop_rect
    x_edges = {cx0 + s + d for s in tile_starts(cx1 - cx0, 640, 512) for d in (0, 640)}
    y_edges = {cy0 + s + d for s in tile_starts(cy1 - cy0, 640, 512) for d in (0, 640)}
    rng = np.random.default_rng(7)
    boxes = []
    for y in range(60, 1260, 37):
        for x in range(60, 1440, 37):
            b = [x + int(rng.integers(0, 8)), y + int(rng.integers(0, 8))]
            b += [b[0] + 14, b[1] + 12]
            corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            if any(b[0] < e < b[2] for e in x_edges) or any(b[1] < e < b[3] for e in y_edges):
                continue  # straddles a tile edge: a truncated duplicate is a separate, documented case
            if points_in_polygon(corners, roi).all() or b[2] < cx0 or b[0] > cx1 or b[3] < cy0 or b[1] > cy1:
                boxes.append(b)
    boxes = np.array(boxes, dtype=float)
    img = squares_image(1500, 1300, boxes)
    res = P.predict_image(img, region, PixelDetector(), TILES, INFER)
    expected = points_in_polygon(box_centers(boxes), roi)
    sel = res.select(0.25)
    assert expected.sum() > 500
    assert res.count(0.25) == int(expected.sum())
    m = match_boxes(res.boxes[sel], res.scores[sel], boxes[expected], 0.5)
    assert (m.tp, m.fp, m.fn) == (int(expected.sum()), 0, 0)


# ---- inference isolation ---------------------------------------------------------------

def write_photo(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def test_prediction_needs_no_mask_and_ignores_mask_files(tmp_path):
    roi = [[0, 0], [700, 0], [700, 700], [0, 700]]
    img = squares_image(700, 700, [[100, 100, 120, 120], [300, 300, 330, 330]])
    photo = write_photo(tmp_path / "照片" / "img9.png", img)
    a = P.predict_region(photo, roi, PixelDetector(), TILES, INFER)
    mask = np.zeros((700, 700, 4), np.uint8)
    mask[400:450, 400:450] = (39, 57, 191, 255)
    Image.fromarray(mask, "RGBA").save(tmp_path / "照片" / "mask9.png")
    b = P.predict_region(photo, roi, PixelDetector(), TILES, INFER)
    assert a.count(0.25) == b.count(0.25) == 2
    assert np.array_equal(a.boxes, b.boxes)


def test_source_role_guard_rejects_masks_and_overlays(tmp_path):
    photo = write_photo(tmp_path / "img1.png", squares_image(50, 50, []))
    overlay = write_photo(tmp_path / "masked_img1.png", squares_image(50, 50, [[1, 1, 5, 5]]))
    from barnacles.imaging import sha256_file
    roles = {"files": {"a/img1.png": sha256_file(photo), "a/masked_img1.png": sha256_file(overlay)}}
    P.check_source_role(photo, roles)
    with pytest.raises(ValueError, match="photograph"):
        P.check_source_role(overlay, roles)
    shutil.copy(overlay, tmp_path / "renamed.png")
    with pytest.raises(ValueError, match="photograph"):
        P.check_source_role(tmp_path / "renamed.png", roles)


def test_predict_cli_writes_demo_outputs_and_refuses_overlays(tmp_path):
    from barnacles.config import load_config
    from barnacles.imaging import sha256_file
    cfg = load_config(ROOT / "configs" / "experiment.yaml")
    photo = write_photo(tmp_path / "照片" / "unseen_x.png", squares_image(800, 700, [[100, 100, 120, 120],
                                                                                 [500, 400, 530, 430]]))
    rois = tmp_path / "rois.json"
    rois.write_text(json.dumps({"rois": {"unseen_x": {"polygon": [[50, 50], [700, 50], [700, 650], [50, 650]]}}}),
                    encoding="utf-8")
    out = tmp_path / "out"
    doc = P.predict_cli(cfg, image=str(photo), roi_file=str(rois), whole_image=False, checkpoint=None, conf=0.25,
                        out_dir=str(out), detector=PixelDetector())
    assert doc["count"] == 2 and doc["status"].startswith("demonstration")
    assert (out / "predictions.json").exists() and (out / "overlay.jpg").exists()
    whole = P.predict_cli(cfg, image=str(photo), roi_file=None, whole_image=True, checkpoint=None, conf=0.25,
                          out_dir=str(tmp_path / "whole"), detector=PixelDetector())
    assert whole["region"]["name"] == "whole_image"
    overlay = cfg["paths"]["raw_dir"] / "masked_img1.png"
    if overlay.exists() and sha256_file(overlay):
        with pytest.raises(ValueError, match="photograph"):
            P.predict_cli(cfg, image=str(overlay), roi_file=None, whole_image=True, checkpoint=None, conf=0.25,
                          out_dir=str(tmp_path / "bad"), detector=PixelDetector())


def test_prediction_json_round_trip(tmp_path):
    roi = [[0, 0], [700, 0], [700, 700], [0, 700]]
    img = squares_image(700, 700, [[100, 100, 120, 120]])
    res = P.predict_image(img, make_region("s", roi, 700, 700), PixelDetector(), TILES, INFER)
    doc = res.to_json(conf=0.25, meta={"image_id": "s"})
    text = json.dumps(doc)
    back = json.loads(text)
    assert back["count"] == 1
    assert back["detections"][0]["bbox_xyxy"] == [100.0, 100.0, 120.0, 120.0]
    assert back["detections"][0]["center"] == [110.0, 110.0]
    assert back["detections"][0]["class_id"] == 0
