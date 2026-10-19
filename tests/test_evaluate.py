import numpy as np
import pytest

from barnacles import evaluate as E
from barnacles.geometry import distance_to_polygon_edges


def row(conf, f1, recall):
    return {"confidence": conf, "f1": f1, "recall": recall}


def test_choose_threshold_maximises_f1():
    rows = [row(0.1, 0.70, 0.9), row(0.25, 0.80, 0.8), row(0.5, 0.75, 0.6)]
    assert E.choose_threshold(rows)["confidence"] == 0.25


def test_choose_threshold_breaks_f1_ties_by_recall_then_higher_confidence():
    assert E.choose_threshold([row(0.1, 0.8, 0.85), row(0.2, 0.8, 0.9)])["confidence"] == 0.2
    assert E.choose_threshold([row(0.1, 0.8, 0.9), row(0.3, 0.8, 0.9)])["confidence"] == 0.3


def test_score_predictions_hand_calculated():
    refs = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]], dtype=float)
    preds = np.array([[0, 0, 10, 10], [21, 0, 31, 10], [100, 0, 110, 10], [40, 0, 50, 10]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    s = E.score_predictions(preds, scores, refs, conf=0.5, iou=0.5, center_factor=0.5)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 2)
    assert s["gt_count"] == 4 and s["pred_count"] == 3 and s["signed_count_error"] == -1
    assert s["absolute_percentage_error"] == 25.0
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == 0.5
    assert s["center_match"]["tp"] == 2


def test_seam_band_flags_points_near_interior_tile_edges():
    # crop 0..1152 wide: tiles at x 0 and 512 -> interior edges at x=512 and x=640 (region edges excluded)
    crop = (0, 0, 1152, 640)
    pts = np.array([[500, 300], [700, 300], [100, 300], [1140, 300], [660, 300]], dtype=float)
    band = E.seam_band(pts, crop, tile=640, stride=512, band=32)
    assert band.tolist() == [True, False, False, False, True]


def test_size_bins_assignment():
    bins = {"small_lt": 18.0, "large_ge": 24.0}
    boxes = np.array([[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]], dtype=float)
    assert E.size_bin(boxes, bins).tolist() == ["small", "medium", "large"]


def test_distance_to_polygon_edges():
    sq = [[0, 0], [100, 0], [100, 100], [0, 100]]
    d = distance_to_polygon_edges(np.array([[50, 50], [10, 50], [0, 0], [150, 50]], dtype=float), sq)
    assert d.tolist() == pytest.approx([50, 10, 0, 50])


def test_breakdown_reports_recall_per_group():
    refs = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], dtype=float)
    matched = np.array([0, -1, 2])
    groups = np.array(["a", "a", "b"])
    out = E.recall_by_group(matched, groups)
    assert out == {"a": {"n": 2, "tp": 1, "recall": 0.5}, "b": {"n": 1, "tp": 1, "recall": 1.0}}
