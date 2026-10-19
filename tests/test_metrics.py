import numpy as np
import pytest

from barnacles import metrics as m

GT = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], dtype=float)


def test_exact_predictions_all_match():
    r = m.match_boxes(GT.copy(), np.array([0.9, 0.8, 0.7]), GT, iou_threshold=0.5)
    assert (r.tp, r.fp, r.fn) == (3, 0, 0)
    assert r.pred_to_ref.tolist() == [0, 1, 2]


def test_all_missed():
    r = m.match_boxes(np.zeros((0, 4)), np.zeros(0), GT, 0.5)
    assert (r.tp, r.fp, r.fn) == (0, 0, 3)


def test_all_false_positives():
    preds = np.array([[100, 100, 110, 110], [200, 200, 210, 210]], dtype=float)
    r = m.match_boxes(preds, np.array([0.5, 0.4]), GT, 0.5)
    assert (r.tp, r.fp, r.fn) == (0, 2, 3)


def test_duplicate_predictions_only_one_matches():
    preds = np.array([[0, 0, 10, 10], [0.5, 0, 10.5, 10]], dtype=float)
    r = m.match_boxes(preds, np.array([0.6, 0.9]), GT[:1], 0.5)
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)
    assert r.pred_to_ref.tolist() == [-1, 0]  # higher score claims the reference


def test_one_prediction_between_two_targets_matches_at_most_one():
    refs = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], dtype=float)
    pred = np.array([[4, 0, 14, 10]], dtype=float)  # IoU 60/140 with ref0, 80/120 with ref1
    iou = m.box_iou(pred, refs)[0]
    r = m.match_boxes(pred, np.array([0.9]), refs, iou_threshold=0.4)
    assert r.tp == 1 and r.fn == 1
    assert r.pred_to_ref[0] == int(np.argmax(iou))


def test_iou_threshold_is_inclusive():
    pred = np.array([[0, 0, 10, 10]], dtype=float)
    ref = np.array([[0, 0, 10, 20]], dtype=float)  # IoU exactly 0.5
    assert m.match_boxes(pred, np.array([1.0]), ref, 0.5).tp == 1
    assert m.match_boxes(pred, np.array([1.0]), ref, 0.5000001).tp == 0


def test_confidence_ties_break_by_prediction_order():
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    r = m.match_boxes(preds, np.array([0.5, 0.5]), GT[:1], 0.5)
    assert r.pred_to_ref.tolist() == [0, -1]


def test_empty_reference_and_predictions():
    r = m.match_boxes(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), 0.5)
    assert (r.tp, r.fp, r.fn) == (0, 0, 0)


def test_prf_conventions():
    assert m.prf(0, 0, 5) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert m.prf(0, 3, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert m.prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    out = m.prf(8, 2, 4)
    assert out["precision"] == pytest.approx(0.8)
    assert out["recall"] == pytest.approx(8 / 12)
    assert out["f1"] == pytest.approx(2 * 8 / (2 * 8 + 2 + 4))


def test_count_errors_hand_calculated():
    assert m.count_errors(n_ref=50, n_pred=46) == {
        "gt_count": 50, "pred_count": 46, "signed_count_error": -4,
        "absolute_count_error": 4, "absolute_percentage_error": 8.0}
    zero = m.count_errors(n_ref=0, n_pred=3)
    assert zero["absolute_percentage_error"] is None
    assert zero["false_positive_count_on_empty"] == 3
    assert m.count_errors(0, 0)["empty_region_exact"] is True


def test_center_matching_uses_per_reference_tolerance():
    refs = np.array([[0, 0, 20, 20], [100, 100, 104, 104]], dtype=float)  # tolerances 10 and 2
    preds = np.array([[8, 0, 28, 20], [103, 100, 107, 104]], dtype=float)  # centre offsets 8 and 3
    r = m.match_centers(preds, np.array([0.9, 0.8]), refs, tolerance_factor=0.5)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)
    assert r.pred_to_ref.tolist() == [0, -1]


def test_center_matching_picks_nearest_unmatched():
    refs = np.array([[0, 0, 20, 20], [6, 0, 26, 20]], dtype=float)
    preds = np.array([[5, 0, 25, 20]], dtype=float)
    r = m.match_centers(preds, np.array([1.0]), refs, 0.5)
    assert r.pred_to_ref.tolist() == [1]


def test_aggregate_macro_counts_and_micro_detection():
    per_fold = [
        {"tp": 8, "fp": 2, "fn": 2, "gt_count": 10, "pred_count": 10,
         "absolute_count_error": 0, "absolute_percentage_error": 0.0},
        {"tp": 1, "fp": 0, "fn": 9, "gt_count": 10, "pred_count": 1,
         "absolute_count_error": 9, "absolute_percentage_error": 90.0},
    ]
    agg = m.aggregate(per_fold)
    assert agg["macro_absolute_count_error"] == 4.5
    assert agg["macro_absolute_percentage_error"] == 45.0
    assert (agg["micro_tp"], agg["micro_fp"], agg["micro_fn"]) == (9, 2, 11)
    assert agg["micro_recall"] == pytest.approx(9 / 20)
