import csv

import pytest

from barnacles import reporting as R


def test_render_template_fills_placeholders_and_formats_numbers():
    text = "F1 {{A.f1}}, error {{A.ape}}%, n={{A.gt}}, missing {{A.none}}"
    out = R.render(text, {"A.f1": 0.50512, "A.ape": 46.347, "A.gt": 397, "A.none": None}, pct_keys={"A.ape"})
    assert out == "F1 0.505, error 46.3%, n=397, missing null (not measured)"


def test_render_template_rejects_unknown_placeholders():
    with pytest.raises(KeyError, match="A.f2"):
        R.render("{{A.f2}}", {"A.f1": 0.5})


def test_results_csv_has_required_columns(tmp_path):
    row = {c: None for c in R.RESULT_COLUMNS}
    row.update({"experiment_id": "x", "fold": "A", "gt_count": 10})
    p = tmp_path / "results.csv"
    R.write_results_csv(p, [row])
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert set(R.RESULT_COLUMNS) <= set(rows[0])
    assert rows[0]["latency_median_ms"] == "null"
    for required in ("experiment_id", "fold", "train_source", "heldout_source", "annotation_version", "review_status",
                     "scored_region", "reviewed_area_fraction", "checkpoint_sha256", "selected_epoch", "seed",
                     "confidence", "local_nms_iou", "global_nms_iou", "gt_count", "pred_count", "signed_count_error",
                     "absolute_count_error", "absolute_percentage_error", "tp", "fp", "fn", "precision", "recall",
                     "f1", "tile_size", "tile_stride", "tile_count", "device", "precision_mode",
                     "latency_median_ms", "latency_p90_ms"):
        assert required in R.RESULT_COLUMNS


def test_result_row_from_records_marks_provisional_status():
    metrics = {"fold": "A", "experiment_id": "e", "train_source": "img1", "heldout_source": "img2",
               "annotation_version": "v1", "review_status": "ai_visual_inspection_provisional", "provisional": True,
               "scored_region": "roi", "reviewed_area_fraction": 1.0, "checkpoint_sha256": "s", "selected_epoch": 80,
               "seed": 42, "confidence": 0.5, "local_nms_iou": 0.5, "global_nms_iou": 0.5, "tiles": 9,
               "tile_size": 640, "tile_stride": 512, "device": "0", "precision_mode": "fp32",
               "primary": {"gt_count": 10, "pred_count": 8, "signed_count_error": -2, "absolute_count_error": 2,
                           "absolute_percentage_error": 20.0, "tp": 7, "fp": 1, "fn": 3, "precision": 0.875,
                           "recall": 0.7, "f1": 0.778}}
    row = R.result_row(metrics, bench=None, result_type="primary")
    assert row["result_status"] == "provisional-label held-out result"
    assert row["latency_median_ms"] is None and row["f1"] == 0.778


def test_junit_summary_counts(tmp_path):
    p = tmp_path / "j.xml"
    p.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="0" failures="1" skipped="2" '
                 'tests="10" time="1.0"></testsuite></testsuites>', encoding="utf-8")
    assert R.junit_summary(p) == {"tests": 10, "failures": 1, "errors": 0, "skipped": 2, "passed": 7}
    assert R.junit_summary(tmp_path / "missing.xml") is None


def test_meets_targets_requires_both_criteria():
    assert R.meets_targets({"absolute_percentage_error": 4.3, "f1": 0.71}) is False
    assert R.meets_targets({"absolute_percentage_error": 9.9, "f1": 0.80}) is True
    assert R.meets_targets({"absolute_percentage_error": None, "f1": 0.9}) is False
