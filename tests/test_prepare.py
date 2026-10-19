import pytest

from barnacles.annotations import Candidate, CandidateSet, validate_annotations
from barnacles.prepare import assemble_annotation

ROI = [[0, 0], [200, 0], [200, 100], [0, 100]]


def cset():
    return CandidateSet("t", "Barnacles/x.png", "Barnacles/x_mask.png", "a" * 64, "b" * 64, 200, 100, [
        Candidate("t_c000001", 40, [10.0, 10.0, 22.0, 22.0], 1, []),
        Candidate("t_c000002", 3, [50.0, 50.0, 52.0, 51.0], 0, ["tiny"])])


def test_missing_corrections_means_pending_and_no_reviewed_region():
    ann = assemble_annotation(cset(), None, ROI, "v0")
    assert ann["review_status"] == "pending"
    assert ann["reviewed_region_polygons"] == []
    assert ann["review_provenance"] is None
    assert [o["id"] for o in ann["objects"]] == ["t_000001", "t_000002"]
    assert validate_annotations(ann, {"width": 200, "height": 100, "sha256": "a" * 64}).ok


def test_ai_review_is_provisional_never_human_verified():
    corr = {"schema_version": 1, "image_id": "t",
            "reviewer": {"type": "ai_visual_inspection", "name": "model"},
            "reviewed_region_polygons": [ROI],
            "operations": [{"op": "delete_artifact", "ids": ["t_c000002"], "reason": "speck"}]}
    ann = assemble_annotation(cset(), corr, ROI, "v1")
    assert ann["review_status"] == "ai_visual_inspection_provisional"
    assert ann["review_provenance"]["human_verified"] is False
    assert ann["reviewed_region_polygons"] == [ROI]
    assert [o["id"] for o in ann["objects"]] == ["t_000001"]
    assert ann["summary"]["n_candidates"] == 2 and ann["summary"]["n_objects"] == 1


def test_corrections_for_another_image_rejected():
    corr = {"image_id": "other", "reviewer": {"type": "human"}, "operations": []}
    with pytest.raises(ValueError, match="image"):
        assemble_annotation(cset(), corr, ROI, "v1")
