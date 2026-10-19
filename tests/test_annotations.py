import math

import numpy as np
import pytest
from PIL import Image

from barnacles import annotations as A
from barnacles.imaging import load_rgb

BLUE1 = (39, 57, 191, 255)
BLUE2 = (59, 76, 160, 255)


def blank_mask(w=120, h=80):
    m = np.zeros((h, w, 4), dtype=np.uint8)
    m[..., :3] = 255  # transparent white background, as in the supplied masks
    return m


def draw_ring(m, x0, y0, size=12, color=BLUE1):
    m[y0, x0:x0 + size] = color
    m[y0 + size - 1, x0:x0 + size] = color
    m[y0:y0 + size, x0] = color
    m[y0:y0 + size, x0 + size - 1] = color


def write_pair(dirpath, mask, name="img"):
    dirpath.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape[:2]
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    ip, mp = dirpath / f"{name}.png", dirpath / f"{name}_mask.png"
    Image.fromarray(img).save(ip)
    Image.fromarray(mask, "RGBA").save(mp)
    return ip, mp


# ---- mask parsing --------------------------------------------------------

def test_transparent_white_is_background():
    assert not A.stroke_mask(blank_mask()).any()


def test_both_blue_colours_are_foreground():
    m = blank_mask()
    m[5, 5] = BLUE1
    m[5, 50] = BLUE2
    assert A.stroke_mask(m).sum() == 2


def test_one_closed_ring_yields_one_unflagged_candidate():
    m = blank_mask()
    draw_ring(m, 10, 10, 12)
    cands = A.candidates_from_mask(m, image_id="t")
    assert len(cands) == 1
    c = cands[0]
    assert c.bbox_xyxy == [10.0, 10.0, 22.0, 22.0]  # half-open: pixels 10..21
    assert c.holes == 1 and c.flags == []


def test_pinholes_inside_thick_stroke_are_not_extra_rings():
    m = blank_mask()
    m[10:24, 10:24] = BLUE1
    m[13:21, 13:21] = 0       # real interior, 64 px
    m[11, 15] = 0             # 1-px pinholes inside the stroke
    m[22, 18] = 0
    cands = A.candidates_from_mask(m, image_id="t")
    assert len(cands) == 1
    assert cands[0].holes == 1 and cands[0].flags == []


def test_two_colour_ring_is_still_one_candidate():
    m = blank_mask()
    draw_ring(m, 10, 10, 12, BLUE1)
    m[10, 10:16] = BLUE2
    assert len(A.candidates_from_mask(m, image_id="t")) == 1


def test_touching_rings_are_flagged_multi_ring():
    m = blank_mask()
    draw_ring(m, 10, 10, 12)
    draw_ring(m, 21, 10, 12)  # shares a column with the first ring
    cands = A.candidates_from_mask(m, image_id="t")
    assert len(cands) == 1
    assert cands[0].holes == 2 and "multi_ring" in cands[0].flags


def test_split_proposal_separates_touching_rings():
    m = blank_mask()
    draw_ring(m, 10, 10, 12)
    draw_ring(m, 21, 14, 12)
    comp = A.stroke_mask(m)[10:26, 10:33]
    boxes = A.propose_ring_split(comp, origin=(10, 10), min_hole_area=5)
    assert len(boxes) == 2
    expected = [[10, 10, 22, 22], [21, 14, 33, 26]]
    for got, exp in zip(sorted(boxes), expected):
        assert np.abs(np.array(got) - exp).max() <= 1.0


def test_split_proposal_single_ring_returns_its_box():
    m = blank_mask()
    draw_ring(m, 30, 20, 14)
    comp = A.stroke_mask(m)[20:34, 30:44]
    assert A.propose_ring_split(comp, origin=(30, 20), min_hole_area=5) == [[30.0, 20.0, 44.0, 34.0]]


def test_open_fragment_and_speck_are_flagged_not_dropped():
    m = blank_mask()
    m[30, 40:52] = BLUE1             # open arc / fragment
    m[60, 90] = BLUE1                # one-pixel speck
    m[5:7, 100:118] = BLUE1          # long thin stroke
    cands = A.candidates_from_mask(m, image_id="t")
    flags = {tuple(c.bbox_xyxy): set(c.flags) for c in cands}
    assert len(cands) == 3
    assert "open_contour" in flags[(40.0, 30.0, 52.0, 31.0)]
    assert "tiny" in flags[(90.0, 60.0, 91.0, 61.0)]
    assert "thin" in flags[(100.0, 5.0, 118.0, 7.0)]


def test_unusually_large_component_is_flagged():
    m = blank_mask(200, 120)
    for k in range(5):
        draw_ring(m, 5 + 15 * k, 5, 12)
    draw_ring(m, 100, 40, 60)
    cands = A.candidates_from_mask(m, image_id="t")
    big = [c for c in cands if c.bbox_xyxy[0] == 100.0][0]
    assert "large" in big.flags


def test_empty_mask_returns_empty_candidates():
    assert A.candidates_from_mask(blank_mask(), image_id="t") == []


def test_candidate_ids_are_stable_and_unique():
    m = blank_mask()
    draw_ring(m, 50, 40, 12)
    draw_ring(m, 10, 10, 12)
    a = [c.id for c in A.candidates_from_mask(m, image_id="t")]
    b = [c.id for c in A.candidates_from_mask(m.copy(), image_id="t")]
    assert a == b == ["t_c000001", "t_c000002"]


def test_extract_candidates_from_unicode_path(tmp_path):
    m = blank_mask()
    draw_ring(m, 10, 10, 12)
    ip, mp = write_pair(tmp_path / "数据 目录", m)
    cs = A.extract_candidates(ip, mp, image_id="t")
    assert (cs.width, cs.height) == (120, 80)
    assert len(cs.candidates) == 1
    assert len(cs.image_sha256) == 64


def test_extract_candidates_rejects_size_mismatch(tmp_path):
    ip, _ = write_pair(tmp_path, blank_mask(120, 80))
    _, mp = write_pair(tmp_path, blank_mask(100, 80), name="other")
    with pytest.raises(ValueError, match="dimensions"):
        A.extract_candidates(ip, mp, image_id="t")


def test_load_rgb_drops_alpha(tmp_path):
    p = tmp_path / "rgba.png"
    Image.fromarray(np.full((4, 5, 4), 200, dtype=np.uint8), "RGBA").save(p)
    assert load_rgb(p).shape == (4, 5, 3)


# ---- corrections -----------------------------------------------------------

def make_candidates():
    m = blank_mask(200, 80)
    for k in range(4):
        draw_ring(m, 10 + 30 * k, 10, 12)
    m[60, 150] = BLUE1
    return A.candidates_from_mask(m, image_id="t")


def corrections(*ops, reviewer_type="ai_visual_inspection"):
    return {"schema_version": 1, "image_id": "t",
            "reviewer": {"type": reviewer_type, "name": "test"}, "operations": list(ops)}


def test_no_corrections_keeps_every_candidate_with_carried_ids():
    cands = make_candidates()
    objs = A.apply_corrections(cands, corrections())
    assert [o["id"] for o in objs] == ["t_000001", "t_000002", "t_000003", "t_000004", "t_000005"]
    assert objs[0]["origin_candidate_ids"] == ["t_c000001"]
    assert objs[0]["review_status"] == "unreviewed_candidate"


def test_delete_merge_split_add_preserve_provenance():
    cands = make_candidates()
    ids = [c.id for c in cands]
    ops = corrections(
        {"op": "delete_artifact", "ids": [ids[4]], "reason": "speck"},
        {"op": "merge", "ids": [ids[0], ids[1]], "reason": "fragments of one target"},
        {"op": "split", "id": ids[2], "boxes": [[70, 10, 76, 22], [76, 10, 82, 22]], "reason": "two targets"},
        {"op": "replace_box", "id": ids[3], "box": [100, 10, 111, 21], "reason": "tighter"},
        {"op": "add_missing", "box": [150, 40, 160, 50], "reason": "visible unannotated target"},
        {"op": "keep", "ids": [], "reason": "noop"},
    )
    objs = {o["id"]: o for o in A.apply_corrections(cands, ops)}
    assert set(objs) == {"t_m000001", "t_000003_s1", "t_000003_s2", "t_000004", "t_a0001"}
    merged = objs["t_m000001"]
    assert merged["bbox_xyxy"] == [10.0, 10.0, 52.0, 22.0]
    assert merged["origin_candidate_ids"] == [ids[0], ids[1]]
    assert merged["correction"]["op"] == "merge" and merged["correction"]["reason"]
    assert objs["t_000004"]["bbox_xyxy"] == [100.0, 10.0, 111.0, 21.0]
    assert objs["t_a0001"]["origin_candidate_ids"] == []
    assert all(o["review_status"] == "ai_reviewed" for o in objs.values())


def test_keep_marks_flagged_candidate_reviewed():
    cands = make_candidates()
    objs = A.apply_corrections(cands, corrections({"op": "keep", "ids": [cands[4].id], "reason": "real"}))
    assert objs[4]["review_status"] == "ai_reviewed"
    assert objs[0]["review_status"] == "unreviewed_candidate"


@pytest.mark.parametrize("op,msg", [
    ({"op": "delete_artifact", "ids": ["t_c999999"], "reason": "x"}, "unknown candidate"),
    ({"op": "replace_box", "id": "t_c000001", "box": [1, 1, 1, 5], "reason": "x"}, "area"),
    ({"op": "replace_box", "id": "t_c000001", "box": [1, 1, math.nan, 5], "reason": "x"}, "finite"),
    ({"op": "merge", "ids": ["t_c000001"], "reason": "x"}, "at least two"),
    ({"op": "split", "id": "t_c000001", "boxes": [[0, 0, 5, 5]], "reason": "x"}, "at least two"),
    ({"op": "frobnicate", "ids": [], "reason": "x"}, "unknown operation"),
    ({"op": "delete_artifact", "ids": ["t_c000001"]}, "reason"),
])
def test_invalid_corrections_fail_clearly(op, msg):
    with pytest.raises(ValueError, match=msg):
        A.apply_corrections(make_candidates(), corrections(op))


def test_candidate_used_by_two_operations_fails():
    ops = corrections({"op": "delete_artifact", "ids": ["t_c000001"], "reason": "a"},
                      {"op": "replace_box", "id": "t_c000001", "box": [0, 0, 4, 4], "reason": "b"})
    with pytest.raises(ValueError, match="more than one"):
        A.apply_corrections(make_candidates(), ops)


# ---- validation --------------------------------------------------------------

def annotation(objects, **kw):
    base = {"schema_version": 1, "image_id": "t", "image_path": "x.png", "image_sha256": "a" * 64,
            "width": 200, "height": 100,
            "roi_polygon": [[0, 0], [200, 0], [200, 100], [0, 100]],
            "reviewed_region_polygons": [[[0, 0], [200, 0], [200, 100], [0, 100]]],
            "review_status": "ai_visual_inspection_provisional",
            "review_provenance": {"reviewer_type": "ai_visual_inspection"},
            "objects": objects}
    base.update(kw)
    return base


def obj(i, box):
    return {"id": f"t_{i:06d}", "class_id": 0, "bbox_xyxy": box, "origin_candidate_ids": [], "review_status": "x"}


def test_valid_annotation_passes():
    rep = A.validate_annotations(annotation([obj(1, [1, 1, 5, 5])]), {"width": 200, "height": 100,
                                                                         "sha256": "a" * 64})
    assert rep.ok, rep.errors


@pytest.mark.parametrize("objects,kw,msg", [
    ([obj(1, [1, 1, 5, 5]), obj(1, [10, 1, 15, 5])], {}, "duplicate"),
    ([obj(1, [1, 1, 1, 5])], {}, "area"),
    ([obj(1, [1, 1, float("inf"), 5])], {}, "finite"),
    ([obj(1, [190, 1, 205, 5])], {}, "bounds"),
    ([], {"roi_polygon": [[0, 0], [10, 10], [10, 0], [0, 10]]}, "roi"),
    ([], {"review_provenance": None}, "provenance"),
    ([], {"image_sha256": "b" * 64}, "sha256"),
])
def test_invalid_annotations_reported(objects, kw, msg):
    rep = A.validate_annotations(annotation(objects, **kw), {"width": 200, "height": 100, "sha256": "a" * 64})
    assert not rep.ok
    assert any(msg in e for e in rep.errors), rep.errors
