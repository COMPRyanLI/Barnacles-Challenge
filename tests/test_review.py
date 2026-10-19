import numpy as np

from barnacles import review as R
from barnacles.annotations import Candidate


def cands():
    return [Candidate("t_c000001", 40, [10.0, 10.0, 22.0, 22.0], 1, []),
            Candidate("t_c000002", 3, [50.0, 50.0, 52.0, 51.0], 0, ["tiny"]),
            Candidate("t_c000003", 90, [100.0, 20.0, 130.0, 44.0], 2, ["multi_ring"])]


def test_contact_sheet_indexes_every_item_in_order():
    img = np.full((200, 300, 3), 90, np.uint8)
    stroke = np.zeros((200, 300), bool)
    sheet, index = R.contact_sheet(img, stroke, cands(), panel=96, cols=2)
    assert [e["candidate_id"] for e in index] == ["t_c000001", "t_c000002", "t_c000003"]
    assert [e["panel"] for e in index] == [1, 2, 3]
    assert sheet.ndim == 3 and sheet.shape[1] == 2 * 96 and sheet.shape[0] >= 2 * 96


def test_contact_sheet_draws_extra_boxes_per_item():
    img = np.full((200, 300, 3), 90, np.uint8)
    stroke = np.zeros((200, 300), bool)
    plain, _ = R.contact_sheet(img, stroke, cands()[2:], panel=96, cols=1)
    extra = {"t_c000003": [([100, 20, 115, 44], (0, 255, 0)), ([115, 20, 130, 44], (0, 255, 0))]}
    marked, index = R.contact_sheet(img, stroke, cands()[2:], panel=96, cols=1, extra=extra)
    assert (marked != plain).any()
    assert index[0]["extra_boxes"] == [[100, 20, 115, 44], [115, 20, 130, 44]]


def test_contact_sheet_handles_candidates_at_image_border():
    img = np.zeros((40, 40, 3), np.uint8)
    c = [Candidate("t_c000001", 4, [0.0, 0.0, 2.0, 2.0], 0, ["tiny"])]
    sheet, index = R.contact_sheet(img, np.zeros((40, 40), bool), c, panel=64, cols=4)
    assert len(index) == 1 and sheet.shape[0] == 64


def test_overlay_draws_without_changing_input():
    img = np.full((100, 160, 3), 50, np.uint8)
    before = img.copy()
    out = R.draw_boxes(img, [[10, 10, 30, 30]], color=(255, 0, 0))
    assert np.array_equal(img, before)
    assert (out != before).any()


def test_review_tiles_cover_polygon_bbox():
    img = np.zeros((300, 400, 3), np.uint8)
    roi = [[20, 30], [380, 30], [380, 290], [20, 290]]
    tiles = R.review_tiles(img, np.zeros((300, 400), bool), roi, size=200)
    covered = np.zeros((300, 400), bool)
    for t in tiles:
        x0, y0 = t["origin"]
        h, w = t["image"].shape[:2]
        covered[y0:y0 + h, x0:x0 + w] = True
    assert covered[30:290, 20:380].all()
