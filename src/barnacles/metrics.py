"""Instance matching, count errors and fold aggregation (pure, no I/O)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import box_centers, box_iou


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    pred_to_ref: np.ndarray  # index of matched reference per prediction, -1 if unmatched
    ref_to_pred: np.ndarray  # index of matched prediction per reference, -1 if unmatched


def _score_order(scores) -> np.ndarray:
    # descending score, ties broken by original prediction order
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def _greedy(order, n_ref, candidates) -> MatchResult:
    """candidates(i, free) -> chosen reference index or -1."""
    n_pred = len(order)
    pred_to_ref = np.full(n_pred, -1, dtype=int)
    ref_to_pred = np.full(n_ref, -1, dtype=int)
    for i in order:
        j = candidates(i, ref_to_pred < 0)
        if j >= 0:
            pred_to_ref[i], ref_to_pred[j] = j, i
    tp = int((pred_to_ref >= 0).sum())
    return MatchResult(tp, n_pred - tp, n_ref - tp, pred_to_ref, ref_to_pred)


def match_boxes(pred_boxes, scores, ref_boxes, iou_threshold: float = 0.5) -> MatchResult:
    """One-to-one matching: predictions in descending score each take their
    highest-IoU unmatched reference with IoU >= threshold."""
    iou = box_iou(pred_boxes, ref_boxes)

    def pick(i, free):
        cand = np.where(free & (iou[i] >= iou_threshold))[0]
        return int(cand[np.argmax(iou[i, cand])]) if len(cand) else -1

    return _greedy(_score_order(scores), iou.shape[1], pick)


def match_centers(pred_boxes, scores, ref_boxes, tolerance_factor: float = 0.5) -> MatchResult:
    """Secondary metric: nearest unmatched reference centre within
    tolerance_factor * sqrt(ref_w * ref_h) of the prediction centre."""
    refs = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    pc, rc = box_centers(pred_boxes), box_centers(refs)
    tol = tolerance_factor * np.sqrt((refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1]))
    dist = np.sqrt(((pc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)) if len(pc) and len(rc) \
        else np.zeros((len(pc), len(rc)))

    def pick(i, free):
        cand = np.where(free & (dist[i] <= tol))[0]
        return int(cand[np.argmin(dist[i, cand])]) if len(cand) else -1

    return _greedy(_score_order(scores), len(refs), pick)


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision is 0 when nothing is predicted; recall is 0 when there is no reference."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def count_errors(n_ref: int, n_pred: int) -> dict:
    out = {
        "gt_count": int(n_ref),
        "pred_count": int(n_pred),
        "signed_count_error": int(n_pred - n_ref),
        "absolute_count_error": int(abs(n_pred - n_ref)),
        "absolute_percentage_error": 100.0 * abs(n_pred - n_ref) / n_ref if n_ref > 0 else None,
    }
    if n_ref == 0:
        out["false_positive_count_on_empty"] = int(n_pred)
        out["empty_region_exact"] = n_pred == 0
    return out


def aggregate(per_fold: list[dict]) -> dict:
    """Macro-average per-image count error; micro-aggregate TP/FP/FN."""
    tp = sum(f["tp"] for f in per_fold)
    fp = sum(f["fp"] for f in per_fold)
    fn = sum(f["fn"] for f in per_fold)
    apes = [f["absolute_percentage_error"] for f in per_fold if f["absolute_percentage_error"] is not None]
    micro = prf(tp, fp, fn)
    return {
        "n_images": len(per_fold),
        "macro_absolute_count_error": float(np.mean([f["absolute_count_error"] for f in per_fold])),
        "macro_absolute_percentage_error": float(np.mean(apes)) if apes else None,
        "micro_tp": tp, "micro_fp": fp, "micro_fn": fn,
        "micro_precision": micro["precision"], "micro_recall": micro["recall"], "micro_f1": micro["f1"],
    }
