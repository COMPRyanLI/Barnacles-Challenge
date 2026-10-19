The diagnostics above pointed to scale shift, so **one** bounded follow-up was run after the
held-out results were seen: experiment `{{X.experiment_id}}` (`configs/exploratory_scale_aug.yaml`). It is
identical to the primary recipe except that the random scale augmentation is widened from ±10 % to
±50 % (`scale: 0.50`). The splits, threshold-selection rule, evaluation code and annotations are
the same.

**Status: exploratory.** The change was chosen because of what the held-out photos showed. These
numbers are therefore *not* an untouched test and do not replace section 1.

| Fold | Count N / P (P − N, abs %) | TP / FP / FN | Precision | Recall | F1 (box) | F1 (centre) | selected conf | meets targets |
|---|---|---|---|---|---|---|---|---|
| A, primary | {{A.gt_count}} / {{A.pred_count}} ({{A.signed_count_error}}, {{A.ape}} %) | {{A.tp}} / {{A.fp}} / {{A.fn}} | {{A.precision}} | {{A.recall}} | {{A.f1}} | {{A.center_f1}} | {{A.conf}} | {{A.meets_targets}} |
| A, exploratory | {{X.A.gt_count}} / {{X.A.pred_count}} ({{X.A.signed_count_error}}, {{X.A.ape}} %) | {{X.A.tp}} / {{X.A.fp}} / {{X.A.fn}} | {{X.A.precision}} | {{X.A.recall}} | {{X.A.f1}} | {{X.A.center_f1}} | {{X.A.conf}} | {{X.A.meets_targets}} |
| B, primary | {{B.gt_count}} / {{B.pred_count}} ({{B.signed_count_error}}, {{B.ape}} %) | {{B.tp}} / {{B.fp}} / {{B.fn}} | {{B.precision}} | {{B.recall}} | {{B.f1}} | {{B.center_f1}} | {{B.conf}} | {{B.meets_targets}} |
| B, exploratory | {{X.B.gt_count}} / {{X.B.pred_count}} ({{X.B.signed_count_error}}, {{X.B.ape}} %) | {{X.B.tp}} / {{X.B.fp}} / {{X.B.fn}} | {{X.B.precision}} | {{X.B.recall}} | {{X.B.f1}} | {{X.B.center_f1}} | {{X.B.conf}} | {{X.B.meets_targets}} |

What it shows:

* **Recall on the out-of-scale targets rose.**
  * Fold A's recall on img2's "large" targets went from {{A.size.large.recall}} to
    {{X.A.size.large.recall}}.
  * Fold B's recall on img1's "small" targets went from {{B.size.small.recall}} to
    {{X.B.size.small.recall}}.
  * Both folds' F1 improved, which supports scale shift as a dominant error mechanism.
* **Precision fell in fold A** ({{A.precision}} → {{X.A.precision}}). The fold A model now
  over-counts img2 by {{X.A.signed_count_error}} ({{X.A.ape}} %). Fold B's count error is
  {{X.B.ape}} %, but its F1 is still {{X.B.f1}}, below 0.80.
* **A small count error is not enough.** Fold B's count lands within 10 %, but
  {{X.B.fp}} false positives and {{X.B.fn}} misses offset each other. Both targets would have to
  hold together.
* **Inner validation barely moved.** Fold A inner F1 went from {{A.inner_f1}} to {{X.A.inner_f1}},
  and fold B's from {{B.inner_f1}} to {{X.B.inner_f1}}. Within-photo validation could not have
  revealed this transfer gain. This is exactly why two photos are a weak basis for model selection.
* **Threshold at the grid edge again.** The inner threshold again hit the top of the fixed grid
  ({{X.A.conf}} and {{X.B.conf}}).

The honest reading is that stronger scale augmentation is a promising direction to test on **new**
labeled scenes. It is not a validated improvement.
