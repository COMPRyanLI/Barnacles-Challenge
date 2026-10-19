# Barnacle counting prototype: tiled YOLO11n with honest two-scene evaluation

A small, reproducible prototype for the [DALI Barnacles challenge](https://os.dali.dartmouth.edu/documents/cmuva04g407lrj4pxhhzwwaqu/public). It finds
individual barnacle targets in a field photograph, counts those whose centres lie inside the inner
square of the green frame, and writes an overlay a person can check and correct.

> **Status of the evidence.**  Only two independent scenes exist, so there is no untouched test set. Read
> [`reports/final_report.md`](reports/final_report.md) for the measured results. The short version:
> the model does **not** reach the proposed targets (≤10 % count error and F1 ≥ 0.80 per fold) when
> it is trained on one photo and tested on the other. The dominant error is the change of barnacle
> size and appearance between the two scenes.

## Results (held-out photo, provisional labels; full detail in `reports/final_report.md`)

| Fold | Train → test | Reference N | Predicted P | Count error | Precision | Recall | F1 (box IoU ≥ 0.5) | F1 (centre match) |
|---|---|---|---|---|---|---|---|---|
| A | img1 → img2 | 397 | 213 | −184 (46.3 %) | 0.723 | 0.388 | 0.505 | 0.607 |
| B | img2 → img1 | 1732 | 1464 | −268 (15.5 %) | 0.730 | 0.617 | 0.668 | 0.782 |

* **Exploratory follow-up, not an untouched test:** the same recipe with stronger scale
  augmentation reached F1 0.647 (A) and 0.712 (B). I chose that change after seeing the held-out
  results.
* **Latency:** a complete image (disk load to count) takes a median of 364 / 755 ms on an RTX 4060
  Laptop GPU and 847 / 1,069 ms on CPU.
* **Unseen photos (demonstration only, no labels):** the final model counts 1,189 detections in
  unseen_img1's frame and 133 in the unseen_img2 close-up.

## Conclusions

* **The pipeline works on a single scene but does not yet transfer reliably between scenes.** When
  a model is tested on the other photo, the dominant errors are:
  * barnacle size and appearance shift between photos (≈21 px vs ≈37 px targets);
  * mis-sized boxes on correctly located barnacles;
  * false positives on empty shells, seaweed and shadow.
* **Tiling and merging were not the limiting factors.** Overlapping native-resolution tiles beat
  both no-overlap tiles and a single resized pass on small targets. Seam and boundary diagnostics
  were not the main error.
* **The model should be used as a review aid, not an automatic counter.** The overlay gives a person
  a box per candidate to accept, delete or add. Whether that is faster than manual counting was not
  measured.
* **The most valuable next step is more labeled scenes**, plus a human decision on whether empty
  shells count. More epochs or a bigger model come after that.

## Reflection

**What worked.** A small pretrained detector, YOLO11n, was enough. Splitting each photo into
full-resolution 640 px tiles mattered more than model size: when I shrank the whole frame into a
single image, small barnacles became almost impossible to find (recall fell from 0.62 to 0.003).
Overlapping tiles plus one global duplicate-removal step gave a clean count, and a box over each
barnacle for a person to check.

**What did not work yet.** I had only two labeled photos, and their barnacles look different: about
21 px across in one and 37 px in the other. A model trained on one photo missed many barnacles in
the other (F1 0.51 and 0.67). That taught me that a test on a different scene matters far more than
a good score inside the same photo. My inner validation said 0.93 while the other photo said 0.51.
Training with stronger random zoom helped (F1 0.65 and 0.71), but I chose it after seeing the test
results, so it is a lead to try on new photos, not a proven fix.

**The surprise.** I expected the neural network to be the slow part. Timing every step showed it
took only about 0.1 s per photo. Most of the 3 s went into drawing the frame's pixel mask, which my
code checked point by point. A simple row-by-row rewrite produced exactly the same results and cut
the time to 0.36–0.75 s per photo.

**Next.** More labeled scenes with different barnacle sizes, a person checking the labels, and a
timed test of whether correcting the model's overlay really beats counting by hand.

## Approach

```
photo + inner-frame polygon
  -> crop to the polygon's bounding box, paint pixels outside the polygon grey (114)
  -> native-resolution 640 x 640 tiles, stride 512 (128 px overlap)
  -> YOLO11n (COCO-pretrained, fine-tuned, one class "barnacle_target")
  -> drop detections centred in padding, restore original coordinates
  -> one global torchvision NMS (IoU 0.5) across tiles
  -> count detections whose box centre is inside (or on) the polygon
  -> predictions JSON + overlay (green = counted, orange = lower-score candidates to review)
```

Why this design:

* **Detection, not classification.** A detector directly gives one box and score per barnacle, so
  counts and review overlays come out of a single model. A patch classifier would still need a
  proposal step. Mask R-CNN would need per-instance masks, but the supplied labels are contour
  drawings.
* **Native-resolution tiles.** Annotated targets are about 21 px (img1) and 37 px (img2) across. If
  the whole 3264 px photo were resized to 640 px, they would shrink to 4–7 px.
* **YOLO11n through Ultralytics (PyTorch).** It is a small (2.6 M parameter) maintained detector with
  a standard training loop. Only the challenge-specific parts are written here: label conversion,
  splits, tiling, merging, metrics and reports.

## Input / output contract

* **Input:** an original photograph (never the masks or the blue overlay images, which the code
  rejects by content hash), plus either an inner-frame polygon from `data/annotations/rois.json` or
  `--whole-image` for a frameless close-up.
* **Output:** `predictions.json`. It contains one record per detection: `bbox_xyxy` in original
  pixels (x2/y2 exclusive), `center`, `confidence`, `class_id` 0 and `in_region`. It also contains
  the region count, raw-versus-merged counts, timings, settings and the checkpoint hash. The
  `overlay.jpg` shows the counted detections. The confidence values are model scores, **not
  calibrated probabilities**.
* **Counting rule:** a detection counts when its box centre is inside the polygon or on its
  boundary (tolerance 1e-6 px).

## Setup (Windows PowerShell; the same steps work on Linux with `source .venv/bin/activate`)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# CUDA 12.8 build used here (RTX 4060 Laptop, driver 576.88). For CPU-only, use the plain PyPI torch wheels.
.venv\Scripts\python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
# Official COCO-pretrained weights (sha256 recorded in data/manifests/weights.json):
.venv\Scripts\python -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('weights/yolo11n.pt')"
```

`requirements-lock.txt` pins the exact versions that passed the GPU training smoke test (Python
3.12.10, torch 2.11.0+cu128, ultralytics 8.4.182). Ultralytics' settings are kept project-local in
`.ultralytics/` with usage analytics (`sync`) switched off. On its first training run, Ultralytics
also downloads `yolo26n.pt` once, only for its AMP self-check. That file is never used for training.

## Reproduce everything

Run these from the repository root with the venv's Python (`.venv\Scripts\python`). Each stage
writes machine-readable JSON. Evaluated runs are never overwritten: re-running `train` or `evaluate`
for an existing run directory fails on purpose.

```powershell
python -m barnacles audit                                   # read-only data audit -> analysis/data_audit.json
python -m barnacles prepare-labels                          # source hashes + contour candidates
python -m barnacles render-label-review                     # overlays, numbered contact sheets, ROI review tiles
python -m barnacles validate-data                           # apply data/annotations/corrections/*.json, validate
python -m barnacles build-dataset --allow-provisional-labels   # leakage-checked splits + YOLO tiles
python -m pytest -m "not training and not gpu" --junitxml=reports/tests/pytest_offline.xml   # CPU-only tests, no weights needed
python -m pytest -m training --junitxml=reports/tests/pytest_training.xml                    # training round trip (needs weights)
python -m barnacles smoke-train                             # 1-epoch smoke test on 4 fold-A tiles
python -m barnacles overfit-check                           # <=200-update overfit diagnostic
python -m barnacles train --fold A
python -m barnacles select-threshold --fold A               # inner-validation sweep only
python -m barnacles evaluate --fold A                       # held-out img2, scored once
python -m barnacles train --fold B
python -m barnacles select-threshold --fold B
python -m barnacles evaluate --fold B                       # held-out img1
python -m barnacles ablate --fold A ; python -m barnacles ablate --fold B   # no-overlap and single-resize diagnostics
python -m barnacles benchmark --fold all                    # complete-image latency, GPU and CPU
python -m barnacles train-final                             # fixed-epoch refit on both photos (demonstration model)
python -m barnacles predict --image Barnacles/Barnacles/unseen_img1.png --roi data/annotations/rois.json --checkpoint runs/yolo11n_tile640_v1/final/weights/last.pt
python -m barnacles predict --image Barnacles/Barnacles/unseen_img2.png --whole-image --checkpoint runs/yolo11n_tile640_v1/final/weights/last.pt
python -m barnacles report                                  # reports/results.csv + reports/final_report.md
```

The exploratory follow-up, which differs only in its stronger scale augmentation, uses the same
commands with `--config configs/exploratory_scale_aug.yaml` for `train`, `select-threshold`,
`evaluate` and `ablate`. Each run directory has its own record. Training on the RTX 4060 Laptop
took about 2 minutes per fold. Bit-identical results are expected only on the recorded software and
hardware (`data/manifests/environment.json`).

## Labels, review and provenance

* The masks are blue contour strokes on a transparent background. The foreground is taken as
  `alpha > 0`, and 8-connected components become **candidates** (img1: 1,747; img2: 521). These
  are not verified counts.
* Each candidate is flagged when needed: `tiny`, `open_contour`, `multi_ring` (several enclosed
  interiors; pinholes under 5 px inside thick strokes are ignored), `thin` or `large`. The flagged
  candidates were reviewed on numbered contact sheets. Every decision is an explicit,
  replayable operation with a reason in `data/annotations/corrections/<id>_corrections.json`:
  `keep`, `delete_artifact`, `split`, `merge`, `replace_box` or `add_missing`. The raw candidates
  (`data/annotations/candidates/`) and the corrected annotations (`data/annotations/img1.json`,
  `img2.json`) are kept separately.
* **Provenance:** reviewer = AI visual inspection, `human_verified: false`, status
  `ai_visual_inspection_provisional`. The same applies to the inner-frame polygons in
  `data/annotations/rois.json`. They were measured from the wire colour and checked in zoomed crops.
* **Annotation convention inferred from the labels:** one outline per barnacle, drawn around the
  opercular opening. Empty shells (hollow, dark interior) are not outlined in either photo. This
  describes the supplied labels; it is not a live/dead classification.
* **Cases that need a person:** see the `ambiguous_cases_for_human_review` lists in the two
  correction files. **Decision:** empty shells are **not** counted, which matches the supplied labels. Training
  refuses non-verified labels unless `--allow-provisional-labels` is given, and every output then
  carries the provisional status.

## Evaluation protocol

* Source photos are split **before** tiling. Fold A trains on img1 and tests on img2; fold B is the
  reverse. Each fold restarts from the original COCO weights.
* Inside the development photo, the left about 65 % of the ROI is used for training and the right
  about 35 % for validation, with a 64 px unused gap. Objects touching the gap are excluded and
  masked in both parts. The code asserts that no pixel or object ID is shared between the parts and
  that the held-out photo never appears in a training tile.
* The checkpoint (best inner-validation mAP50-95) and the confidence threshold (best inner F1 on
  the fixed grid 0.10–0.50) are chosen on inner validation only. The held-out photo is then scored
  once over the full region: count error, one-to-one box matching at IoU ≥ 0.5, centre matching,
  and seam, size and boundary diagnostics.
* The final `train-all` model is a **demonstration model**. Its outputs on the unseen photos have no
  reference labels and are not accuracy measurements.

## Repository layout

```
configs/experiment.yaml              primary experiment (all settings)
configs/exploratory_scale_aug.yaml   exploratory follow-up (scale augmentation only)
src/barnacles/                       package (cli, annotations, prepare, review, geometry, dataset,
                                     train, predict, metrics, evaluate, benchmark, reporting)
tests/                               pytest suite (CPU-only by default; `training` marker for GPU/weights)
data/annotations/                    rois, candidates, corrections, corrected annotations
data/manifests/                      source/weights hashes, environment, split + tile manifests
data/processed/                      generated tiles (git-ignored; regenerate with build-dataset)
runs/<experiment>/fold_X/            run_record.json, threshold sweep, evaluation/ (metrics, predictions,
                                     matches.csv, overlays), ablations, benchmark; weights/ git-ignored
runs/yolo11n_tile640_v1/final/       train-all demonstration model + deployment.json
reports/                             final_report.md, results.csv, figures/, predictions/, label_review/
scripts/audit_data.py                original read-only audit (reused by `barnacles audit`)
analysis/data_audit.json             measured image metadata and mask diagnostics
```

Checkpoints (`*.pt`) are git-ignored by default. Their SHA-256 hashes are in each `run_record.json`.

## Limitations

Two photographs are two scenes, not a sample of field conditions. The labels have not been checked
by a person, and empty shells are unlabeled by convention. The fold models received only about 80
optimizer updates (6 training tiles per epoch). Counts depend on a hand-supplied frame polygon. See
the final report for the measured consequences and what would most improve confidence.
