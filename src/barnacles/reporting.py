"""reports/results.csv, figures and reports/final_report.md, generated only from recorded run outputs.

The narrative lives in reports/report_template.md; every number in it is a {{placeholder}} filled
from run JSON files here, and an unknown placeholder is an error.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

RESULT_COLUMNS = [
    "experiment_id", "result_status", "fold", "train_source", "heldout_source", "annotation_version",
    "review_status", "scored_region", "reviewed_area_fraction", "checkpoint_sha256", "selected_epoch", "seed",
    "confidence", "local_nms_iou", "global_nms_iou", "gt_count", "pred_count", "signed_count_error",
    "absolute_count_error", "absolute_percentage_error", "tp", "fp", "fn", "precision", "recall", "f1",
    "center_match_f1", "tile_size", "tile_stride", "tile_count", "device", "precision_mode",
    "latency_median_ms", "latency_p90_ms",
]
STATUS = {"primary": "provisional-label held-out result",
          "exploratory": "EXPLORATORY (held-out-informed recipe change; not an untouched test)"}

# palette: validated categorical slots 1-2 and chart chrome (dataviz reference palette, light surface)
SERIES = ["#2a78d6", "#eb6834"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def result_row(metrics: dict, bench: dict | None, result_type: str) -> dict:
    p = metrics["primary"]
    row = {k: metrics.get(k) for k in RESULT_COLUMNS}
    row.update({k: p.get(k) for k in ("gt_count", "pred_count", "signed_count_error", "absolute_count_error",
                                      "absolute_percentage_error", "tp", "fp", "fn", "precision", "recall", "f1")})
    row["result_status"] = STATUS[result_type] if metrics.get("provisional") or result_type != "primary" \
        else "verified-label held-out result"
    row["center_match_f1"] = p.get("center_match", {}).get("f1")
    row["tile_count"] = metrics.get("tiles")
    gpu = (bench or {}).get("gpu_tiled") or (bench or {}).get("cpu_tiled")
    row["latency_median_ms"] = gpu["latency"]["median_ms"] if gpu else None
    row["latency_p90_ms"] = gpu["latency"]["p90_ms"] if gpu else None
    return row


def write_results_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = RESULT_COLUMNS + [c for c in rows[0] if c not in RESULT_COLUMNS] if rows else RESULT_COLUMNS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("null" if r.get(c) is None else r.get(c)) for c in cols})


def junit_summary(path: Path) -> dict | None:
    """Counts from a pytest --junitxml file (None when the file does not exist)."""
    import xml.etree.ElementTree as ET
    path = Path(path)
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    out = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    out["passed"] = out["tests"] - out["failures"] - out["errors"] - out["skipped"]
    return out


def meets_targets(primary: dict, max_ape: float = 10.0, min_f1: float = 0.80) -> bool:
    """Proposed (aspirational) per-fold targets from the plan: count error <= 10 % AND F1 >= 0.80."""
    ape = primary.get("absolute_percentage_error")
    return ape is not None and ape <= max_ape and primary["f1"] >= min_f1


def _fmt(v, pct: bool) -> str:
    if v is None:
        return "null (not measured)"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.1f}" if pct else f"{v:.3f}"
    return str(v)


def render(text: str, values: dict, pct_keys=frozenset()) -> str:
    def sub(m):
        key = m.group(1).strip()
        if key not in values:
            raise KeyError(f"report placeholder {key!r} has no recorded value")
        return _fmt(values[key], key in pct_keys)
    return re.sub(r"\{\{([^}]+)\}\}", sub, text)


# ---- collection ---------------------------------------------------------------------

def _read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def _fold_records(exp_dir: Path, fold: str) -> dict | None:
    run = exp_dir / f"fold_{fold}"
    metrics = _read(run / "evaluation" / "metrics.json")
    if metrics is None:
        return None
    bench = _read(run / "benchmark.json")
    v0 = _read(run / "benchmark_v0_slow_rasterizer.json")
    if v0:
        metrics = {**metrics, "_v0_load_region_ms": v0["folds"][fold]["gpu_tiled"]["stages"]["load_and_region"]["median_ms"]}
    return {"run": run, "record": _read(run / "run_record.json"), "selection": _read(run / "threshold_selection.json"),
            "metrics": metrics, "ablations": _read(run / "evaluation" / "ablations.json"),
            "bench": (bench or {}).get("folds", {}).get(fold)}


def _values_for(prefix: str, r: dict, values: dict, pct: set) -> None:
    m, p, sel, rec = r["metrics"], r["metrics"]["primary"], r["selection"], r["record"]
    put = values.__setitem__
    for k in ("gt_count", "pred_count", "signed_count_error", "absolute_count_error", "tp", "fp", "fn",
              "precision", "recall", "f1"):
        put(f"{prefix}.{k}", p[k])
    put(f"{prefix}.ape", p["absolute_percentage_error"]); pct.add(f"{prefix}.ape")
    put(f"{prefix}.center_f1", p["center_match"]["f1"])
    put(f"{prefix}.center_recall", p["center_match"]["recall"])
    put(f"{prefix}.center_precision", p["center_match"]["precision"])
    put(f"{prefix}.conf", m["confidence"])
    put(f"{prefix}.meets_targets", meets_targets(p))
    fx = m["fixed_confidence_sensitivity"]
    for k in ("pred_count", "signed_count_error", "precision", "recall", "f1"):
        put(f"{prefix}.fixed25.{k}", fx[k])
    put(f"{prefix}.fixed25.ape", fx["absolute_percentage_error"]); pct.add(f"{prefix}.fixed25.ape")
    put(f"{prefix}.inner_f1", sel["selected"]["f1"])
    put(f"{prefix}.inner_precision", sel["selected"]["precision"])
    put(f"{prefix}.inner_recall", sel["selected"]["recall"])
    put(f"{prefix}.inner_refs", sel["n_val_references"])
    put(f"{prefix}.best_epoch", rec["best_epoch"])
    put(f"{prefix}.epochs", rec["epochs_completed"])
    put(f"{prefix}.steps", rec["optimizer_steps"])
    put(f"{prefix}.applied", rec["grad_checks"]["applied_updates"])
    put(f"{prefix}.amp_skipped", rec["grad_checks"]["amp_skipped_steps"])
    put(f"{prefix}.train_s", rec["elapsed_s"]); pct.add(f"{prefix}.train_s")
    put(f"{prefix}.inner_map", rec["checkpoints"]["best"]["fitness_at_save"])
    put(f"{prefix}.train_objects", rec.get("train_objects"))
    put(f"{prefix}.train_tiles", rec.get("train_tiles"))
    put(f"{prefix}.raw_tile_dets", m["merge"]["raw_tile_detections_at_conf"])
    put(f"{prefix}.after_nms", m["merge"]["after_global_nms_at_conf"])
    put(f"{prefix}.seam_edge", m["merge"]["seam_edge_detections_counted"])
    sd = m["seam_diagnostic"]
    for g in ("seam", "interior"):
        put(f"{prefix}.seam.{g}.recall", sd["recall"].get(g, {}).get("recall"))
        put(f"{prefix}.seam.{g}.n", sd["recall"].get(g, {}).get("n"))
        put(f"{prefix}.seam.{g}.precision", sd["precision"].get(g, {}).get("precision"))
    sz = m["size_diagnostic"]
    put(f"{prefix}.size.small_lt", sz["bins"]["small_lt"]); pct.add(f"{prefix}.size.small_lt")
    put(f"{prefix}.size.large_ge", sz["bins"]["large_ge"]); pct.add(f"{prefix}.size.large_ge")
    for g in ("small", "medium", "large"):
        put(f"{prefix}.size.{g}.n", sz["recall"].get(g, {}).get("n", 0))
        put(f"{prefix}.size.{g}.recall", sz["recall"].get(g, {}).get("recall"))
    b = m["boundary"]
    put(f"{prefix}.boundary_refs", b["references_crossing_roi_edge"])
    if "_v0_load_region_ms" in m:
        put(f"{prefix}.lat_v0.gpu.load_region", m["_v0_load_region_ms"]); pct.add(f"{prefix}.lat_v0.gpu.load_region")
    put(f"{prefix}.eroded_f1", b["eroded_interior_scores"]["f1"])
    curve = {r_["confidence"]: r_ for r_ in m["heldout_confidence_curve"]["rows"]}
    best = max(curve.values(), key=lambda r_: r_["f1"])
    put(f"{prefix}.heldout_best_f1", best["f1"]); put(f"{prefix}.heldout_best_conf", best["confidence"])
    for c in (0.2, 0.3):
        put(f"{prefix}.curve{int(c * 100)}.pred_count", curve[c]["pred_count"])
        put(f"{prefix}.curve{int(c * 100)}.f1", curve[c]["f1"])
    if r["ablations"]:
        for k, short in (("overlap_640_512", "overlap"), ("no_overlap_640", "nooverlap"),
                         ("single_resized_roi_640", "single")):
            a = r["ablations"][k]
            for f in ("pred_count", "signed_count_error", "recall", "precision", "f1"):
                put(f"{prefix}.abl.{short}.{f}", a[f])
            put(f"{prefix}.abl.{short}.tiles", a["tiles"])
        wh = r["ablations"]["single_resized_roi_640"]["region_wh"]
        put(f"{prefix}.abl.single.scale", 640 / max(wh)); pct.add(f"{prefix}.abl.single.scale")
    if r["bench"]:
        for dev in ("gpu", "cpu"):
            for mode, short in (("tiled", "tiled"), ("single_resized_roi", "single")):
                bb = r["bench"].get(f"{dev}_{mode}")
                if bb:
                    put(f"{prefix}.lat.{dev}.{short}.median", bb["latency"]["median_ms"]); pct.add(f"{prefix}.lat.{dev}.{short}.median")
                    put(f"{prefix}.lat.{dev}.{short}.p90", bb["latency"]["p90_ms"]); pct.add(f"{prefix}.lat.{dev}.{short}.p90")
            g = r["bench"].get("gpu_tiled")
            if g:
                put(f"{prefix}.lat.gpu.infer", g["stages"]["tile_inference"]["median_ms"]); pct.add(f"{prefix}.lat.gpu.infer")
                put(f"{prefix}.lat.gpu.load_region", g["stages"]["load_and_region"]["median_ms"]); pct.add(f"{prefix}.lat.gpu.load_region")
                put(f"{prefix}.lat.gpu.mem", g["peak_cuda_memory_mib"]); pct.add(f"{prefix}.lat.gpu.mem")
                put(f"{prefix}.lat.model_load_s", g["model_load_s"])
                put(f"{prefix}.lat.overlay_s", g["output_generation_s"]["overlay_jpg"])


# ---- figures ------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_thresholds(folds: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(folds), figsize=(5.2 * len(folds), 3.8), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (fold, r) in zip(axes, folds.items()):
        _style(ax)
        inner = r["selection"]["rows"]
        held = r["metrics"]["heldout_confidence_curve"]["rows"]
        ax.plot([x["confidence"] for x in inner], [x["f1"] for x in inner], color=SERIES[0], lw=2, marker="o", ms=5,
                label="inner validation (used for selection)")
        ax.plot([x["confidence"] for x in held], [x["f1"] for x in held], color=SERIES[1], lw=2, ls="--", marker="s",
                ms=5, label="held-out photo (diagnostic only)")
        c = r["metrics"]["confidence"]
        ax.axvline(c, color=MUTED, lw=1, ls=":")
        ax.text(c, 0.03, f" selected {c}", color=INK2, fontsize=8, va="bottom")
        ax.text(inner[-1]["confidence"] + 0.01, inner[-1]["f1"], "inner", color=INK2, fontsize=8, va="center")
        ax.text(held[-1]["confidence"] + 0.01, held[-1]["f1"], "held-out", color=INK2, fontsize=8, va="center")
        m = r["metrics"]
        ax.set_title(f"Fold {fold}: {m['train_source']} -> {m['heldout_source']}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("confidence threshold", color=INK2, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("instance F1 (box IoU >= 0.5)", color=INK2, fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK2, bbox_to_anchor=(0, 0.08))
    fig.text(0.01, 0.005, "Provisional labels (AI visual review, not human-verified). Held-out curve computed after "
             "selection; it did not influence the threshold.", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_training(folds: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    _style(ax)
    for i, (fold, r) in enumerate(folds.items()):
        ef = r["record"]["epoch_fitness"]
        xs, ys = [e for e, _ in ef], [f for _, f in ef]
        ax.plot(xs, ys, color=SERIES[i], lw=2, ls="-" if i == 0 else "--", label=f"fold {fold} ({r['metrics']['train_source']} inner val)")
        ax.text(xs[-1] + 1, ys[-1], f"fold {fold}", color=INK2, fontsize=8, va="center")
        be = r["record"]["best_epoch"]
        ax.plot([be], [dict(ef)[be]], marker="o", ms=8, color=SERIES[i], mec=SURFACE, mew=2)
    ax.set_xlabel("epoch (one optimizer step per epoch: 6 training tiles, batch 8)", color=INK2, fontsize=9)
    ax.set_ylabel("inner-val mAP50-95 (checkpoint fitness)", color=INK2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _thumb(src: Path, dst: Path, max_side: int = 1600) -> str:
    im = Image.open(src)
    im.thumbnail((max_side, max_side))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=85)
    return dst.name


# ---- command --------------------------------------------------------------------------

def build_report(cfg) -> dict:
    root, reports = cfg["root"], cfg["paths"]["reports_dir"]
    runs = cfg["paths"]["runs_dir"]
    primary_id = cfg["experiment_id"]
    values, pct = {"experiment_id": primary_id}, set()
    rows, folds = [], {}
    for fold in cfg["folds"]:
        r = _fold_records(runs / primary_id, fold)
        if r is None:
            raise SystemExit(f"fold {fold} of {primary_id} has no evaluation; run the folds first")
        folds[fold] = r
        _values_for(fold, r, values, pct)
        rows.append(result_row(r["metrics"], r["bench"], "primary"))
    from .metrics import aggregate
    agg = aggregate([r["metrics"]["primary"] for r in folds.values()])
    for k, v in agg.items():
        values[f"agg.{k}"] = v
    pct.update({"agg.macro_absolute_percentage_error", "agg.macro_absolute_count_error"})
    values["provisional"] = any(r["metrics"]["provisional"] for r in folds.values())
    values["review_status"] = folds["A"]["metrics"]["review_status"]

    # labels, diagnostics and test runs (all from recorded files)
    val = _read(cfg["paths"]["manifests_dir"] / "annotation_validation.json") or {}
    for sid, v in val.items():
        values[f"labels.{sid}.n_candidates"], values[f"labels.{sid}.n_objects"] = v["n_candidates"], v["n_objects"]
    diag = runs / primary_id / "diagnostics"
    over = sorted(diag.glob("overfit_*/overfit_result.json")) if diag.exists() else []
    if over:
        o = _read(over[-1])
        values.update({"overfit.passed": o["passed"], "overfit.steps": o["optimizer_steps"],
                       "overfit.before_f1": o["before"]["f1"], "overfit.after_f1": o["after"]["f1"],
                       "overfit.tiles": len(o["tiles"])})
    smoke = sorted(diag.glob("smoke_*/run_record.json")) if diag.exists() else []
    if smoke:
        sr = _read(smoke[-1])
        values["smoke.ok"] = sr["losses_finite"] and sr["reload_ok"] and sr["parameters_changed"]
    for name in ("offline", "training"):
        js = junit_summary(reports / "tests" / f"pytest_{name}.xml")
        for k in ("tests", "passed", "failures", "errors", "skipped"):
            values[f"tests.{name}.{k}"] = None if js is None else js[k]

    # exploratory experiments: any other experiment directory with evaluated folds
    for exp_dir in sorted(p for p in runs.iterdir() if p.is_dir() and p.name != primary_id):
        ex = {f: _fold_records(exp_dir, f) for f in cfg["folds"]}
        if not all(ex.values()):
            continue
        tag = "X" if "scaleaug" in exp_dir.name else exp_dir.name
        values[f"{tag}.experiment_id"] = exp_dir.name
        for fold, r in ex.items():
            _values_for(f"{tag}.{fold}", r, values, pct)
            rows.append(result_row(r["metrics"], r["bench"], "exploratory"))
        xagg = aggregate([r["metrics"]["primary"] for r in ex.values()])
        for k, v in xagg.items():
            values[f"{tag}.agg.{k}"] = v
        pct.add(f"{tag}.agg.macro_absolute_percentage_error")
    write_results_csv(reports / "results.csv", rows)

    # final train-all demonstration model
    final_dir = runs / primary_id / cfg["final"]["run_name"]
    frec, deploy = _read(final_dir / "run_record.json"), _read(final_dir / "deployment.json")
    values["final.status"] = "trained" if frec else "not trained"
    if frec:
        values.update({"final.epochs": frec["epochs_completed"], "final.steps": frec["optimizer_steps"],
                       "final.applied": frec["grad_checks"]["applied_updates"], "final.conf": deploy["confidence"],
                       "final.sha": deploy["checkpoint_sha256"][:12], "final.train_s": frec["elapsed_s"],
                       "final.tiles": frec["trainer"]["train_images"],
                       "final.steps_ratio": frec["optimizer_steps"] / np.mean([values["A.steps"], values["B.steps"]])})
    for name in ("unseen_img1", "unseen_img2"):
        pj = _read(reports / "predictions" / name / "predictions.json")
        values[f"{name}.count"] = pj["count"] if pj else None
        values[f"{name}.region"] = pj["region"]["name"] if pj else None

    env = _read(cfg["paths"]["manifests_dir"] / "environment.json") or {}
    for k in ("gpu", "torch", "ultralytics", "python", "platform"):
        values[f"env.{k}"] = env.get(k)

    figs = reports / "figures"
    figure_thresholds(folds, figs / "threshold_curves.png")
    figure_training(folds, figs / "training_curves.png")
    from .evaluate import inner_val_overlay
    for fold, r in folds.items():
        iv = inner_val_overlay(cfg, fold, figs / f"fold_{fold}_inner_val_errors.jpg")
        if (iv["tp"], iv["fp"], iv["fn"]) != (r["selection"]["selected"]["tp"], r["selection"]["selected"]["fp"],
                                             r["selection"]["selected"]["fn"]):
            raise RuntimeError(f"fold {fold} inner-val overlay does not reproduce the recorded selection")
        values[f"{fold}.inner_fp"] = iv["fp"]
        ev = r["run"] / "evaluation"
        _thumb(ev / "overlay_full.jpg", figs / f"fold_{fold}_heldout_overlay.jpg")
        for z in sorted(ev.glob("zoom_*.png")):
            kind = z.stem.split("_")[1]
            _thumb(z, figs / f"fold_{fold}_zoom_{kind}.jpg", 640)
    for name in ("unseen_img1", "unseen_img2"):
        src = reports / "predictions" / name / "overlay.jpg"
        if src.exists():
            _thumb(src, figs / f"{name}_demo_overlay.jpg")

    ex_tpl = reports / "report_exploratory_template.md"
    if "X.experiment_id" in values and ex_tpl.exists():
        values["EXPLORATORY_SECTION"] = render(ex_tpl.read_text(encoding="utf-8"), values, pct)
    else:
        values["EXPLORATORY_SECTION"] = "No exploratory experiment was recorded."
    template = (reports / "report_template.md").read_text(encoding="utf-8")
    (reports / "final_report.md").write_text(render(template, values, pct), encoding="utf-8")
    (reports / "report_values.json").write_text(json.dumps(values, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {reports / 'results.csv'} ({len(rows)} rows), {reports / 'final_report.md'}, figures in {figs}")
    return values
