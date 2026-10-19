"""Command-line interface: python -m barnacles <command> [options]."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/experiment.yaml"


def _cfg(args):
    from .config import load_config
    return load_config(args.config)


def cmd_audit(args):
    from .config import find_root
    script = find_root(Path.cwd()) / "scripts" / "audit_data.py"
    runpy.run_path(str(script), run_name="__main__")


def cmd_prepare_labels(args):
    from .prepare import prepare_labels
    prepare_labels(_cfg(args))


def cmd_render_label_review(args):
    from .prepare import render_label_review
    render_label_review(_cfg(args))


def cmd_validate_data(args):
    from .prepare import build_annotations
    build_annotations(_cfg(args))


def cmd_build_dataset(args):
    from .dataset import build_all
    build_all(_cfg(args), allow_provisional=args.allow_provisional_labels)


def cmd_smoke_train(args):
    from .train import smoke_train
    smoke_train(_cfg(args))


def cmd_overfit_check(args):
    from .train import overfit_check
    overfit_check(_cfg(args))


def cmd_train(args):
    from .train import train_outer_fold
    train_outer_fold(_cfg(args), args.fold)


def cmd_select_threshold(args):
    from .evaluate import select_threshold
    select_threshold(_cfg(args), args.fold)


def cmd_evaluate(args):
    from .evaluate import evaluate_fold
    evaluate_fold(_cfg(args), args.fold)


def cmd_ablate(args):
    from .evaluate import run_ablations
    run_ablations(_cfg(args), args.fold)


def cmd_benchmark(args):
    from .benchmark import benchmark_folds
    benchmark_folds(_cfg(args), args.fold)


def cmd_report(args):
    from .reporting import build_report
    build_report(_cfg(args))


def cmd_train_final(args):
    from .train import train_final
    train_final(_cfg(args))


def cmd_predict(args):
    from .predict import predict_cli
    predict_cli(_cfg(args), image=args.image, roi_file=args.roi, whole_image=args.whole_image,
                checkpoint=args.checkpoint, conf=args.conf, out_dir=args.out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="barnacles", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, fold=None):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        if fold:
            sp.add_argument("--fold", required=True, choices=fold)
        sp.set_defaults(func=fn)
        return sp

    add("audit", cmd_audit, "run the read-only data audit (scripts/audit_data.py)")
    add("prepare-labels", cmd_prepare_labels, "hash sources and extract contour candidates")
    add("render-label-review", cmd_render_label_review, "render overlays, contact sheets and ROI review tiles")
    add("validate-data", cmd_validate_data, "apply corrections, validate and write annotations")
    sp = add("build-dataset", cmd_build_dataset, "build split manifests and YOLO tiles for every fold")
    sp.add_argument("--allow-provisional-labels", action="store_true",
                    help="permit labels that are not human-verified; every output is marked provisional")
    add("smoke-train", cmd_smoke_train, "one-epoch training/validation smoke test")
    add("overfit-check", cmd_overfit_check, "tiny overfit diagnostic on a few training tiles")
    add("train", cmd_train, "train one outer fold from the pretrained weights", fold=["A", "B"])
    add("select-threshold", cmd_select_threshold, "inner-validation confidence sweep", fold=["A", "B"])
    add("evaluate", cmd_evaluate, "full-region held-out evaluation", fold=["A", "B"])
    add("ablate", cmd_ablate, "inference ablations: no-overlap tiles and single resized ROI pass", fold=["A", "B"])
    add("benchmark", cmd_benchmark, "complete-image latency benchmark", fold=["A", "B", "all"])
    add("report", cmd_report, "write reports/results.csv and reports/final_report.md")
    add("train-final", cmd_train_final, "fixed-epoch refit on both labeled photos (demonstration model)")
    sp = add("predict", cmd_predict, "tiled prediction on one photograph")
    sp.add_argument("--image", required=True)
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--roi", help="rois.json containing a polygon for this image id (file stem)")
    g.add_argument("--whole-image", action="store_true", help="score the whole image (close-up demo)")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--conf", type=float, default=None, help="confidence threshold (default: final model setting)")
    sp.add_argument("--out", default=None, help="output directory (default: reports/predictions/<image id>)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
