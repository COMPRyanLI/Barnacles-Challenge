from pathlib import Path

import pytest
import yaml

from barnacles.config import load_config, load_rois

ROOT = Path(__file__).resolve().parents[1]


def test_project_config_resolves_paths_against_repo_root():
    cfg = load_config(ROOT / "configs" / "experiment.yaml")
    assert cfg["root"] == ROOT
    assert cfg["paths"]["raw_dir"] == ROOT / "Barnacles" / "Barnacles"
    assert cfg["source_paths"]["img1"]["image"] == ROOT / "Barnacles" / "Barnacles" / "img1.png"
    assert cfg["source_paths"]["img1"]["mask"].name == "mask1.png"
    assert "mask" not in cfg["source_paths"]["unseen_img1"]


def test_config_rejects_fold_using_unlabeled_source(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    raw["folds"]["A"]["dev"] = "unseen_img1"
    p = tmp_path / "configs" / "bad.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="labeled"):
        load_config(p, root=ROOT)


def test_rois_file_polygons_are_valid():
    rois = load_rois(ROOT / "data" / "annotations" / "rois.json")
    assert set(rois) >= {"img1", "img2", "unseen_img1"}
    assert len(rois["img1"]) == 4
    assert rois["unseen_img2"] is None  # whole-image demo, no polygon
