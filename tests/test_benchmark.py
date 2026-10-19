import numpy as np
from PIL import Image

from barnacles.benchmark import benchmark_region
from test_predict import INFER, TILES, PixelDetector, squares_image


def test_benchmark_protocol_counts_runs_and_reports_stages(tmp_path):
    img = squares_image(900, 700, [[100, 100, 120, 120], [600, 300, 630, 330]])
    p = tmp_path / "图像.png"
    Image.fromarray(img).save(p)
    det = PixelDetector()
    det.device = "cpu"
    rep = benchmark_region(p, [[0, 0], [900, 0], [900, 700], [0, 700]], None, INFER, TILES, warmup=3, runs=10,
                           detector=det)
    assert rep["latency"]["n"] == 10
    assert det.calls == 13 * rep["tiles"] // INFER["batch"] + 13 * (rep["tiles"] % INFER["batch"] > 0)
    assert rep["latency"]["median_ms"] > 0 and rep["latency"]["p90_ms"] >= rep["latency"]["median_ms"]
    assert set(rep["stages"]) >= {"load_and_region", "prepare_tiles", "tile_inference", "restore_merge_count"}
    assert rep["output_generation_s"]["overlay_jpg"] > 0
    assert rep["count_at_conf"] == 2


def test_benchmark_single_pass_mode(tmp_path):
    img = squares_image(900, 700, [[100, 100, 120, 120]])
    p = tmp_path / "x.png"
    Image.fromarray(img).save(p)
    det = PixelDetector()
    det.device = "cpu"
    rep = benchmark_region(p, None, None, INFER, TILES, warmup=1, runs=2, detector=det, single_pass=True)
    assert rep["mode"] == "single resized ROI pass" and rep["tiles"] == 1 and rep["count_at_conf"] == 1
