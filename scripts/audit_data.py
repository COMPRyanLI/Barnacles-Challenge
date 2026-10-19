"""Inspect supplied PNGs and contour connectivity; does not create ground truth.

Run: python scripts/audit_data.py
Requires numpy and Pillow. Output: analysis/data_audit.json.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Barnacles" / "Barnacles"


def component_boxes(binary):
    """8-connected foreground components, using a padded bytearray flood fill."""
    padded = np.pad(binary, 1)
    stride = padded.shape[1]
    remaining = bytearray(padded.tobytes())
    offsets = (-stride - 1, -stride, -stride + 1, -1, 1,
               stride - 1, stride, stride + 1)
    boxes = []
    start = remaining.find(1)
    while start != -1:
        stack = [start]
        remaining[start] = 0
        y, x = divmod(start, stride)
        xmin = xmax = x
        ymin = ymax = y
        area = 0
        while stack:
            pos = stack.pop()
            y, x = divmod(pos, stride)
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)
            area += 1
            for offset in offsets:
                neighbor = pos + offset
                if remaining[neighbor]:
                    remaining[neighbor] = 0
                    stack.append(neighbor)
        boxes.append([xmin - 1, ymin - 1, xmax, ymax, area])
        start = remaining.find(1, start + 1)
    return np.asarray(boxes, dtype=np.int64)


def main():
    report = {"warning": "Connected contour components are NOT verified barnacle counts. Touching outlines can merge; broken outlines can fragment.",
              "images": [], "masks": {}}
    for path in sorted(DATA.glob("*.png")):
        with Image.open(path) as image:
            report["images"].append({"file": path.name, "width": image.width,
                                     "height": image.height, "mode": image.mode})
            if path.stem not in ("mask1", "mask2"):
                continue
            rgba = np.array(image.convert("RGBA"))
            binary = rgba[:, :, 3] > 0
            boxes = component_boxes(binary)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            colors, counts = np.unique(rgba[binary], axis=0, return_counts=True)
            report["masks"][path.name] = {
                "foreground_pixels": int(binary.sum()),
                "foreground_rgba_counts": [
                    {"rgba": c.tolist(), "pixels": int(n)}
                    for c, n in zip(colors, counts)],
                "raw_connected_components": len(boxes),
                "width_percentiles_10_50_90": np.percentile(widths, [10, 50, 90]).tolist(),
                "height_percentiles_10_50_90": np.percentile(heights, [10, 50, 90]).tolist(),
                "components_under_10_pixels": int((boxes[:, 4] < 10).sum()),
                "foreground_bbox_xyxy": [int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                                           int(boxes[:, 2].max()), int(boxes[:, 3].max())],
            }
    output = ROOT / "analysis" / "data_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
