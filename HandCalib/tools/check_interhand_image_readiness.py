"""Decode frozen InterHand samples and exercise the existing AnyCalib preprocessing only."""

import argparse
import csv
import json
import math
import sys
import statistics
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.gigahands import GigaHandsRayDataset
from dataloaders.interhand26m import read_image
from anycalib.cameras import CameraFactory
from siclib.utils.image_rays import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT / "datasets/interhand2.6m/raw/images")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/interhand_external_test_v1.csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke limit; default checks all frozen manifest rows.")
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.limit is None and len(rows) != 5760:
        raise RuntimeError("Expected all 5760 frozen manifest rows, found %d" % len(rows))
    preprocessor = ImagePreprocessor(OmegaConf.create({"edge_divisible_by": 14, "random_center": False, "resize_backend": "kornia"}))
    camera = CameraFactory.create_from_id("pinhole")
    passed = 0
    errors = {key: [] for key in ("fx", "fy", "cx", "cy")}
    output_shapes = set()
    for row in rows:
        path = args.image_root / "test" / row["file_name"]
        image = read_image(path)
        if image.ndim != 3 or image.shape[0] != 3:
            raise RuntimeError("Decoded image does not have three RGB channels: %s" % path)
        if tuple(image.shape[-2:]) != (int(row["height"]), int(row["width"])):
            raise RuntimeError("Decoded resolution mismatch: %s" % path)
        target = GigaHandsRayDataset.compute_target_size(int(row["height"]), int(row["width"]))
        processed = preprocessor(image, target, crop=None, change_pix_ar=False)
        if not torch.isfinite(processed["image"]).all().item():
            raise RuntimeError("Non-finite preprocessed image: %s" % path)
        values = processed["image"]
        if values.dtype != torch.float32 or not torch.isfinite(values).all().item():
            raise RuntimeError("Invalid preprocessed pixel values: %s" % path)
        if values.shape[-2:] != target or not all(math.isfinite(float(value)) for value in processed["scale_xy"]):
            raise RuntimeError("Invalid preprocessing output: %s" % path)
        output_shapes.add(tuple(int(value) for value in values.shape))
        original = torch.tensor([float(row[key]) for key in ("fx", "fy", "cx", "cy")])
        transformed = camera.scale_and_shift(original, processed["scale_xy"], processed["shift_xy"])
        recovered = camera.reverse_scale_and_shift(transformed, processed["scale_xy"], processed["shift_xy"])
        for index, key in enumerate(errors):
            errors[key].append(abs(float(recovered[index] - original[index])))
        passed += 1
    result = {
        "decoded": passed,
        "total": len(rows),
        "preprocessing": "PASS",
        "inference": 0,
        "output_shapes": sorted(output_shapes),
        "k_round_trip": {
            key: {"mean_abs_error": statistics.fmean(values), "median_abs_error": statistics.median(values), "max_abs_error": max(values)}
            for key, values in errors.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
