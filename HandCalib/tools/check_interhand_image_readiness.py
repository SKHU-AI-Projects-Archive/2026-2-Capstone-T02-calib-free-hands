"""Decode frozen InterHand samples and exercise the existing AnyCalib preprocessing only."""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.gigahands import GigaHandsRayDataset
from dataloaders.interhand26m import read_image
from siclib.utils.image_rays import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT / "datasets/interhand2.6m/raw/images")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/interhand_external_test_v1.csv")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[:args.limit]
    preprocessor = ImagePreprocessor(OmegaConf.create({"edge_divisible_by": 14, "random_center": False, "resize_backend": "kornia"}))
    passed = 0
    for row in rows:
        path = args.image_root / "test" / row["file_name"]
        image = read_image(path)
        if tuple(image.shape[-2:]) != (int(row["height"]), int(row["width"])):
            raise RuntimeError("Decoded resolution mismatch: %s" % path)
        target = GigaHandsRayDataset.compute_target_size(int(row["height"]), int(row["width"]))
        processed = preprocessor(image, target, crop=None, change_pix_ar=False)
        if not torch.isfinite(processed["image"]).all().item():
            raise RuntimeError("Non-finite preprocessed image: %s" % path)
        values = processed["image"]
        if values.shape[-2:] != target or not all(math.isfinite(float(value)) for value in processed["scale_xy"]):
            raise RuntimeError("Invalid preprocessing output: %s" % path)
        passed += 1
    print("decoded=%d preprocessing=PASS inference=0" % passed)


if __name__ == "__main__":
    main()
