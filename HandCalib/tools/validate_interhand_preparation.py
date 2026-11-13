"""Validate the image-free InterHand external-evaluation preparation."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/interhand_external_test_v1.csv"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_trip(rows):
    errors = {key: [] for key in ("fx", "fy", "cx", "cy")}
    scale = (0.73, 1.17)
    shift = (-11.5, 8.25)
    for row in rows:
        original = {key: float(row[key]) for key in errors}
        transformed = {"fx": original["fx"] * scale[0], "fy": original["fy"] * scale[1], "cx": original["cx"] * scale[0] + shift[0], "cy": original["cy"] * scale[1] + shift[1]}
        recovered = {"fx": transformed["fx"] / scale[0], "fy": transformed["fy"] / scale[1], "cx": (transformed["cx"] - shift[0]) / scale[0], "cy": (transformed["cy"] - shift[1]) / scale[1]}
        for key in errors:
            errors[key].append(abs(recovered[key] - original[key]))
    return {key: {"max_abs_error": max(values), "mean_abs_error": statistics.mean(values)} for key, values in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-result", action="store_true")
    args = parser.parse_args()
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    units = {row["calibration_unit"] for row in rows}
    result = {
        "protocol_version": "02-H.0-v1",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST),
        "official_test_only": all(row["split"] == "test" and row["dataset"] == "InterHand2.6M" for row in rows),
        "calibration_units": len(units), "selected_frames": len(rows), "max_frames_per_unit": max(sum(row["calibration_unit"] == unit for row in rows) for unit in units),
        "sampling_rule": "sort (seq_name, frame_idx, image_id); midpoint quantiles round((i+0.5)*N/K-0.5); nearest unused collision resolution",
        "metric": "per-unit median prediction for fx/fy/cx/cy; max(rel_fx, rel_fy); equal-weight mean over 360 units",
        "camera_model": "pinhole; distortion metadata unavailable in inspected InterHand annotation",
        "dataset_paths": {"annotation_root": "datasets/interhand2.6m/raw/annotations", "image_root": "datasets/interhand2.6m/raw/images", "archive_root": "datasets/interhand2.6m/archives"},
        "expected_image_archive": {"parts": 44, "suffix": "partaa..partbr", "bytes": 81718036480, "asset": "InterHand2.6M.images.5.fps.v1.0.tar"},
        "checkpoint_identities": {"pretrained_sha256": "e73b174563bcb90dc9a0e348ee64fbb898901d2a1786623e62fc0e3b176a1a38", "finetuned_path": "runs/02_anycalib_finetune/train/checkpoint_best.pt", "finetuned_sha256": "1b87efa27c847886f5987084a95000bccf2491f1abc719efcfa1c232cd8c2d41", "finetuned_epoch": 5, "finetuned_global_step": 31975},
        "backward_compatibility": "existing GigaHands paths preserved; InterHand selected only when dataset.name is InterHand2.6M",
        "metadata_round_trip": round_trip(rows),
        "readiness_state": "NOT READY: image archive is not present",
        "remaining_requirements": ["download all 44 official image parts and CHECKSUM", "verify part sizes/checksum", "scan all tar members and match manifest 5760/5760", "selectively extract frozen manifest images", "run image decode and preprocessing smoke tests", "run old GigaHands regression checks"],
        "inference_run": False,
    }
    print(json.dumps(result, indent=2))
    if args.write_result:
        (ROOT / "results/02h0_interhand_external_eval_preparation.yaml").write_text("# Image-free preparation record.\n" + json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
