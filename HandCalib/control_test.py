"""Measure AnyCalib image dependence without changing production evaluation."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dataloaders.gigahands import GigaHandsDataset
from models.anycalib import AnyCalibAdapter


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "runs/02_anycalib_finetune/image_dependence_control"
MANIFEST = PROJECT_ROOT / "data/manifests/gigahands.csv"
DATASET_ROOT = PROJECT_ROOT / "datasets/gigahands"
TRAIN_SPLIT = PROJECT_ROOT / "data/splits/gigahands/train.txt"
TEST_SPLIT = PROJECT_ROOT / "data/splits/gigahands/test.txt"
CHECKPOINT = PROJECT_ROOT / "runs/02_anycalib_finetune/train/checkpoint_best.pt"
LOADER_CHECKPOINT = Path("/tmp/handcalib_control_best.tar")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(split):
    return GigaHandsDataset(DATASET_ROOT, split, MANIFEST)


def grouped_indices(dataset):
    groups = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[sample["row"]["camera_key"]].append(index)
    return dict(sorted(groups.items()))


class AlteredDataset(Dataset):
    def __init__(self, target, source, mapping, fixed_image=None):
        self.target = target
        self.source = source
        self.mapping = mapping
        self.fixed_image = fixed_image

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        target = self.target.samples[index]
        target_row = target["row"]
        source_index = self.mapping[index]
        image = self.fixed_image if self.fixed_image is not None else self.source[source_index]["image"]
        return {
            "image": image,
            "intrinsics": torch.tensor(
                [target_row["fx"], target_row["fy"], target_row["cx"], target_row["cy"]],
                dtype=torch.float32,
            ),
            "meta": {
                "participant": target_row["participant"],
                "sequence": target_row["sequence"],
                "camera": target_row["camera"],
                "camera_key": target_row["camera_key"],
                "video_path": target_row["video_path"],
                "video_name": target_row["video_name"],
                "frame_index": target["frame_index"],
                "width": int(target_row["width"]),
                "height": int(target_row["height"]),
            },
            "source_index": source_index,
        }


def write_gzip_csv(path, rows, fields):
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_metrics(frame_rows):
    groups = defaultdict(list)
    for row in frame_rows:
        groups[row["camera_key"]].append(row)
    pair_rows = []
    for key, rows in sorted(groups.items()):
        pred = {
            name: statistics.median(float(row[name]) for row in rows)
            for name in ("pred_fx", "pred_fy", "pred_cx", "pred_cy")
        }
        gt = {name: float(rows[0][name]) for name in ("gt_fx", "gt_fy", "gt_cx", "gt_cy")}
        rel_fx = abs(pred["pred_fx"] - gt["gt_fx"]) / abs(gt["gt_fx"])
        rel_fy = abs(pred["pred_fy"] - gt["gt_fy"]) / abs(gt["gt_fy"])
        pair_rows.append({
            "camera_key": key,
            "total_frames": len(rows),
            **pred,
            **gt,
            "pair_rel_fx_error": rel_fx,
            "pair_rel_fy_error": rel_fy,
            "pair_max_rel_f_error": max(rel_fx, rel_fy),
        })
    values = [row["pair_max_rel_f_error"] for row in pair_rows]
    return pair_rows, {
        "frames": len(frame_rows),
        "pairs": len(pair_rows),
        "mean_max_rel_f": sum(values) / len(values),
        "median_max_rel_f": statistics.median(values),
        "std_max_rel_f": statistics.pstdev(values),
        "within_1pct": sum(value <= 0.01 for value in values),
        "within_2pct": sum(value <= 0.02 for value in values),
        "within_3pct": sum(value <= 0.03 for value in values),
        "within_5pct": sum(value <= 0.05 for value in values),
        "within_10pct": sum(value <= 0.10 for value in values),
        "within_20pct": sum(value <= 0.20 for value in values),
    }


def make_mapping(condition, target, train):
    target_groups = grouped_indices(target)
    train_groups = grouped_indices(train)
    if condition == "fixed":
        first_key = sorted(train_groups, key=lambda key: (key, train.samples[train_groups[key][0]]["video_path"]))[0]
        source_index = train_groups[first_key][0]
        return [source_index] * len(target), {
            "fixed_source_participant": train.samples[source_index]["row"]["participant"],
            "fixed_source_camera_key": first_key,
            "fixed_source_video": train.samples[source_index]["row"]["video_path"],
            "fixed_source_frame": train.samples[source_index]["frame_index"],
        }
    keys = sorted(target_groups)
    mapping = {}
    for index, key in enumerate(keys):
        source_key = keys[(index + 1) % len(keys)]
        for position, target_index in enumerate(target_groups[key]):
            mapping[target_index] = target_groups[source_key][position % len(target_groups[source_key])]
    return [mapping[index] for index in range(len(target))], {"shuffle_shift": 1}


def run(condition, dry_run=False):
    target = load_dataset(TEST_SPLIT)
    train = load_dataset(TRAIN_SPLIT)
    mapping, condition_metadata = make_mapping(condition, target, train)
    fixed_image = None
    if condition == "fixed":
        fixed_image = train[mapping[0]]["image"]
    dataset = AlteredDataset(target, target, mapping, fixed_image=fixed_image)
    output = ROOT / condition
    if output.exists():
        if dry_run:
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    if LOADER_CHECKPOINT.exists() or LOADER_CHECKPOINT.is_symlink():
        LOADER_CHECKPOINT.unlink()
    LOADER_CHECKPOINT.symlink_to(CHECKPOINT)
    adapter = AnyCalibAdapter(checkpoint=LOADER_CHECKPOINT, cam_id="pinhole", device="cuda:1").build()
    frame_rows = []
    mapping_rows = []
    limit = 4 if dry_run else len(dataset)
    for batch_index, batch in enumerate(loader):
        start = batch_index * loader.batch_size
        if start >= limit:
            break
        keep = min(len(batch["image"]), limit - start)
        prediction = adapter.predict(batch["image"][:keep])
        intrinsics = prediction["intrinsics"]
        if isinstance(intrinsics, (list, tuple)):
            intrinsics = torch.stack([torch.as_tensor(value) for value in intrinsics])
        for offset in range(keep):
            target_index = start + offset
            meta = {}
            for key, value in batch["meta"].items():
                item = value[offset] if isinstance(value, (list, tuple)) else value[offset].item()
                meta[key] = item
            pred = intrinsics[offset].detach().cpu().tolist()
            row = {
                **meta,
                "gt_fx": float(batch["intrinsics"][offset, 0]),
                "gt_fy": float(batch["intrinsics"][offset, 1]),
                "gt_cx": float(batch["intrinsics"][offset, 2]),
                "gt_cy": float(batch["intrinsics"][offset, 3]),
                "pred_fx": pred[0], "pred_fy": pred[1], "pred_cx": pred[2], "pred_cy": pred[3],
                "success": True,
            }
            frame_rows.append(row)
            if condition == "shuffled":
                source = target.samples[mapping[target_index]]
                mapping_rows.append({
                    "target_camera_key": meta["camera_key"],
                    "source_camera_key": source["row"]["camera_key"],
                    "target_frame_index": meta["frame_index"],
                    "source_frame_index": source["frame_index"],
                })
    pair_rows, metrics = pair_metrics(frame_rows)
    if dry_run:
        print(json.dumps({"condition": condition, "frames": len(frame_rows), "pairs": len({r['camera_key'] for r in frame_rows}), "metadata": condition_metadata}, indent=2))
        shutil.rmtree(output)
        return
    frame_fields = list(frame_rows[0])
    pair_fields = list(pair_rows[0])
    write_gzip_csv(output / "frame_predictions.csv.gz", frame_rows, frame_fields)
    with (output / "pair_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    if mapping_rows:
        with (output / "mapping.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
            writer.writeheader()
            writer.writerows(mapping_rows)
    metadata = {
        "condition": f"{condition}_image",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_epoch": 5,
        "checkpoint_global_step": 31975,
        "cam_id": "pinhole",
        "batch_size": 2,
        "num_workers": 4,
        "test_participant": "p52",
        "test_sequence": "p52-instrument-0034",
        "gt_used_only_for_metrics": True,
        **condition_metadata,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"condition": condition, **metrics}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("fixed", "shuffled"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.condition, args.dry_run)


if __name__ == "__main__":
    main()
