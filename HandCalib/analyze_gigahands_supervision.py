"""GigaHands의 distortion-aware ray supervision과 pinhole oracle을 CPU에서 점검한다."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import cv2
import torch
from omegaconf import OmegaConf

from anycalib.cameras import CameraFactory
from siclib.models.networks.anycalib_net import Calibrator


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "data/manifests/gigahands.csv"
EXPECTED_PAIRS = {"train": 147, "val": 49}
TARGET_RESOLUTION = 102400
EDGE_DIVISIBLE_BY = 14
ASPECT_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class Pair:
    participant: str
    sequence: str
    camera: str
    camera_key: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(split):
    if split not in EXPECTED_PAIRS:
        raise ValueError("--split must be train or val; Test audit is intentionally unsupported.")
    split_path = PROJECT_ROOT / "data/splits/gigahands" / f"{split}.txt"
    participants = {line.strip() for line in split_path.read_text().splitlines() if line.strip()}
    pairs = {}
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["participant"] not in participants:
                continue
            pairs[row["camera_key"]] = Pair(
                participant=row["participant"], sequence=row["sequence"], camera=row["camera"],
                camera_key=row["camera_key"], width=int(float(row["width"])), height=int(float(row["height"])),
                fx=float(row["fx"]), fy=float(row["fy"]), cx=float(row["cx"]), cy=float(row["cy"]),
                k1=float(row["k1"]), k2=float(row["k2"]), p1=float(row["p1"]), p2=float(row["p2"]),
            )
    if len(pairs) != EXPECTED_PAIRS[split]:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS[split]} {split} pairs, found {len(pairs)}")
    return list(pairs.values()), split_path


def compute_target_size(height, width):
    aspect = max(ASPECT_RANGE[0], min(height / width, ASPECT_RANGE[1]))
    target_width = (TARGET_RESOLUTION / aspect) ** 0.5
    target_height = aspect * target_width
    return (round(target_height / EDGE_DIVISIBLE_BY) * EDGE_DIVISIBLE_BY,
            round(target_width / EDGE_DIVISIBLE_BY) * EDGE_DIVISIBLE_BY)


def transform_spec(pair, target_size):
    target_height, target_width = target_size
    height, width = pair.height, pair.width
    aspect = target_width / target_height
    shift_x = shift_y = 0.0
    if width / height > aspect:
        crop_width = round(width - height * aspect)
        shift_x = -(crop_width // 2)
        width -= crop_width
    else:
        crop_height = round(height - width / aspect)
        shift_y = -(crop_height // 2)
        height -= crop_height
    scale_x = target_width / width
    scale_y = target_height / height
    shift_x *= scale_x
    shift_y *= scale_y
    scale = torch.tensor([scale_x, scale_y], dtype=torch.float32)
    shift = torch.tensor([shift_x, shift_y], dtype=torch.float32)
    params = torch.tensor([
        pair.fx * scale_x, pair.fy * scale_y,
        pair.cx * scale_x + shift_x, pair.cy * scale_y + shift_y,
    ], dtype=torch.float32)
    return params, scale, shift


def target_pixel_grid(target_size):
    height, width = target_size
    x = torch.arange(width, dtype=torch.float32) + 0.5
    y = torch.arange(height, dtype=torch.float32) + 0.5
    return torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1).reshape(-1, 2)


def distortion_aware_rays(pair, target_size):
    target_height, target_width = target_size
    params, scale, shift = transform_spec(pair, target_size)
    grid = target_pixel_grid(target_size)
    original_pixels = (grid - shift) / scale
    camera_matrix = torch.tensor([[pair.fx, 0.0, pair.cx], [0.0, pair.fy, pair.cy], [0.0, 0.0, 1.0]], dtype=torch.float32).numpy()
    distortion = torch.tensor([pair.k1, pair.k2, pair.p1, pair.p2], dtype=torch.float32).numpy()
    normalized = cv2.undistortPoints(original_pixels.numpy().reshape(-1, 1, 2), camera_matrix, distortion).reshape(-1, 2)
    rays = torch.from_numpy(normalized).to(torch.float32)
    rays = torch.cat((rays, torch.ones((len(rays), 1), dtype=torch.float32)), dim=-1)
    return torch.nn.functional.normalize(rays, dim=-1), params, grid


def pinhole_rays(params, grid):
    rays = torch.cat(((grid - params[2:]) / params[:2], torch.ones((len(grid), 1), dtype=grid.dtype)), dim=-1)
    return torch.nn.functional.normalize(rays, dim=-1)


def angular_degrees(first, second):
    cosine = (first * second).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "p90": sorted(values)[int(0.90 * (len(values) - 1))],
        "p95": sorted(values)[int(0.95 * (len(values) - 1))], "max": max(values),
    }


def official_pinhole_oracle(calibrator, rays, target_size):
    height, width = target_size
    prediction = calibrator(
        {"rays": rays.unsqueeze(0)},
        {"image": torch.zeros((1, 3, height, width), dtype=torch.float32), "cam_id": ["pinhole"]},
    )
    return prediction["intrinsics"][0].detach().cpu(), bool(prediction["success"][0].item())


def audit_pair(pair, calibrator):
    target_size = compute_target_size(pair.height, pair.width)
    rays, transformed, grid = distortion_aware_rays(pair, target_size)
    pinhole = pinhole_rays(transformed, grid)
    angular = angular_degrees(rays, pinhole)
    center = (grid[:, 0] >= target_size[1] * 0.25) & (grid[:, 0] < target_size[1] * 0.75) & (grid[:, 1] >= target_size[0] * 0.25) & (grid[:, 1] < target_size[0] * 0.75)
    border = ~center
    oracle, success = official_pinhole_oracle(calibrator, rays, target_size)
    focal = ((oracle[:2] - transformed[:2]).abs() / transformed[:2].abs()).max()
    principal = 2 * torch.maximum((oracle[2] - transformed[2]).abs() / target_size[1], (oracle[3] - transformed[3]).abs() / target_size[0])
    return {
        "participant": pair.participant, "sequence": pair.sequence, "camera": pair.camera,
        "camera_key": pair.camera_key, "width": pair.width, "height": pair.height,
        "target_height": target_size[0], "target_width": target_size[1], "pixel_offset": 0.5,
        "ray_count": len(rays), "angular_mean_deg": stats(angular.tolist())["mean"],
        "angular_median_deg": stats(angular.tolist())["median"], "angular_p90_deg": stats(angular.tolist())["p90"],
        "angular_p95_deg": stats(angular.tolist())["p95"], "angular_max_deg": stats(angular.tolist())["max"],
        "center_angular_mean_deg": stats(angular[center].tolist())["mean"],
        "border_angular_mean_deg": stats(angular[border].tolist())["mean"],
        "oracle_success": success, "oracle_fx": float(oracle[0]), "oracle_fy": float(oracle[1]),
        "oracle_cx": float(oracle[2]), "oracle_cy": float(oracle[3]),
        "oracle_max_rel_f_error": float(focal), "oracle_max_rel_c_error": float(principal),
        "oracle_within_5pct": bool(success and focal <= 0.05),
    }


def _mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def compatibility_report():
    weight = Path(torch.hub.get_dir()) / "anycalib" / "anycalib_pinhole.pt"
    state = torch.load(weight, map_location="cpu", weights_only=False) if weight.exists() else None
    state = state if isinstance(state, dict) else {}
    state = state.get("model", state.get("state_dict", state))
    dino = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    result = {"status": "not_run", "strict_load": None, "pretrained_key_count": len(state), "training_key_count": None, "missing": None, "unexpected": None, "shape_mismatch": None,
              "pretrained_weight_sha256": sha256_file(weight) if weight.exists() else None}
    result["reason"] = ("DINOv2 training backbone cache is absent; no download was attempted." if not dino.exists()
                        else "Static compatibility hook is intentionally not instantiating the training model in this CPU-only preparation step.")
    return result


def run(split):
    pairs, split_path = load_pairs(split)
    calibrator = Calibrator(OmegaConf.create({"loss": {"name": None}}))
    rows = [audit_pair(pair, calibrator) for pair in pairs]
    output = PROJECT_ROOT / "runs/02_anycalib_finetune/supervision_audit"
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{split}_pair_geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    oracle_valid = [row for row in rows if row["oracle_success"]]
    report = {
        "split": split, "pair_count": len(rows), "target_resolution": TARGET_RESOLUTION,
        "edge_divisible_by": EDGE_DIVISIBLE_BY, "aspect_range": ASPECT_RANGE, "pixel_convention": "pixel centers, offset 0.5",
        "ray_definition": "cv2.undistortPoints Brown-Conrady [k1,k2,p1,p2] then normalize([x,y,1])",
        "pinhole_comparison": {key: _mean(rows, key) for key in ("angular_mean_deg", "angular_median_deg", "angular_p90_deg", "angular_p95_deg", "angular_max_deg")},
        "oracle": {"valid_pair_count": len(oracle_valid), "max_rel_f_mean": _mean(oracle_valid, "oracle_max_rel_f_error"), "max_rel_f_median": statistics.median(row["oracle_max_rel_f_error"] for row in oracle_valid), "within_5pct_count": sum(row["oracle_within_5pct"] for row in oracle_valid), "within_5pct_rate": sum(row["oracle_within_5pct"] for row in oracle_valid) / len(oracle_valid)},
        "compatibility": compatibility_report(),
        "manifest_sha256": sha256_file(MANIFEST), "split_sha256": sha256_file(split_path),
        "opencv_version": cv2.__version__, "python_version": platform.python_version(), "anycalib_commit": "027a8497d893f4b2596f23d6324c05e4b81064ed",
        "test_data_used": False, "training_run": False,
    }
    (output / f"summary_{split}.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / f"metadata_{split}.json").open("w") as handle:
        json.dump({"repository_commit": __import__("subprocess").check_output(["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], text=True).strip(), **report}, handle, indent=2)
    summary = [
        f"Train pairs: {len(rows)}" if split == "train" else "Validation pairs: 49",
        f"Target network geometry: {TARGET_RESOLUTION} pixels, edge divisible by {EDGE_DIVISIBLE_BY}, aspect {ASPECT_RANGE}",
        "Pixel convention: pixel centers with offset 0.5",
        f"Pinhole vs distortion-aware ray ({split} mean/median/p90/p95/max degrees): {report['pinhole_comparison']}",
        f"Perfect-ray -> pinhole oracle ({split}) focal mean/median: {report['oracle']['max_rel_f_mean']} / {report['oracle']['max_rel_f_median']}",
        f"Oracle focal within 5%: {report['oracle']['within_5pct_count']} / {report['oracle']['valid_pair_count']} ({report['oracle']['within_5pct_rate']})",
        f"Weight compatibility: {report['compatibility']}", "No training was run.", "No Test data was used.",
    ]
    (output / f"summary_{split}.txt").write_text("\n".join(summary) + "\n")
    reports = {name: json.loads((output / f"summary_{name}.json").read_text()) for name in ("train", "val") if (output / f"summary_{name}.json").exists()}
    if len(reports) == 2:
        combined = {"splits": reports, "test_data_used": False, "training_run": False}
        (output / "summary.json").write_text(json.dumps(combined, indent=2) + "\n")
        repository_commit = __import__("subprocess").check_output(["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], text=True).strip()
        (output / "metadata.json").write_text(json.dumps({"repository_commit": repository_commit, **combined}, indent=2) + "\n")
        (output / "summary.txt").write_text("\n\n".join((output / f"summary_{name}.txt").read_text().rstrip() for name in ("train", "val")) + "\n")
    print("\n".join(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    run(args.split)


if __name__ == "__main__":
    main()
