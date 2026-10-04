"""GigaHands의 distortion-aware ray supervision과 pinhole oracle을 CPU에서 점검한다."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import statistics

import cv2
import torch
from omegaconf import OmegaConf

from siclib.models.networks.anycalib_net import Calibrator
from siclib.models.networks.anycalib_net import AnyCalib as TrainingAnyCalib


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


def oracle_metrics(prediction, target, target_size):
    rel_f = (prediction[:2] - target[:2]).abs() / target[:2].abs()
    rel_c = 2 * torch.stack(
        ((prediction[2] - target[2]).abs() / target_size[1],
         (prediction[3] - target[3]).abs() / target_size[0])
    )
    return {
        "rel_fx_error": float(rel_f[0]),
        "rel_fy_error": float(rel_f[1]),
        "max_rel_f_error": float(rel_f.max()),
        "max_rel_c_error": float(rel_c.max()),
    }


def audit_pair(pair, calibrator):
    target_size = compute_target_size(pair.height, pair.width)
    rays, transformed, grid = distortion_aware_rays(pair, target_size)
    pinhole = pinhole_rays(transformed, grid)
    angular = angular_degrees(rays, pinhole)
    center = (grid[:, 0] >= target_size[1] * 0.25) & (grid[:, 0] < target_size[1] * 0.75) & (grid[:, 1] >= target_size[0] * 0.25) & (grid[:, 1] < target_size[0] * 0.75)
    border = ~center
    canonical = pinhole_rays(transformed, grid)
    physical_oracle, physical_success = official_pinhole_oracle(calibrator, rays, target_size)
    pinhole_oracle, pinhole_success = official_pinhole_oracle(calibrator, canonical, target_size)
    physical_metrics = oracle_metrics(physical_oracle, transformed, target_size)
    pinhole_metrics = oracle_metrics(pinhole_oracle, transformed, target_size)
    return {
        "participant": pair.participant, "sequence": pair.sequence, "camera": pair.camera,
        "camera_key": pair.camera_key, "width": pair.width, "height": pair.height,
        "target_height": target_size[0], "target_width": target_size[1], "pixel_offset": 0.5,
        "ray_count": len(rays), "angular_mean_deg": stats(angular.tolist())["mean"],
        "angular_median_deg": stats(angular.tolist())["median"], "angular_p90_deg": stats(angular.tolist())["p90"],
        "angular_p95_deg": stats(angular.tolist())["p95"], "angular_max_deg": stats(angular.tolist())["max"],
        "center_angular_mean_deg": stats(angular[center].tolist())["mean"],
        "border_angular_mean_deg": stats(angular[border].tolist())["mean"],
        "physical_oracle_success": physical_success,
        "physical_oracle_max_rel_f_error": physical_metrics["max_rel_f_error"],
        "physical_oracle_max_rel_c_error": physical_metrics["max_rel_c_error"],
        "physical_oracle_within_5pct": bool(physical_success and physical_metrics["max_rel_f_error"] <= 0.05),
        "pinhole_oracle_success": pinhole_success,
        "pinhole_oracle_fx": float(pinhole_oracle[0]), "pinhole_oracle_fy": float(pinhole_oracle[1]),
        "pinhole_oracle_cx": float(pinhole_oracle[2]), "pinhole_oracle_cy": float(pinhole_oracle[3]),
        "pinhole_oracle_rel_fx_error": pinhole_metrics["rel_fx_error"],
        "pinhole_oracle_rel_fy_error": pinhole_metrics["rel_fy_error"],
        "pinhole_oracle_max_rel_f_error": pinhole_metrics["max_rel_f_error"],
        "pinhole_oracle_max_rel_c_error": pinhole_metrics["max_rel_c_error"],
        "pinhole_oracle_within_5pct": bool(pinhole_success and pinhole_metrics["max_rel_f_error"] <= 0.05),
        "pinhole_oracle_within_0_1pct": bool(pinhole_success and pinhole_metrics["max_rel_f_error"] <= 0.001),
    }


def _mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def oracle_summary(rows, prefix):
    valid = [row for row in rows if row[f"{prefix}_oracle_success"]]
    focal = f"{prefix}_oracle_max_rel_f_error"
    center = f"{prefix}_oracle_max_rel_c_error"
    within_5 = f"{prefix}_oracle_within_5pct"
    result = {
        "valid_pair_count": len(valid),
        "max_rel_f_mean": _mean(valid, focal),
        "max_rel_f_median": statistics.median(row[focal] for row in valid),
        "max_rel_f_max": max(row[focal] for row in valid),
        "max_rel_c_mean": _mean(valid, center),
        "max_rel_c_median": statistics.median(row[center] for row in valid),
        "max_rel_c_max": max(row[center] for row in valid),
        "within_5pct_count": sum(row[within_5] for row in valid),
        "within_5pct_rate": sum(row[within_5] for row in valid) / len(valid),
    }
    if prefix == "pinhole":
        within_01 = "pinhole_oracle_within_0_1pct"
        result["within_0_1pct_count"] = sum(row[within_01] for row in valid)
        result["within_0_1pct_rate"] = sum(row[within_01] for row in valid) / len(valid)
    return result


def validate_config(config_path):
    path = Path(config_path)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            config = OmegaConf.load(candidate)
            break
    else:
        raise FileNotFoundError(f"Could not find config: {config_path}")
    expected = {
        "model.model_id": "anycalib_pinhole",
        "model.cam_id": "pinhole",
        "supervision.input": "raw_rgb",
        "supervision.target": "canonical_pinhole_rays",
    }
    for key, value in expected.items():
        actual = OmegaConf.select(config, key)
        if actual != value:
            raise ValueError(f"Config {key} must be {value!r}, found {actual!r}")
    if OmegaConf.select(config, "supervision.distortion_for_target") is not False:
        raise ValueError("supervision.distortion_for_target must be false")
    return config


def synthetic_canonical_oracle(calibrator):
    pair = Pair("synthetic", "synthetic", "pinhole", "synthetic/pinhole", 1280, 720, 913.0, 877.0, 601.0, 319.0, 0.0, 0.0, 0.0, 0.0)
    target_size = compute_target_size(pair.height, pair.width)
    transformed, _, _ = transform_spec(pair, target_size)
    rays = pinhole_rays(transformed, target_pixel_grid(target_size))
    prediction, success = official_pinhole_oracle(calibrator, rays, target_size)
    metrics = oracle_metrics(prediction, transformed, target_size)
    if not success or metrics["max_rel_f_error"] > 0.001:
        raise RuntimeError(f"Synthetic canonical pinhole oracle failed: {success=}, {metrics=}")
    return {"target_size": target_size, "success": success, **metrics}


def compatibility_report():
    weight = Path(torch.hub.get_dir()) / "anycalib" / "anycalib_pinhole.pt"
    state = torch.load(weight, map_location="cpu", weights_only=False) if weight.exists() else None
    state = state if isinstance(state, dict) else {}
    state = state.get("model", state.get("state_dict", state))
    dino = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    result = {"status": "not_run", "strict_load": None, "pretrained_key_count": len(state), "training_key_count": None, "matching_key_count": None, "missing": None, "unexpected": None, "shape_mismatch": None,
              "pretrained_weight_sha256": sha256_file(weight) if weight.exists() else None, "dino_path": str(dino)}
    result["reason"] = ("DINOv2 training backbone cache is absent; no download was attempted." if not dino.exists()
                        else "")
    if dino.exists():
        model_conf = OmegaConf.load(PROJECT_ROOT / "AnyCalib/siclib/configs/model/anycalib.yaml")
        training_model = TrainingAnyCalib(model_conf)
        training_state = training_model.state_dict()
        pretrained_keys = set(state)
        training_keys = set(training_state)
        shape_mismatch = sorted(key for key in pretrained_keys & training_keys if state[key].shape != training_state[key].shape)
        result.update({
            "status": "pass" if not shape_mismatch and pretrained_keys == training_keys else "fail",
            "training_key_count": len(training_keys),
            "matching_key_count": len(pretrained_keys & training_keys),
            "missing": sorted(training_keys - pretrained_keys),
            "unexpected": sorted(pretrained_keys - training_keys),
            "shape_mismatch": shape_mismatch,
        })
        try:
            training_model.load_state_dict(state, strict=True)
            result["strict_load"] = True
        except RuntimeError as exc:
            result["strict_load"] = False
            result["strict_load_error"] = str(exc)
        del training_model, training_state
    return result


def run(split, config_path):
    validate_config(config_path)
    pairs, split_path = load_pairs(split)
    calibrator = Calibrator(OmegaConf.create({"loss": {"name": None}}))
    synthetic = synthetic_canonical_oracle(calibrator)
    rows = [audit_pair(pair, calibrator) for pair in pairs]
    output = PROJECT_ROOT / "runs/02_anycalib_finetune/supervision_audit"
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{split}_pair_geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "split": split, "pair_count": len(rows), "target_resolution": TARGET_RESOLUTION,
        "edge_divisible_by": EDGE_DIVISIBLE_BY, "aspect_range": ASPECT_RANGE, "pixel_convention": "pixel centers, offset 0.5",
        "ray_definition": "cv2.undistortPoints Brown-Conrady [k1,k2,p1,p2] then normalize([x,y,1])",
        "pinhole_comparison": {key: _mean(rows, key) for key in ("angular_mean_deg", "angular_median_deg", "angular_p90_deg", "angular_p95_deg", "angular_max_deg", "center_angular_mean_deg", "border_angular_mean_deg")},
        "physical_distortion_oracle": oracle_summary(rows, "physical"),
        "canonical_pinhole_oracle": oracle_summary(rows, "pinhole"),
        "synthetic_canonical_oracle": synthetic,
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
        f"Physical distortion-aware ray -> pinhole oracle: {report['physical_distortion_oracle']}",
        f"Canonical pinhole ray -> pinhole oracle: {report['canonical_pinhole_oracle']}",
        "Angular p90/p95/max are pair-equal means of per-pair statistics.",
        "Selected supervision: raw RGB + canonical pinhole target rays",
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
    run(args.split, args.config)


if __name__ == "__main__":
    main()
