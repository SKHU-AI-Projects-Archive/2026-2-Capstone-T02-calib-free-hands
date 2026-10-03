"""GigaHands 02-B dry-run과 one-step smoke 경로를 제공한다."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "AnyCalib"))

from dataloaders.gigahands import GigaHandsRayDataset, PairBalancedSampler
from models.anycalib import build_training_anycalib


EXPECTED_TRAIN_FRAMES = 44254
EXPECTED_TRAIN_PAIRS = 147
EXPECTED_VAL_FRAMES = 18652
EXPECTED_VAL_PAIRS = 49
TRAIN_FRAMES_PER_PAIR = 174


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT.parent), *args], text=True).strip()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, PROJECT_ROOT / path, PROJECT_ROOT.parent / path):
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def load_experiment(config_path):
    path = resolve_path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["model"]["model_id"] != "anycalib_pinhole":
        raise ValueError("02-B requires model_id=anycalib_pinhole")
    if config["model"]["cam_id"] != "pinhole":
        raise ValueError("02-B requires cam_id=pinhole")
    if config["supervision"]["target"] != "canonical_pinhole_rays":
        raise ValueError("02-B requires canonical_pinhole_rays supervision")
    return config, path


def make_datasets(config):
    root = resolve_path(config["dataset"]["root"])
    manifest = PROJECT_ROOT / "data/manifests/gigahands.csv"
    train = GigaHandsRayDataset(root, resolve_path(config["dataset"]["train_split"]), manifest)
    val = GigaHandsRayDataset(root, resolve_path(config["dataset"]["val_split"]), manifest)
    if len(train) != EXPECTED_TRAIN_FRAMES or len(train.pair_indices) != EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(f"Unexpected Train dataset count: {len(train)=}, {len(train.pair_indices)=}")
    if len(val) != EXPECTED_VAL_FRAMES or len(val.pair_indices) != EXPECTED_VAL_PAIRS:
        raise RuntimeError(f"Unexpected Validation dataset count: {len(val)=}, {len(val.pair_indices)=}")
    return train, val, root, manifest


def dry_run(config, batch_size, num_workers):
    train, val, _, _ = make_datasets(config)
    sampler = PairBalancedSampler(train, TRAIN_FRAMES_PER_PAIR, seed=config["experiment"]["seed"])
    sampler.set_epoch(0)
    print(f"Train frames = {len(train)}")
    print(f"Train pairs = {len(train.pair_indices)}")
    print(f"sampler samples/epoch = {len(sampler)}")
    print(f"Validation frames = {len(val)}")
    print(f"Validation pairs = {len(val.pair_indices)}")
    print("target geometry = 238x420 for 1280x720")
    print(f"batch_size = {batch_size}")
    print(f"num_workers = {num_workers}")
    print("supervision = raw_rgb + canonical_pinhole_rays")


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def parse_single_cuda_selector(selector):
    entries = [entry.strip() for entry in selector.split(",")] if selector else []
    entries = [entry for entry in entries if entry]
    if len(entries) != 1:
        raise RuntimeError("--smoke requires exactly one CUDA_VISIBLE_DEVICES entry")
    return selector


def require_single_cuda_device():
    """사용자 physical selector와 process-local logical device를 분리한다."""
    selector = parse_single_cuda_selector(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not torch.cuda.is_available():
        raise RuntimeError("--smoke requires CUDA; CPU fallback is disabled")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"--smoke requires exactly one visible GPU, found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    return torch.device("cuda:0"), selector


def sync_cuda(device):
    torch.cuda.synchronize(device)


def autocast_context(precision):
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 smoke requested but this GPU does not support bf16")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def gpu_metadata(device, selector):
    properties = torch.cuda.get_device_properties(device)
    try:
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()[0].strip()
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
        uuid = None
    return {
        "visible_gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": selector,
        "logical_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": uuid,
        "gpu_total_vram_mib": properties.total_memory / (1024**2),
    }


def smoke(config, config_path, batch_size, num_workers, precision, overwrite):
    device, selector = require_single_cuda_device()
    if batch_size != 1:
        raise ValueError("02-B smoke requires batch_size=1")

    output = PROJECT_ROOT / "runs/02_anycalib_finetune/smoke"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Smoke output exists; pass --overwrite to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output.glob("*.json"):
            path.unlink()
        for path in output.glob("*.txt"):
            path.unlink()

    train, val, _, manifest = make_datasets(config)
    train_sampler = PairBalancedSampler(train, TRAIN_FRAMES_PER_PAIR, seed=config["experiment"]["seed"])
    train_sampler.set_epoch(0)
    train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    model_start = time.perf_counter()
    model, model_info = build_training_anycalib(device=device)
    model_load_seconds = time.perf_counter() - model_start
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
    representative = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    before = representative.detach().clone()

    sync_cuda(device)
    data_start = time.perf_counter()
    train_batch = move_to_device(next(train_iter), device)
    sync_cuda(device)
    data_load_seconds = time.perf_counter() - data_start
    optimizer.zero_grad(set_to_none=True)

    forward_start = time.perf_counter()
    with autocast_context(precision):
        train_pred = model(train_batch)
    sync_cuda(device)
    train_forward_seconds = time.perf_counter() - forward_start

    loss_start = time.perf_counter()
    with autocast_context(precision):
        train_losses, _ = model.loss(train_pred, train_batch)
        train_loss = train_losses["total"].mean()
    sync_cuda(device)
    loss_seconds = time.perf_counter() - loss_start
    if not torch.isfinite(train_loss) or not train_loss.requires_grad:
        raise RuntimeError("Smoke train loss is not finite or does not require gradients")

    backward_start = time.perf_counter()
    train_loss.backward()
    sync_cuda(device)
    backward_seconds = time.perf_counter() - backward_start
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite_gradients = [gradient for gradient in gradients if torch.isfinite(gradient).all()]
    nonzero_gradients = [gradient for gradient in finite_gradients if torch.count_nonzero(gradient).item() > 0]
    if not gradients or len(finite_gradients) != len(gradients) or not nonzero_gradients:
        raise RuntimeError("Smoke gradients are missing, non-finite, or all zero")

    step_start = time.perf_counter()
    optimizer.step()
    sync_cuda(device)
    optimizer_step_seconds = time.perf_counter() - step_start
    parameter_changed = not torch.equal(before, representative.detach())
    if not parameter_changed:
        raise RuntimeError("Smoke optimizer step did not change the representative parameter")

    model.eval()
    sync_cuda(device)
    val_data_start = time.perf_counter()
    val_batch = move_to_device(next(val_iter), device)
    sync_cuda(device)
    data_load_seconds += time.perf_counter() - val_data_start
    validation_start = time.perf_counter()
    with torch.no_grad(), autocast_context(precision):
        val_pred = model(val_batch)
        val_losses, _ = model.loss(val_pred, val_batch)
        val_loss = val_losses["total"].mean()
    sync_cuda(device)
    validation_forward_seconds = time.perf_counter() - validation_start
    if not torch.isfinite(val_loss):
        raise RuntimeError("Smoke validation loss is not finite")
    if "intrinsics" not in val_pred or "success" not in val_pred:
        raise RuntimeError("Smoke validation did not run official pinhole fitting")

    memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }
    total_seconds = time.perf_counter() - total_start
    metadata = {
        "experiment": config["experiment"]["name"], "mode": "training_smoke",
        "repository_commit": git_output("rev-parse", "HEAD"),
        "repository_dirty": bool(git_output("status", "--porcelain")),
        "anycalib_commit": git_output("-C", "AnyCalib", "rev-parse", "HEAD"),
        "config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest),
        "train_split_sha256": sha256_file(resolve_path(config["dataset"]["train_split"])),
        "val_split_sha256": sha256_file(resolve_path(config["dataset"]["val_split"])),
        "model": "official siclib.models.networks.anycalib_net.AnyCalib",
        "initialization": "anycalib_pinhole pretrained",
        "pretrained_weight_sha256": model_info["pretrained_weight_sha256"],
        "dinov2_weight_sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
        "model_parameter_count": model_info["parameter_count"],
        "supervision": "raw_rgb_canonical_pinhole_rays",
        "train_frames": len(train), "train_pairs": len(train.pair_indices), "samples_per_epoch": len(train_sampler),
        "validation_frames": len(val), "validation_pairs": len(val.pair_indices),
        "batch_size": batch_size, "num_workers": num_workers, "precision": precision,
        **gpu_metadata(device, selector), "command_line": " ".join(os.sys.argv),
    }
    tensor_shapes = {key: list(value.shape) for key, value in train_batch.items() if isinstance(value, torch.Tensor)}
    val_shapes = {key: list(value.shape) for key, value in val_batch.items() if isinstance(value, torch.Tensor)}
    smoke_result = {
        "train_batch_shape": tensor_shapes, "ray_shape": list(train_batch["rays"].shape),
        "train_loss_total": float(train_loss.detach().cpu()), "train_loss_finite": True,
        "gradient_parameter_count": len(gradients), "finite_gradient_parameter_count": len(finite_gradients),
        "nonzero_gradient_parameter_count": len(nonzero_gradients), "optimizer_step_success": True,
        "parameter_changed": parameter_changed, "validation_batch_shape": val_shapes,
        "validation_loss_total": float(val_loss.detach().cpu()), "validation_loss_finite": True,
        "validation_fit_success_count": int(val_pred["success"].sum().item()),
        "validation_batch_size": len(val_batch["cam_id"]),
    }
    runtime = {
        "model_load_seconds": model_load_seconds, "data_load_seconds": data_load_seconds,
        "train_forward_seconds": train_forward_seconds, "loss_seconds": loss_seconds,
        "backward_seconds": backward_seconds, "optimizer_step_seconds": optimizer_step_seconds,
        "validation_forward_seconds": validation_forward_seconds, "total_seconds": total_seconds,
        **memory, "gpu_total_vram_mib": metadata["gpu_total_vram_mib"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "smoke.json").write_text(json.dumps(smoke_result, indent=2) + "\n")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (output / "summary.txt").write_text(
        "\n".join(
            [
                "02-B Training Smoke", "Model: official AnyCalib training architecture",
                "Initialization: anycalib_pinhole pretrained", "Supervision: raw RGB + canonical pinhole rays",
                f"Train batch: {tensor_shapes}", f"Validation batch: {val_shapes}",
                "Loss finite: YES", "Backward: PASS", f"Finite gradients: {len(finite_gradients)} / {len(gradients)}",
                "Optimizer step: PASS", "Parameter changed: YES", "Validation pinhole fitting: PASS",
                f"Peak allocated: {runtime['peak_allocated_mib']:.2f} MiB", f"Peak reserved: {runtime['peak_reserved_mib']:.2f} MiB",
                f"GPU: {metadata['gpu_name']}", "No full training was run.", "No Test data was used.",
            ]
        ) + "\n"
    )
    print((output / "summary.txt").read_text(), end="")


def parse_args():
    parser = argparse.ArgumentParser(description="GigaHands 02-B smoke entry point.")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.smoke:
        parser.error("Full fine-tuning is not enabled yet. Use --dry-run or --smoke.")
    return args


def main():
    args = parse_args()
    config, config_path = load_experiment(args.config)
    if args.dry_run:
        dry_run(config, args.batch_size, args.num_workers)
    else:
        smoke(config, config_path, args.batch_size, args.num_workers, args.precision, args.overwrite)


if __name__ == "__main__":
    main()
