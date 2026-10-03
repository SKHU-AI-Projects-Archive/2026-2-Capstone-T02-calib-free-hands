"""GigaHands 02-B dry-run과 one-step smoke 경로를 제공한다."""

import argparse
import csv
from contextlib import nullcontext
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

from utils.metrics import camera_errors
from utils.results import write_frame_predictions, write_summaries

REPO_ROOT = Path(__file__).resolve().parent.parent
HANDCALIB_ROOT = REPO_ROOT / "HandCalib"
ANYCALIB_ROOT = HANDCALIB_ROOT / "AnyCalib"
PROJECT_ROOT = HANDCALIB_ROOT
sys.path.insert(0, str(ANYCALIB_ROOT))

from dataloaders.gigahands import GigaHandsRayDataset, PairBalancedSampler
from models.anycalib import build_training_anycalib


EXPECTED_TRAIN_FRAMES = 44254
EXPECTED_TRAIN_PAIRS = 147
EXPECTED_VAL_FRAMES = 18652
EXPECTED_VAL_PAIRS = 49
TRAIN_FRAMES_PER_PAIR = 174
BENCHMARK_WARMUP_STEPS = 3
BENCHMARK_MEASURE_STEPS = 10
VALIDATION_STEP0_OUTPUT = PROJECT_ROOT / "runs/02_anycalib_finetune/validation_step0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args, cwd=REPO_ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path, HANDCALIB_ROOT / path):
        if candidate.exists():
            return candidate
    return HANDCALIB_ROOT / path


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


def make_train_dataset(config):
    root = resolve_path(config["dataset"]["root"])
    manifest = PROJECT_ROOT / "data/manifests/gigahands.csv"
    train = GigaHandsRayDataset(root, resolve_path(config["dataset"]["train_split"]), manifest)
    if len(train) != EXPECTED_TRAIN_FRAMES or len(train.pair_indices) != EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(f"Unexpected Train dataset count: {len(train)=}, {len(train.pair_indices)=}")
    return train, root, manifest


def make_validation_dataset(config):
    root = resolve_path(config["dataset"]["root"])
    manifest = PROJECT_ROOT / "data/manifests/gigahands.csv"
    val = GigaHandsRayDataset(root, resolve_path(config["dataset"]["val_split"]), manifest)
    if len(val) != EXPECTED_VAL_FRAMES or len(val.pair_indices) != EXPECTED_VAL_PAIRS:
        raise RuntimeError(f"Unexpected Validation dataset count: {len(val)=}, {len(val.pair_indices)=}")
    participants = {sample["row"]["participant"] for sample in val.samples}
    if participants != {"p36"}:
        raise RuntimeError(f"Step-0 Validation requires participant p36, found {sorted(participants)}")
    return val, resolve_path(config["dataset"]["val_split"]), manifest


def make_datasets(config):
    train, root, manifest = make_train_dataset(config)
    val = GigaHandsRayDataset(root, resolve_path(config["dataset"]["val_split"]), manifest)
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
        raise RuntimeError("execution requires exactly one CUDA_VISIBLE_DEVICES entry")
    return entries[0]


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
            raise RuntimeError("bf16 execution requested but this GPU does not support bf16")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def selected_gpu_query(selector):
    command = [
        "nvidia-smi", "-i", selector,
        "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not query selected GPU metadata for selector {selector!r}") from exc
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi metadata for selector {selector!r}: {output!r}")
    index, uuid, pci_bus_id, name, memory_total = (value.strip() for value in rows[0])
    try:
        physical_index = int(index)
        memory_total_mib = float(memory_total)
    except ValueError as exc:
        raise RuntimeError(f"Invalid nvidia-smi metadata for selector {selector!r}: {rows[0]!r}") from exc
    return {
        "gpu_physical_index": physical_index,
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": pci_bus_id,
        "gpu_name": name,
        "gpu_total_vram_mib": memory_total_mib,
    }


def gpu_metadata(device, selector):
    properties = torch.cuda.get_device_properties(device)
    selected = selected_gpu_query(selector)
    torch_name = torch.cuda.get_device_name(device)
    if selected["gpu_name"].lower() not in torch_name.lower() and torch_name.lower() not in selected["gpu_name"].lower():
        raise RuntimeError(f"Selected GPU mismatch: nvidia-smi={selected['gpu_name']!r}, torch={torch_name!r}")
    torch_memory_mib = properties.total_memory / (1024**2)
    if abs(selected["gpu_total_vram_mib"] - torch_memory_mib) > max(1024.0, torch_memory_mib * 0.10):
        raise RuntimeError(
            f"Selected GPU memory mismatch: nvidia-smi={selected['gpu_total_vram_mib']}, torch={torch_memory_mib}"
        )
    return {
        "visible_gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": selector,
        "logical_device": str(device),
        **selected,
        "gpu_total_vram_mib": torch_memory_mib,
    }


def write_smoke_outputs(output, metadata, smoke_result, runtime, summary, overwrite):
    serialized = {
        "metadata.json": json.dumps(metadata, indent=2) + "\n",
        "smoke.json": json.dumps(smoke_result, indent=2) + "\n",
        "runtime.json": json.dumps(runtime, indent=2) + "\n",
        "summary.txt": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output.glob("*.json"):
            path.unlink()
        for path in output.glob("*.txt"):
            path.unlink()
    for name, contents in serialized.items():
        (output / name).write_text(contents)


def percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def timing_summary(rows, key):
    values = [row[key] for row in rows]
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "mean": sum(values) / len(values),
        "median": median,
        "p95": percentile95(values),
    }


def write_benchmark_outputs(output, metadata, runtime, benchmark_result, rows, summary, overwrite):
    fieldnames = [
        "step", "batch_size", "loader_wait_seconds", "host_to_device_seconds",
        "forward_seconds", "loss_seconds", "backward_seconds",
        "optimizer_step_seconds", "step_total_seconds",
    ]
    csv_lines = []
    with io.StringIO() as buffer:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        csv_lines.append(buffer.getvalue())
    serialized = {
        "metadata.json": json.dumps(metadata, indent=2) + "\n",
        "runtime.json": json.dumps(runtime, indent=2) + "\n",
        "benchmark.json": json.dumps(benchmark_result, indent=2) + "\n",
        "steps.csv": csv_lines[0],
        "summary.txt": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output.iterdir():
            if path.is_file():
                path.unlink()
    for name, contents in serialized.items():
        (output / name).write_text(contents)


def validate_benchmark_args(batch_size, num_workers, precision):
    if batch_size <= 0:
        raise ValueError("benchmark batch_size must be a positive integer")
    if num_workers < 0:
        raise ValueError("benchmark num_workers must be non-negative")
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported benchmark precision: {precision}")


def benchmark(config, config_path, batch_size, num_workers, precision, overwrite):
    validate_benchmark_args(batch_size, num_workers, precision)
    device, selector = require_single_cuda_device()
    output = PROJECT_ROOT / "runs/02_anycalib_finetune/benchmark_train" / f"bs{batch_size}_nw{num_workers}_{precision}"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Benchmark output exists; pass --overwrite to replace it: {output}")

    train, _, manifest = make_train_dataset(config)
    train_sampler = PairBalancedSampler(train, TRAIN_FRAMES_PER_PAIR, seed=config["experiment"]["seed"])
    train_sampler.set_epoch(0)
    loader_kwargs = {
        "batch_size": batch_size,
        "sampler": train_sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train, **loader_kwargs)
    train_iter = iter(train_loader)
    model_info = None
    rows = []
    try:
        model_start = time.perf_counter()
        model, model_info = build_training_anycalib(device=device)
        model_load_seconds = time.perf_counter() - model_start
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)

        for _ in range(BENCHMARK_WARMUP_STEPS):
            batch = move_to_device(next(train_iter), device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(precision):
                prediction = model(batch)
                losses, _ = model.loss(prediction, batch)
                loss = losses["total"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Benchmark warmup loss is not finite")
            loss.backward()
            if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                raise RuntimeError("Benchmark warmup gradient is not finite")
            optimizer.step()
            sync_cuda(device)

        torch.cuda.reset_peak_memory_stats(device)
        allocated_before_measure_mib = torch.cuda.memory_allocated(device) / (1024**2)
        reserved_before_measure_mib = torch.cuda.memory_reserved(device) / (1024**2)
        for step in range(1, BENCHMARK_MEASURE_STEPS + 1):
            step_start = time.perf_counter()
            loader_start = time.perf_counter()
            batch = next(train_iter)
            loader_wait_seconds = time.perf_counter() - loader_start
            host_start = time.perf_counter()
            batch = move_to_device(batch, device)
            sync_cuda(device)
            host_to_device_seconds = time.perf_counter() - host_start
            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            with autocast_context(precision):
                prediction = model(batch)
            sync_cuda(device)
            forward_seconds = time.perf_counter() - forward_start
            loss_start = time.perf_counter()
            with autocast_context(precision):
                losses, _ = model.loss(prediction, batch)
                loss = losses["total"].mean()
            sync_cuda(device)
            loss_seconds = time.perf_counter() - loss_start
            if not torch.isfinite(loss):
                raise RuntimeError("Benchmark loss is not finite")
            backward_start = time.perf_counter()
            loss.backward()
            sync_cuda(device)
            backward_seconds = time.perf_counter() - backward_start
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError("Benchmark gradient is not finite or missing")
            optimizer_start = time.perf_counter()
            optimizer.step()
            sync_cuda(device)
            optimizer_step_seconds = time.perf_counter() - optimizer_start
            rows.append({
                "step": step, "batch_size": len(batch["image"]),
                "loader_wait_seconds": loader_wait_seconds,
                "host_to_device_seconds": host_to_device_seconds,
                "forward_seconds": forward_seconds, "loss_seconds": loss_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_step_seconds": optimizer_step_seconds,
                "step_total_seconds": time.perf_counter() - step_start,
            })
        memory = {
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
            "allocated_before_measure_mib": allocated_before_measure_mib,
            "reserved_before_measure_mib": reserved_before_measure_mib,
        }
        metric_keys = ["loader_wait_seconds", "host_to_device_seconds", "forward_seconds", "loss_seconds", "backward_seconds", "optimizer_step_seconds", "step_total_seconds"]
        timings = {key: timing_summary(rows, key) for key in metric_keys}
        measured_total_seconds = sum(row["step_total_seconds"] for row in rows)
        gpu = gpu_metadata(device, selector)
        metadata = {
            "experiment": config["experiment"]["name"], "mode": "training_benchmark",
            "repository_commit": git_output("rev-parse", "HEAD"),
            "repository_dirty": bool(git_output("status", "--porcelain")),
            "anycalib_commit": git_output("rev-parse", "HEAD", cwd=ANYCALIB_ROOT),
            "config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest),
            "train_split_sha256": sha256_file(resolve_path(config["dataset"]["train_split"])),
            "model": "official siclib.models.networks.anycalib_net.AnyCalib",
            "initialization": "anycalib_pinhole pretrained", "pretrained_weight_sha256": model_info["pretrained_weight_sha256"],
            "dinov2_weight_sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
            "model_parameter_count": model_info["parameter_count"], "supervision": "raw_rgb_canonical_pinhole_rays",
            "train_frames": len(train), "train_pairs": len(train.pair_indices), "samples_per_epoch": len(train_sampler),
            "batch_size": batch_size, "num_workers": num_workers, "precision": precision,
            "warmup_steps": BENCHMARK_WARMUP_STEPS, "measure_steps": BENCHMARK_MEASURE_STEPS,
            "benchmark_only_optimizer": {"name": "AdamW", "lr": 1e-6, "weight_decay": 0.0},
            **gpu, "command_line": " ".join(os.sys.argv),
        }
        benchmark_result = {
            "status": "pass", "samples_per_second": batch_size * BENCHMARK_MEASURE_STEPS / measured_total_seconds,
            "steps_per_second": BENCHMARK_MEASURE_STEPS / measured_total_seconds, "timings": timings, **memory,
            "gpu_memory_fraction_reserved": memory["peak_reserved_mib"] / gpu["gpu_total_vram_mib"],
        }
        runtime = {"status": "pass", "model_load_seconds": model_load_seconds, "measured_total_seconds": measured_total_seconds, **memory}
        summary = "\n".join([
            "02-C Training Benchmark", "Status: PASS", f"Batch size: {batch_size}",
            f"Workers: {num_workers}", f"Precision: {precision}",
            f"Warmup steps: {BENCHMARK_WARMUP_STEPS}", f"Measured steps: {BENCHMARK_MEASURE_STEPS}",
            f"Samples/s: {benchmark_result['samples_per_second']:.4f}",
            f"Step median: {timings['step_total_seconds']['median']:.6f}",
            f"Loader wait median: {timings['loader_wait_seconds']['median']:.6f}",
            f"Forward median: {timings['forward_seconds']['median']:.6f}",
            f"Backward median: {timings['backward_seconds']['median']:.6f}",
            f"Peak allocated: {memory['peak_allocated_mib']:.2f} MiB",
            f"Peak reserved: {memory['peak_reserved_mib']:.2f} MiB",
            f"VRAM reserved fraction: {benchmark_result['gpu_memory_fraction_reserved']:.4f}",
            f"GPU: {gpu['gpu_name']}", "No full training was run.", "No Validation/Test evaluation was run.",
        ]) + "\n"
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "fail"
        gpu = gpu_metadata(device, selector)
        metadata = {
            "experiment": config["experiment"]["name"], "mode": "training_benchmark", "status": status,
            "repository_commit": git_output("rev-parse", "HEAD"),
            "anycalib_commit": git_output("rev-parse", "HEAD", cwd=ANYCALIB_ROOT),
            "config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest),
            "train_split_sha256": sha256_file(resolve_path(config["dataset"]["train_split"])),
            "batch_size": batch_size, "num_workers": num_workers, "precision": precision,
            "warmup_steps": BENCHMARK_WARMUP_STEPS, "measure_steps": BENCHMARK_MEASURE_STEPS,
            **gpu, "command_line": " ".join(os.sys.argv),
        }
        benchmark_result = {"status": status, "error": str(exc)}
        runtime = {"status": status}
        summary = f"02-C Training Benchmark\nStatus: {status.upper()}\nError: {exc}\n"
    write_benchmark_outputs(output, metadata, runtime, benchmark_result, rows, summary, overwrite)
    print((output / "summary.txt").read_text(), end="")


def _validation_prediction(prediction, batch_size, pred_size):
    intrinsics = prediction["intrinsics"]
    if isinstance(intrinsics, (list, tuple)):
        intrinsics = torch.stack([torch.as_tensor(value) for value in intrinsics])
    intrinsics = torch.as_tensor(intrinsics).detach().cpu()
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0)
    success = torch.as_tensor(prediction["success"]).reshape(-1).detach().cpu().bool()
    if tuple(intrinsics.shape) != (batch_size, 4) or tuple(success.shape) != (batch_size,):
        raise RuntimeError("Validation prediction shape does not match the batch")
    return intrinsics, success, pred_size


def _batch_meta_value(values, index):
    value = values[index]
    return value.item() if isinstance(value, torch.Tensor) and value.ndim == 0 else value


def _step0_metric_aliases(metrics):
    pair = metrics["pair_level"]
    frame = metrics["frame_level"]
    return {
        "val_pair_max_rel_f_mean": pair["pair_max_rel_f_error_mean"],
        "val_pair_max_rel_f_median": pair["pair_max_rel_f_error_median"],
        "val_pair_rel_fx_mean": pair["pair_rel_fx_error_mean"],
        "val_pair_rel_fx_median": pair["pair_rel_fx_error_median"],
        "val_pair_rel_fy_mean": pair["pair_rel_fy_error_mean"],
        "val_pair_rel_fy_median": pair["pair_rel_fy_error_median"],
        "val_pair_max_rel_c_mean": pair["pair_max_rel_c_error_mean"],
        "val_pair_max_rel_c_median": pair["pair_max_rel_c_error_median"],
        "val_pair_within_5pct_count": pair["pair_focal_within_5pct_count"],
        "val_pair_within_5pct_rate": pair["pair_focal_within_5pct_rate"],
        "val_frame_total": frame["total_frames"],
        "val_frame_successful": frame["successful_frames"],
        "val_frame_failed": frame["total_frames"] - frame["successful_frames"],
        "val_frame_success_rate": frame["success_rate"],
        "val_pair_count": pair["pair_count"],
        "val_valid_pair_count": pair["valid_pair_count"],
    }


def _write_step0_summary(output, metrics, runtime, metadata):
    aliases = _step0_metric_aliases(metrics)
    lines = [
        "02-D.0 Validation step 0",
        "Mode: pre-finetuning reference",
        "Model: official training AnyCalib with anycalib_pinhole pretrained weights",
        "Validation precision: fp32",
        "Split: Validation / p36",
        f"Frames: {aliases['val_frame_successful']} successful / {aliases['val_frame_total']} total",
        f"Pairs: {aliases['val_valid_pair_count']} valid / {aliases['val_pair_count']} total",
        f"Primary val_pair_max_rel_f_mean: {aliases['val_pair_max_rel_f_mean']}",
        f"val_pair_max_rel_f_median: {aliases['val_pair_max_rel_f_median']}",
        f"val_pair_rel_fx_mean/median: {aliases['val_pair_rel_fx_mean']} / {aliases['val_pair_rel_fx_median']}",
        f"val_pair_rel_fy_mean/median: {aliases['val_pair_rel_fy_mean']} / {aliases['val_pair_rel_fy_median']}",
        f"val_pair_max_rel_c_mean/median: {aliases['val_pair_max_rel_c_mean']} / {aliases['val_pair_max_rel_c_median']}",
        f"Within 5% focal: {aliases['val_pair_within_5pct_count']} / {aliases['val_valid_pair_count']} / {aliases['val_pair_within_5pct_rate']}",
        f"Model load seconds: {runtime['model_load_seconds']:.3f}",
        f"Validation seconds: {runtime['validation_seconds']:.3f}",
        f"Frames/sec: {runtime['frames_per_second']:.3f}",
        f"Peak allocated: {runtime['peak_allocated_mib']:.2f} MiB",
        f"Peak reserved: {runtime['peak_reserved_mib']:.2f} MiB",
        f"Repository commit: {metadata['repository_commit']}",
        f"AnyCalib commit: {metadata['anycalib_commit']}",
        "No fine-tuning was run.",
        "Test data was not used.",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")


def validate_step0(config, config_path, batch_size=4, num_workers=4, overwrite=False):
    if batch_size != 4 or num_workers != 4:
        raise ValueError("step-0 Validation is fixed to batch_size=4 and num_workers=4")
    device, selector = require_single_cuda_device()
    if VALIDATION_STEP0_OUTPUT.exists() and not overwrite:
        raise FileExistsError(f"Validation output exists; pass --overwrite to replace it: {VALIDATION_STEP0_OUTPUT}")
    val, split_path, manifest = make_validation_dataset(config)
    loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    model_start = time.perf_counter()
    model, model_info = build_training_anycalib(device=device)
    model_load_seconds = time.perf_counter() - model_start
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    validation_start = time.perf_counter()
    rows = []
    success_count = 0
    iterator = iter(loader)
    for batch in iterator:
        batch_size_actual = len(batch["image"])
        batch = move_to_device(batch, device)
        sync_cuda(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                prediction = model(batch)
        sync_cuda(device)
        pred_size = (int(batch["image"].shape[-2]), int(batch["image"].shape[-1]))
        pred_intrinsics, success, pred_size = _validation_prediction(prediction, batch_size_actual, pred_size)
        pred_height, pred_width = pred_size
        for offset in range(batch_size_actual):
            meta = {key: _batch_meta_value(values, offset) for key, values in batch["meta"].items()}
            gt = batch["intrinsics"][offset].detach().cpu()
            frame_success = bool(success[offset].item())
            row = {
                "participant": meta["participant"], "sequence": meta["sequence"], "camera": meta["camera"],
                "camera_key": meta["camera_key"], "video_path": "", "video_name": meta["video_name"],
                "frame_index": int(meta["frame_index"]), "width": int(meta["target_width"]), "height": int(meta["target_height"]),
                "gt_fx": float(gt[0]), "gt_fy": float(gt[1]), "gt_cx": float(gt[2]), "gt_cy": float(gt[3]),
                "pred_width": pred_width, "pred_height": pred_height, "success": frame_success,
            }
            if frame_success:
                pred = pred_intrinsics[offset, :4]
                row.update({f"pred_{key}": float(pred[index]) for index, key in enumerate(("fx", "fy", "cx", "cy"))})
                row.update({key: float(value) for key, value in camera_errors(pred, gt, meta["target_width"], meta["target_height"]).items()})
                success_count += 1
            rows.append(row)
    sync_cuda(device)
    validation_seconds = time.perf_counter() - validation_start
    gpu = gpu_metadata(device, selector)
    metrics_output = VALIDATION_STEP0_OUTPUT
    if metrics_output.exists():
        shutil.rmtree(metrics_output)
    metrics_output.mkdir(parents=True, exist_ok=True)
    raw_path = metrics_output / "frame_predictions.csv.gz"
    write_frame_predictions(raw_path, rows)
    metrics = write_summaries(raw_path, metrics_output)
    metrics.update(_step0_metric_aliases(metrics))
    metrics["checkpoint_selection"] = {"primary_metric": "val_pair_max_rel_f_mean", "direction": "minimize"}
    (metrics_output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    shutil.copyfile(config_path, metrics_output / "config.yaml")
    runtime = {
        "model_load_seconds": model_load_seconds, "validation_seconds": validation_seconds,
        "frames_per_second": len(rows) / validation_seconds if validation_seconds else 0,
        "frames_processed": len(rows), "successful_frames": success_count,
        "failed_frames": len(rows) - success_count, "success_rate": success_count / len(rows),
        "batch_size": batch_size, "num_workers": num_workers,
        "batch_count": len(loader), "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }
    metadata = {
        "experiment": config["experiment"]["name"], "mode": "validation_step0",
        "repository_commit": git_output("rev-parse", "HEAD"),
        "repository_dirty": bool(git_output("status", "--porcelain")),
        "anycalib_commit": git_output("rev-parse", "HEAD", cwd=ANYCALIB_ROOT),
        "config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest),
        "validation_split_sha256": sha256_file(split_path), "dataset_split": "validation",
        "validation_participant": "p36", "validation_frames": len(rows), "validation_pairs": len(val.pair_indices),
        "model": "official siclib.models.networks.anycalib_net.AnyCalib",
        "initialization": "anycalib_pinhole pretrained", "pretrained_weight_sha256": model_info["pretrained_weight_sha256"],
        "dinov2_weight_sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
        "model_parameter_count": model_info["parameter_count"], "input_width": 1280, "input_height": 720,
        "pred_width": 420, "pred_height": 238, "batch_size": batch_size, "num_workers": num_workers,
        "training_precision": "bf16", "validation_precision": "fp32", "cuda_visible_devices": selector,
        **gpu, "command_line": " ".join(os.sys.argv), "output_directory": str(metrics_output.relative_to(PROJECT_ROOT)),
    }
    (metrics_output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (metrics_output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    _write_step0_summary(metrics_output, metrics, runtime, metadata)
    print((metrics_output / "summary.txt").read_text(), end="")


def smoke(config, config_path, batch_size, num_workers, precision, overwrite):
    device, selector = require_single_cuda_device()
    if batch_size != 1:
        raise ValueError("02-B smoke requires batch_size=1")

    output = PROJECT_ROOT / "runs/02_anycalib_finetune/smoke"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Smoke output exists; pass --overwrite to replace it: {output}")

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
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=False):
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
        "anycalib_commit": git_output("rev-parse", "HEAD", cwd=ANYCALIB_ROOT),
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
        "training_precision": precision, "validation_precision": "fp32",
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
    summary = "\n".join(
        [
            "02-B Training Smoke", "Model: official AnyCalib training architecture",
            "Initialization: anycalib_pinhole pretrained", "Supervision: raw RGB + canonical pinhole rays",
            f"Train batch: {tensor_shapes}", f"Validation batch: {val_shapes}",
            "Loss finite: YES", "Validation precision: fp32", "Backward: PASS", f"Finite gradients: {len(finite_gradients)} / {len(gradients)}",
            "Optimizer step: PASS", "Parameter changed: YES", "Validation pinhole fitting: PASS",
            f"Peak allocated: {runtime['peak_allocated_mib']:.2f} MiB", f"Peak reserved: {runtime['peak_reserved_mib']:.2f} MiB",
            f"GPU: {metadata['gpu_name']}", "No full training was run.", "No Test data was used.",
        ]
    ) + "\n"
    write_smoke_outputs(output, metadata, smoke_result, runtime, summary, overwrite)
    print((output / "summary.txt").read_text(), end="")


def parse_args():
    parser = argparse.ArgumentParser(description="GigaHands 02-B smoke and 02-C benchmark entry point.")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--validate-step0", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.smoke and not args.benchmark and not args.validate_step0:
        parser.error("Full fine-tuning is not enabled yet. Use --dry-run, --smoke, --benchmark, or --validate-step0.")
    return args


def main():
    args = parse_args()
    config, config_path = load_experiment(args.config)
    if args.dry_run:
        dry_run(config, args.batch_size, args.num_workers)
    elif args.benchmark:
        benchmark(config, config_path, args.batch_size, args.num_workers, args.precision, args.overwrite)
    elif args.validate_step0:
        validate_step0(config, config_path, overwrite=args.overwrite)
    else:
        smoke(config, config_path, args.batch_size, args.num_workers, args.precision, args.overwrite)


if __name__ == "__main__":
    main()
