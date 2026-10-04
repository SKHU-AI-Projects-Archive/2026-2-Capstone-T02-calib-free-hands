"""AnyCalib smoke test와 Validation clip benchmark 실행 진입점."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloaders.gigahands import GigaHandsDataset
from models.anycalib import AnyCalibAdapter
from utils.config import load_config
from utils.metrics import camera_errors
from utils.results import read_frame_predictions, write_frame_predictions, write_summaries
from utils.runtime import (
    TelemetrySampler,
    finite_max,
    finite_mean,
    format_duration,
    percentile,
    physical_gpu_token,
    query_nvidia_smi,
    sha256_file,
    utc_now,
    write_gzip_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
EXPECTED_VALIDATION_SIZE = 18652
EXPECTED_TEST_FRAMES = 7667
SMOKE_OUTPUT = PROJECT_ROOT / "runs/01_anycalib_pretrained/smoke_val"
BENCHMARK_ROOT = PROJECT_ROOT / "runs/01_anycalib_pretrained/benchmark_val"
TEST_OUTPUT = PROJECT_ROOT / "runs/01_anycalib_pretrained/test"
EXPECTED_TEST_SIZE = 7667
EXPECTED_TEST_PARTICIPANT = "p52"
EXPECTED_TEST_SEQUENCE = "p52-instrument-0034"
EXPECTED_TEST_PAIRS = 49
TELEMETRY_FIELDS = (
    "elapsed_seconds", "frames_done", "frames_total", "success_count", "telemetry_available",
    "gpu_util_percent", "gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_temperature_c",
    "gpu_power_w", "torch_allocated_mib", "torch_reserved_mib",
)
BATCH_FIELDS = (
    "batch_index", "first_dataset_index", "last_dataset_index", "batch_size", "loader_wait_seconds",
    "inference_seconds", "postprocess_seconds", "batch_total_seconds", "frames_per_second",
    "torch_allocated_mib", "torch_reserved_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AnyCalib smoke or Validation clip benchmark.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="Run one validation frame on one visible GPU.")
    modes.add_argument("--benchmark", action="store_true", help="Benchmark the first complete Validation RGB clip.")
    modes.add_argument("--test", action="store_true", help="Run the locked final Test evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the selected mode without a model or inference.")
    parser.add_argument("--batch-size", type=int, help="Required for --benchmark.")
    parser.add_argument("--num-workers", type=int, help="Required for --benchmark.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing benchmark output directory.")
    return parser.parse_args()


def _config_path(value):
    candidate = Path(value)
    for option in (candidate, PROJECT_ROOT / candidate, REPO_ROOT / candidate):
        if option.exists():
            return option.resolve()
    raise FileNotFoundError(f"Config does not exist: {value}")


def _path_from_config(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dataset(config, split_name="val"):
    dataset_config = config["dataset"]
    split_path = _path_from_config(dataset_config[f"{split_name}_split"])
    dataset = GigaHandsDataset(
        root=_path_from_config(dataset_config["root"]),
        split_file=split_path,
        manifest_path=PROJECT_ROOT / "data/manifests/gigahands.csv",
    )
    expected_size = EXPECTED_TEST_SIZE if split_name == "test" else EXPECTED_VALIDATION_SIZE
    if len(dataset) != expected_size:
        raise RuntimeError(f"Expected {expected_size} {split_name} frames, found {len(dataset)}")
    if split_name == "test":
        participants = {sample["row"]["participant"] for sample in dataset.samples}
        sequences = {sample["row"]["sequence"] for sample in dataset.samples}
        pairs = {sample["row"]["camera_key"] for sample in dataset.samples}
        if participants != {EXPECTED_TEST_PARTICIPANT}:
            raise RuntimeError(f"Unexpected Test participants: {sorted(participants)}")
        if sequences != {EXPECTED_TEST_SEQUENCE}:
            raise RuntimeError(f"Unexpected Test sequences: {sorted(sequences)}")
        if len(pairs) != EXPECTED_TEST_PAIRS:
            raise RuntimeError(f"Expected {EXPECTED_TEST_PAIRS} Test pairs, found {len(pairs)}")
    return dataset, split_path


def _first_clip(dataset):
    first = dataset.samples[0]["row"]
    indices = [
        index for index, sample in enumerate(dataset.samples)
        if sample["row"]["camera_key"] == first["camera_key"]
        and sample["row"]["video_name"] == first["video_name"]
    ]
    return Subset(dataset, indices), indices, first


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _normalize_batched_prediction(prediction, batch_size):
    """Normalize official AnyCalib output while preserving batch order."""
    intrinsics = prediction["intrinsics"]
    if isinstance(intrinsics, (list, tuple)):
        pred_intrinsics = torch.stack([torch.as_tensor(value) for value in intrinsics])
    else:
        pred_intrinsics = torch.as_tensor(intrinsics)
    if pred_intrinsics.ndim == 1:
        pred_intrinsics = pred_intrinsics.unsqueeze(0)
    pred_intrinsics = pred_intrinsics.detach().cpu()
    success = torch.as_tensor(prediction["success"]).reshape(-1).detach().cpu().to(torch.bool)
    if pred_intrinsics.shape != (batch_size, 4):
        raise RuntimeError(f"Expected batched pinhole intrinsics shape {(batch_size, 4)}, got {tuple(pred_intrinsics.shape)}")
    if success.shape != (batch_size,):
        raise RuntimeError(f"Expected success shape {(batch_size,)}, got {tuple(success.shape)}")
    pred_size = tuple(int(value) for value in prediction["pred_size"])
    if len(pred_size) != 2:
        raise RuntimeError(f"Expected pred_size=(height, width), got {pred_size}")
    return pred_intrinsics, success, pred_size


def _validate_batch(batch):
    batch_size = int(batch["image"].shape[0])
    if len(batch["intrinsics"]) != batch_size:
        raise RuntimeError("Batch image and ground-truth intrinsics lengths differ.")
    for key, values in batch["meta"].items():
        if len(values) != batch_size:
            raise RuntimeError(f"Batch image and metadata field '{key}' lengths differ.")
    return batch_size


def _git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty():
    try:
        return bool(subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_visible_devices():
    return __import__("os").environ.get("CUDA_VISIBLE_DEVICES")


def _require_benchmark_args(args):
    if args.batch_size is None or args.num_workers is None:
        raise ValueError("--benchmark requires both --batch-size and --num-workers.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")


def _require_single_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Benchmark requires CUDA with exactly one visible GPU; no CPU fallback is available.")
    return physical_gpu_token()


def _smoke(config, config_path, overwrite):
    dataset, split_path = _dataset(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke evaluation requires CUDA with exactly one visible GPU; no CPU fallback is available.")
    if SMOKE_OUTPUT.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {SMOKE_OUTPUT}")
    if SMOKE_OUTPUT.exists():
        shutil.rmtree(SMOKE_OUTPUT)
    SMOKE_OUTPUT.mkdir(parents=True)
    sample = dataset[0]
    adapter = AnyCalibAdapter(
        model_id=config["model"]["model_id"], cam_id=config["model"]["cam_id"],
        checkpoint=config["model"].get("checkpoint"), device="cuda:0",
    ).build()
    prediction = adapter.predict(sample["image"])
    intrinsics = torch.as_tensor(prediction["intrinsics"]).detach().cpu().flatten()
    success = bool(_scalar(prediction["success"]))
    pred_height, pred_width = (int(value) for value in prediction["pred_size"])
    row = {**sample["meta"], **{f"gt_{key}": float(value) for key, value in zip(("fx", "fy", "cx", "cy"), sample["intrinsics"])},
           "pred_width": pred_width, "pred_height": pred_height, "success": success}
    if success:
        row.update({f"pred_{key}": float(intrinsics[index]) for index, key in enumerate(("fx", "fy", "cx", "cy"))})
        row.update({key: float(value) for key, value in camera_errors(intrinsics[:4], sample["intrinsics"], sample["meta"]["width"], sample["meta"]["height"]).items()})
    raw_path = SMOKE_OUTPUT / "frame_predictions.csv.gz"
    write_frame_predictions(raw_path, [row])
    write_summaries(raw_path, SMOKE_OUTPUT)
    shutil.copyfile(config_path, SMOKE_OUTPUT / "config.yaml")
    (SMOKE_OUTPUT / "metadata.json").write_text(json.dumps({
        "experiment": config["experiment"]["name"], "mode": "smoke_val",
        "model_id": config["model"]["model_id"], "cam_id": config["model"]["cam_id"],
        "repo_head": _git_head(PROJECT_ROOT), "anycalib_head": _git_head(PROJECT_ROOT / "AnyCalib"),
        "dataset_split": str(split_path.relative_to(PROJECT_ROOT)), "dataset_size": len(dataset),
        "smoke_index": 0, "input_width": sample["meta"]["width"], "input_height": sample["meta"]["height"],
        "pred_width": pred_width, "pred_height": pred_height, "success": success,
    }, indent=2) + "\n")
    print(f"smoke_output={SMOKE_OUTPUT}")
    print(f"success={success} pred_size={(pred_height, pred_width)}")


def _benchmark_metadata(config, config_path, split_path, first, clip, batch_size, num_workers, final_test, token, output_dir, adapter, weight_path, weight_present_before, started):
    gpu = query_nvidia_smi(token) or {}
    input_width = int(first["width"])
    input_height = int(first["height"])
    metadata = {
        "experiment": config["experiment"]["name"], "mode": "final_test" if final_test else "benchmark_val",
        "model_id": config["model"]["model_id"], "cam_id": config["model"]["cam_id"],
        "repository_commit": _git_head(REPO_ROOT), "repository_dirty": _git_dirty(),
        "anycalib_commit": _git_head(PROJECT_ROOT / "AnyCalib"),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(PROJECT_ROOT / "data/manifests/gigahands.csv"),
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "torchvision_version": _version("torchvision"), "numpy_version": _version("numpy"),
        "opencv_version": cv2.__version__, "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "visible_gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": _cuda_visible_devices(), "gpu_name": gpu.get("gpu_name"),
        "gpu_uuid": gpu.get("gpu_uuid"), "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_vram_mib": gpu.get("gpu_memory_total_mib"), "nvidia_driver_version": gpu.get("nvidia_driver_version"),
        "torch_num_threads": torch.get_num_threads(), "cpu_count": __import__("os").cpu_count(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "model_parameter_count": sum(parameter.numel() for parameter in adapter.model.parameters()),
        "batch_size": batch_size, "num_workers": num_workers,
        "dataloader_prefetch_factor": None if num_workers == 0 else 2,
        "input_width": input_width, "input_height": input_height, "pred_width": None, "pred_height": None,
        "weight_cache_path": f"torch.hub.get_dir()/anycalib/{config['model']['model_id']}.pt",
        "weight_cache_present_before_run": weight_present_before,
        "pretrained_weight_sha256": sha256_file(weight_path) if weight_path.exists() else None,
        "command_line": sys.argv, "started_at_utc": started,
        "finished_at_utc": None, "output_directory": str(output_dir.relative_to(PROJECT_ROOT)),
    }
    if final_test:
        metadata.update({
            "dataset_split": "test", "test_participant": first["participant"],
            "test_sequence": first["sequence"], "test_frames": len(clip),
            "test_pairs": len({sample["row"]["camera_key"] for sample in clip.samples}),
            "evaluation_setting_source": "validation_benchmark",
            "test_split_sha256": sha256_file(split_path),
        })
    else:
        metadata.update({
            "dataset_split": "validation", "validation_split_sha256": sha256_file(split_path),
            "benchmark_split": "validation", "benchmark_sequence": first["sequence"],
            "benchmark_camera": first["camera"], "benchmark_camera_key": first["camera_key"],
            "benchmark_video_name": first["video_name"], "benchmark_frames": len(clip),
        })
    return metadata


def _display(value):
    return "None" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def _write_summary(output_dir, metadata, runtime, metrics):
    pair = metrics["pair_level"]
    lines = [
        "Experiment: 01_anycalib_pretrained",
        "Mode: final_test",
        "Model: anycalib_pinhole",
        "Camera model: pinhole",
        f"Split: Test / {metadata['test_participant']}",
        f"Frames: {runtime['frames_processed']}",
        f"Pairs: {pair['pair_count']} / valid {pair['valid_pair_count']}",
        f"Success: {runtime['successful_frames']} / {runtime['frames_processed']} / {runtime['success_rate']:.6f}",
        "Pair-level focal:",
        f"  rel_fx mean={_display(pair['pair_rel_fx_error_mean'])} median={_display(pair['pair_rel_fx_error_median'])}",
        f"  rel_fy mean={_display(pair['pair_rel_fy_error_mean'])} median={_display(pair['pair_rel_fy_error_median'])}",
        f"  max_rel_f mean={_display(pair['pair_max_rel_f_error_mean'])} median={_display(pair['pair_max_rel_f_error_median'])}",
        "Pair-level principal-point:",
        f"  max_rel_c mean={_display(pair['pair_max_rel_c_error_mean'])} median={_display(pair['pair_max_rel_c_error_median'])}",
        f"Pairs within +/-5% focal target: {pair['pair_focal_within_5pct_count']} / {pair['valid_pair_count']} / {_display(pair['pair_focal_within_5pct_rate'])}",
        "Runtime:",
        f"  evaluation_seconds={_display(runtime['evaluation_seconds'])} FPS={_display(runtime['frames_per_second'])}",
        f"  GPU={metadata.get('gpu_name')} batch_size={metadata['batch_size']} num_workers={metadata['num_workers']}",
        f"Repository commit: {metadata['repository_commit']}",
        f"AnyCalib commit: {metadata['anycalib_commit']}",
        f"Result directory: {output_dir}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def _benchmark(config, config_path, args, final_test=False):
    if final_test:
        if args.batch_size is not None or args.num_workers is not None:
            raise ValueError("--test uses batch_size=2 and num_workers=4 from config; CLI overrides are not allowed.")
        batch_size = config["evaluation"]["batch_size"]
        num_workers = config["evaluation"]["num_workers"]
        if batch_size != 2 or num_workers != 4:
            raise RuntimeError("Final Test requires evaluation.batch_size=2 and evaluation.num_workers=4.")
        dataset, split_path = _dataset(config, "test")
        clip = dataset
        indices = list(range(len(dataset)))
        first = dataset.samples[0]["row"]
        output_dir = PROJECT_ROOT / config["evaluation"].get("output_dir", "runs/01_anycalib_pretrained/test")
    else:
        _require_benchmark_args(args)
        batch_size = args.batch_size
        num_workers = args.num_workers
        dataset, split_path = _dataset(config)
        clip, indices, first = _first_clip(dataset)
        output_dir = BENCHMARK_ROOT / f"bs{batch_size}_nw{num_workers}"
    if args.dry_run:
        print("mode=test" if final_test else "benchmark mode")
        print(f"{'test_split' if final_test else 'validation_split'}={split_path}")
        print(f"split={'test' if final_test else 'validation'}")
        print(f"participant={first['participant']}")
        print(f"selected_sequence={first['sequence']}")
        print(f"selected_camera={first['camera']}")
        print(f"selected_video_name={first['video_name']}")
        print(f"{'frames' if final_test else 'benchmark_frame_count'}={len(clip)}")
        print(f"pairs={len({sample['row']['camera_key'] for sample in dataset.samples}) if final_test else 1}")
        print(f"batch_size={batch_size}")
        print(f"num_workers={num_workers}")
        print(f"planned_batches={(len(clip) + batch_size - 1) // batch_size}")
        print(f"planned_output={output_dir}")
        print("dry-run: no model, weights, GPU, or inference used")
        return

    if final_test and _git_dirty():
        raise RuntimeError("Final Test evaluation requires a clean working tree.")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
    if output_dir.exists():
        if final_test:
            print("WARNING: overwriting final Test result")
        shutil.rmtree(output_dir)

    token = _require_single_gpu()
    output_dir.mkdir(parents=True)
    started_at = utc_now()
    total_start = time.perf_counter()
    weight_path = Path(torch.hub.get_dir()) / "anycalib" / f"{config['model']['model_id']}.pt"
    weight_present_before = weight_path.exists()
    model_start = time.perf_counter()
    adapter = AnyCalibAdapter(
        model_id=config["model"]["model_id"], cam_id=config["model"]["cam_id"],
        checkpoint=config["model"].get("checkpoint"), device="cuda:0",
    ).build()
    model_load_seconds = time.perf_counter() - model_start
    metadata = _benchmark_metadata(config, config_path, split_path, first, clip, batch_size, num_workers, final_test, token, output_dir, adapter, weight_path, weight_present_before, started_at)
    loader = DataLoader(clip, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    torch.cuda.reset_peak_memory_stats()
    state = {"frames_done": 0, "frames_total": len(clip), "success_count": 0}

    def telemetry_state():
        return {
            **state,
            "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        }

    sampler = TelemetrySampler(token, telemetry_state)
    sampler.start()
    rows = []
    timings = []
    success_count = 0
    evaluation_start = None
    iterator = iter(loader)
    with tqdm(total=len(clip), desc="01 Final Test" if final_test else "01 Val Benchmark", unit="frame", dynamic_ncols=True) as progress:
        while state["frames_done"] < len(clip):
            wait_start = time.perf_counter()
            batch = next(iterator)
            loader_wait = time.perf_counter() - wait_start
            if evaluation_start is None:
                evaluation_start = wait_start
            batch_size = _validate_batch(batch)
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            prediction = adapter.predict(batch["image"].to("cuda:0"))
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            postprocess_start = time.perf_counter()
            pred_intrinsics, success, pred_size = _normalize_batched_prediction(prediction, batch_size)
            pred_height, pred_width = pred_size
            for offset in range(batch_size):
                frame_success = bool(success[offset].item())
                meta = {key: _scalar(value[offset]) for key, value in batch["meta"].items()}
                gt = batch["intrinsics"][offset]
                row = {**meta, **{f"gt_{key}": float(value) for key, value in zip(("fx", "fy", "cx", "cy"), gt)},
                       "pred_width": pred_width, "pred_height": pred_height, "success": frame_success}
                if frame_success:
                    prediction_values = pred_intrinsics[offset, :4]
                    row.update({f"pred_{key}": float(prediction_values[index]) for index, key in enumerate(("fx", "fy", "cx", "cy"))})
                    row.update({key: float(value) for key, value in camera_errors(prediction_values, gt, meta["width"], meta["height"]).items()})
                rows.append(row)
                success_count += int(frame_success)
            postprocess_seconds = time.perf_counter() - postprocess_start
            batch_total = time.perf_counter() - wait_start
            state["frames_done"] += batch_size
            state["success_count"] = success_count
            allocated = torch.cuda.memory_allocated() / 2**20
            reserved = torch.cuda.memory_reserved() / 2**20
            timings.append({
                "batch_index": len(timings), "first_dataset_index": indices[state["frames_done"] - batch_size],
                "last_dataset_index": indices[state["frames_done"] - 1], "batch_size": batch_size,
                "loader_wait_seconds": loader_wait, "inference_seconds": inference_seconds,
                "postprocess_seconds": postprocess_seconds, "batch_total_seconds": batch_total,
                "frames_per_second": batch_size / batch_total if batch_total else None,
                "torch_allocated_mib": allocated, "torch_reserved_mib": reserved,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            })
            latest_gpu = sampler.rows[-1] if sampler.rows else {}
            progress.update(batch_size)
            progress.set_postfix_str(f"success={success_count}/{state['frames_done']} GPU={latest_gpu.get('gpu_util_percent', 'n/a')}% VRAM={latest_gpu.get('gpu_memory_used_mib', 'n/a')}/{latest_gpu.get('gpu_memory_total_mib', 'n/a')}MiB Torch={allocated:.1f}/{reserved:.1f}MiB")
    sampler.stop()
    evaluation_seconds = time.perf_counter() - evaluation_start
    result_start = time.perf_counter()
    raw_path = output_dir / "frame_predictions.csv.gz"
    write_frame_predictions(raw_path, rows)
    write_summaries(raw_path, output_dir)
    write_gzip_csv(output_dir / "batch_timings.csv.gz", BATCH_FIELDS, timings)
    write_gzip_csv(output_dir / "telemetry.csv.gz", TELEMETRY_FIELDS, sampler.rows)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    result_write_seconds = time.perf_counter() - result_start
    fps = len(rows) / evaluation_seconds if evaluation_seconds else 0
    inference_values = [row["inference_seconds"] for row in timings]
    batch_values = [row["batch_total_seconds"] for row in timings]
    telemetry = sampler.rows
    runtime = {
        "model_load_seconds": model_load_seconds, "evaluation_seconds": evaluation_seconds,
        "total_seconds_including_model_load": time.perf_counter() - total_start, "frames_processed": len(rows),
        "successful_frames": success_count, "success_rate": success_count / len(rows), "batch_count": len(timings),
        "frames_per_second": fps, "first_batch_seconds": batch_values[0],
        "loader_wait_total_seconds": sum(row["loader_wait_seconds"] for row in timings),
        "loader_wait_fraction": sum(row["loader_wait_seconds"] for row in timings) / evaluation_seconds,
        "inference_total_seconds": sum(inference_values), "postprocess_total_seconds": sum(row["postprocess_seconds"] for row in timings),
        "result_write_seconds": result_write_seconds, "batch_total_p50_ms": percentile(batch_values, .5) * 1000,
        "batch_total_p95_ms": percentile(batch_values, .95) * 1000, "inference_p50_ms": percentile(inference_values, .5) * 1000,
        "inference_p95_ms": percentile(inference_values, .95) * 1000, "torch_peak_allocated_mib": max(row["torch_peak_allocated_mib"] for row in timings),
        "torch_peak_reserved_mib": max(row["torch_peak_reserved_mib"] for row in timings), "gpu_telemetry_samples": len(telemetry),
        "gpu_util_mean_percent": finite_mean(telemetry, "gpu_util_percent"), "gpu_util_max_percent": finite_max(telemetry, "gpu_util_percent"),
        "device_vram_used_mean_mib": finite_mean(telemetry, "gpu_memory_used_mib"), "device_vram_used_peak_mib": finite_max(telemetry, "gpu_memory_used_mib"),
        "gpu_temperature_mean_c": finite_mean(telemetry, "gpu_temperature_c"), "gpu_temperature_max_c": finite_max(telemetry, "gpu_temperature_c"),
        "gpu_power_mean_w": finite_mean(telemetry, "gpu_power_w"), "gpu_power_max_w": finite_max(telemetry, "gpu_power_w"),
        "estimated_test_frames": EXPECTED_TEST_FRAMES, "estimated_test_seconds": EXPECTED_TEST_FRAMES / fps if fps else None,
        "estimated_test_human": format_duration(EXPECTED_TEST_FRAMES / fps) if fps else None,
        "nonfinite_prediction_count": sum(not torch.isfinite(torch.tensor([row.get("pred_fx"), row.get("pred_fy"), row.get("pred_cx"), row.get("pred_cy")], dtype=torch.float64)).all().item() for row in rows if row["success"]),
        "nonpositive_focal_count": sum(row["success"] and (row["pred_fx"] <= 0 or row["pred_fy"] <= 0) for row in rows),
    }
    if final_test:
        for field in ("estimated_test_frames", "estimated_test_seconds", "estimated_test_human"):
            runtime.pop(field)
        stored_rows = read_frame_predictions(raw_path)
        with (output_dir / "pair_summary.csv").open(newline="") as handle:
            stored_pairs = list(csv.DictReader(handle))
        if len(stored_rows) != EXPECTED_TEST_SIZE:
            raise RuntimeError(f"Expected {EXPECTED_TEST_SIZE} raw Test rows, found {len(stored_rows)}")
        if len(stored_pairs) != EXPECTED_TEST_PAIRS:
            raise RuntimeError(f"Expected {EXPECTED_TEST_PAIRS} Test pair rows, found {len(stored_pairs)}")
        if len({row["camera_key"] for row in stored_rows}) != EXPECTED_TEST_PAIRS:
            raise RuntimeError("Stored Test rows do not contain the expected 49 camera pairs.")
    metadata.update({"pred_width": rows[0]["pred_width"], "pred_height": rows[0]["pred_height"], "finished_at_utc": utc_now()})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    if final_test:
        metrics = json.loads((output_dir / "metrics.json").read_text())
        _write_summary(output_dir, metadata, runtime, metrics)
        pair = metrics["pair_level"]
        print("Final Test completed")
        print(f"Frames: {len(rows)}")
        print(f"Success: {success_count} / {len(rows)} / {success_count / len(rows):.1%}")
        print(f"Pairs: {pair['pair_count']} / valid {pair['valid_pair_count']}")
        print(f"Pair max focal error: mean={_display(pair['pair_max_rel_f_error_mean'])} median={_display(pair['pair_max_rel_f_error_median'])}")
        print(f"Within 5%: {pair['pair_focal_within_5pct_count']} / {pair['valid_pair_count']} ({_display(pair['pair_focal_within_5pct_rate'])})")
        print(f"Pair principal-point error: mean={_display(pair['pair_max_rel_c_error_mean'])} median={_display(pair['pair_max_rel_c_error_median'])}")
        print(f"Runtime: {runtime['evaluation_seconds']:.2f}s FPS: {runtime['frames_per_second']:.2f}")
    else:
        print(f"Benchmark completed: frames={len(rows)} success_rate={success_count / len(rows):.1%} fps={fps:.2f} model_load={model_load_seconds:.2f}s")
        print(f"Inference p50/p95={runtime['inference_p50_ms']:.1f}/{runtime['inference_p95_ms']:.1f}ms GPU mean/max={runtime['gpu_util_mean_percent']}/{runtime['gpu_util_max_percent']}%")
        print(f"Estimated Test ({EXPECTED_TEST_FRAMES} frames): {runtime['estimated_test_human']}")
    print(f"Output: {output_dir}")


def _dry_run(config):
    dataset, split_path = _dataset(config)
    print(f"experiment={config['experiment']['name']}")
    print(f"model_id={config['model']['model_id']} cam_id={config['model']['cam_id']}")
    print(f"validation_split={split_path}")
    print(f"validation_size={len(dataset)}")
    first = dataset.samples[0]["row"]
    print(f"first_input={int(first['width'])}x{int(first['height'])}")
    print(f"smoke_output={SMOKE_OUTPUT}")
    print("dry-run: no model, weights, GPU, or inference used")


def main():
    args = parse_args()
    config_path = _config_path(args.config)
    config = load_config(config_path)
    if args.benchmark:
        _benchmark(config, config_path, args)
        return 0
    if args.test:
        _benchmark(config, config_path, args, final_test=True)
        return 0
    if args.dry_run:
        _dry_run(config)
        return 0
    if not args.smoke:
        print("Full evaluation is not enabled in 3-A. Use --smoke or --dry-run.")
        return 1
    _smoke(config, config_path, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
