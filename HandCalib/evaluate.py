"""AnyCalib 3-A smoke evaluation entry point."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from dataloaders.gigahands import GigaHandsDataset
from models.anycalib import AnyCalibAdapter
from utils.config import load_config
from utils.metrics import camera_errors
from utils.results import write_frame_predictions, write_summaries


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_VALIDATION_SIZE = 18652
SMOKE_OUTPUT = PROJECT_ROOT / "runs/01_anycalib_pretrained/smoke_val"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AnyCalib pretrained validation smoke test.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run one validation frame on one visible GPU.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and validation data without a model.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing smoke output directory.")
    return parser.parse_args()


def _config_path(value):
    candidate = Path(value)
    for option in (candidate, PROJECT_ROOT / candidate, PROJECT_ROOT.parent / candidate):
        if option.exists():
            return option.resolve()
    raise FileNotFoundError(f"Config does not exist: {value}")


def _path_from_config(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dataset(config):
    dataset_config = config["dataset"]
    split_path = _path_from_config(dataset_config["val_split"])
    dataset = GigaHandsDataset(
        root=_path_from_config(dataset_config["root"]),
        split_file=split_path,
        manifest_path=PROJECT_ROOT / "data/manifests/gigahands.csv",
    )
    if len(dataset) != EXPECTED_VALIDATION_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_VALIDATION_SIZE} validation frames, found {len(dataset)}")
    return dataset, split_path


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dry_run(config):
    dataset, split_path = _dataset(config)
    sample = dataset[0]
    print(f"experiment={config['experiment']['name']}")
    print(f"model_id={config['model']['model_id']} cam_id={config['model']['cam_id']}")
    print(f"validation_split={split_path}")
    print(f"validation_size={len(dataset)}")
    print(f"first_input={sample['image'].shape[-1]}x{sample['image'].shape[-2]}")
    print(f"smoke_output={SMOKE_OUTPUT}")
    print("dry-run: no model, weights, GPU, or inference used")


def _require_single_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke evaluation requires CUDA with exactly one visible GPU; no CPU fallback is available.")


def _metadata(config, split_path, sample, prediction):
    pred_size = tuple(int(value) for value in prediction["pred_size"])
    image = sample["image"]
    return {
        "experiment": config["experiment"]["name"], "mode": "smoke_val",
        "model_id": config["model"]["model_id"], "cam_id": config["model"]["cam_id"],
        "repo_head": _git_head(PROJECT_ROOT), "anycalib_head": _git_head(PROJECT_ROOT / "AnyCalib"),
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "dataset_split": str(split_path.relative_to(PROJECT_ROOT)), "dataset_size": EXPECTED_VALIDATION_SIZE,
        "smoke_index": 0, "input_width": int(image.shape[-1]), "input_height": int(image.shape[-2]),
        "pred_width": pred_size[1], "pred_height": pred_size[0],
        "success": bool(_scalar(prediction["success"])),
    }


def _smoke(config, config_path, overwrite):
    dataset, split_path = _dataset(config)
    _require_single_gpu()
    if SMOKE_OUTPUT.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {SMOKE_OUTPUT}")
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
    pred_size = tuple(int(value) for value in prediction["pred_size"])
    row = {**sample["meta"], **{f"gt_{key}": float(value) for key, value in zip(("fx", "fy", "cx", "cy"), sample["intrinsics"])},
           "pred_width": pred_size[1], "pred_height": pred_size[0], "success": success}
    if success:
        row.update({f"pred_{key}": float(intrinsics[index]) for index, key in enumerate(("fx", "fy", "cx", "cy"))})
        row.update({key: float(value) for key, value in camera_errors(intrinsics[:4], sample["intrinsics"], sample["meta"]["width"], sample["meta"]["height"]).items()})
    raw_path = SMOKE_OUTPUT / "frame_predictions.csv.gz"
    write_frame_predictions(raw_path, [row])
    write_summaries(raw_path, SMOKE_OUTPUT)
    shutil.copyfile(config_path, SMOKE_OUTPUT / "config.yaml")
    (SMOKE_OUTPUT / "metadata.json").write_text(json.dumps(_metadata(config, split_path, sample, prediction), indent=2) + "\n")
    print(f"smoke_output={SMOKE_OUTPUT}")
    print(f"success={success} pred_size={pred_size}")


def main():
    args = parse_args()
    config_path = _config_path(args.config)
    config = load_config(config_path)
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
