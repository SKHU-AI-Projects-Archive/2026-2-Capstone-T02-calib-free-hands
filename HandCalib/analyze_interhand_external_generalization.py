"""Post-hoc diagnostics for the completed InterHand external evaluations.

This script only reads existing result files and GigaHands metadata. It does not
build a model, load a checkpoint, access images, or modify an original result.
"""

import csv
import gzip
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PRETRAINED = ROOT / "runs/02_interhand_external_eval/pretrained"
FINETUNED = ROOT / "runs/02_interhand_external_eval/gigahands_finetuned"
GIGA_PRETRAINED = ROOT / "runs/01_anycalib_pretrained/test"
GIGA_FINETUNED = ROOT / "runs/02_anycalib_finetune/test"
GIGA_MANIFEST = ROOT / "data/manifests/gigahands.csv"
GIGA_TRAIN_SPLIT = ROOT / "data/splits/gigahands/train.txt"
OUTPUT = ROOT / "runs/02_interhand_external_eval/diagnostics"
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000
TIE_TOLERANCE = 1e-12
MANIFEST_SHA256 = "af910a754c3449c258335a3baf574cb2650d8fabe6cc3458ee69cee3f429b750"
PRETRAINED_SHA256 = "e73b174563bcb90dc9a0e348ee64fbb898901d2a1786623e62fc0e3b176a1a38"
FINETUNED_SHA256 = "1b87efa27c847886f5987084a95000bccf2491f1abc719efcfa1c232cd8c2d41"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head():
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q * 100))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": float(np.max(values)),
    }


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path):
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows, fields):
    return [{key: (float(row[key]) if key not in {"success"} else row[key] == "True") for key in fields} for row in rows]


def result_snapshot(directory):
    required = ["frame_predictions.csv.gz", "pair_summary.csv", "metrics.json", "metadata.json", "runtime.json", "config.yaml", "summary.txt"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Missing result files in {directory}: {missing}")
    files = {name: sha256(directory / name) for name in ("frame_predictions.csv.gz", "pair_summary.csv", "metrics.json", "metadata.json")}
    metadata = json.loads((directory / "metadata.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    frames = read_gzip_csv(directory / "frame_predictions.csv.gz")
    pairs = read_csv(directory / "pair_summary.csv")
    if len(frames) != 5760 or len(pairs) != 360:
        raise RuntimeError(f"Unexpected result counts in {directory}: frames={len(frames)}, pairs={len(pairs)}")
    if len({row["camera_key"] for row in frames}) != 360 or len({row["camera_key"] for row in pairs}) != 360:
        raise RuntimeError(f"Duplicate or missing camera keys in {directory}")
    if sum(row["success"] == "True" for row in frames) != 5760:
        raise RuntimeError(f"Not all frames succeeded in {directory}")
    return {"directory": directory, "files": files, "metadata": metadata, "metrics": metrics, "frames": frames, "pairs": pairs}


def verify_pair_aggregation(snapshot):
    by_key = {}
    for row in snapshot["frames"]:
        if row["success"] != "True":
            continue
        by_key.setdefault(row["camera_key"], []).append(row)
    stored = {row["camera_key"]: row for row in snapshot["pairs"]}
    mismatches = []
    for key, rows in by_key.items():
        first = rows[0]
        pred_fx = float(np.median([float(row["pred_fx"]) for row in rows]))
        pred_fy = float(np.median([float(row["pred_fy"]) for row in rows]))
        pred_cx = float(np.median([float(row["pred_cx"]) for row in rows]))
        pred_cy = float(np.median([float(row["pred_cy"]) for row in rows]))
        expected = {
            "median_pred_fx": pred_fx,
            "median_pred_fy": pred_fy,
            "pair_rel_fx_error": abs(pred_fx - float(first["gt_fx"])) / abs(float(first["gt_fx"])),
            "pair_rel_fy_error": abs(pred_fy - float(first["gt_fy"])) / abs(float(first["gt_fy"])),
            "pair_max_rel_c_error": 2 * max(abs(pred_cx - float(first["gt_cx"])) / float(first["width"]), abs(pred_cy - float(first["gt_cy"])) / float(first["height"])),
        }
        expected["pair_max_rel_f_error"] = max(expected["pair_rel_fx_error"], expected["pair_rel_fy_error"])
        for field, value in expected.items():
            if abs(float(stored[key][field]) - value) > 1e-9:
                mismatches.append({"camera_key": key, "field": field, "stored": float(stored[key][field]), "recomputed": value})
    return {"status": "PASS" if not mismatches else "FAIL", "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def join_pairs(pre, fine):
    left = {row["camera_key"]: row for row in pre["pairs"]}
    right = {row["camera_key"]: row for row in fine["pairs"]}
    keys = sorted(set(left) & set(right))
    if len(left) != 360 or len(right) != 360 or len(keys) != 360:
        raise RuntimeError(f"Pair join failed: pretrained={len(left)}, fine={len(right)}, intersection={len(keys)}")
    rows = []
    for key in keys:
        a, b = left[key], right[key]
        pre_error = float(a["pair_max_rel_f_error"])
        fine_error = float(b["pair_max_rel_f_error"])
        rows.append({
            "camera_key": key, "sequence": a["sequence"], "camera": a["camera"],
            "gt_fx": float(next(row["gt_fx"] for row in pre["frames"] if row["camera_key"] == key)),
            "gt_fy": float(next(row["gt_fy"] for row in pre["frames"] if row["camera_key"] == key)),
            "width": int(float(next(row["width"] for row in pre["frames"] if row["camera_key"] == key))),
            "height": int(float(next(row["height"] for row in pre["frames"] if row["camera_key"] == key))),
            "pre_pair_rel_fx_error": float(a["pair_rel_fx_error"]), "fine_pair_rel_fx_error": float(b["pair_rel_fx_error"]),
            "pre_pair_rel_fy_error": float(a["pair_rel_fy_error"]), "fine_pair_rel_fy_error": float(b["pair_rel_fy_error"]),
            "pre_pair_max_rel_f_error": pre_error, "fine_pair_max_rel_f_error": fine_error,
            "delta_focal": fine_error - pre_error,
            "pre_pair_max_rel_c_error": float(a["pair_max_rel_c_error"]),
            "fine_pair_max_rel_c_error": float(b["pair_max_rel_c_error"]),
            "delta_center": float(b["pair_max_rel_c_error"]) - float(a["pair_max_rel_c_error"]),
            "pre_median_pred_fx": float(a["median_pred_fx"]), "fine_median_pred_fx": float(b["median_pred_fx"]),
            "pre_median_pred_fy": float(a["median_pred_fy"]), "fine_median_pred_fy": float(b["median_pred_fy"]),
        })
    for row in rows:
        row["gt_nfx"] = row["gt_fx"] / row["width"]
        row["gt_nfy"] = row["gt_fy"] / row["height"]
        row["pre_nfx"] = row["pre_median_pred_fx"] / row["width"]
        row["pre_nfy"] = row["pre_median_pred_fy"] / row["height"]
        row["fine_nfx"] = row["fine_median_pred_fx"] / row["width"]
        row["fine_nfy"] = row["fine_median_pred_fy"] / row["height"]
        row["gt_nf_mean"] = (row["gt_nfx"] + row["gt_nfy"]) / 2
        row["pre_signed_fx"] = (row["pre_median_pred_fx"] - row["gt_fx"]) / row["gt_fx"]
        row["pre_signed_fy"] = (row["pre_median_pred_fy"] - row["gt_fy"]) / row["gt_fy"]
        row["fine_signed_fx"] = (row["fine_median_pred_fx"] - row["gt_fx"]) / row["gt_fx"]
        row["fine_signed_fy"] = (row["fine_median_pred_fy"] - row["gt_fy"]) / row["gt_fy"]
    return rows


def bootstrap(values):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    medians = np.median(values[indices], axis=1)
    return {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES, "mean_delta_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "median_delta_ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(x, y).statistic)
    except ImportError:
        spearman = None
    return {"pearson": pearson, "spearman": spearman}


def sign_test(values):
    positive = int(np.sum(np.asarray(values) > TIE_TOLERANCE))
    negative = int(np.sum(np.asarray(values) < -TIE_TOLERANCE))
    try:
        from scipy.stats import binomtest
        p_value = float(binomtest(min(positive, negative), positive + negative, 0.5).pvalue)
        return {"name": "two-sided paired sign test", "positive": positive, "negative": negative, "p_value": p_value}
    except ImportError:
        return {"status": "skipped", "reason": "scipy not installed"}


def load_giga_train_prior():
    participants = {line.strip() for line in GIGA_TRAIN_SPLIT.read_text().splitlines() if line.strip()}
    rows = [row for row in read_csv(GIGA_MANIFEST) if row["participant"] in participants]
    units = {row["camera_key"]: row for row in rows}
    nfx = np.asarray([float(row["fx"]) / float(row["width"]) for row in units.values()])
    nfy = np.asarray([float(row["fy"]) / float(row["height"]) for row in units.values()])
    return {"participants": sorted(participants), "unit_count": len(units), "mean": [float(nfx.mean()), float(nfy.mean())], "median": [float(np.median(nfx)), float(np.median(nfy))], "nfx_range": [float(nfx.min()), float(nfx.max())], "nfy_range": [float(nfy.min()), float(nfy.max())]}


def load_giga_train_normalized_values():
    participants = {line.strip() for line in GIGA_TRAIN_SPLIT.read_text().splitlines() if line.strip()}
    units = {row["camera_key"]: row for row in read_csv(GIGA_MANIFEST) if row["participant"] in participants}
    return np.asarray([float(row["fx"]) / float(row["width"]) for row in units.values()])


def prior_reference(rows, prior):
    errors = []
    for row in rows:
        pred_fx, pred_fy = prior[0] * row["width"], prior[1] * row["height"]
        errors.append(max(abs(pred_fx - row["gt_fx"]) / abs(row["gt_fx"]), abs(pred_fy - row["gt_fy"]) / abs(row["gt_fy"])))
    return {**stats(errors), "within_20pct": int(np.sum(np.asarray(errors) <= 0.20)), "within_5pct": int(np.sum(np.asarray(errors) <= 0.05))}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures" / f"{name}.png", dpi=220)
    fig.savefig(OUTPUT / "figures" / f"{name}.pdf")
    plt.close(fig)


def make_figures(rows, pre_giga, fine_giga, prior):
    errors_pre = np.asarray([row["pre_pair_max_rel_f_error"] for row in rows])
    errors_fine = np.asarray([row["fine_pair_max_rel_f_error"] for row in rows])
    delta = errors_fine - errors_pre
    plt.style.use("grayscale")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["GigaHands\nPretrained", "GigaHands\nFine-tuned", "InterHand\nPretrained", "InterHand\nFine-tuned"]
    values = [pre_giga, fine_giga, float(errors_pre.mean()), float(errors_fine.mean())]
    bars = ax.bar(np.arange(4), np.asarray(values) * 100, edgecolor="black", color="white")
    for bar, hatch in zip(bars, ["///", "\\\\", "...", ""]):
        bar.set_hatch(hatch)
    ax.set_xticks(range(4), labels); ax.set_ylabel("Pair mean max focal error (%)")
    figure_save(fig, "figure_1_dataset_model_comparison")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(errors_pre * 100, errors_fine * 100, s=12, facecolors="white", edgecolors="black")
    limit = max(errors_pre.max(), errors_fine.max()) * 100 * 1.05
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Pretrained focal error (%)", ylabel="Fine-tuned focal error (%)")
    figure_save(fig, "figure_2_paired_focal_scatter")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delta * 100, bins=30, edgecolor="black", color="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Fine-tuned - Pretrained focal error (percentage points)", ylabel="Calibration units")
    figure_save(fig, "figure_3_delta_distribution")

    fig, ax = plt.subplots(figsize=(6, 5))
    gt = np.asarray([row["gt_nf_mean"] for row in rows])
    ax.scatter(gt, [row["pre_nfx"] for row in rows], marker="o", facecolors="none", edgecolors="black", s=14, label="Pretrained fx")
    ax.scatter(gt, [row["fine_nfx"] for row in rows], marker="x", color="black", s=14, label="Fine-tuned fx")
    limits = [min(gt.min(), min(row["pre_nfx"] for row in rows), min(row["fine_nfx"] for row in rows)), max(gt.max(), max(row["pre_nfx"] for row in rows), max(row["fine_nfx"] for row in rows))]
    ax.plot(limits, limits, "k--", linewidth=1, label="identity")
    ax.axhline(prior["mean"][0], color="black", linestyle=":", label="Giga Train mean prior fx")
    ax.set(xlabel="InterHand GT normalized focal mean", ylabel="Predicted normalized fx"); ax.legend(fontsize=8)
    figure_save(fig, "figure_4_gt_vs_predicted_normalized_focal")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(load_giga_train_normalized_values(), bins=20, histtype="step", linewidth=2, label="GigaHands Train nfx")
    ax.hist([row["gt_nfx"] for row in rows], bins=20, histtype="step", linewidth=2, linestyle="--", label="InterHand GT nfx")
    ax.set(xlabel="Normalized focal fx / width", ylabel="Calibration units"); ax.legend()
    figure_save(fig, "figure_5_gigahands_vs_interhand_focal_distribution")

    order = np.argsort([row["gt_nf_mean"] for row in rows])
    chunks = np.array_split(order, 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, 5)
    ax.plot(x, [errors_pre[c].mean() * 100 for c in chunks], "ko-", label="Pretrained")
    ax.plot(x, [errors_fine[c].mean() * 100 for c in chunks], "ks--", markerfacecolor="white", label="Fine-tuned")
    ax.set(xticks=x, xlabel="GT normalized focal quartile", ylabel="Mean max focal error (%)"); ax.legend()
    figure_save(fig, "figure_6_focal_quartile_performance")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    protected_before = {str(directory.relative_to(ROOT)): result_snapshot(directory) for directory in (PRETRAINED, FINETUNED)}
    pre = protected_before[str(PRETRAINED.relative_to(ROOT))]
    fine = protected_before[str(FINETUNED.relative_to(ROOT))]
    if pre["metadata"]["manifest_sha256"] != fine["metadata"]["manifest_sha256"] != MANIFEST_SHA256:
        raise RuntimeError("InterHand manifest provenance mismatch")
    if pre["metadata"]["model_weight_sha256"] != PRETRAINED_SHA256 or fine["metadata"]["model_weight_sha256"] != FINETUNED_SHA256:
        raise RuntimeError("Checkpoint provenance mismatch")
    if fine["metadata"].get("checkpoint_epoch") != 5 or fine["metadata"].get("checkpoint_global_step") != 31975:
        raise RuntimeError("Fine-tuned checkpoint metadata mismatch")
    aggregation_checks = {"pretrained": verify_pair_aggregation(pre), "finetuned": verify_pair_aggregation(fine)}
    rows = join_pairs(pre, fine)
    protocol_fields = ("dataset_name", "test_manifest", "manifest_sha256", "test_frames", "test_calibration_units", "test_pairs", "dataset_split", "test_split_sha256", "evaluation_setting_source", "batch_size", "num_workers", "cam_id", "model_id", "input_width", "input_height", "pred_width", "pred_height", "dataloader_prefetch_factor")
    protocol_diff = {field: (pre["metadata"].get(field), fine["metadata"].get(field)) for field in protocol_fields if pre["metadata"].get(field) != fine["metadata"].get(field)}
    if protocol_diff:
        raise RuntimeError(f"Protocol parity failed: {protocol_diff}")
    delta = np.asarray([row["delta_focal"] for row in rows])
    delta_center = np.asarray([row["delta_center"] for row in rows])
    better = int(np.sum(delta < -TIE_TOLERANCE)); equal = int(np.sum(np.abs(delta) <= TIE_TOLERANCE)); worse = int(np.sum(delta > TIE_TOLERANCE))
    prior = load_giga_train_prior()
    prior_mean = prior_reference(rows, prior["mean"])
    prior_median = prior_reference(rows, prior["median"])
    pre_dist = np.asarray([[abs(row["pre_nfx"] - prior["median"][0]), abs(row["pre_nfy"] - prior["median"][1])] for row in rows])
    fine_dist = np.asarray([[abs(row["fine_nfx"] - prior["median"][0]), abs(row["fine_nfy"] - prior["median"][1])] for row in rows])
    pre_l2, fine_l2 = np.linalg.norm(pre_dist, axis=1), np.linalg.norm(fine_dist, axis=1)
    attraction = {"fine_closer": int(np.sum(fine_l2 < pre_l2 - TIE_TOLERANCE)), "equal": int(np.sum(np.abs(fine_l2 - pre_l2) <= TIE_TOLERANCE)), "pretrained_closer": int(np.sum(fine_l2 > pre_l2 + TIE_TOLERANCE))}
    signed = {}
    for model in ("pre", "fine"):
        for axis in ("fx", "fy"):
            values = np.asarray([row[f"{model}_signed_{axis}"] for row in rows])
            signed[f"{model}_{axis}"] = {"mean": float(values.mean()), "median": float(np.median(values)), "positive_count": int(np.sum(values > TIE_TOLERANCE)), "negative_count": int(np.sum(values < -TIE_TOLERANCE)), "over_count": int(np.sum(values > TIE_TOLERANCE)), "under_count": int(np.sum(values < -TIE_TOLERANCE))}
    gt_focal = np.asarray([row["gt_nf_mean"] for row in rows])
    quartile_rows = []
    for index, chunk in enumerate(np.array_split(np.argsort(gt_focal), 4), 1):
        quartile_rows.append({"quartile": f"Q{index}", "unit_count": len(chunk), "gt_focal_mean": float(gt_focal[chunk].mean()), "pretrained_mean_error": float(np.mean([rows[i]["pre_pair_max_rel_f_error"] for i in chunk])), "pretrained_median_error": float(np.median([rows[i]["pre_pair_max_rel_f_error"] for i in chunk])), "finetuned_mean_error": float(np.mean([rows[i]["fine_pair_max_rel_f_error"] for i in chunk])), "finetuned_median_error": float(np.median([rows[i]["fine_pair_max_rel_f_error"] for i in chunk])), "mean_delta": float(delta[chunk].mean())})
    pre_frames = {row["camera_key"]: [] for row in pre["frames"]}
    fine_frames = {row["camera_key"]: [] for row in fine["frames"]}
    for row in pre["frames"]: pre_frames[row["camera_key"]].append(float(row["pred_fx"]))
    for row in fine["frames"]: fine_frames[row["camera_key"]].append(float(row["pred_fx"]))
    stability = {}
    for name, frame_map in (("pretrained", pre_frames), ("finetuned", fine_frames)):
        fx_std = np.asarray([np.std(values) for values in frame_map.values()])
        fy_map = {key: [] for key in frame_map}
        source = pre["frames"] if name == "pretrained" else fine["frames"]
        for row in source: fy_map[row["camera_key"]].append(float(row["pred_fy"]))
        fy_std = np.asarray([np.std(values) for values in fy_map.values()])
        errors = np.asarray([row["pre_pair_max_rel_f_error"] if name == "pretrained" else row["fine_pair_max_rel_f_error"] for row in rows])
        pair_by_key = {row["camera_key"]: row for row in (pre if name == "pretrained" else fine)["pairs"]}
        pair_fx_std = np.asarray([float(pair_by_key[key]["pred_fx_std"]) for key in frame_map])
        stability[name] = {"fx_std_mean": float(fx_std.mean()), "fx_std_median": float(np.median(fx_std)), "fy_std_mean": float(fy_std.mean()), "fy_std_median": float(np.median(fy_std)), "fx_std_vs_focal_error": correlation(fx_std, errors), "pair_summary_fx_std_max_abs_difference": float(np.max(np.abs(fx_std - pair_fx_std)))}
    ratio_values = [row["fine_pair_max_rel_f_error"] / row["pre_pair_max_rel_f_error"] for row in rows if row["pre_pair_max_rel_f_error"] > 1e-12]
    summary = {
        "analysis": {"repository_commit": git_head(), "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "bootstrap": bootstrap(delta), "tie_tolerance": TIE_TOLERANCE, "new_inference_runs": 0, "training_runs": 0, "checkpoint_modifications": 0},
        "provenance": {"pretrained_result": str(PRETRAINED.relative_to(ROOT)), "finetuned_result": str(FINETUNED.relative_to(ROOT)), "manifest_sha256": MANIFEST_SHA256, "pretrained_weight_sha256": PRETRAINED_SHA256, "finetuned_checkpoint_sha256": FINETUNED_SHA256, "pretrained_result_files_before": pre["files"], "finetuned_result_files_before": fine["files"]},
        "integrity": {"pretrained_frames": 5760, "finetuned_frames": 5760, "pretrained_units": 360, "finetuned_units": 360, "protocol_parity": "PASS", "aggregation_recheck": aggregation_checks, "paired_join": {"pretrained_unique": 360, "finetuned_unique": 360, "intersection": 360, "missing": 0, "duplicate": 0}},
        "paired_focal": {"fine_better": better, "equal": equal, "pretrained_better": worse, "fine_better_rate": better / 360, "equal_rate": equal / 360, "pretrained_better_rate": worse / 360, "delta": stats(delta.tolist()), "mean_absolute_change": float(np.mean(np.abs(delta))), "median_absolute_change": float(np.median(np.abs(delta))), "fine_to_pretrained_ratio": stats(ratio_values), "sign_test": sign_test(delta)},
        "focal_components": {"pretrained_fx_error": stats([row["pre_pair_rel_fx_error"] for row in rows]), "pretrained_fy_error": stats([row["pre_pair_rel_fy_error"] for row in rows]), "finetuned_fx_error": stats([row["fine_pair_rel_fx_error"] for row in rows]), "finetuned_fy_error": stats([row["fine_pair_rel_fy_error"] for row in rows]), "signed_bias": signed},
        "giga_train_prior": {**prior, "mean_prior_interhand": prior_mean, "median_prior_interhand": prior_median, "prediction_distance_to_median_prior": {"pretrained_component_mean": pre_dist.mean(axis=0).tolist(), "finetuned_component_mean": fine_dist.mean(axis=0).tolist(), "pretrained_l2_mean": float(pre_l2.mean()), "finetuned_l2_mean": float(fine_l2.mean())}, "attraction": attraction},
        "gt_focal_range": {"correlation_with_pretrained_error": correlation(gt_focal, [row["pre_pair_max_rel_f_error"] for row in rows]), "correlation_with_finetuned_error": correlation(gt_focal, [row["fine_pair_max_rel_f_error"] for row in rows]), "correlation_with_delta": correlation(gt_focal, delta), "quartiles": quartile_rows},
        "principal_point": {"pretrained": stats([row["pre_pair_max_rel_c_error"] for row in rows]), "finetuned": stats([row["fine_pair_max_rel_c_error"] for row in rows]), "delta": stats(delta_center.tolist()), "fine_better": int(np.sum(delta_center < -TIE_TOLERANCE)), "pretrained_better": int(np.sum(delta_center > TIE_TOLERANCE)), "equal": int(np.sum(np.abs(delta_center) <= TIE_TOLERANCE)), "focal_delta_mean": float(delta.mean()), "center_delta_mean": float(delta_center.mean())},
        "frame_stability": stability,
        "giga_test_reference": {"pretrained_mean_max_rel_f": json.loads((GIGA_PRETRAINED / "metrics.json").read_text())["pair_level"]["pair_max_rel_f_error_mean"], "finetuned_mean_max_rel_f": json.loads((GIGA_FINETUNED / "metrics.json").read_text())["pair_level"]["pair_max_rel_f_error_mean"], "sources": [str(GIGA_PRETRAINED.relative_to(ROOT)), str(GIGA_FINETUNED.relative_to(ROOT))]},
    }
    write_csv(OUTPUT / "paired_units.csv", rows)
    write_csv(OUTPUT / "focal_quartiles.csv", quartile_rows)
    write_csv(OUTPUT / "bias_summary.csv", [{"model": model, "component": axis, **signed[f"{model}_{axis}"]} for model in ("pre", "fine") for axis in ("fx", "fy")])
    write_csv(OUTPUT / "prior_comparison.csv", [{"model": "pretrained", "median_prior_component_distance_nfx": float(pre_dist[:, 0].mean()), "median_prior_component_distance_nfy": float(pre_dist[:, 1].mean()), "median_prior_l2_distance": float(pre_l2.mean())}, {"model": "finetuned", "median_prior_component_distance_nfx": float(fine_dist[:, 0].mean()), "median_prior_component_distance_nfy": float(fine_dist[:, 1].mean()), "median_prior_l2_distance": float(fine_l2.mean())}])
    giga_pre_metrics = json.loads((GIGA_PRETRAINED / "metrics.json").read_text())["pair_level"]
    giga_fine_metrics = json.loads((GIGA_FINETUNED / "metrics.json").read_text())["pair_level"]
    summary_rows = []
    for dataset, model, directory, metrics in (("GigaHands", "Pretrained", GIGA_PRETRAINED, giga_pre_metrics), ("GigaHands", "Fine-tuned", GIGA_FINETUNED, giga_fine_metrics), ("InterHand", "Pretrained", PRETRAINED, pre["metrics"]["pair_level"]), ("InterHand", "Fine-tuned", FINETUNED, fine["metrics"]["pair_level"])):
        pair_values = [float(row["pair_max_rel_f_error"]) for row in read_csv(directory / "pair_summary.csv")]
        summary_rows.append({"dataset": dataset, "model": model, "units": len(pair_values), "mean_focal_error": float(np.mean(pair_values)), "median_focal_error": float(np.median(pair_values)), "p90": percentile(pair_values, 0.90), "p95": percentile(pair_values, 0.95), "max": float(np.max(pair_values)), "within_5pct": int(np.sum(np.asarray(pair_values) <= 0.05)), "within_10pct": int(np.sum(np.asarray(pair_values) <= 0.10)), "within_20pct": int(np.sum(np.asarray(pair_values) <= 0.20)), "principal_point_mean": float(metrics["pair_max_rel_c_error_mean"])})
    write_csv(OUTPUT / "summary_table.csv", summary_rows)
    paired_table = [{"fine_better": better, "equal": equal, "pretrained_better": worse, "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)), "bootstrap_ci95_low": summary["analysis"]["bootstrap"]["mean_delta_ci95"][0], "bootstrap_ci95_high": summary["analysis"]["bootstrap"]["mean_delta_ci95"][1]}]
    write_csv(OUTPUT / "paired_comparison.csv", paired_table)
    stability_rows = [{"model": model, **values} for model, values in stability.items()]
    write_csv(OUTPUT / "stability_summary.csv", stability_rows)
    (OUTPUT / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(rows, summary["giga_test_reference"]["pretrained_mean_max_rel_f"], summary["giga_test_reference"]["finetuned_mean_max_rel_f"], prior)
    protected_after = {str(directory.relative_to(ROOT)): result_snapshot(directory) for directory in (PRETRAINED, FINETUNED)}
    summary["provenance"]["original_result_files_unchanged"] = protected_before == protected_after
    summary["provenance"]["pretrained_result_files_after"] = protected_after[str(PRETRAINED.relative_to(ROOT))]["files"]
    summary["provenance"]["finetuned_result_files_after"] = protected_after[str(FINETUNED.relative_to(ROOT))]["files"]
    if not summary["provenance"]["original_result_files_unchanged"]:
        raise RuntimeError("Original result files changed during analysis")
    (OUTPUT / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "summary_table.md").write_text("| Dataset | Model | Units | Mean focal error | Median | P90 | P95 | Max | Within 5% | Within 10% | Within 20% | Principal-point mean |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {r['dataset']} | {r['model']} | {r['units']} | {r['mean_focal_error']:.6f} | {r['median_focal_error']:.6f} | {r['p90']:.6f} | {r['p95']:.6f} | {r['max']:.6f} | {r['within_5pct']} | {r['within_10pct']} | {r['within_20pct']} | {r['principal_point_mean']:.6f} |" for r in summary_rows) + "\n")
    (OUTPUT / "paired_comparison.md").write_text("| Fine better | Equal | Pretrained better | Mean delta | Median delta | Bootstrap mean-delta 95% CI |\n|---:|---:|---:|---:|---:|---|\n" + f"| {better} | {equal} | {worse} | {delta.mean():.6f} | {np.median(delta):.6f} | [{paired_table[0]['bootstrap_ci95_low']:.6f}, {paired_table[0]['bootstrap_ci95_high']:.6f}] |\n")
    lines = ["# InterHand external generalization post-hoc diagnostics", "", "No model inference, training, fine-tuning, or checkpoint modification was performed.", "", f"- Paired units: {better} fine-tuned better, {equal} equal, {worse} pretrained better.", f"- Mean focal delta (fine - pretrained): {delta.mean():.6f}; bootstrap 95% CI: {summary['analysis']['bootstrap']['mean_delta_ci95']}.", f"- GigaHands Train median-prior attraction: {attraction['fine_closer']}/360 fine-tuned closer, {attraction['pretrained_closer']}/360 pretrained closer.", f"- Mean principal-point delta: {delta_center.mean():.6f}; mean focal delta: {delta.mean():.6f}.", "", "These are descriptive post-hoc diagnostics and do not establish causality or prove calibration memorization."]
    (OUTPUT / "analysis_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUTPUT), "paired": {"fine_better": better, "equal": equal, "pretrained_better": worse}, "delta_mean": float(delta.mean()), "delta_median": float(np.median(delta)), "bootstrap_mean_ci95": summary["analysis"]["bootstrap"]["mean_delta_ci95"], "prior_attraction": attraction, "original_result_files_unchanged": summary["provenance"]["original_result_files_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
