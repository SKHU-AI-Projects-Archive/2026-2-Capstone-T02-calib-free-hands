#!/usr/bin/env python3
"""Reaudit stored GigaHands focal predictions without model execution."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


PREDICTION_FIELDS = ("pred_fx", "pred_fy", "pred_cx", "pred_cy")
HANDCALIB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = HANDCALIB_ROOT / "analysis/focal_metric_reaudit_20261014"
LOCKED_FX_EDGES = [
    824.9450073242188,
    863.3640502929687,
    901.7830932617187,
    940.2021362304688,
    978.6211791992188,
    1017.0402221679688,
]
MODEL_LABELS = {
    "pretrained": "Pretrained AnyCalib",
    "finetuned": "GigaHands Fine-tuned AnyCalib",
    "train_constant": "Train Constant Baseline",
}


def relative_focal_errors(pred_fx, pred_fy, gt_fx, gt_fy):
    error_fx = abs(float(pred_fx) - float(gt_fx)) / abs(float(gt_fx)) * 100.0
    error_fy = abs(float(pred_fy) - float(gt_fy)) / abs(float(gt_fy)) * 100.0
    return {
        "relative_fx_percent": error_fx,
        "relative_fy_percent": error_fy,
        "relative_focal_percent": max(error_fx, error_fy),
    }


def distribution_stats(values):
    values = [float(value) for value in values]
    if not values:
        return {
            key: None
            for key in (
                "mean", "median", "population_std", "population_variance",
                "p90", "p95", "min", "max", "within_5_percent", "count",
            )
        }
    variance = statistics.pvariance(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": math.sqrt(variance),
        "population_variance": variance,
        "p90": float(np.percentile(np.asarray(values), 90)),
        "p95": float(np.percentile(np.asarray(values), 95)),
        "min": min(values),
        "max": max(values),
        "within_5_percent": sum(value <= 5.0 for value in values) / len(values),
        "count": len(values),
    }


def _constant_value(rows, field, camera_key):
    values = {float(row[field]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"inconsistent {field} within camera pair {camera_key}")
    return next(iter(values))


def aggregate_frame_predictions(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("success", False):
            groups[row["camera_key"]].append(row)
    result = []
    for camera_key in sorted(groups):
        group = groups[camera_key]
        first = group[0]
        pair = {
            "participant": first["participant"],
            "sequence": first["sequence"],
            "camera": first["camera"],
            "camera_key": camera_key,
            "frame_count": len(group),
        }
        for field in ("width", "height", "gt_fx", "gt_fy", "gt_cx", "gt_cy"):
            pair[field] = _constant_value(group, field, camera_key)
        pair["width"] = int(pair["width"])
        pair["height"] = int(pair["height"])
        for field in PREDICTION_FIELDS:
            pair[field] = statistics.median(float(row[field]) for row in group)
        pair.update(relative_focal_errors(
            pair["pred_fx"], pair["pred_fy"], pair["gt_fx"], pair["gt_fy"]
        ))
        result.append(pair)
    return result


def range_errors(pred_fx, pred_fy, gt_fx, gt_fy, fx_range, fy_range):
    if fx_range <= 0 or fy_range <= 0:
        raise ValueError("focal ranges must be positive")
    error_fx = abs(float(pred_fx) - float(gt_fx)) / float(fx_range) * 100.0
    error_fy = abs(float(pred_fy) - float(gt_fy)) / float(fy_range) * 100.0
    return {
        "range_fx_percent": error_fx,
        "range_fy_percent": error_fy,
        "range_focal_percent": max(error_fx, error_fy),
    }


def fov_degrees(focal, extent):
    return math.degrees(2.0 * math.atan(float(extent) / (2.0 * float(focal))))


def fov_errors(pred_fx, pred_fy, gt_fx, gt_fy, width, height):
    return {
        "hfov_absolute_error_deg": abs(
            fov_degrees(pred_fx, width) - fov_degrees(gt_fx, width)
        ),
        "vfov_absolute_error_deg": abs(
            fov_degrees(pred_fy, height) - fov_degrees(gt_fy, height)
        ),
    }


def _manifest_signature(row):
    return tuple(float(row[field]) for field in ("width", "height", "fx", "fy", "cx", "cy"))


def deduplicate_manifest(rows):
    result = {}
    signatures = {}
    for row in rows:
        key = row["camera_key"]
        signature = _manifest_signature(row)
        if key in signatures and signatures[key] != signature:
            raise ValueError(f"inconsistent manifest duplicate for {key}")
        signatures[key] = signature
        if key not in result:
            item = dict(row)
            for field in ("width", "height"):
                item[field] = int(float(item[field]))
            for field in ("fx", "fy", "cx", "cy"):
                item[field] = float(item[field])
            result[key] = item
    return [result[key] for key in sorted(result)]


def train_constant_prior(train_rows):
    if not train_rows:
        raise ValueError("Train rows cannot be empty")
    return {
        "camera_pair_count": len(train_rows),
        "median_nfx": statistics.median(float(row["fx"]) / float(row["width"]) for row in train_rows),
        "median_nfy": statistics.median(float(row["fy"]) / float(row["height"]) for row in train_rows),
    }


def constant_prediction(prior, width, height):
    return prior["median_nfx"] * float(width), prior["median_nfy"] * float(height)


def pearson_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_correlation(left, right):
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def response_diagnostics(pairs, axis):
    gt = np.asarray([float(pair[f"gt_{axis}"]) for pair in pairs], dtype=np.float64)
    pred = np.asarray([float(pair[f"pred_{axis}"]) for pair in pairs], dtype=np.float64)
    if gt.size < 2 or np.std(gt) == 0:
        raise ValueError(f"GT {axis} must vary for response diagnostics")
    slope, intercept = np.polyfit(gt, pred, 1)
    fitted = slope * gt + intercept
    pred_variance_sum = float(np.sum((pred - np.mean(pred)) ** 2))
    residual_sum = float(np.sum((pred - fitted) ** 2))
    r_squared = None if pred_variance_sum == 0 else 1.0 - residual_sum / pred_variance_sum
    gt_std = float(np.std(gt))
    pred_std = float(np.std(pred))
    return {
        "axis": axis,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "pearson": pearson_correlation(gt, pred),
        "spearman": spearman_correlation(gt.tolist(), pred.tolist()),
        "gt_min": float(np.min(gt)),
        "gt_max": float(np.max(gt)),
        "gt_range": float(np.max(gt) - np.min(gt)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "prediction_range": float(np.max(pred) - np.min(pred)),
        "gt_std": gt_std,
        "prediction_std": pred_std,
        "pred_std_over_gt_std": pred_std / gt_std,
        "count": int(gt.size),
    }


def assign_bin(value, edges):
    value = float(value)
    if value < edges[0] or value > edges[-1]:
        raise ValueError(f"value {value} falls outside fixed bin support")
    if value == edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(np.asarray(edges), value, side="right") - 1)


def fixed_bin_groups(rows, edges):
    groups = [[] for _ in range(len(edges) - 1)]
    for row in rows:
        groups[assign_bin(row["gt_fx"], edges)].append(row)
    return groups


def prepare_output(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_legacy_reproduction(pairs, stored_metrics, tolerance=1e-12):
    if not pairs:
        raise RuntimeError("no pairs available for legacy reproduction")
    recomputed_ratio = statistics.fmean(
        float(pair["relative_focal_percent"]) / 100.0 for pair in pairs
    )
    pair_level = stored_metrics["pair_level"]
    recorded_ratio = float(pair_level["pair_max_rel_f_error_mean"])
    if abs(recomputed_ratio - recorded_ratio) > tolerance:
        raise RuntimeError(
            "legacy focal mean mismatch: "
            f"recomputed={recomputed_ratio:.17g} recorded={recorded_ratio:.17g}"
        )
    recorded_count = int(pair_level.get("valid_pair_count", pair_level.get("pair_count", -1)))
    if recorded_count != len(pairs):
        raise RuntimeError(
            f"legacy pair count mismatch: recomputed={len(pairs)} recorded={recorded_count}"
        )
    return {
        "recomputed_ratio": recomputed_ratio,
        "recorded_ratio": recorded_ratio,
        "absolute_difference": abs(recomputed_ratio - recorded_ratio),
        "pair_count": len(pairs),
        "tolerance": tolerance,
        "status": "PASS",
    }


def validate_model_pair_alignment(model_pairs):
    key_sets = {
        model: {pair["camera_key"] for pair in pairs}
        for model, pairs in model_pairs.items()
    }
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        raise ValueError(f"model camera-key mismatch: {key_sets}")
    by_model = {
        model: {pair["camera_key"]: pair for pair in pairs}
        for model, pairs in model_pairs.items()
    }
    first_model = next(iter(by_model))
    for key in sorted(key_sets[first_model]):
        reference = by_model[first_model][key]
        for model, indexed in by_model.items():
            current = indexed[key]
            for field in ("width", "height", "gt_fx", "gt_fy", "gt_cx", "gt_cy"):
                if current[field] != reference[field]:
                    raise ValueError(
                        f"model GT mismatch for {key} field={field}: "
                        f"{first_model}={reference[field]} {model}={current[field]}"
                    )
    return sorted(key_sets[first_model])


def validate_single_resolution(rows):
    resolutions = sorted({(int(row["width"]), int(row["height"])) for row in rows})
    if len(resolutions) != 1:
        raise ValueError(f"single-resolution raw-pixel analysis required, found {resolutions}")
    return resolutions[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(paths):
    return {str(Path(path).resolve()): sha256_file(path) for path in paths}


def verify_source_snapshot(snapshot):
    current = {path: sha256_file(path) for path in snapshot}
    if current != snapshot:
        changed = sorted(path for path in snapshot if current[path] != snapshot[path])
        raise RuntimeError(f"immutable source changed: {changed}")
    return True


def response_summary_fields(pairs):
    result = {}
    for axis in ("fx", "fy"):
        diagnostic = response_diagnostics(pairs, axis)
        for source, target in (
            ("slope", "slope"),
            ("intercept", "intercept"),
            ("r_squared", "r_squared"),
            ("pearson", "pearson"),
            ("spearman", "spearman"),
            ("gt_min", "gt_min"),
            ("gt_max", "gt_max"),
            ("gt_range", "gt_range"),
            ("prediction_min", "prediction_min"),
            ("prediction_max", "prediction_max"),
            ("prediction_range", "prediction_range"),
            ("gt_std", "gt_std"),
            ("prediction_std", "prediction_std"),
            ("pred_std_over_gt_std", "pred_std_over_gt_std"),
        ):
            result[f"{target}_{axis}"] = diagnostic[source]
    return result


def bin_level_rows(model_pairs, edges):
    result = []
    for model, pairs in model_pairs.items():
        groups = fixed_bin_groups(pairs, edges)
        for index, group in enumerate(groups):
            relative = distribution_stats([row["relative_focal_percent"] for row in group])
            test_range = distribution_stats([row["test_range_focal_percent"] for row in group])
            global_range = distribution_stats([row["global_range_focal_percent"] for row in group])
            pred_fx = distribution_stats([row["pred_fx"] for row in group])
            pred_fy = distribution_stats([row["pred_fy"] for row in group])
            result.append({
                "model": model,
                "bin_index": index + 1,
                "lower_edge_fx": edges[index],
                "upper_edge_fx": edges[index + 1],
                "upper_edge_inclusive": index == len(groups) - 1,
                "pair_count": len(group),
                "relative_focal_mean_percent": relative["mean"],
                "relative_focal_median_percent": relative["median"],
                "test_range_mean_percent": test_range["mean"],
                "global_range_mean_percent": global_range["mean"],
                "prediction_fx_mean": pred_fx["mean"],
                "prediction_fx_population_std": pred_fx["population_std"],
                "prediction_fy_mean": pred_fy["mean"],
                "prediction_fy_population_std": pred_fy["population_std"],
            })
    return result


def identity_line_extent(pairs, axis):
    values = [float(pair[f"gt_{axis}"]) for pair in pairs]
    values.extend(float(pair[f"pred_{axis}"]) for pair in pairs)
    return min(values), max(values)


def scatter_annotation(diagnostic):
    r_squared = diagnostic["r_squared"]
    r2_text = "undefined" if r_squared is None else f"{r_squared:.3f}"
    return (
        f"slope={diagnostic['slope']:.3f}\n"
        f"R²={r2_text}\n"
        f"GT range={diagnostic['gt_range']:.2f}\n"
        f"Pred range={diagnostic['prediction_range']:.2f}\n"
        f"GT σ={diagnostic['gt_std']:.2f}\n"
        f"Pred σ={diagnostic['prediction_std']:.2f}"
    )


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_frame_csv(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = (
        "frame_index", "width", "height", "gt_fx", "gt_fy", "gt_cx", "gt_cy",
        "pred_fx", "pred_fy", "pred_cx", "pred_cy",
    )
    for row in rows:
        for field in numeric:
            row[field] = int(row[field]) if field in ("frame_index", "width", "height") else float(row[field])
        row["success"] = row["success"].lower() in {"1", "true", "yes"}
    return rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def focal_range_definition(rows):
    fx = [float(row["gt_fx"] if "gt_fx" in row else row["fx"]) for row in rows]
    fy = [float(row["gt_fy"] if "gt_fy" in row else row["fy"]) for row in rows]
    return {
        "fx_min": min(fx), "fx_max": max(fx), "fx_range": max(fx) - min(fx),
        "fy_min": min(fy), "fy_max": max(fy), "fy_range": max(fy) - min(fy),
        "pair_count": len(rows),
    }


def enrich_pairs(pairs, test_range, global_range):
    result = []
    for source in pairs:
        row = dict(source)
        test = range_errors(
            row["pred_fx"], row["pred_fy"], row["gt_fx"], row["gt_fy"],
            test_range["fx_range"], test_range["fy_range"],
        )
        global_values = range_errors(
            row["pred_fx"], row["pred_fy"], row["gt_fx"], row["gt_fy"],
            global_range["fx_range"], global_range["fy_range"],
        )
        row.update({f"test_{key}": value for key, value in test.items()})
        row.update({f"global_{key}": value for key, value in global_values.items()})
        row.update(fov_errors(
            row["pred_fx"], row["pred_fy"], row["gt_fx"], row["gt_fy"],
            row["width"], row["height"],
        ))
        result.append(row)
    return result


def constant_pairs(reference_pairs, prior):
    result = []
    for reference in reference_pairs:
        row = {
            key: reference[key]
            for key in (
                "participant", "sequence", "camera", "camera_key", "frame_count",
                "width", "height", "gt_fx", "gt_fy", "gt_cx", "gt_cy",
            )
        }
        row["pred_fx"], row["pred_fy"] = constant_prediction(prior, row["width"], row["height"])
        row["pred_cx"] = None
        row["pred_cy"] = None
        row.update(relative_focal_errors(
            row["pred_fx"], row["pred_fy"], row["gt_fx"], row["gt_fy"]
        ))
        result.append(row)
    return result


def _add_stats(row, prefix, values):
    for key, value in distribution_stats(values).items():
        row[f"{prefix}_{key}"] = value


def model_summary(model, pairs):
    row = {"model": model, "model_label": MODEL_LABELS[model], "pair_count": len(pairs)}
    for prefix, field in (
        ("relative_fx_percent", "relative_fx_percent"),
        ("relative_fy_percent", "relative_fy_percent"),
        ("relative_focal_percent", "relative_focal_percent"),
        ("test_range_focal_percent", "test_range_focal_percent"),
        ("global_range_focal_percent", "global_range_focal_percent"),
        ("vfov_absolute_error_deg", "vfov_absolute_error_deg"),
        ("hfov_absolute_error_deg", "hfov_absolute_error_deg"),
    ):
        _add_stats(row, prefix, [pair[field] for pair in pairs])
    row.update(response_summary_fields(pairs))
    return row


def validate_pair_summary(recomputed_pairs, stored_rows, tolerance=1e-12):
    stored = {row["camera_key"]: row for row in stored_rows}
    if set(stored) != {row["camera_key"] for row in recomputed_pairs}:
        raise RuntimeError("stored pair summary camera-key mismatch")
    maximum = 0.0
    for pair in recomputed_pairs:
        source = stored[pair["camera_key"]]
        for axis in ("fx", "fy", "cx", "cy"):
            difference = abs(pair[f"pred_{axis}"] - float(source[f"median_pred_{axis}"]))
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise RuntimeError(
                    f"stored pair median mismatch for {pair['camera_key']} pred_{axis}"
                )
    return {"status": "PASS", "maximum_absolute_difference": maximum, "tolerance": tolerance}


def pair_level_rows(model_pairs):
    result = []
    for model, pairs in model_pairs.items():
        for pair in pairs:
            result.append({"model": model, "model_label": MODEL_LABELS[model], **pair})
    return result


def _none_to_nan(values):
    return [float("nan") if value is None else value for value in values]


def plot_figures(output, model_pairs, summary_rows, bin_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "0.85",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })
    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = list(model_pairs)
    labels = [MODEL_LABELS[model] for model in models]
    x = np.arange(len(models))

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    means = [next(row for row in summary_rows if row["model"] == model)["relative_focal_percent_mean"] for model in models]
    medians = [next(row for row in summary_rows if row["model"] == model)["relative_focal_percent_median"] for model in models]
    axis.bar(x - 0.18, means, 0.36, color="0.35", label="Mean")
    axis.bar(x + 0.18, medians, 0.36, color="white", edgecolor="black", hatch="//", label="Median")
    axis.set(xticks=x, xticklabels=labels, ylabel="Relative focal error (%)", title="Relative focal error")
    axis.tick_params(axis="x", rotation=12)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "relative_error_comparison.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    test_values = [next(row for row in summary_rows if row["model"] == model)["test_range_focal_percent_mean"] for model in models]
    global_values = [next(row for row in summary_rows if row["model"] == model)["global_range_focal_percent_mean"] for model in models]
    axis.bar(x - 0.18, test_values, 0.36, color="0.25", label="Test range")
    axis.bar(x + 0.18, global_values, 0.36, color="0.8", edgecolor="black", hatch="..", label="Demo-global range")
    axis.set(xticks=x, xticklabels=labels, ylabel="Range-normalized error (%)", title="Error relative to focal range")
    axis.tick_params(axis="x", rotation=12)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "range_error_comparison.png")
    plt.close(fig)

    markers = ("o", "s", "^")
    shades = ("white", "0.55", "black")
    for axis_name in ("fx", "fy"):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
        for panel, model, marker, shade in zip(axes, models, markers, shades):
            pairs = model_pairs[model]
            low, high = identity_line_extent(pairs, axis_name)
            padding = (high - low) * 0.06 or 1.0
            panel.scatter(
                [pair[f"gt_{axis_name}"] for pair in pairs],
                [pair[f"pred_{axis_name}"] for pair in pairs],
                marker=marker, facecolors=shade, edgecolors="black", s=28,
            )
            panel.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1, label="y=x")
            diagnostic = response_diagnostics(pairs, axis_name)
            panel.set_title(MODEL_LABELS[model])
            panel.text(
                0.04, 0.96,
                scatter_annotation(diagnostic),
                transform=panel.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.9},
            )
            panel.set(xlim=(low - padding, high + padding), ylim=(low - padding, high + padding), xlabel=f"GT {axis_name} (px)")
        axes[0].set_ylabel(f"Predicted {axis_name} (px)")
        fig.suptitle(f"GT vs predicted {axis_name}", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        fig.savefig(figures / f"gt_vs_pred_{axis_name}.png")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    reference = model_pairs["finetuned"]
    bins = np.linspace(
        min(pair["gt_fx"] for pair in reference),
        max(pair["gt_fx"] for pair in reference),
        12,
    )
    axis.hist([pair["gt_fx"] for pair in reference], bins=bins, histtype="step", color="black", linewidth=2, label="GT fx")
    axis.hist([pair["pred_fx"] for pair in reference], bins=bins, histtype="step", color="0.35", linestyle="--", linewidth=2, label="Fine-tuned prediction")
    axis.hist([pair["pred_fx"] for pair in model_pairs["train_constant"]], bins=bins, histtype="step", color="0.6", linestyle=":", linewidth=2.5, label="Train constant")
    axis.set(xlabel="Focal length fx (px)", ylabel="Independent camera pairs", title="Test focal distributions")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "focal_distribution.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.3))
    positions = np.arange(1, 6)
    width = 0.24
    for offset, model, shade, hatch in zip((-width, 0, width), models, ("0.25", "0.6", "white"), (None, "//", "..")):
        selected = [row for row in bin_rows if row["model"] == model]
        values = _none_to_nan([row["relative_focal_mean_percent"] for row in selected])
        axis.bar(positions + offset, values, width, color=shade, edgecolor="black", hatch=hatch, label=MODEL_LABELS[model])
    counts = [row["pair_count"] for row in bin_rows if row["model"] == "pretrained"]
    for position, count in zip(positions, counts):
        axis.text(position, 0, f"n={count}", ha="center", va="bottom", fontsize=9, rotation=90)
    axis.set(xticks=positions, xlabel="Precommitted Demo-global GT fx bin", ylabel="Mean relative focal error (%)", title="Relative error by fixed focal bin")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "bin_error.png")
    plt.close(fig)


def metric_definition_text(test_range, global_range):
    return f"""# Metric definitions

## Existing relative focal error

`e_fx = abs(fx_pred - fx_gt) / abs(fx_gt) * 100`

`e_fy = abs(fy_pred - fy_gt) / abs(fy_gt) * 100`

`e_f = max(e_fx, e_fy)`

The denominator is GT, not prediction. Predictions are component-wise frame medians within each sequence-camera pair, and all 49 independent camera pairs receive equal weight.

## Raw-pixel focal ranges

All Demo and Test samples are 1280 by 720, so this analysis uses raw focal pixels.

|Scope|fx min|fx max|fx range|fy min|fy max|fy range|Pairs|
|---|---:|---:|---:|---:|---:|---:|---:|
|Test|{test_range['fx_min']:.12g}|{test_range['fx_max']:.12g}|{test_range['fx_range']:.12g}|{test_range['fy_min']:.12g}|{test_range['fy_max']:.12g}|{test_range['fy_range']:.12g}|{test_range['pair_count']}|
|Demo-global|{global_range['fx_min']:.12g}|{global_range['fx_max']:.12g}|{global_range['fx_range']:.12g}|{global_range['fy_min']:.12g}|{global_range['fy_max']:.12g}|{global_range['fy_range']:.12g}|{global_range['pair_count']}|

For either scope, `RangeError_axis = abs(pred_axis - gt_axis) / axis_range * 100`, and RangeError is the maximum of the two components. This auxiliary metric describes absolute prediction error relative to dataset variation. It does not define a new model ranking because all models share the same denominator.

## FoV errors

`hFoV = 2 * atan(width / (2 * fx))` and `vFoV = 2 * atan(height / (2 * fy))`, reported as absolute angular errors in degrees.

## Response diagnostics

OLS slope/intercept, R-squared, Pearson, Spearman, GT/prediction population standard deviations, and their ratio are reported separately for fx and fy. Undefined constant-prediction correlations and R-squared values remain empty.

## Fixed bins

Test pairs are binned by GT fx using the precommitted Demo-global edges `{LOCKED_FX_EDGES}`. Empty bins remain in the table.
"""


def report_text(summary, bin_rows):
    rows = {row["model"]: row for row in summary["models"]}
    pre = rows["pretrained"]
    fine = rows["finetuned"]
    constant = rows["train_constant"]
    counts = [row["pair_count"] for row in bin_rows if row["model"] == "pretrained"]
    return f"""# GigaHands focal 평가 지표 재검증 및 분포 민감도 분석

## 1. 분석 목적

기존 저장 prediction만 이용해 상대 focal 오차를 독립 재계산하고, 낮은 오차가 GT focal 변화 추적 능력을 얼마나 반영하는지 보조 지표로 점검했다. 학습, 추론, checkpoint load, GPU 연산은 수행하지 않았다.

## 2. 기존 결과 계산과 재현

각 camera-key의 frame prediction `[fx, fy, cx, cy]`를 component-wise median으로 집계한 뒤 49개 camera pair를 동일 가중 평균했다. Pretrained 평균은 `{pre['relative_focal_percent_mean']:.6f}%`, fine-tuned 평균은 `{fine['relative_focal_percent_mean']:.6f}%`로 기존 기록과 허용 오차 안에서 일치했다.

## 3. 실제 denominator

실제 evaluator는 `abs(prediction - GT) / abs(GT)`를 사용한다. 과거 메시지의 prediction denominator 설명은 코드와 일치하지 않는다.

## 4. Range-normalized 보조 지표

모든 표본이 1280x720이므로 raw pixel focal range를 사용했다. Test와 Demo-global min/max/range는 `metric_definition.md`와 `summary.json`에 고정했다. 같은 Test에서는 모든 모델이 같은 denominator를 사용하므로 이 값은 새로운 순위 metric이 아니라 절대 오차가 focal variation 폭에서 차지하는 크기다.

## 5. 기존 relative error를 유지하는 이유

상대 focal 오차는 GT 크기에 비례한 예측 오차를 직접 측정하고 기존 연구와 비교 가능한 primary metric이다. Range-normalized metric은 이를 폐기하거나 대체하지 않는다.

## 6. Train constant baseline

p41/p44의 147개 고유 Train camera-key에서 normalized focal 중앙값만 계산했다. Test GT는 constant 결정에 사용하지 않았다. Constant baseline 평균 상대오차는 `{constant['relative_focal_percent_mean']:.6f}%`이다.

## 7. GT-prediction correlation

Fine-tuned slope는 fx `{fine['slope_fx']:.6f}`, fy `{fine['slope_fy']:.6f}`이며 R²는 fx `{fine['r_squared_fx']:.6f}`, fy `{fine['r_squared_fy']:.6f}`이다. Prediction/GT std 비율은 fx `{fine['pred_std_over_gt_std_fx']:.6f}`, fy `{fine['pred_std_over_gt_std_fy']:.6f}`이다. 이 값들은 frame variation이 아니라 camera-pair GT variation에 대한 반응을 나타낸다.

## 8. 고정 5-bin 결과

기존 Demo-global GT fx 경계를 그대로 사용했다. Test pair count는 `{counts}`이며 빈 bin과 희소 bin을 합치거나 숨기지 않았다. 세 모델의 상세 오차와 prediction 분포는 `bin_level_results.csv`에 있다.

## 9. 주장할 수 있는 내용

좁은 focal 분포에서는 낮은 상대오차만으로 이미지 기반 focal variation 식별 능력을 충분히 설명하지 못할 가능성이 있다. Train constant baseline과 축별 slope, correlation, 표준편차 비율은 dataset prior와 실제 variation 추적을 구분하기 위한 근거를 제공한다.

## 10. 아직 주장할 수 없는 내용

이 결과만으로 fine-tuned 모델이 focal을 외웠다거나 이미지를 보지 않는다고 단정할 수 없다. 기존 1.617%가 잘못됐다는 결론도 아니다. 한 참가자와 49개 camera pair에 대한 사후 기술 분석이며 물리 카메라 독립 일반화를 증명하지 않는다.

## 11. 다음 일반화 실험

참가자와 calibration이 독립인 사전 확정 split, 더 넓은 focal support, 여러 seed, 실제 RGB 가용성 검증, 동일 평가 코드의 외부 데이터 적용이 필요하다.
"""


def readme_text():
    return """# GigaHands focal metric reaudit

This directory contains a CPU-only post-hoc analysis of existing GigaHands Demo Test predictions. No model was built, no checkpoint was loaded, and no training or inference was run.

Primary files:

- `summary.csv` and `summary.json`: model-level metrics and separate fx/fy response diagnostics
- `pair_level_results.csv`: equal-weight independent camera-pair results
- `bin_level_results.csv`: all precommitted five-bin rows, including empty bins
- `constant_baseline.json`: Train-only normalized focal median
- `resolution_audit.csv`: input population and resolution checks
- `metric_definition.md`: exact denominators and metric equations
- `analysis_report.md`: Korean research interpretation
- `source_integrity.json`: immutable input SHA256 and execution restrictions
- `figures/`: six grayscale presentation figures
"""


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def run_analysis(output, overwrite=False):
    pretrained_dir = HANDCALIB_ROOT / "runs/01_anycalib_pretrained/test"
    finetuned_dir = HANDCALIB_ROOT / "runs/02_anycalib_finetune/test"
    manifest_path = HANDCALIB_ROOT / "data/manifests/gigahands.csv"
    split_dir = HANDCALIB_ROOT / "data/splits/gigahands"
    source_paths = [
        pretrained_dir / "frame_predictions.csv.gz", pretrained_dir / "pair_summary.csv",
        pretrained_dir / "metrics.json", pretrained_dir / "metadata.json",
        finetuned_dir / "frame_predictions.csv.gz", finetuned_dir / "pair_summary.csv",
        finetuned_dir / "metrics.json", finetuned_dir / "metadata.json",
        manifest_path, split_dir / "train.txt", split_dir / "val.txt",
        split_dir / "test.txt", split_dir / "split_report.json",
    ]
    before = source_snapshot(source_paths)

    manifest = deduplicate_manifest(read_csv(manifest_path))
    if len(manifest) != 245:
        raise RuntimeError(f"expected 245 unique Demo camera pairs, found {len(manifest)}")
    resolution = validate_single_resolution(manifest)
    if resolution != (1280, 720):
        raise RuntimeError(f"expected Demo resolution 1280x720, found {resolution}")
    train_participants = set((split_dir / "train.txt").read_text(encoding="utf-8").split())
    train_rows = [row for row in manifest if row["participant"] in train_participants]
    if len(train_rows) != 147:
        raise RuntimeError(f"expected 147 unique Train camera pairs, found {len(train_rows)}")

    model_dirs = {"pretrained": pretrained_dir, "finetuned": finetuned_dir}
    model_pairs = {}
    reproduction = {}
    pair_summary_checks = {}
    for model, directory in model_dirs.items():
        frames = read_frame_csv(directory / "frame_predictions.csv.gz")
        if len(frames) != 7667:
            raise RuntimeError(f"expected 7667 {model} frames, found {len(frames)}")
        validate_single_resolution(frames)
        pairs = aggregate_frame_predictions(frames)
        if len(pairs) != 49:
            raise RuntimeError(f"expected 49 {model} pairs, found {len(pairs)}")
        reproduction[model] = validate_legacy_reproduction(pairs, read_json(directory / "metrics.json"))
        pair_summary_checks[model] = validate_pair_summary(pairs, read_csv(directory / "pair_summary.csv"))
        model_pairs[model] = pairs
    validate_model_pair_alignment(model_pairs)

    # The output tree is created only after both legacy reproduction checks pass.
    output = prepare_output(output, overwrite=overwrite)
    test_range = focal_range_definition(model_pairs["finetuned"])
    global_range = focal_range_definition(manifest)
    prior = train_constant_prior(train_rows)
    model_pairs["train_constant"] = constant_pairs(model_pairs["finetuned"], prior)
    model_pairs = {
        model: enrich_pairs(pairs, test_range, global_range)
        for model, pairs in model_pairs.items()
    }
    summary_rows = [model_summary(model, model_pairs[model]) for model in model_pairs]
    bins = bin_level_rows(model_pairs, LOCKED_FX_EDGES)
    pairs = pair_level_rows(model_pairs)

    summary = {
        "experiment": "gigahands_focal_metric_reaudit_20261014",
        "status": "completed",
        "analysis_type": "stored_prediction_posthoc_only",
        "model_training_runs": 0,
        "model_inference_runs": 0,
        "checkpoint_loads": 0,
        "gpu_operations": 0,
        "rgb_decodes": 0,
        "resolution": {"width": resolution[0], "height": resolution[1], "unique_resolution_count": 1},
        "population": {"demo_unique_camera_pairs": 245, "train_unique_camera_pairs": 147, "test_unique_camera_pairs": 49, "test_frames_per_model": 7667},
        "aggregation": "component-wise frame median per camera_key; equal weight across 49 camera pairs",
        "relative_error_denominator": "GT",
        "test_focal_range": test_range,
        "demo_global_focal_range": global_range,
        "fixed_five_bin_variable": "GT fx in raw pixels",
        "fixed_five_bin_edges": LOCKED_FX_EDGES,
        "legacy_reproduction": reproduction,
        "pair_summary_reproduction": pair_summary_checks,
        "constant_baseline": prior,
        "models": summary_rows,
    }

    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "pair_level_results.csv", pairs)
    write_csv(output / "bin_level_results.csv", bins)
    write_csv(output / "resolution_audit.csv", [
        {"scope": "Demo-global unique camera pairs", "row_count": 245, "width": 1280, "height": 720, "resolution_count": 1},
        {"scope": "Train unique camera pairs", "row_count": 147, "width": 1280, "height": 720, "resolution_count": 1},
        {"scope": "Test independent camera pairs", "row_count": 49, "width": 1280, "height": 720, "resolution_count": 1},
        {"scope": "Pretrained stored frames", "row_count": 7667, "width": 1280, "height": 720, "resolution_count": 1},
        {"scope": "Fine-tuned stored frames", "row_count": 7667, "width": 1280, "height": 720, "resolution_count": 1},
    ])
    (output / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "constant_baseline.json").write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
    (output / "metric_definition.md").write_text(metric_definition_text(test_range, global_range), encoding="utf-8")
    (output / "README.md").write_text(readme_text(), encoding="utf-8")
    (output / "analysis_report.md").write_text(report_text(summary, bins), encoding="utf-8")
    plot_figures(output, model_pairs, summary_rows, bins)

    verify_source_snapshot(before)
    integrity = {
        "source_sha256_before": before,
        "source_sha256_after": source_snapshot(source_paths),
        "all_sources_unchanged": True,
        "model_training_runs": 0,
        "model_inference_runs": 0,
        "checkpoint_loads": 0,
        "gpu_operations": 0,
        "rgb_decodes": 0,
    }
    (output / "source_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_analysis(args.output.resolve(), overwrite=args.overwrite)
    print("legacy_reproduction=PASS")
    for row in summary["models"]:
        print(
            f"{row['model']}: pairs={row['pair_count']} "
            f"relative_mean={row['relative_focal_percent_mean']:.6f}% "
            f"slope_fx={row['slope_fx']:.6f} slope_fy={row['slope_fy']:.6f}"
        )
    print(f"output={args.output.resolve()}")
    print("training=0 inference=0 checkpoint_load=0 gpu=0")


if __name__ == "__main__":
    main()
