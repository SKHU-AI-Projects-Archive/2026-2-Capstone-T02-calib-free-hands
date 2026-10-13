"""Audit GigaHands demo focal coverage without decoding video or running models."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


HANDCALIB_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = HANDCALIB_ROOT.parent
DEFAULT_MANIFEST = HANDCALIB_ROOT / "data/manifests/gigahands.csv"
DEFAULT_SPLIT_DIR = HANDCALIB_ROOT / "data/splits/gigahands"
DEFAULT_OUTPUT = HANDCALIB_ROOT / "runs/02_gigahands_focal_coverage_audit"
DEFAULT_RESULT = HANDCALIB_ROOT / "results/02i0_gigahands_focal_coverage_feasibility.yaml"
FULL_FIELDS = ("width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
INTRINSIC_FIELDS = ("width", "height", "fx", "fy", "cx", "cy")
FOCAL_FIELDS = ("fx", "fy", "nfx", "nfy", "hfov_deg", "vfov_deg", "log_nfx", "log_nfy")
SPLIT_NAMES = ("train", "val", "test")
BIN_COUNTS = (3, 5, 7, 10)
NEAR_MATCH_RELATIVE_TOLERANCES = (1e-4, 1e-3, 1e-2)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [convert_row(row) for row in csv.DictReader(handle)]


def convert_row(row):
    converted = dict(row)
    for field in ("frame_count", "width", "height"):
        converted[field] = int(converted[field])
    for field in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"):
        converted[field] = float(converted[field])
    return converted


def derive_focal(row):
    width, height = float(row["width"]), float(row["height"])
    fx, fy = float(row["fx"]), float(row["fy"])
    nfx, nfy = fx / width, fy / height
    return {
        "fx": fx,
        "fy": fy,
        "nfx": nfx,
        "nfy": nfy,
        "hfov_deg": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
        "vfov_deg": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
        "log_nfx": math.log(nfx),
        "log_nfy": math.log(nfy),
    }


def signature(row, fields):
    return tuple(int(row[field]) if field in ("width", "height") else float(row[field]) for field in fields)


def unique_rows(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], row)
    return list(result.values())


def dataset_inventory(rows):
    camera_rows = unique_rows(rows, "camera_key")
    videos = Counter(row["video_path"] for row in rows)
    keys = Counter(row["camera_key"] for row in rows)
    return {
        "row_count": len(rows),
        "total_frame_count": sum(int(row["frame_count"]) for row in rows),
        "participant_count": len({row["participant"] for row in rows}),
        "sequence_count": len({row["sequence"] for row in rows}),
        "video_count": len(rows),
        "unique_video_path_count": len(videos),
        "duplicate_video_path_count": sum(count - 1 for count in videos.values() if count > 1),
        "unique_camera_key_count": len(camera_rows),
        "duplicate_camera_key_row_count": sum(count - 1 for count in keys.values() if count > 1),
        "unique_camera_identifier_count": len({row["camera"] for row in camera_rows}),
        "unique_fx_count": len({float(row["fx"]) for row in camera_rows}),
        "unique_fy_count": len({float(row["fy"]) for row in camera_rows}),
        "unique_fx_fy_count": len({(float(row["fx"]), float(row["fy"])) for row in camera_rows}),
        "unique_exact_intrinsic_signature_count": len({signature(row, INTRINSIC_FIELDS) for row in camera_rows}),
        "unique_exact_full_signature_count": len({signature(row, FULL_FIELDS) for row in camera_rows}),
        "confirmed_physical_camera_group_count": 0,
        "actual_physical_camera_count": "unknown",
        "physical_camera_identity_status": "not_verifiable",
        "camera_key_physical_identity_unverified_count": len(camera_rows),
    }


def validate_manifest(rows):
    required = {
        "participant", "sequence", "camera", "camera_key", "video_path", "video_name",
        "frame_count", "width", "height", *FULL_FIELDS,
    }
    missing_metadata = 0
    invalid_focal = 0
    invalid_dimensions = 0
    inconsistent_camera_keys = 0
    grouped = defaultdict(list)
    for row in rows:
        missing_metadata += int(any(field not in row or row[field] in (None, "") for field in required))
        invalid_focal += int(not math.isfinite(float(row["fx"])) or not math.isfinite(float(row["fy"])) or float(row["fx"]) <= 0 or float(row["fy"]) <= 0)
        invalid_dimensions += int(int(row["width"]) <= 0 or int(row["height"]) <= 0 or int(row["frame_count"]) <= 0)
        grouped[row["camera_key"]].append(row)
    for group in grouped.values():
        inconsistent_camera_keys += int(len({signature(row, FULL_FIELDS) for row in group}) > 1)
    return {
        "missing_calibration_metadata_rows": missing_metadata,
        "invalid_focal_rows": invalid_focal,
        "invalid_dimension_rows": invalid_dimensions,
        "inconsistent_camera_key_calibration_count": inconsistent_camera_keys,
        "valid": not any((missing_metadata, invalid_focal, invalid_dimensions, inconsistent_camera_keys)),
    }


def distribution(values):
    values = [float(value) for value in values]
    ordered = sorted(values)
    quantiles = np.quantile(np.asarray(ordered), [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    result = {
        "count": len(ordered),
        "unique_count": len(set(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "std_population": statistics.pstdev(ordered),
        "variance_population": statistics.pvariance(ordered),
        "range": ordered[-1] - ordered[0],
    }
    for name, value in zip(("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"), quantiles):
        result[name] = float(value)
    result["iqr"] = result["q75"] - result["q25"]
    return result


def derived_rows(rows):
    result = []
    for row in rows:
        item = dict(row)
        item.update(derive_focal(row))
        result.append(item)
    return result


def assign_bin(value, edges):
    if value < edges[0] or value > edges[-1]:
        raise ValueError(f"value outside bin range: {value}")
    if value == edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(np.asarray(edges), value, side="right") - 1)


def equal_width_edges(values, bins):
    low, high = min(values), max(values)
    if low == high:
        return [low, high]
    return [float(value) for value in np.linspace(low, high, bins + 1)]


def quantile_edges(values, bins):
    edges = [float(value) for value in np.quantile(np.asarray(values), np.linspace(0, 1, bins + 1))]
    return edges


def histogram_counts(values, edges):
    counts = [0] * (len(edges) - 1)
    for value in values:
        counts[assign_bin(float(value), edges)] += 1
    return counts


def balance_feasibility(counts):
    total = sum(counts)
    minimum = min(counts) if counts else 0
    maximum = max(counts) if counts else 0
    retained = minimum * len(counts)
    discarded = total - retained
    return {
        "minimum_bin_count": minimum,
        "maximum_bin_count": maximum,
        "empty_bin_count": sum(count == 0 for count in counts),
        "balanced_per_bin": minimum,
        "balanced_subset_size": retained,
        "retained_groups": retained,
        "discarded_groups": discarded,
        "retention_rate": retained / total if total else 0.0,
        "discard_rate": discarded / total if total else 0.0,
        "maximum_oversampling_multiplier": (maximum / minimum) if minimum else None,
    }


def three_way_capacity(counts, minimum_per_split):
    return all(count >= 3 * minimum_per_split for count in counts)


def macro_evaluation_feasibility(counts):
    empty = sum(count == 0 for count in counts)
    return {
        "status": "TECHNICALLY_FEASIBLE" if empty == 0 else "TECHNICALLY_INFEASIBLE",
        "empty_bin_count": empty,
        "minimum_bin_count": min(counts) if counts else 0,
    }


def frame_retention_range(frame_counts_by_bin, groups_per_bin):
    if groups_per_bin <= 0:
        return {"minimum_retained_frames": 0, "maximum_retained_frames": 0}
    return {
        "minimum_retained_frames": sum(sum(sorted(counts)[:groups_per_bin]) for counts in frame_counts_by_bin),
        "maximum_retained_frames": sum(sum(sorted(counts, reverse=True)[:groups_per_bin]) for counts in frame_counts_by_bin),
    }


def participant_conflicts(groups):
    owners = defaultdict(set)
    for name, rows in groups.items():
        for row in rows:
            owners[row["participant"]].add(name)
    return sorted(participant for participant, names in owners.items() if len(names) > 1)


def compare_splits(left_name, left_rows, right_name, right_rows):
    def values(rows, field):
        return {row[field] for row in rows}

    return {
        "left": left_name,
        "right": right_name,
        "exact_intrinsic_signature_overlap": len({signature(row, INTRINSIC_FIELDS) for row in left_rows} & {signature(row, INTRINSIC_FIELDS) for row in right_rows}),
        "exact_full_signature_overlap": len({signature(row, FULL_FIELDS) for row in left_rows} & {signature(row, FULL_FIELDS) for row in right_rows}),
        "camera_identifier_overlap": len(values(left_rows, "camera") & values(right_rows, "camera")),
        "participant_overlap": len(values(left_rows, "participant") & values(right_rows, "participant")),
        "sequence_overlap": len(values(left_rows, "sequence") & values(right_rows, "sequence")),
        "video_path_overlap": len(values(left_rows, "video_path") & values(right_rows, "video_path")),
        "physical_camera_overlap": "not_verifiable",
    }


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_participants(split_dir):
    return {name: (split_dir / f"{name}.txt").read_text(encoding="utf-8").split() for name in SPLIT_NAMES}


def split_camera_rows(rows, participants):
    return {
        name: unique_rows([row for row in rows if row["participant"] in members], "camera_key")
        for name, members in participants.items()
    }


def distribution_rows(rows, unit_name, group_name="all"):
    enriched = derived_rows(rows)
    return [
        {"unit": unit_name, "group": group_name, "variable": field, **distribution([row[field] for row in enriched])}
        for field in FOCAL_FIELDS
    ]


def focal_distribution_table(all_rows, camera_rows):
    intrinsic_rows = unique_rows_by_signature(camera_rows, INTRINSIC_FIELDS)
    full_rows = unique_rows_by_signature(camera_rows, FULL_FIELDS)
    table = []
    table.extend(distribution_rows(all_rows, "video"))
    table.extend(distribution_rows(camera_rows, "camera_key"))
    table.extend(distribution_rows(intrinsic_rows, "exact_intrinsic_signature"))
    table.extend(distribution_rows(full_rows, "exact_full_signature"))
    for participant in sorted({row["participant"] for row in camera_rows}):
        table.extend(distribution_rows([row for row in camera_rows if row["participant"] == participant], "participant_camera_key", participant))
    return table


def unique_rows_by_signature(rows, fields):
    result = {}
    for row in rows:
        result.setdefault(signature(row, fields), row)
    return list(result.values())


def existing_split_table(split_rows, global_edges):
    table = []
    for split, rows in split_rows.items():
        enriched = derived_rows(rows)
        for field in FOCAL_FIELDS:
            item = {"split": split, "variable": field, **distribution([row[field] for row in enriched])}
            if field in global_edges:
                counts = histogram_counts([row[field] for row in enriched], global_edges[field])
                item.update({f"existing_bin_{index + 1}_count": count for index, count in enumerate(counts)})
                item["central_bin_rate"] = counts[len(counts) // 2] / len(rows)
            table.append(item)
    return table


def coverage_table(camera_rows):
    enriched = derived_rows(camera_rows)
    table = []
    for field in ("fx", "fy", "nfx", "nfy"):
        values = [row[field] for row in enriched]
        for bins in BIN_COUNTS:
            for method, edges in (("equal_width", equal_width_edges(values, bins)), ("quantile", quantile_edges(values, bins))):
                if len(set(edges)) != len(edges):
                    continue
                assignments = [assign_bin(row[field], edges) for row in enriched]
                for index in range(bins):
                    group = [row for row, assigned in zip(enriched, assignments) if assigned == index]
                    table.append({
                        "variable": field,
                        "method": method,
                        "bin_count": bins,
                        "bin_index": index + 1,
                        "lower_edge": edges[index],
                        "upper_edge": edges[index + 1],
                        "upper_inclusive": index == bins - 1,
                        "camera_key_count": len(group),
                        "exact_intrinsic_signature_count": len({signature(row, INTRINSIC_FIELDS) for row in group}),
                        "exact_full_signature_count": len({signature(row, FULL_FIELDS) for row in group}),
                        "participant_count": len({row["participant"] for row in group}),
                        "sequence_count": len({row["sequence"] for row in group}),
                        "video_count": sum(1 for row in all_manifest_rows if row["camera_key"] in {item["camera_key"] for item in group}),
                        "frame_count": sum(int(row["frame_count"]) for row in all_manifest_rows if row["camera_key"] in {item["camera_key"] for item in group}),
                    })
    return table


def balance_table(coverage):
    grouped = defaultdict(list)
    for row in coverage:
        grouped[(row["variable"], row["method"], row["bin_count"])].append(row)
    table = []
    for (variable, method, bins), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["bin_index"])
        counts = [row["camera_key_count"] for row in rows]
        frames = [row["frame_count"] for row in rows]
        info = balance_feasibility(counts)
        edges = [row["lower_edge"] for row in rows] + [rows[-1]["upper_edge"]]
        camera_frame_counts = defaultdict(int)
        for item in all_manifest_rows:
            camera_frame_counts[item["camera_key"]] += int(item["frame_count"])
        camera_derived = derived_rows(unique_rows(all_manifest_rows, "camera_key"))
        frames_by_bin = [
            [camera_frame_counts[item["camera_key"]] for item in camera_derived if assign_bin(item[variable], edges) == index]
            for index in range(bins)
        ]
        frame_range = frame_retention_range(frames_by_bin, info["balanced_per_bin"])
        table.append({
            "variable": variable,
            "method": method,
            "bin_count": bins,
            **info,
            "total_frame_count": sum(frames),
            **frame_range,
            "minimum_discarded_frames": sum(frames) - frame_range["maximum_retained_frames"],
            "maximum_discarded_frames": sum(frames) - frame_range["minimum_retained_frames"],
            "camera_key_three_way_at_least_1": three_way_capacity(counts, 1),
            "camera_key_three_way_at_least_2": three_way_capacity(counts, 2),
            "camera_key_three_way_at_least_5": three_way_capacity(counts, 5),
            "exact_intrinsic_three_way_at_least_1": three_way_capacity([row["exact_intrinsic_signature_count"] for row in rows], 1),
            "exact_full_three_way_at_least_1": three_way_capacity([row["exact_full_signature_count"] for row in rows], 1),
            "training_balance_interpretation": "group subsampling or inverse-frequency weighting candidate; not implemented",
            "evaluation_balance_interpretation": "bin-macro aggregation can retain all groups; metric not adopted",
        })
    return table


def near_match_counts(camera_rows):
    pairs = list(itertools.combinations(camera_rows, 2))
    result = []
    for tolerance in NEAR_MATCH_RELATIVE_TOLERANCES:
        count = 0
        for left, right in pairs:
            rel_fx = abs(float(left["fx"]) - float(right["fx"])) / max(abs(float(left["fx"])), abs(float(right["fx"])))
            rel_fy = abs(float(left["fy"]) - float(right["fy"])) / max(abs(float(left["fy"])), abs(float(right["fy"])))
            if max(rel_fx, rel_fy) <= tolerance and (float(left["fx"]), float(left["fy"])) != (float(right["fx"]), float(right["fy"])):
                count += 1
        result.append({"relative_tolerance": tolerance, "near_match_pair_count": count})
    return result


def train_range_analysis(split_rows):
    train = derived_rows(split_rows["train"])
    result = {}
    for field in ("fx", "fy", "nfx", "nfy"):
        low, high = min(row[field] for row in train), max(row[field] for row in train)
        median = statistics.median(row[field] for row in train)
        result[field] = {"train_min": low, "train_max": high, "train_median": median}
        for name in ("val", "test"):
            values = [derive_focal(row)[field] for row in split_rows[name]]
            inside = sum(low <= value <= high for value in values)
            differences = [abs(value - median) for value in values]
            result[field][f"{name}_inside_count"] = inside
            result[field][f"{name}_inside_rate"] = inside / len(values)
            result[field][f"{name}_outside_count"] = len(values) - inside
            result[field][f"{name}_median_absolute_difference"] = statistics.median(differences)
    return result


def participant_split_feasibility(camera_rows):
    participants = sorted({row["participant"] for row in camera_rows})
    enriched = derived_rows(camera_rows)
    output = []
    for bins in BIN_COUNTS:
        edges_by_field = {field: equal_width_edges([row[field] for row in enriched], bins) for field in ("fx", "fy")}
        for train in itertools.combinations(participants, 2):
            remaining = sorted(set(participants) - set(train))
            for val, test in itertools.permutations(remaining):
                assignments = {"train": set(train), "val": {val}, "test": {test}}
                record = {"bin_count": bins, "train_participants": list(train), "val_participant": val, "test_participant": test}
                all_minimums = []
                for field, edges in edges_by_field.items():
                    for split, members in assignments.items():
                        values = [row[field] for row in enriched if row["participant"] in members]
                        counts = histogram_counts(values, edges)
                        record[f"{field}_{split}_counts"] = counts
                        record[f"{field}_{split}_empty_bins"] = sum(count == 0 for count in counts)
                        all_minimums.append(min(counts))
                record["minimum_groups_per_split_bin"] = min(all_minimums)
                for threshold in (1, 2, 5):
                    record[f"at_least_{threshold}_per_split_bin"] = record["minimum_groups_per_split_bin"] >= threshold
                groups = {name: [row for row in camera_rows if row["participant"] in members] for name, members in assignments.items()}
                overlaps = [compare_splits(a, groups[a], b, groups[b]) for a, b in itertools.combinations(SPLIT_NAMES, 2)]
                record["participant_independent"] = not participant_conflicts(groups)
                record["exact_intrinsic_independent"] = all(item["exact_intrinsic_signature_overlap"] == 0 for item in overlaps)
                record["exact_full_independent"] = all(item["exact_full_signature_overlap"] == 0 for item in overlaps)
                record["physical_camera_independent"] = "not_verifiable"
                output.append(record)
    return output


def two_dimensional_coverage(camera_rows):
    enriched = derived_rows(camera_rows)
    result = []
    for xfield, yfield in (("fx", "fy"), ("nfx", "nfy")):
        for bins in BIN_COUNTS:
            xedges = equal_width_edges([row[xfield] for row in enriched], bins)
            yedges = equal_width_edges([row[yfield] for row in enriched], bins)
            occupied = {(assign_bin(row[xfield], xedges), assign_bin(row[yfield], yedges)) for row in enriched}
            result.append({"x": xfield, "y": yfield, "bins_per_axis": bins, "cell_count": bins * bins, "occupied_cell_count": len(occupied), "empty_cell_count": bins * bins - len(occupied), "occupancy_rate": len(occupied) / (bins * bins)})
    return result


def scenario_rows(summary):
    return [
        {"scenario": "S0", "description": "기존 분할과 원본 분포 유지", "data_retention": 1.0, "training_balance": "no", "evaluation_balance": "existing unit-equal metric", "technical_status": "TECHNICALLY_FEASIBLE", "academic_limit": "focal coverage imbalance remains"},
        {"scenario": "S1", "description": "기존 Test에서 focal-bin macro 후보", "data_retention": 1.0, "training_balance": "no", "evaluation_balance": "bin means receive equal weight", "technical_status": summary["macro_evaluation_status"], "academic_limit": "sparse bins and post-hoc protocol selection"},
        {"scenario": "S2", "description": "동일 수 calibration group 가상 추출", "data_retention": summary["fx_five_bin_retention"], "training_balance": "group-balanced subset", "evaluation_balance": "possible on retained groups", "technical_status": "CONDITIONALLY_FEASIBLE", "academic_limit": "large information loss and few independent extremes"},
        {"scenario": "S3", "description": "원본 Train과 bin-balanced sampling/weighting", "data_retention": 1.0, "training_balance": "weights or repeated exposure", "evaluation_balance": "unchanged unless separately defined", "technical_status": "CONDITIONALLY_FEASIBLE", "academic_limit": "oversampling adds no calibration diversity"},
        {"scenario": "S4", "description": "추가 데이터셋 후보", "data_retention": None, "training_balance": "not evaluated", "evaluation_balance": "requires new protocol", "technical_status": "UNKNOWN", "academic_limit": "domain confounding and new independence audit required"},
    ]


def compact_result(summary):
    fields = (
        "experiment", "status", "analysis_scope", "source_manifest", "source_manifest_sha256",
        "source_split_files", "source_split_hashes", "repository_commit", "dataset_population",
        "manifest_validation", "analysis_units", "focal_definitions", "distribution_summary",
        "participant_distribution", "existing_histogram", "existing_split_distribution",
        "train_range_analysis", "calibration_overlap", "near_match_sensitivity",
        "candidate_bin_definitions", "two_dimensional_coverage", "training_balance_assessment",
        "evaluation_balance_assessment", "existing_pair_balanced_sampler",
        "participant_split_feasibility", "physical_camera_identity_limitations",
        "statistical_limitations", "scenario_comparison", "decision_status", "next_stage_questions",
        "original_files_preserved", "training_runs", "inference_runs", "checkpoint_modifications",
        "existing_evaluation_results_modified",
    )
    result = {field: summary[field] for field in fields if field in summary}
    grouped = defaultdict(list)
    for row in summary.get("bin_coverage", []):
        grouped[(row["variable"], row["method"], row["bin_count"])].append(row)
    result["bin_coverage"] = {}
    for (variable, method, bins), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["bin_index"])
        result["bin_coverage"][f"{variable}_{method}_{bins}"] = {
            "edges": [row["lower_edge"] for row in rows] + [rows[-1]["upper_edge"]],
            "camera_key_counts": [row["camera_key_count"] for row in rows],
            "exact_intrinsic_signature_counts": [row["exact_intrinsic_signature_count"] for row in rows],
            "exact_full_signature_counts": [row["exact_full_signature_count"] for row in rows],
        }
    keep = (
        "variable", "method", "bin_count", "minimum_bin_count", "maximum_bin_count",
        "empty_bin_count", "balanced_per_bin", "retained_groups", "discarded_groups",
        "retention_rate", "discard_rate", "maximum_oversampling_multiplier",
        "minimum_retained_frames", "maximum_retained_frames", "minimum_discarded_frames",
        "maximum_discarded_frames", "camera_key_three_way_at_least_1",
        "camera_key_three_way_at_least_2", "camera_key_three_way_at_least_5",
        "exact_intrinsic_three_way_at_least_1", "exact_full_three_way_at_least_1",
    )
    result["balancing_feasibility"] = {
        name: {field: row[field] for field in keep if field in row}
        for name, row in summary.get("balancing_feasibility", {}).items()
    }
    return result


def plot_figures(output, camera_rows, split_rows, global_edges, balance_rows):
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    enriched = derived_rows(camera_rows)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, field in zip(axes, ("fx", "fy")):
        values = [row[field] for row in enriched]
        axis.hist(values, bins=30, color="0.75", edgecolor="black")
        for edge in global_edges[field]: axis.axvline(edge, color="black", linestyle="--", linewidth=0.8)
        counts = histogram_counts(values, global_edges[field])
        axis.set_title(f"GigaHands {field} (5-bin counts: {counts})")
        axis.set_xlabel(f"{field} (pixels)"); axis.set_ylabel("Calibration units")
    save_figure(fig, figure_dir / "01_overall_focal_distribution")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    shades = {"train": "0.25", "val": "0.55", "test": "0.8"}
    for axis, field in zip(axes, ("fx", "fy")):
        x = np.arange(5); width = 0.24
        for offset, name in enumerate(SPLIT_NAMES):
            counts = histogram_counts([float(row[field]) for row in split_rows[name]], global_edges[field])
            rates = np.asarray(counts) / len(split_rows[name])
            axis.bar(x + (offset - 1) * width, rates, width, label=name, color=shades[name], edgecolor="black")
        axis.set_title(f"Locked split: {field}"); axis.set_ylabel("Proportion"); axis.set_xticks(x, [str(i) for i in range(1, 6)]); axis.legend()
    axes[-1].set_xlabel("Existing equal-width bin")
    save_figure(fig, figure_dir / "02_locked_split_distribution")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, field, label in zip(axes.flat, ("nfx", "nfy", "hfov_deg", "vfov_deg"), ("fx / width", "fy / height", "HFOV (degrees)", "VFOV (degrees)")):
        axis.hist([row[field] for row in enriched], bins=25, color="0.7", edgecolor="black")
        axis.set_xlabel(label); axis.set_ylabel("Calibration units")
    save_figure(fig, figure_dir / "03_normalized_focal_and_fov")

    fig, axis = plt.subplots(figsize=(7, 6))
    markers = {"p36": "o", "p41": "s", "p44": "^", "p52": "x"}
    for participant, marker in markers.items():
        group = [row for row in enriched if row["participant"] == participant]
        style = {"marker": marker, "color": "black", "label": participant}
        if marker != "x":
            style.update({"facecolors": "none", "edgecolors": "black"})
        axis.scatter([row["fx"] for row in group], [row["fy"] for row in group], **style)
    axis.set_xlabel("fx (pixels)"); axis.set_ylabel("fy (pixels)"); axis.legend(); axis.set_title("Calibration-unit focal pairs")
    save_figure(fig, figure_dir / "04_fx_fy_scatter")

    equal_fx = [row for row in balance_rows if row["variable"] == "fx" and row["method"] == "equal_width"]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(equal_fx)); width = 0.25
    axis.bar(x - width, [row["minimum_bin_count"] for row in equal_fx], width, label="minimum", color="0.2")
    axis.bar(x, [row["maximum_bin_count"] for row in equal_fx], width, label="maximum", color="0.55")
    axis.bar(x + width, [row["balanced_subset_size"] for row in equal_fx], width, label="balanced subset", color="0.8", edgecolor="black")
    axis.set_xticks(x, [str(row["bin_count"]) for row in equal_fx]); axis.set_xlabel("Equal-width fx bins"); axis.set_ylabel("Calibration units"); axis.legend()
    save_figure(fig, figure_dir / "05_bin_sensitivity")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(equal_fx)); retained = [row["retained_groups"] for row in equal_fx]; discarded = [row["discarded_groups"] for row in equal_fx]
    axis.bar(x, retained, label="retained", color="0.25")
    axis.bar(x, discarded, bottom=retained, label="discarded", color="0.8", edgecolor="black")
    axis.set_xticks(x, [str(row["bin_count"]) for row in equal_fx]); axis.set_xlabel("Equal-width fx bins (hypothetical)"); axis.set_ylabel("Calibration units"); axis.legend()
    save_figure(fig, figure_dir / "06_hypothetical_balance_information_loss")


def save_figure(fig, base):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def markdown_report(summary):
    inventory = summary["dataset_population"]
    fx = summary["distribution_summary"]["camera_key"]["fx"]
    fy = summary["distribution_summary"]["camera_key"]["fy"]
    five = summary["balancing_feasibility"]["fx_equal_width_5"]
    independent = summary["participant_split_feasibility"]
    focal_lines = []
    for field in ("fx", "fy", "nfx", "nfy", "hfov_deg", "vfov_deg"):
        item = summary["distribution_summary"]["camera_key"][field]
        focal_lines.append(f"| {field} | {item['min']:.6f} | {item['mean']:.6f} | {item['median']:.6f} | {item['std_population']:.6f} | {item['max']:.6f} |")
    balance_lines = []
    for bins in BIN_COUNTS:
        x = summary["balancing_feasibility"][f"fx_equal_width_{bins}"]
        y = summary["balancing_feasibility"][f"fy_equal_width_{bins}"]
        balance_lines.append(f"| {bins} | fx | {x['minimum_bin_count']} | {x['maximum_bin_count']} | {x['retained_groups']} | {x['discarded_groups']} | {x['retention_rate']:.2%} |")
        balance_lines.append(f"| {bins} | fy | {y['minimum_bin_count']} | {y['maximum_bin_count']} | {y['retained_groups']} | {y['discarded_groups']} | {y['retention_rate']:.2%} |")
    overlap_lines = [f"| {row['left']}-{row['right']} | {row['exact_intrinsic_signature_overlap']} | {row['exact_full_signature_overlap']} | {row['camera_identifier_overlap']} | {row['video_path_overlap']} | {row['physical_camera_overlap']} |" for row in summary["calibration_overlap"]]
    participant_lines = [f"| {bins} | {item['candidate_assignment_count']} | {item['at_least_1_per_split_bin']} | {item['at_least_2_per_split_bin']} | {item['at_least_5_per_split_bin']} | {item['at_least_1_and_exact_full_independent']} |" for bins, item in independent["by_bin_count"].items()]
    scenario_lines = [f"| {row['scenario']} | {row['description']} | {row['technical_status']} | {row['academic_limit']} |" for row in summary["scenario_comparison"]]
    return f"""# 02-I.0 GigaHands focal 분포 및 균형 실험 가능성 감사

## 1. 연구 목적

현재 GigaHands 데모의 focal 범위와 독립 보정 표본 수를 확인하고, 새로운 학습이나 평가를 실행하기 전에 구간 균형 실험이 가능한지 검토한다.

## 2. 문제 정의 및 기존 실험의 한계

기존 분할은 participant를 분리하면서 전체 focal histogram과 비슷한 분포를 선택했다. 그러나 분포 자체가 중앙 구간에 집중되어 있으므로 다양한 focal 범위를 균등하게 대표한다는 뜻은 아니다.

## 3. 연구 질문과 검증 가능한 가설

같은 frame과 video 수가 많아도 독립 camera calibration 수는 늘지 않는다고 구분한다. 동일 폭 focal 구간의 최소 독립 표본 수, 정보 손실, participant 독립 분할 가능성을 데이터로 검사한다.

## 4. 분석 데이터와 범위

분석 모집단은 공식 전체 GigaHands가 아니라 현재 저장소의 데모 하위집합이다. Participant {inventory['participant_count']}명, sequence {inventory['sequence_count']}개, video {inventory['video_count']}개, frame {inventory['total_frame_count']:,}개, camera key {inventory['unique_camera_key_count']}개다.

## 5. 분석 단위 및 용어 정의

- Frame은 한 영상의 장면이며 독립 카메라 표본으로 계산하지 않는다.
- Video는 manifest의 한 행이다.
- Camera calibration unit은 sequence-camera 조합인 `camera_key`다.
- Exact intrinsic signature는 width, height, fx, fy, cx, cy가 모두 같은 경우다.
- Exact full calibration signature는 intrinsic에 k1, k2, p1, p2까지 같은 경우다.
- 실제 물리 카메라 identity와 수는 metadata만으로 검증할 수 없다. Confirmed group 0은 실제 카메라가 0대라는 뜻이 아니다.

## 6. 분석 방법

Camera key를 동일 가중치로 사용했다. 분산은 모집단 분산이다. 동일 폭 3/5/7/10-bin과 quantile 진단을 비교했고 마지막 bin은 최댓값을 포함한다. 모델 성능은 경계나 후보 선택에 사용하지 않았다.

## 7. 통제 변수와 비교 변수

원본 manifest와 locked split은 고정했다. 비교 변수는 fx, fy, 정규화 focal, HFOV, VFOV, bin 수, grouping 단위다.

## 8. 평가 및 판단 기준

산술적 최소 조건은 모든 split과 bin에 1/2/5개의 calibration group이 존재하는지다. 이는 학술적으로 충분한 표본 수를 뜻하지 않는다. 물리 카메라 독립성은 `not_verifiable`로 유지한다.

## 9. Focal 분포 분석 결과

Camera-key 기준 fx는 {fx['min']:.3f}~{fx['max']:.3f} pixel, 평균 {fx['mean']:.3f}, 중앙값 {fx['median']:.3f}이다. fy는 {fy['min']:.3f}~{fy['max']:.3f} pixel, 평균 {fy['mean']:.3f}, 중앙값 {fy['median']:.3f}이다. 기존 fx 5-bin 중앙 구간에는 {summary['existing_histogram']['fx_counts'][2]}/{inventory['unique_camera_key_count']}개가 존재한다.

| 변수 | 최솟값 | 평균 | 중앙값 | 모집단 표준편차 | 최댓값 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(focal_lines)}

Train 범위에는 Validation과 Test의 fx, fy, nfx, nfy가 각각 49/49개 포함된다. 이는 범위 포함 여부만 뜻하며, 분포 균형이나 독립성을 증명하지 않는다.

## 10. Calibration 중복 분석 결과

고유 exact intrinsic signature는 {inventory['unique_exact_intrinsic_signature_count']}개, full signature는 {inventory['unique_exact_full_signature_count']}개다. 기존 Train-Validation full calibration overlap은 {summary['calibration_overlap'][0]['exact_full_signature_overlap']}개다. Participant 분리는 물리 카메라 분리를 증명하지 않는다.

| 비교 | Exact intrinsic | Exact full | Camera identifier | Video path | Physical camera |
|---|---:|---:|---:|---:|---|
{chr(10).join(overlap_lines)}

49개 camera identifier는 모든 split에서 반복된다. 이는 이름 중복이며 동일 물리 장치라는 증거는 아니다. 실제 물리 카메라 수와 identity는 `not_verifiable`이다.

## 11. 구간 균형 가능성 분석 결과

fx 5-bin의 최소/최대 수는 {five['minimum_bin_count']}/{five['maximum_bin_count']}개다. 비복원 균형 subset은 최대 {five['retained_groups']}개이며 {five['discarded_groups']}개를 제외한다. 이 계산은 가상 가능성 분석이며 실제 split이나 subset을 만들지 않았다.

| Bin 수 | 축 | 최소 bin | 최대 bin | 균형 subset | 제외 group | 유지율 |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(balance_lines)}

| Bin 수 | 2/1/1 후보 | split/bin 1개 이상 | 2개 이상 | 5개 이상 | 1개 이상 + exact full 독립 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(participant_lines)}

3-bin은 6개 participant 배치에서 split/bin 최소 1개를 산술적으로 만족하지만 최소 2개를 만족하는 배치는 없다. 그 6개 중 exact full calibration까지 split 간 독립인 배치는 0개다. 5/7/10-bin은 최소 1개 조건도 만족하는 participant 배치가 없다.

학습 균형과 평가 균형은 다르다. 학습에서는 group subsampling 또는 bin weighting으로 노출 비중을 조정할 수 있지만 희소 calibration의 다양성은 늘지 않는다. 평가는 모든 데이터를 보존한 채 bin별 평균을 같은 가중치로 합칠 수 있지만, 희소 bin의 불확실성과 기존 지표와 다른 평가량이라는 문제가 남는다.

기존 Test의 전역 5-bin count는 fx `[0, 3, 43, 3, 0]`, fy `[0, 6, 41, 2, 0]`이다. 양 끝 bin이 비어 있으므로 사전에 정한 5개 전역 구간 전체의 macro 평균은 현재 Test에서 계산할 수 없다. 관측된 bin만 남기거나 경계를 바꾸는 것은 새로운 평가 정의이며 이번 작업에서 채택하지 않았다.

## 12. 학술적 해석

현재 데모에서 focal 분포 편중은 확인된다. 균형 sampling은 분포 기여도를 바꿀 수 있지만 일반화 향상을 증명하지 않는다. 동일한 calibration의 frame 반복은 pseudoreplication이며 독립 표본 증가가 아니다.

| 시나리오 | 내용 | 기술 판정 | 핵심 한계 |
|---|---|---|---|
{chr(10).join(scenario_lines)}

## 13. 한계

- Selection bias: 이미 본 Test 성능을 구간 선택에 사용하지 않았지만, 기존 Test metadata를 탐색했다.
- Distribution mismatch: 극단 구간의 독립 calibration이 적다.
- Calibration leakage: exact overlap은 계산했으나 물리 장치 중복은 확인할 수 없다.
- Sampling uncertainty: 희소 bin 평균은 불안정하다.
- Domain confounding: 추가 데이터셋은 focal 외 환경도 바꾼다.
- Metric comparability: 기존 unit 평균과 bin-macro 평균은 직접 비교할 수 없다.
- Post-hoc protocol selection: 후속 확증 평가는 새 독립 protocol이 필요하다.

## 14. 다음 단계에서 결정할 사항

최종 focal 축과 bin 경계, 학습 weighting 여부, 평가 macro 지표의 보조 사용 여부, 최소 독립 calibration 수, 추가 데이터 확보 여부를 사전에 결정해야 한다.

## 15. 결론

데이터 감사는 완료했다. GigaHands 데모만으로 학습 노출 비중을 조정하는 것은 조건부로 가능하다. 그러나 다양한 focal 구간마다 충분한 독립 표본을 두고 Train/Validation/Test를 모두 균형 있게 구성하는 것은 {independent['academic_verdict_ko']} 물리 카메라 독립성도 확인할 수 없으므로 추가 calibration과 독립 평가 데이터가 필요하다.
"""


def protected_paths(manifest, split_dir):
    candidates = [
        manifest,
        split_dir / "train.txt", split_dir / "val.txt", split_dir / "test.txt",
        split_dir / "split_report.json", split_dir / "split_report.md",
        HANDCALIB_ROOT / "configs/01_anycalib_pretrained.yaml",
        HANDCALIB_ROOT / "configs/02_anycalib_finetune.yaml",
        HANDCALIB_ROOT / "runs/02_anycalib_finetune/train/checkpoint_best.pt",
        HANDCALIB_ROOT / "runs/01_anycalib_pretrained/test/pair_summary.csv",
        HANDCALIB_ROOT / "runs/02_anycalib_finetune/test/pair_summary.csv",
        HANDCALIB_ROOT / "runs/02_interhand_external_eval/pretrained/pair_summary.csv",
        HANDCALIB_ROOT / "runs/02_interhand_external_eval/gigahands_finetuned/pair_summary.csv",
    ]
    return [path for path in candidates if path.exists()]


def audit(manifest, split_dir, output, result_yaml, overwrite=False):
    global all_manifest_rows
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"; tables.mkdir(exist_ok=True)
    source_hashes_before = {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in protected_paths(manifest, split_dir)}

    all_manifest_rows = read_manifest(manifest)
    camera_rows = unique_rows(all_manifest_rows, "camera_key")
    inventory = dataset_inventory(all_manifest_rows)
    validation = validate_manifest(all_manifest_rows)
    participants = split_participants(split_dir)
    split_rows = split_camera_rows(all_manifest_rows, participants)
    split_all_rows = {name: [row for row in all_manifest_rows if row["participant"] in members] for name, members in participants.items()}
    split_report = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))
    global_edges = {"fx": split_report["histogram"]["fx_bin_edges"], "fy": split_report["histogram"]["fy_bin_edges"]}
    existing_counts = {field: histogram_counts([float(row[field]) for row in camera_rows], edges) for field, edges in global_edges.items()}
    expected_counts = {field: split_report["histogram"]["global"][field]["global"]["counts"] for field in ("fx", "fy")}
    if existing_counts != expected_counts:
        raise RuntimeError(f"Existing histogram reproduction failed: {existing_counts} != {expected_counts}")

    inventory_rows = [{"metric": key, "value": value} for key, value in {**inventory, **validation}.items()]
    focal_rows = focal_distribution_table(all_manifest_rows, camera_rows)
    split_distribution = existing_split_table(split_rows, global_edges)
    overlaps = [compare_splits(a, split_rows[a], b, split_rows[b]) for a, b in itertools.combinations(SPLIT_NAMES, 2)]
    coverage = coverage_table(camera_rows)
    balances = balance_table(coverage)
    participant_candidates = participant_split_feasibility(camera_rows)
    two_d = two_dimensional_coverage(camera_rows)

    five_fx = next(row for row in balances if row["variable"] == "fx" and row["method"] == "equal_width" and row["bin_count"] == 5)
    one_group = [row for row in participant_candidates if row["at_least_1_per_split_bin"]]
    two_groups = [row for row in participant_candidates if row["at_least_2_per_split_bin"]]
    five_groups = [row for row in participant_candidates if row["at_least_5_per_split_bin"]]
    exact_full_independent = [row for row in one_group if row["exact_full_independent"]]
    participant_by_bin = {}
    for bins in BIN_COUNTS:
        candidates = [row for row in participant_candidates if row["bin_count"] == bins]
        participant_by_bin[str(bins)] = {
            "candidate_assignment_count": len(candidates),
            "at_least_1_per_split_bin": sum(row["at_least_1_per_split_bin"] for row in candidates),
            "at_least_2_per_split_bin": sum(row["at_least_2_per_split_bin"] for row in candidates),
            "at_least_5_per_split_bin": sum(row["at_least_5_per_split_bin"] for row in candidates),
            "at_least_1_and_exact_full_independent": sum(row["at_least_1_per_split_bin"] and row["exact_full_independent"] for row in candidates),
        }
    participant_summary = {
        "candidate_assignment_count_across_all_bin_counts": len(participant_candidates),
        "candidate_assignment_count_per_bin_count": 12,
        "by_bin_count": participant_by_bin,
        "assignments_with_at_least_1_per_split_bin": len(one_group),
        "assignments_with_at_least_2_per_split_bin": len(two_groups),
        "assignments_with_at_least_5_per_split_bin": len(five_groups),
        "assignments_with_minimum_coverage_and_exact_full_independence": len(exact_full_independent),
        "physical_camera_independence": "not_verifiable",
        "technical_verdict": "CONDITIONALLY_FEASIBLE" if one_group else "TECHNICALLY_INFEASIBLE",
        "academic_verdict": "INSUFFICIENT_FOR_RELIABLE_INDEPENDENT_BALANCED_EVALUATION",
        "academic_verdict_ko": "현재 수치로는 학술적으로 신뢰할 만한 독립 균형 평가에 충분하지 않다.",
    }
    dist_by_unit = {}
    for row in focal_rows:
        if row["group"] == "all":
            dist_by_unit.setdefault(row["unit"], {})[row["variable"]] = {k: v for k, v in row.items() if k not in ("unit", "group", "variable")}
    test_macro_fx = macro_evaluation_feasibility(histogram_counts([float(row["fx"]) for row in split_rows["test"]], global_edges["fx"]))
    test_macro_fy = macro_evaluation_feasibility(histogram_counts([float(row["fy"]) for row in split_rows["test"]], global_edges["fy"]))
    macro_status = "CONDITIONALLY_FEASIBLE" if test_macro_fx["status"] == test_macro_fy["status"] == "TECHNICALLY_FEASIBLE" else "TECHNICALLY_INFEASIBLE_FOR_ALL_EXISTING_5_BINS"
    scenario_input = {"macro_evaluation_status": macro_status, "fx_five_bin_retention": five_fx["retention_rate"]}
    scenarios = scenario_rows(scenario_input)

    write_csv(tables / "dataset_inventory.csv", inventory_rows)
    write_csv(tables / "focal_distribution_summary.csv", focal_rows)
    write_csv(tables / "existing_split_distribution.csv", split_distribution)
    write_csv(tables / "calibration_overlap.csv", overlaps)
    write_csv(tables / "bin_coverage.csv", coverage)
    write_csv(tables / "balance_feasibility.csv", balances)
    write_csv(tables / "scenario_comparison.csv", scenarios)
    write_csv(tables / "participant_split_feasibility.csv", [{**row, **{k: json.dumps(v) for k, v in row.items() if isinstance(v, list)}} for row in participant_candidates])
    write_csv(tables / "two_dimensional_coverage.csv", two_d)
    write_csv(tables / "near_match_sensitivity.csv", near_match_counts(camera_rows))
    plot_figures(output, camera_rows, split_rows, global_edges, balances)

    summary = {
        "experiment": "02i0_gigahands_focal_coverage_feasibility",
        "status": "FOCAL_DISTRIBUTION_AUDIT_COMPLETE",
        "analysis_scope": "current local GigaHands demo subset used by experiment 02; not the full official GigaHands dataset",
        "source_manifest": str(manifest.relative_to(HANDCALIB_ROOT)),
        "source_manifest_sha256": sha256_file(manifest),
        "source_split_files": {name: str((split_dir / f"{name}.txt").relative_to(HANDCALIB_ROOT)) for name in SPLIT_NAMES},
        "source_split_hashes": {name: sha256_file(split_dir / f"{name}.txt") for name in SPLIT_NAMES},
        "repository_commit": git_output("rev-parse", "HEAD"),
        "dataset_population": inventory,
        "manifest_validation": validation,
        "analysis_units": {"frame": inventory["total_frame_count"], "video": inventory["video_count"], "camera_key": inventory["unique_camera_key_count"], "exact_intrinsic_signature": inventory["unique_exact_intrinsic_signature_count"], "exact_full_calibration_signature": inventory["unique_exact_full_signature_count"], "unique_camera_identifier": inventory["unique_camera_identifier_count"], "physical_camera_identity": "not_verifiable", "confirmed_physical_camera_groups": 0, "actual_physical_camera_count": "unknown"},
        "focal_definitions": {"fx": "pixels", "fy": "pixels", "nfx": "fx / width", "nfy": "fy / height", "hfov_deg": "degrees(2 * atan(width / (2 * fx)))", "vfov_deg": "degrees(2 * atan(height / (2 * fy)))", "single_axis_status": "candidate_only; no final representative focal variable adopted"},
        "distribution_summary": dist_by_unit,
        "participant_distribution": {row["group"]: {item["variable"]: {k: v for k, v in item.items() if k not in ("unit", "group", "variable")} for item in focal_rows if item["group"] == row["group"]} for row in focal_rows if row["unit"] == "participant_camera_key"},
        "existing_histogram": {"fx_edges": global_edges["fx"], "fx_counts": existing_counts["fx"], "fy_edges": global_edges["fy"], "fy_counts": existing_counts["fy"], "reproduced": True},
        "existing_split_distribution": {name: {field: distribution([derive_focal(row)[field] for row in rows]) for field in ("fx", "fy", "nfx", "nfy", "hfov_deg", "vfov_deg")} for name, rows in split_rows.items()},
        "train_range_analysis": train_range_analysis(split_rows),
        "calibration_overlap": overlaps,
        "near_match_sensitivity": near_match_counts(camera_rows),
        "candidate_bin_definitions": {"methods": ["equal_width", "quantile_diagnostic"], "variables": ["fx", "fy", "nfx", "nfy"], "bin_counts": list(BIN_COUNTS), "last_bin_includes_maximum": True, "selection_status": "not_finalized"},
        "bin_coverage": coverage,
        "two_dimensional_coverage": two_d,
        "balancing_feasibility": {f"{row['variable']}_{row['method']}_{row['bin_count']}": row for row in balances},
        "training_balance_assessment": {"group_subsampling": "conditionally feasible but may discard most central-bin groups", "bin_weighting_or_sampling": "technically feasible without deleting data; repeated exposure does not add independent calibration", "implemented": False},
        "evaluation_balance_assessment": {"bin_macro_candidate": "technically computable only where every precommitted target bin is nonempty", "existing_test_fx_five_bin": test_macro_fx, "existing_test_fy_five_bin": test_macro_fy, "data_retention": "all groups can be retained", "limitations": "existing Test has empty global extreme bins; different estimand, sparse-bin uncertainty, post-hoc selection risk", "adopted": False},
        "existing_pair_balanced_sampler": {"frames_per_camera_key": 174, "sampling": "without replacement within each camera_key per epoch, then globally shuffled", "balances": "frame exposure per camera_key", "does_not_balance": "focal bins or independent calibration signatures"},
        "participant_split_feasibility": participant_summary,
        "physical_camera_identity_limitations": {"status": "not_verifiable", "confirmed_group_count": 0, "actual_count": "unknown", "camera_identifier_overlap_reported_separately": True, "calibration_overlap_reported_separately": True},
        "statistical_limitations": ["pseudoreplication", "selection_bias", "distribution_mismatch", "calibration_leakage", "sampling_uncertainty", "domain_confounding", "metric_comparability", "post_hoc_protocol_selection"],
        "scenario_comparison": scenarios,
        "decision_status": {"new_bin_definition_finalized": False, "new_split_finalized": False, "new_metric_adopted": False, "overall": "CONDITIONALLY_FEASIBLE_FOR_TRAINING_REWEIGHTING_BUT_INSUFFICIENT_FOR_RELIABLE_INDEPENDENT_BALANCED_EVALUATION"},
        "next_stage_questions": ["Which focal variable and precommitted bin definition should be used?", "What minimum independent calibration count is required per split and bin?", "Should training balance use weighting or group subsampling?", "Should bin-macro reporting remain a secondary metric?", "Is additional independent camera data required?"],
        "original_files_preserved": True,
        "training_runs": 0,
        "inference_runs": 0,
        "checkpoint_modifications": 0,
        "existing_evaluation_results_modified": 0,
    }
    metadata = {
        "command": list(os.sys.argv),
        "started_and_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": summary["repository_commit"],
        "deterministic": True,
        "random_sampling": False,
        "gpu_used": False,
        "video_decoding": False,
        "source_sha256_before": source_hashes_before,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "analysis_report.md").write_text(markdown_report(summary), encoding="utf-8")
    result_yaml.parent.mkdir(parents=True, exist_ok=True)
    result_yaml.write_text(yaml.safe_dump(compact_result(summary), allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    source_hashes_after = {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in protected_paths(manifest, split_dir)}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Protected input or existing result changed during audit")
    metadata["source_sha256_after"] = source_hashes_after
    metadata["source_files_preserved"] = True
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Audit GigaHands demo focal coverage and balance feasibility.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-yaml", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--overwrite", action="store_true", help="Replace files only inside the selected audit output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = audit(args.manifest.resolve(), args.split_dir.resolve(), args.output.resolve(), args.result_yaml.resolve(), args.overwrite)
    print(f"status={summary['status']}")
    print(f"output={args.output.resolve()}")
    print(f"result_yaml={args.result_yaml.resolve()}")
    print("training_runs=0 inference_runs=0 new_split_finalized=false")


if __name__ == "__main__":
    main()
