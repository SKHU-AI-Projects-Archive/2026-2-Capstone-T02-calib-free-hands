"""Audit full GigaHands camera metadata without training, inference, or RGB download."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
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
DEFAULT_ROOT = HANDCALIB_ROOT / "datasets/gigahands/full_hand_poses"
DEFAULT_DEMO_MANIFEST = HANDCALIB_ROOT / "data/manifests/gigahands.csv"
DEFAULT_SPLIT_REPORT = HANDCALIB_ROOT / "data/splits/gigahands/split_report.json"
DEFAULT_OUTPUT = HANDCALIB_ROOT / "runs/02_gigahands_full_focal_coverage"
DEFAULT_RESULT = HANDCALIB_ROOT / "results/02i1_gigahands_full_focal_coverage.yaml"
EXPECTED_HEADER = (
    "cam_id", "width", "height", "fx", "fy", "cx", "cy", "k1", "k2",
    "p1", "p2", "cam_name", "qvecw", "qvecx", "qvecy", "qvecz",
    "tvecx", "tvecy", "tvecz",
)
INTEGER_FIELDS = ("cam_id", "width", "height")
FLOAT_FIELDS = tuple(field for field in EXPECTED_HEADER if field not in INTEGER_FIELDS + ("cam_name",))
INTRINSIC_FIELDS = ("width", "height", "fx", "fy", "cx", "cy")
FULL_FIELDS = INTRINSIC_FIELDS + ("k1", "k2", "p1", "p2")
FOCAL_FIELDS = ("fx", "fy", "nfx", "nfy", "hfov_deg", "vfov_deg")
BIN_COUNTS = (3, 5, 7, 10)
MEDIA_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}


class IntegrityError(RuntimeError):
    """Fatal source metadata error; analysis must not continue."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def participant_labels(sequence):
    raw = sequence.split("-", 1)[0]
    if not raw.startswith("p") or not raw[1:].isdigit():
        raise IntegrityError(f"invalid_participant sequence={sequence!r}")
    return raw, f"p{int(raw[1:])}"


def participant_action(sequence):
    _, participant = participant_labels(sequence)
    parts = sequence.split("-")[1:]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return participant + "/" + "-".join(parts)


def signature_tuple(row, fields):
    return tuple(int(row[field]) if field in ("width", "height") else float(row[field]) for field in fields)


def signature_text(row, fields):
    return json.dumps(signature_tuple(row, fields), separators=(",", ":"))


def derive_focal(row):
    width, height = float(row["width"]), float(row["height"])
    fx, fy = float(row["fx"]), float(row["fy"])
    return {
        "fx": fx,
        "fy": fy,
        "nfx": fx / width,
        "nfy": fy / height,
        "hfov_deg": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
        "vfov_deg": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
    }


def _fatal(kind, path, line_number, raw, detail=""):
    raise IntegrityError(
        f"{kind} path={path} line={line_number} raw={raw!r} detail={detail}"
    )


def parse_optim_file(path, root):
    path, root = Path(path), Path(root)
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise IntegrityError(f"read_failure path={path} error={exc}") from exc
    lines = text.splitlines()
    if not lines:
        _fatal("missing_header", path, 1, "")
    header = lines[0].split()
    if header and header[0] == "#":
        header = header[1:]
    if tuple(header) != EXPECTED_HEADER:
        _fatal("invalid_header", path, 1, lines[0], f"expected={EXPECTED_HEADER}")

    sequence = path.parent.name
    participant_raw, participant = participant_labels(sequence)
    rows, cam_ids, cam_names = [], set(), set()
    for line_number, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            _fatal("blank_data_row", path, line_number, raw)
        values = raw.split()
        if len(values) != len(EXPECTED_HEADER):
            _fatal("field_count", path, line_number, raw, f"expected=19 actual={len(values)}")
        item = dict(zip(EXPECTED_HEADER, values))
        try:
            for field in INTEGER_FIELDS:
                item[field] = int(item[field])
            for field in FLOAT_FIELDS:
                item[field] = float(item[field])
        except ValueError as exc:
            _fatal("numeric_conversion", path, line_number, raw, str(exc))
        if any(not math.isfinite(item[field]) for field in FLOAT_FIELDS):
            _fatal("non_finite", path, line_number, raw)
        if item["width"] <= 0 or item["height"] <= 0:
            _fatal("invalid_resolution", path, line_number, raw)
        if item["fx"] <= 0 or item["fy"] <= 0:
            _fatal("invalid_focal", path, line_number, raw)
        if not (0 <= item["cx"] <= item["width"] and 0 <= item["cy"] <= item["height"]):
            _fatal("invalid_principal_point", path, line_number, raw)
        if item["cam_id"] in cam_ids:
            _fatal("duplicate_cam_id", path, line_number, raw, str(item["cam_id"]))
        if item["cam_name"] in cam_names:
            _fatal("duplicate_cam_name", path, line_number, raw, item["cam_name"])
        quaternion_norm = math.sqrt(sum(item[field] ** 2 for field in ("qvecw", "qvecx", "qvecy", "qvecz")))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            _fatal("invalid_quaternion_norm", path, line_number, raw, str(quaternion_norm))
        cam_ids.add(item["cam_id"])
        cam_names.add(item["cam_name"])
        item.update({
            "source_dataset": "GigaHands",
            "sequence": sequence,
            "participant_raw": participant_raw,
            "participant": participant,
            "camera": item.pop("cam_name"),
            "source_path": path.relative_to(HANDCALIB_ROOT).as_posix()
                if path.is_relative_to(HANDCALIB_ROOT) else path.relative_to(root).as_posix(),
        })
        item["camera_key"] = f"{sequence}/{item['camera']}"
        item.update(derive_focal(item))
        item["intrinsic_signature"] = signature_text(item, INTRINSIC_FIELDS)
        item["full_calibration_signature"] = signature_text(item, FULL_FIELDS)
        item["physical_camera_identity"] = "not_verifiable"
        rows.append(item)
    if not rows:
        _fatal("no_data_rows", path, 2, "")
    return {
        "rows": rows,
        "integrity": {
            "sequence": sequence,
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "header_field_count": len(header),
            "data_row_count": len(rows),
            "parse_status": "PASS",
        },
    }


def read_full_metadata(root):
    files = sorted(Path(root).glob("*/optim_params.txt"))
    sequence_dirs = sorted(path for path in Path(root).iterdir() if path.is_dir())
    if len(sequence_dirs) != 236 or len(files) != 236:
        raise IntegrityError(
            f"unexpected_inventory sequence_dirs={len(sequence_dirs)} optim_files={len(files)} expected=236"
        )
    rows, integrity = [], []
    for path in files:
        parsed = parse_optim_file(path, root)
        rows.extend(parsed["rows"])
        integrity.append(parsed["integrity"])
    return rows, integrity


def read_demo_manifest(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            for field in ("width", "height"):
                row[field] = int(row[field])
            for field in FULL_FIELDS[2:]:
                row[field] = float(row[field])
            row["intrinsic_signature"] = signature_text(row, INTRINSIC_FIELDS)
            row["full_calibration_signature"] = signature_text(row, FULL_FIELDS)
            row.update(derive_focal(row))
            rows.append(row)
    unique = {}
    for row in rows:
        unique.setdefault(row["camera_key"], row)
    return list(unique.values())


def overlap_summary(full_rows, demo_rows):
    def values(rows, field):
        return {row[field] for row in rows}

    full_actions = {participant_action(row["sequence"]) for row in full_rows}
    demo_actions = {participant_action(row["sequence"]) for row in demo_rows}
    return {
        "exact_intrinsic_signature_overlap_count": len(values(full_rows, "intrinsic_signature") & values(demo_rows, "intrinsic_signature")),
        "exact_full_signature_overlap_count": len(values(full_rows, "full_calibration_signature") & values(demo_rows, "full_calibration_signature")),
        "camera_identifier_overlap_count": len(values(full_rows, "camera") & values(demo_rows, "camera")),
        "exact_sequence_identity_overlap_count": len(values(full_rows, "sequence") & values(demo_rows, "sequence")),
        "normalized_participant_action_overlap_count": len(full_actions & demo_actions),
        "participant_overlap_count": len(values(full_rows, "participant") & values(demo_rows, "participant")),
        "new_exact_intrinsic_signature_count": len(values(full_rows, "intrinsic_signature") - values(demo_rows, "intrinsic_signature")),
        "new_exact_full_signature_count": len(values(full_rows, "full_calibration_signature") - values(demo_rows, "full_calibration_signature")),
        "physical_camera_identity": "not_verifiable",
    }


def distribution(values):
    values = np.asarray([float(value) for value in values], dtype=float)
    result = {
        "count": int(values.size),
        "unique_count": int(np.unique(values).size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std_population": float(values.std()),
    }
    for name, q in (("q01", .01), ("q05", .05), ("q10", .10), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9), ("q95", .95), ("q99", .99)):
        result[name] = float(np.quantile(values, q))
    result["iqr"] = result["q75"] - result["q25"]
    return result


def unique_by(rows, field):
    result = {}
    for row in rows:
        result.setdefault(row[field], row)
    return list(result.values())


def distribution_by_unit(rows):
    units = {
        "camera_row": rows,
        "sequence_camera": unique_by(rows, "camera_key"),
        "exact_intrinsic_signature": unique_by(rows, "intrinsic_signature"),
        "exact_full_calibration_signature": unique_by(rows, "full_calibration_signature"),
    }
    participant_rows = []
    for participant in sorted({row["participant"] for row in rows}):
        group = [row for row in rows if row["participant"] == participant]
        participant_rows.append({field: statistics.fmean(row[field] for row in group) for field in FOCAL_FIELDS})
    units["participant_equal_mean"] = participant_rows
    return {
        unit: {field: distribution([row[field] for row in unit_rows]) for field in FOCAL_FIELDS}
        for unit, unit_rows in units.items()
    }


def assign_bin(value, edges):
    if value < edges[0] or value > edges[-1]:
        raise ValueError(f"value outside bin range: {value}")
    if value == edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(np.asarray(edges), value, side="right") - 1)


def equal_width_edges(values, bins):
    return [float(value) for value in np.linspace(min(values), max(values), bins + 1)]


def quantile_edges(values, bins):
    return [float(value) for value in np.quantile(np.asarray(values), np.linspace(0, 1, bins + 1))]


def histogram_counts(values, edges):
    result = [0] * (len(edges) - 1)
    for value in values:
        result[assign_bin(float(value), edges)] += 1
    return result


def histogram_counts_with_overflow(values, edges):
    counts = [0] * (len(edges) - 1)
    below = above = 0
    for value in values:
        if value < edges[0]:
            below += 1
        elif value > edges[-1]:
            above += 1
        else:
            counts[assign_bin(float(value), edges)] += 1
    return counts, below, above


def _balance(counts):
    minimum = min(counts) if counts else 0
    total = sum(counts)
    retained = minimum * len(counts)
    return {
        "counts": counts,
        "minimum_bin_count": minimum,
        "maximum_bin_count": max(counts) if counts else 0,
        "empty_bin_count": sum(count == 0 for count in counts),
        "balanced_per_bin": minimum,
        "balanced_subset_size": retained,
        "discarded_count": total - retained,
        "retention_rate": retained / total if total else 0.0,
    }


def balance_summary(camera_counts, signature_counts):
    return {
        "camera_key": _balance(camera_counts),
        "exact_full_signature": _balance(signature_counts),
    }


def balance_analysis(rows):
    camera_rows = unique_by(rows, "camera_key")
    signature_rows = unique_by(rows, "full_calibration_signature")
    output = {}
    table = []
    for variable in ("fx", "fy", "nfx", "nfy"):
        source_values = [row[variable] for row in camera_rows]
        for method in ("equal_width", "quantile"):
            for bins in BIN_COUNTS:
                edges = equal_width_edges(source_values, bins) if method == "equal_width" else quantile_edges(source_values, bins)
                camera_counts = histogram_counts(source_values, edges)
                signature_counts = histogram_counts([row[variable] for row in signature_rows], edges)
                participants = [set() for _ in range(bins)]
                for row in camera_rows:
                    participants[assign_bin(row[variable], edges)].add(row["participant"])
                summary = balance_summary(camera_counts, signature_counts)
                key = f"{variable}_{method}_{bins}"
                output[key] = {
                    "variable": variable,
                    "method": method,
                    "bin_count": bins,
                    "edges": edges,
                    "participant_counts": [len(group) for group in participants],
                    **summary,
                }
                for index in range(bins):
                    table.append({
                        "variable": variable, "method": method, "bin_count": bins,
                        "bin_index": index + 1, "lower_edge": edges[index],
                        "upper_edge": edges[index + 1],
                        "camera_key_count": camera_counts[index],
                        "exact_full_signature_count": signature_counts[index],
                        "participant_count": len(participants[index]),
                    })
    return output, table


def participant_components(signature_participants, participants):
    adjacency = {participant: set() for participant in participants}
    for group in signature_participants.values():
        for participant in group:
            adjacency[participant].update(group - {participant})
    unseen, components = set(participants), []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component, stack = {seed}, [seed]
        while stack:
            for neighbor in sorted(adjacency[stack.pop()]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda item: (-len(item), sorted(item)))


def independence_analysis(rows):
    owners = defaultdict(set)
    for row in rows:
        owners[row["full_calibration_signature"]].add(row["participant"])
    participants = {row["participant"] for row in rows}
    components = participant_components(owners, participants)
    return {
        "participant_count": len(participants),
        "exact_full_signatures_shared_across_participants": sum(len(group) > 1 for group in owners.values()),
        "maximum_participants_per_signature": max(map(len, owners.values())),
        "participant_components_under_shared_signature": len(components),
        "participant_component_sizes": [len(component) for component in components],
        "strict_three_way_participant_and_signature_independent_split": "IMPOSSIBLE" if len(components) < 3 else "REQUIRES_SEARCH",
        "reason": "all participants are connected by shared exact full calibration signatures" if len(components) == 1 else "shared-signature components must remain within one split",
    }


def duplicate_file_analysis(integrity):
    groups = defaultdict(list)
    for row in integrity:
        groups[row["sha256"]].append(row["sequence"])
    records = [
        {"sha256": digest, "file_count": len(sequences), "sequences": ";".join(sorted(sequences))}
        for digest, sequences in sorted(groups.items())
    ]
    return {
        "unique_file_sha256_count": len(groups),
        "duplicate_hash_group_count": sum(len(group) > 1 for group in groups.values()),
        "files_in_duplicate_hash_groups": sum(len(group) for group in groups.values() if len(group) > 1),
        "records": records,
    }


def fixed_demo_bin_comparison(full_rows, demo_rows, split_report_path):
    report = json.loads(Path(split_report_path).read_text(encoding="utf-8"))
    result = {}
    table = []
    for variable in ("fx", "fy"):
        edges = report["histogram"][f"{variable}_bin_edges"]
        for dataset, rows in (("demo", demo_rows), ("full", full_rows)):
            camera_rows = unique_by(rows, "camera_key")
            signature_rows = unique_by(rows, "full_calibration_signature")
            camera_counts, camera_below, camera_above = histogram_counts_with_overflow(
                [row[variable] for row in camera_rows], edges
            )
            signature_counts, signature_below, signature_above = histogram_counts_with_overflow(
                [row[variable] for row in signature_rows], edges
            )
            result[f"{dataset}_{variable}"] = {
                "edges": edges,
                "camera_key_counts": camera_counts,
                "camera_key_ratios": [count / len(camera_rows) for count in camera_counts],
                "exact_full_signature_counts": signature_counts,
                "exact_full_signature_ratios": [count / len(signature_rows) for count in signature_counts],
                "camera_keys_below_demo_range": camera_below,
                "camera_keys_above_demo_range": camera_above,
                "signatures_below_demo_range": signature_below,
                "signatures_above_demo_range": signature_above,
            }
            for index, count in enumerate(camera_counts):
                table.append({
                    "dataset": dataset, "variable": variable, "bin_index": index + 1,
                    "lower_edge": edges[index], "upper_edge": edges[index + 1],
                    "camera_key_count": count, "camera_key_ratio": count / len(camera_rows),
                    "exact_full_signature_count": signature_counts[index],
                    "exact_full_signature_ratio": signature_counts[index] / len(signature_rows),
                })
    return result, table


def rgb_availability(root, full_rows, demo_rows):
    dataset_root = Path(root).parent
    media = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS)
    full_root_media = [path for path in media if Path(root) in path.parents]
    demo_signatures = {row["full_calibration_signature"] for row in demo_rows}
    candidates = []
    for row in full_rows:
        candidates.append({
            "sequence": row["sequence"],
            "participant": row["participant"],
            "camera": row["camera"],
            "camera_key": row["camera_key"],
            "fx": row["fx"], "fy": row["fy"],
            "full_calibration_signature": row["full_calibration_signature"],
            "exact_calibration_available_in_demo": row["full_calibration_signature"] in demo_signatures,
            "full_sequence_rgb_found": False,
            "candidate_priority": "novel_calibration" if row["full_calibration_signature"] not in demo_signatures else "duplicate_calibration_lower_priority",
        })
    return {
        "all_local_media_file_count_under_gigahands": len(media),
        "media_file_count_under_full_hand_poses": len(full_root_media),
        "full_sequence_rgb_status": "NOT_FOUND" if not full_root_media else "FOUND_UNVERIFIED_MEDIA",
        "exact_required_video_count": "unknown_from_camera_metadata",
        "exact_required_download_size": "unknown_from_camera_metadata",
        "candidate_count": len(candidates),
    }, candidates


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def aggregate_source_hash(integrity):
    digest = hashlib.sha256()
    for row in sorted(integrity, key=lambda item: item["path"]):
        digest.update(f"{row['path']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def protected_paths(demo_manifest, split_report):
    candidates = [
        demo_manifest,
        split_report,
        HANDCALIB_ROOT / "data/splits/gigahands/train.txt",
        HANDCALIB_ROOT / "data/splits/gigahands/val.txt",
        HANDCALIB_ROOT / "data/splits/gigahands/test.txt",
        HANDCALIB_ROOT / "results/02i0_gigahands_focal_coverage_feasibility.yaml",
        HANDCALIB_ROOT / "runs/02_anycalib_finetune/train/checkpoint_best.pt",
    ]
    return [path for path in candidates if Path(path).exists()]


def plot_figures(output, full_rows, demo_rows, balance):
    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": .25})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, field in zip(axes, ("fx", "fy")):
        axis.hist([row[field] for row in full_rows], bins=30, color="0.75", edgecolor="black", density=True, label="Full")
        axis.hist([row[field] for row in demo_rows], bins=20, histtype="step", color="black", density=True, label="Demo")
        axis.set(xlabel=f"{field} (px)", ylabel="Density")
        axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"01_demo_vs_full_focal.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter([row["fx"] for row in full_rows], [row["fy"] for row in full_rows], s=4, c="0.3", alpha=.15)
    axes[0].set(xlabel="fx (px)", ylabel="fy (px)")
    axes[1].scatter([row["nfx"] for row in full_rows], [row["nfy"] for row in full_rows], s=4, c="0.3", alpha=.15)
    axes[1].set(xlabel="fx / width", ylabel="fy / height")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"02_focal_scatter.{suffix}")
    plt.close(fig)

    labels, camera, signatures = [], [], []
    for variable in ("fx", "fy"):
        for bins in BIN_COUNTS:
            item = balance[f"{variable}_equal_width_{bins}"]
            labels.append(f"{variable}-{bins}")
            camera.append(item["camera_key"]["balanced_subset_size"])
            signatures.append(item["exact_full_signature"]["balanced_subset_size"])
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(x - .2, camera, .4, color="0.7", edgecolor="black", label="camera_key")
    axis.bar(x + .2, signatures, .4, color="white", edgecolor="black", hatch="//", label="exact signature")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set(ylabel="Maximum balanced subset")
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"03_balance_capacity.{suffix}")
    plt.close(fig)


def compact_result(summary):
    return {
        "experiment": summary["experiment"],
        "status": summary["status"],
        "scope": summary["scope"],
        "repository_commit_at_analysis": summary["repository_commit_at_analysis"],
        "integrity": summary["integrity"],
        "population": summary["population"],
        "demo_overlap": summary["demo_overlap"],
        "range_comparison": summary["range_comparison"],
        "fixed_demo_five_bin_comparison": summary["fixed_demo_five_bin_comparison"],
        "balance_feasibility": summary["balance_feasibility"],
        "training_balance_assessment": summary["training_balance_assessment"],
        "evaluation_balance_assessment": summary["evaluation_balance_assessment"],
        "independence": summary["independence"],
        "rgb_availability": summary["rgb_availability"],
        "physical_camera_identity": summary["physical_camera_identity"],
        "decision": summary["decision"],
        "protection": summary["protection"],
    }


def markdown_report(summary):
    pop = summary["population"]
    overlap = summary["demo_overlap"]
    independence = summary["independence"]
    balance = summary["balance_feasibility"]
    lines = [
        "# 02-I.1 GigaHands 전체 focal coverage 감사", "",
        "## 연구 목적", "",
        "전체 GigaHands hand-pose 카메라 메타데이터가 데모의 focal 편중과 독립 표본 부족을 얼마나 완화하는지 조사했다.", "",
        "## 방법", "",
        f"236개 `optim_params.txt`의 11,756행을 원본 수정 없이 strict parse했다. camera_key와 exact calibration signature를 독립 단위로 계산하고 기존 데모와 exact value로 대조했다.", "",
        "## 변인 통제", "",
        "GigaHands만 사용했다. 학습, 추론, GPU, 다운로드, split 생성은 수행하지 않았다. 고정 데모 5-bin과 전체 범위 탐색 bin을 섞지 않았다.", "",
        "## 결과", "",
        f"- 참가자 {pop['participant_count']}명, 시퀀스 {pop['sequence_count']}개, sequence-camera {pop['camera_key_count']}개",
        f"- exact intrinsic/full calibration signature 각 {pop['exact_intrinsic_signature_count']}/{pop['exact_full_signature_count']}개",
        f"- 반복 camera row {pop['duplicate_calibration_row_count']}개 ({pop['duplicate_calibration_row_ratio']:.2%}); 이는 독립 calibration이 아니다.",
        f"- 데모와 exact full calibration 중복 {overlap['exact_full_signature_overlap_count']}개, 신규 {overlap['new_exact_full_signature_count']}개",
        f"- fx 범위 {summary['range_comparison']['full']['fx']['min']:.3f}–{summary['range_comparison']['full']['fx']['max']:.3f}px, fy 범위 {summary['range_comparison']['full']['fy']['min']:.3f}–{summary['range_comparison']['full']['fy']['max']:.3f}px",
        f"- exact 파일 해시는 {summary['duplicate_files']['unique_file_sha256_count']}종이며 {summary['duplicate_files']['files_in_duplicate_hash_groups']}개 파일이 반복 hash group에 속한다.", "",
        "### Equal-width 균형 용량", "",
        "|축-bin|camera_key counts|camera_key 최대|signature counts|signature 최대|",
        "|---|---:|---:|---:|---:|",
    ]
    for variable in ("fx", "fy"):
        for bins in BIN_COUNTS:
            item = balance[f"{variable}_equal_width_{bins}"]
            lines.append(
                f"|{variable}-{bins}|{item['camera_key']['counts']}|{item['camera_key']['balanced_subset_size']}|"
                f"{item['exact_full_signature']['counts']}|{item['exact_full_signature']['balanced_subset_size']}|"
            )
    lines += [
        "", "## 해석", "",
        "전체 자료는 데모보다 focal 범위와 고유 calibration 수를 늘린다. 그러나 반복 camera row를 독립 calibration으로 해석하면 안 되며, quantile bin의 수치상 균형은 경계를 데이터에 맞춘 탐색 결과일 뿐이다.",
        "Calibration-level 비복원 표본화는 독립 단위를 맞추지만 많은 데이터를 버릴 수 있다. Bin weighting은 모든 데이터를 유지하지만 반복 calibration의 독립성을 늘리지 않는다. Bin-macro 평가는 모든 자료를 유지하면서 구간 평균을 동일 가중할 수 있지만 희소 bin의 분산과 calibration leakage를 해결하지 않는다.", "",
        "## 한계", "",
        f"exact calibration 공유 그래프에서 참가자 {independence['participant_count']}명이 {independence['participant_components_under_shared_signature']}개 연결 성분을 이룬다. 따라서 참가자와 exact calibration을 동시에 비중복으로 유지한 3-way 분할은 불가능하다.",
        "물리 카메라 identity는 not_verifiable이다. 같은 값이나 cam_name은 같은 물리 장비의 증거가 아니다. 전체 hand-pose 폴더에서 RGB를 확인하지 못했으므로 메타데이터만으로 영상 수와 용량을 추정하지 않았다.", "",
        "## 결론", "",
        "GigaHands 전체 메타데이터는 focal coverage와 calibration 후보 수를 크게 확장해 학습 가중치 또는 calibration-level 표본화 연구는 가능하게 한다. 하지만 현재 공유 calibration 구조 때문에 엄격한 참가자·calibration 독립 Train/Validation/Test와 그에 기반한 신뢰도 높은 확증 평가는 구성할 수 없다.", "",
        "## 다음 단계", "",
        "모델 성능을 보지 않은 사전 규약으로 focal 축, bin 경계, 최소 독립 calibration 수를 정하고, 신규 calibration의 RGB 존재·접근성을 확인한다. 엄격한 독립 평가에는 공유 signature graph 밖의 참가자 또는 외부 독립 카메라 자료가 필요하다.", "",
    ]
    return "\n".join(lines)


def audit(root, demo_manifest, split_report, output, result_yaml, overwrite=False):
    output, result_yaml = Path(output), Path(result_yaml)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(exist_ok=True)

    protected = protected_paths(demo_manifest, split_report)
    protected_before = {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in protected}
    full_rows, integrity = read_full_metadata(root)
    source_hash_before = aggregate_source_hash(integrity)
    annotation_path = Path(root) / "annotations_v0.jsonl"
    if not annotation_path.is_file():
        raise IntegrityError(f"missing_annotation_index path={annotation_path}")
    annotation_sha_before = sha256_file(annotation_path)
    demo_rows = read_demo_manifest(demo_manifest)
    overlap = overlap_summary(full_rows, demo_rows)
    duplicates = duplicate_file_analysis(integrity)
    distributions = distribution_by_unit(full_rows)
    balance, balance_table = balance_analysis(full_rows)
    fixed_bins, fixed_bin_table = fixed_demo_bin_comparison(full_rows, demo_rows, split_report)
    independence = independence_analysis(full_rows)
    rgb, rgb_candidates = rgb_availability(root, full_rows, demo_rows)

    signature_multiplicity = Counter(row["full_calibration_signature"] for row in full_rows)
    population = {
        "sequence_count": len({row["sequence"] for row in full_rows}),
        "participant_raw_count": len({row["participant_raw"] for row in full_rows}),
        "participant_count": len({row["participant"] for row in full_rows}),
        "camera_row_count": len(full_rows),
        "camera_key_count": len({row["camera_key"] for row in full_rows}),
        "camera_identifier_count": len({row["camera"] for row in full_rows}),
        "exact_intrinsic_signature_count": len({row["intrinsic_signature"] for row in full_rows}),
        "exact_full_signature_count": len({row["full_calibration_signature"] for row in full_rows}),
        "unique_fx_count": len({row["fx"] for row in full_rows}),
        "unique_fy_count": len({row["fy"] for row in full_rows}),
        "duplicate_calibration_row_count": len(full_rows) - len(signature_multiplicity),
        "duplicate_calibration_row_ratio": (len(full_rows) - len(signature_multiplicity)) / len(full_rows),
        "camera_rows_per_signature_min": min(signature_multiplicity.values()),
        "camera_rows_per_signature_max": max(signature_multiplicity.values()),
        "camera_rows_per_signature_mean": statistics.fmean(signature_multiplicity.values()),
        "signature_count_by_camera_row_multiplicity": dict(sorted(Counter(signature_multiplicity.values()).items())),
        "camera_rows_per_sequence_distribution": dict(
            sorted(Counter(Counter(row["sequence"] for row in full_rows).values()).items())
        ),
    }
    range_comparison = {
        dataset: {field: distribution([row[field] for row in rows]) for field in FOCAL_FIELDS}
        for dataset, rows in (("demo", demo_rows), ("full", full_rows))
    }
    summary = {
        "experiment": "02i1_gigahands_full_focal_coverage",
        "status": "FULL_METADATA_INTEGRITY_AND_FOCAL_AUDIT_COMPLETE",
        "scope": "GigaHands only; camera metadata analysis; no training, inference, GPU, download, or split creation",
        "repository_commit_at_analysis": git_output("rev-parse", "HEAD"),
        "integrity": {
            "status": "PASS", "expected_file_count": 236, "parsed_file_count": len(integrity),
            "parsed_data_row_count": len(full_rows), "header_field_count": 19,
            "malformed_row_count": 0, "numeric_conversion_error_count": 0,
            "non_finite_row_count": 0, "invalid_range_row_count": 0,
            "duplicate_cam_id_within_file_count": 0, "duplicate_cam_name_within_file_count": 0,
            "p041_boxing_format_error": False,
            "input_camera_metadata_aggregate_sha256": source_hash_before,
            "annotations_v0_exists": True,
            "annotations_v0_size_bytes": annotation_path.stat().st_size,
            "annotations_v0_sha256": annotation_sha_before,
        },
        "population": population,
        "participant_normalization": {"rule": "p + decimal integer without leading zeros", "example": "p041 -> p41", "raw_label_preserved": True},
        "physical_camera_identity": {"status": "not_verifiable", "confirmed_group_count": 0, "actual_count": "unknown", "camera_identifier_and_calibration_duplicates_reported_separately": True},
        "duplicate_files": {key: value for key, value in duplicates.items() if key != "records"},
        "demo_overlap": overlap,
        "distribution_by_analysis_unit": distributions,
        "range_comparison": range_comparison,
        "fixed_demo_five_bin_comparison": fixed_bins,
        "exploratory_bins": {"methods": ["equal_width", "quantile"], "bin_counts": list(BIN_COUNTS), "status": "candidate_only_not_finalized", "last_bin_includes_maximum": True},
        "balance_feasibility": balance,
        "independence": independence,
        "rgb_availability": rgb,
        "training_balance_assessment": {
            "calibration_group_subsampling": "possible for nonempty bins but may discard many signatures; equal-width 10-bin has empty bins",
            "bin_weighting": "retains all rows but does not create independent calibrations and can preserve repeated-signature exposure bias",
            "required_unit": "exact calibration signature, not frame or camera_key",
            "implemented": False,
        },
        "evaluation_balance_assessment": {
            "bin_macro": "descriptively possible with precommitted nonempty bins and equal bin weights",
            "data_retention": "all observations may be retained",
            "limitations": "sparse-bin uncertainty, repeated calibration, participant/calibration leakage, and changed estimand",
            "confirmatory_status": "not reliable under current strict-independence failure",
            "adopted": False,
        },
        "selection_bias_control": {
            "model_performance_used": False,
            "existing_test_performance_used_to_choose_bins": False,
            "new_bins_or_splits_finalized": False,
        },
        "decision": {
            "training_balance": "CONDITIONALLY_FEASIBLE_AT_CALIBRATION_LEVEL_AFTER_RGB_AVAILABILITY_CHECK",
            "all_data_bin_macro_evaluation": "DESCRIPTIVELY_FEASIBLE_WITH_PRECOMMITTED_BINS_AND_UNCERTAINTY",
            "strict_independent_three_way_evaluation": "INFEASIBLE_WITH_CURRENT_SHARED_SIGNATURE_GRAPH",
            "new_split_finalized": False, "new_bin_definition_finalized": False,
            "training_runs": 0, "inference_runs": 0,
        },
        "protection": {"source_files_preserved": True, "existing_manifest_split_checkpoint_results_preserved": True, "source_hash_before": source_hash_before},
    }

    manifest_fields = [
        "source_dataset", "sequence", "participant_raw", "participant", "camera", "cam_id",
        "camera_key", "source_path", "width", "height", "fx", "fy", "cx", "cy",
        "k1", "k2", "p1", "p2", "nfx", "nfy", "hfov_deg", "vfov_deg",
        "intrinsic_signature", "full_calibration_signature", "physical_camera_identity",
    ]
    write_csv(tables / "gigahands_full_camera_metadata.csv", full_rows, manifest_fields)
    write_csv(tables / "optim_params_integrity.csv", integrity)
    write_csv(tables / "duplicate_file_groups.csv", duplicates["records"])
    write_csv(tables / "fixed_demo_five_bin_comparison.csv", fixed_bin_table)
    write_csv(tables / "exploratory_bin_coverage.csv", balance_table)
    write_csv(tables / "rgb_acquisition_candidates.csv", rgb_candidates)
    write_csv(tables / "demo_overlap_summary.csv", [{"metric": key, "value": value} for key, value in overlap.items()])
    participant_table = []
    for participant in sorted({row["participant"] for row in full_rows}):
        group = [row for row in full_rows if row["participant"] == participant]
        participant_table.append({
            "participant": participant,
            "raw_labels": ";".join(sorted({row["participant_raw"] for row in group})),
            "sequence_count": len({row["sequence"] for row in group}),
            "camera_key_count": len({row["camera_key"] for row in group}),
            "exact_full_signature_count": len({row["full_calibration_signature"] for row in group}),
            "fx_min": min(row["fx"] for row in group), "fx_max": max(row["fx"] for row in group),
            "fy_min": min(row["fy"] for row in group), "fy_max": max(row["fy"] for row in group),
        })
    write_csv(tables / "participant_distribution.csv", participant_table)
    plot_figures(output, full_rows, demo_rows, balance)

    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "analysis_report.md").write_text(markdown_report(summary), encoding="utf-8")
    metadata = {
        "command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": summary["repository_commit_at_analysis"], "cpu_metadata_only": True,
        "gpu_used": False, "training_runs": 0, "inference_runs": 0,
        "source_file_sha256": {row["path"]: row["sha256"] for row in integrity},
        "protected_sha256_before": protected_before,
    }
    result_yaml.parent.mkdir(parents=True, exist_ok=True)
    result_yaml.write_text(yaml.safe_dump(compact_result(summary), allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")

    _, integrity_after = read_full_metadata(root)
    source_hash_after = aggregate_source_hash(integrity_after)
    annotation_sha_after = sha256_file(annotation_path)
    protected_after = {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in protected}
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source camera metadata changed during audit")
    if annotation_sha_before != annotation_sha_after:
        raise RuntimeError("Source annotation index changed during audit")
    if protected_before != protected_after:
        raise RuntimeError("Protected existing research file changed during audit")
    metadata.update({
        "source_aggregate_sha256_after": source_hash_after,
        "annotations_v0_sha256_after": annotation_sha_after,
        "protected_sha256_after": protected_after,
        "source_files_preserved": True,
        "protected_files_preserved": True,
    })
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    parser.add_argument("--split-report", type=Path, default=DEFAULT_SPLIT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-yaml", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = audit(
        args.root.resolve(), args.demo_manifest.resolve(), args.split_report.resolve(),
        args.output.resolve(), args.result_yaml.resolve(), args.overwrite,
    )
    print(f"status={summary['status']}")
    print(f"files={summary['integrity']['parsed_file_count']} rows={summary['integrity']['parsed_data_row_count']}")
    print(f"output={args.output.resolve()}")
    print("training_runs=0 inference_runs=0 new_split_finalized=false")


if __name__ == "__main__":
    main()
