"""Design metadata-only GigaHands focal-balanced protocol candidates."""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import random
import shutil
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
DEFAULT_SPLIT_DIR = HANDCALIB_ROOT / "data/splits/gigahands"
DEFAULT_OUTPUT = HANDCALIB_ROOT / "runs/02_gigahands_focal_balance_design"
DEFAULT_RESULT = HANDCALIB_ROOT / "results/02i2_gigahands_focal_balance_protocol_design.yaml"
DEFAULT_SEED = 42
DEFAULT_SEARCH_BUDGET = 12000
SPLITS = ("train", "validation", "test")
BIN_COUNTS = (3, 5, 7, 10)
SAMPLING_PROTOCOLS = ("S0", "S1", "S2", "S3", "S4")

_FULL_AUDIT_PATH = Path(__file__).with_name("audit_gigahands_full_focal_coverage.py")
_spec = importlib.util.spec_from_file_location("gigahands_full_audit_for_design", _FULL_AUDIT_PATH)
full_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(full_audit)


def git_output(*args):
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def unique_rows(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], row)
    return list(result.values())


def duplicate_structure(rows):
    signature_participants = defaultdict(set)
    signature_sequences = defaultdict(set)
    identifier_signatures = defaultdict(set)
    for row in rows:
        signature = row["full_calibration_signature"]
        signature_participants[signature].add(row["participant"])
        signature_sequences[signature].add(row["sequence"])
        identifier_signatures[row["camera"]].add(signature)
    return {
        "record_count": len(rows),
        "unique_full_signature_count": len(signature_participants),
        "cross_participant_signature_count": sum(len(group) > 1 for group in signature_participants.values()),
        "cross_sequence_signature_count": sum(len(group) > 1 for group in signature_sequences.values()),
        "camera_identifier_count": len(identifier_signatures),
        "physical_camera_identity": "not_verifiable",
    }


def _pairwise_overlap(split_rows, field):
    values = {name: {row[field] for row in split_rows.get(name, [])} for name in SPLITS}
    union = set()
    details = {}
    for left, right in itertools.combinations(SPLITS, 2):
        overlap = values[left] & values[right]
        details[f"{left}_{right}"] = len(overlap)
        union.update(overlap)
    return len(union), details


def split_overlaps(split_rows):
    fields = {
        "participant_overlap": "participant",
        "sequence_overlap": "sequence",
        "camera_key_overlap": "camera_key",
        "exact_intrinsic_overlap": "intrinsic_signature",
        "exact_full_overlap": "full_calibration_signature",
        "camera_identifier_overlap": "camera",
    }
    result = {}
    for name, field in fields.items():
        count, details = _pairwise_overlap(split_rows, field)
        result[name] = count
        result[name + "_by_pair"] = details
    return result


def select_matching_records(rows, participant_assignment, owner_assignment, owner_key):
    split_rows = {name: [] for name in SPLITS}
    excluded = []
    for row in rows:
        participant_split = participant_assignment[row["participant"]]
        owner_split = owner_assignment[row[owner_key]]
        if participant_split == owner_split:
            split_rows[participant_split].append(row)
        else:
            excluded.append({**row, "exclusion_reason": f"participant_{participant_split}_owner_{owner_split}_mismatch"})
    return split_rows, excluded


def validate_record_partition(split_rows, excluded):
    locations = defaultdict(list)
    for split, rows in split_rows.items():
        for row in rows:
            locations[row["camera_key"]].append(split)
    for row in excluded:
        locations[row["camera_key"]].append("excluded")
    duplicated = {key: value for key, value in locations.items() if len(value) > 1}
    if duplicated:
        raise ValueError(f"record overlap: {next(iter(duplicated.items()))}")
    return True


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


def histogram(values, edges):
    counts = [0] * (len(edges) - 1)
    for value in values:
        counts[assign_bin(float(value), edges)] += 1
    return counts


def bin_support(counts, required):
    return {
        "counts": list(counts),
        "minimum_bin_support": min(counts) if counts else 0,
        "empty_bin_count": sum(count == 0 for count in counts),
        "required_per_bin": required,
        "meets_requirement": bool(counts) and all(count >= required for count in counts),
    }


def signature_base_weights(rows, protocol):
    counts = Counter(row["full_calibration_signature"] for row in rows)
    if protocol == "S0":
        return {signature: float(count) for signature, count in counts.items()}
    return {signature: 1.0 for signature in counts}


def bin_weight_factors(counts, alpha):
    return [0.0 if count <= 0 else 1.0 / (float(count) ** alpha) for count in counts]


def clip_and_normalize(weights, max_multiple):
    if not weights:
        return {}
    keys = sorted(weights)
    total = sum(max(0.0, weights[key]) for key in keys)
    if total <= 0:
        raise ValueError("weights must contain a positive value")
    probabilities = {key: max(0.0, weights[key]) / total for key in keys}
    cap = max_multiple / len(probabilities)
    free = set(probabilities)
    fixed = {}
    while free:
        fixed_keys = sorted(fixed)
        free_keys = sorted(free)
        remaining = 1.0 - sum(fixed[key] for key in fixed_keys)
        source_total = sum(probabilities[key] for key in free_keys)
        scaled = {key: remaining * probabilities[key] / source_total for key in free_keys}
        over = {key for key, value in scaled.items() if value > cap + 1e-15}
        if not over:
            fixed.update(scaled)
            break
        for key in sorted(over):
            fixed[key] = cap
            free.remove(key)
    return {key: fixed[key] for key in keys}


def effective_sample_size(weights):
    total = sum(weights)
    square_sum = sum(weight * weight for weight in weights)
    return 0.0 if square_sum == 0 else total * total / square_sum


def deterministic_partition(items, fractions, seed):
    names = sorted(items)
    rng = random.Random(seed)
    rng.shuffle(names)
    raw = [len(names) * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))[:len(names) - sum(counts)]:
        counts[index] += 1
    result = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        for item in names[offset:offset + count]:
            result[item] = split
        offset += count
    return result


def prepare_output(path, overwrite):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def search_status(found):
    return "CANDIDATE_FOUND_NOT_PROVEN_OPTIMAL" if found else "NO_CANDIDATE_FOUND_WITHIN_BUDGET"


def rgb_requirement_row(candidate_id, proposed_split, row, focal_bin):
    return {
        "candidate_id": candidate_id,
        "proposed_split": proposed_split,
        "participant": row["participant"],
        "sequence": row["sequence"],
        "camera_name": row["camera"],
        "camera_key": row["camera_key"],
        "calibration_signature_id": row["full_calibration_signature"],
        "fx": row["fx"], "fy": row["fy"],
        "width": row["width"], "height": row["height"],
        "focal_bin": focal_bin,
        "metadata_source": row.get("source_path", "unknown"),
        "existing_rgb_status": "not_available_for_full_sequence",
        "download_required": True,
        "selection_reason": "metadata-only protocol candidate",
        "frame_count": "unknown",
        "video_size_bytes": "unknown",
        "download_url": "unknown",
    }


def calibration_set_labels(rows):
    by_sequence = defaultdict(set)
    for row in rows:
        by_sequence[row["sequence"]].add(row["full_calibration_signature"])
    unique_sets = sorted({tuple(sorted(signatures)) for signatures in by_sequence.values()})
    labels = {
        signatures: f"calset_{index:02d}_{hashlib.sha256(chr(10).join(signatures).encode()).hexdigest()[:10]}"
        for index, signatures in enumerate(unique_sets, start=1)
    }
    sequence_labels = {sequence: labels[tuple(sorted(signatures))] for sequence, signatures in by_sequence.items()}
    enriched = [{**row, "calibration_set": sequence_labels[row["sequence"]]} for row in rows]
    return enriched, sequence_labels


def _random_fixed_counts(items, counts, rng):
    shuffled = sorted(items)
    rng.shuffle(shuffled)
    result, offset = {}, 0
    for split, count in zip(SPLITS, counts):
        for item in shuffled[offset:offset + count]:
            result[item] = split
        offset += count
    return result


def _greedy_capacity_assignment(items, scores, counts, rng):
    random_tie = {item: rng.random() for item in items}
    ordered = sorted(
        items,
        key=lambda item: (
            -(max(scores[item].values()) - min(scores[item].values())),
            random_tie[item], item,
        ),
    )
    remaining = dict(zip(SPLITS, counts))
    assignment = {}
    for item in ordered:
        available = [split for split in SPLITS if remaining[split] > 0]
        best = max(available, key=lambda split: (scores[item][split], -SPLITS.index(split)))
        assignment[item] = best
        remaining[best] -= 1
    return assignment


def active_participant_counts(scores, assignment):
    counts = {split: 0 for split in SPLITS}
    for participant, split in assignment.items():
        counts[split] += int(scores[participant][split] > 0)
    return counts


def _owner_bin_vectors(rows, owner_key, edges):
    representatives = unique_rows(rows, "full_calibration_signature")
    vectors = defaultdict(lambda: [0] * (len(edges) - 1))
    signature_owner = {}
    for row in representatives:
        owner = row[owner_key]
        vectors[owner][assign_bin(float(row["fx"]), edges)] += 1
        signature_owner[row["full_calibration_signature"]] = owner
    return dict(vectors), signature_owner


def _participant_owner_matrix(rows, owner_key):
    matrix = defaultdict(Counter)
    for row in rows:
        matrix[row["participant"]][row[owner_key]] += 1
    return matrix


def _assignment_bin_counts(owner_assignment, vectors):
    bins = len(next(iter(vectors.values())))
    result = {split: [0] * bins for split in SPLITS}
    for owner, split in owner_assignment.items():
        for index, count in enumerate(vectors[owner]):
            result[split][index] += count
    return result


def usable_signatures(rows, participant_assignment, owner_assignment, owner_key):
    signature_participants = defaultdict(set)
    signature_owner = {}
    for row in rows:
        signature = row["full_calibration_signature"]
        signature_participants[signature].add(row["participant"])
        signature_owner[signature] = row[owner_key]
    return {
        signature
        for signature, participants in signature_participants.items()
        if any(
            participant_assignment[participant] == owner_assignment[signature_owner[signature]]
            for participant in participants
        )
    }


def search_joint_assignment(rows, owner_key, budget, seed, owner_counts, participant_counts):
    owners = sorted({row[owner_key] for row in rows})
    participants = sorted({row["participant"] for row in rows})
    signature_rows = unique_rows(rows, "full_calibration_signature")
    edges = equal_width_edges([float(row["fx"]) for row in signature_rows], 5)
    vectors, _ = _owner_bin_vectors(rows, owner_key, edges)
    matrix = _participant_owner_matrix(rows, owner_key)
    signature_representative = {
        row["full_calibration_signature"]: row
        for row in unique_rows(rows, "full_calibration_signature")
    }
    signature_participants = defaultdict(set)
    for row in rows:
        signature_participants[row["full_calibration_signature"]].add(row["participant"])
    signature_info = [
        (
            signature,
            representative[owner_key],
            signature_participants[signature],
            assign_bin(float(representative["fx"]), edges),
        )
        for signature, representative in signature_representative.items()
    ]
    rng = random.Random(seed)
    best_retention = None
    best_coverage = None
    for iteration in range(budget):
        owner_assignment = _random_fixed_counts(owners, owner_counts, rng)
        scores = {
            participant: {
                split: sum(count for owner, count in matrix[participant].items() if owner_assignment[owner] == split)
                for split in SPLITS
            }
            for participant in participants
        }
        participant_assignment = _greedy_capacity_assignment(
            participants, scores, participant_counts, rng
        )
        retained = sum(scores[participant][participant_assignment[participant]] for participant in participants)
        active_participants = active_participant_counts(scores, participant_assignment)
        bin_counts = {split: [0] * (len(edges) - 1) for split in SPLITS}
        retained_signatures = 0
        for _, owner, signature_owners, bin_index in signature_info:
            split = owner_assignment[owner]
            if any(participant_assignment[participant] == split for participant in signature_owners):
                bin_counts[split][bin_index] += 1
                retained_signatures += 1
        evaluation_min = min(min(bin_counts["validation"]), min(bin_counts["test"]))
        split_min = min(sum(bin_counts[split]) for split in SPLITS)
        common = {
            "iteration": iteration,
            "owner_assignment": owner_assignment,
            "participant_assignment": participant_assignment,
            "retained_record_estimate": retained,
            "bin_counts": bin_counts,
            "evaluation_min_bin_support": evaluation_min,
            "retained_signature_estimate": retained_signatures,
            "active_participant_counts": active_participants,
        }
        minimum_evaluation_participants = min(active_participants["validation"], active_participants["test"])
        retention_score = (retained, minimum_evaluation_participants, retained_signatures, evaluation_min, split_min)
        coverage_score = (evaluation_min, minimum_evaluation_participants, split_min, retained_signatures, retained)
        if best_retention is None or retention_score > best_retention[0]:
            best_retention = (retention_score, common)
        if best_coverage is None or coverage_score > best_coverage[0]:
            best_coverage = (coverage_score, common)
    return {
        "retention": best_retention[1] if best_retention else None,
        "coverage": best_coverage[1] if best_coverage else None,
        "budget": budget,
        "seed": seed,
        "focal_edges": edges,
    }


def stratified_owner_assignment(rows, owner_key, bins, fractions, seed):
    representatives = unique_rows(rows, "full_calibration_signature")
    owner_values = defaultdict(list)
    for row in representatives:
        owner_values[row[owner_key]].append(float(row["fx"]))
    values = {owner: statistics.fmean(group) for owner, group in owner_values.items()}
    edges = equal_width_edges(list(values.values()), bins)
    groups = defaultdict(list)
    for owner, value in values.items():
        groups[assign_bin(value, edges)].append(owner)
    assignment = {}
    for bin_index, owners in sorted(groups.items()):
        assignment.update(deterministic_partition(owners, fractions, seed + bin_index))
    return assignment, edges


def owner_split(rows, owner_assignment, owner_key):
    split_rows = {name: [] for name in SPLITS}
    for row in rows:
        split_rows[owner_assignment[row[owner_key]]].append(row)
    return split_rows


def load_demo_candidate(demo_manifest, split_dir):
    rows = full_audit.read_demo_manifest(demo_manifest)
    participants = {
        "train": set((Path(split_dir) / "train.txt").read_text(encoding="utf-8").split()),
        "validation": set((Path(split_dir) / "val.txt").read_text(encoding="utf-8").split()),
        "test": set((Path(split_dir) / "test.txt").read_text(encoding="utf-8").split()),
    }
    split_rows = {
        split: [row for row in rows if row["participant"] in members]
        for split, members in participants.items()
    }
    return {
        "candidate_id": "P0_EXISTING_DEMO",
        "split_protocol": "P0",
        "search_variant": "locked_existing_demo",
        "split_rows": split_rows,
        "excluded": [],
        "feasibility_status": "PROVEN_FEASIBLE_EXISTING_PROTOCOL",
        "search_notes": "Existing locked demo split; not an independent full-data candidate.",
    }


def build_candidates(full_rows, demo_manifest, split_dir, budget, seed):
    rows, _ = calibration_set_labels(full_rows)
    candidates = [load_demo_candidate(demo_manifest, split_dir)]

    p1_search = search_joint_assignment(
        rows, "calibration_set", budget, seed,
        owner_counts=(7, 3, 2), participant_counts=(34, 11, 11),
    )
    for variant in ("retention", "coverage"):
        found = p1_search[variant]
        split_rows, excluded = select_matching_records(
            rows, found["participant_assignment"], found["owner_assignment"], "calibration_set"
        )
        candidates.append({
            "candidate_id": f"P1_{variant.upper()}", "split_protocol": "P1",
            "search_variant": variant, "split_rows": split_rows, "excluded": excluded,
            "feasibility_status": search_status(bool(found)),
            "search_notes": "Participant and calibration-set co-assignment; mismatched records explicitly excluded.",
            "search": found,
        })

    signature_assignment, _ = stratified_owner_assignment(
        rows, "full_calibration_signature", 5, (0.6, 0.2, 0.2), seed
    )
    candidates.append({
        "candidate_id": "P2_CALIBRATION_STRATIFIED", "split_protocol": "P2",
        "search_variant": "signature_stratified", "split_rows": owner_split(rows, signature_assignment, "full_calibration_signature"),
        "excluded": [], "feasibility_status": "PROVEN_FEASIBLE",
        "search_notes": "Exact signature is assigned to one split; participant overlap is allowed.",
    })

    identifier_assignment, _ = stratified_owner_assignment(
        rows, "camera", 5, (0.6, 0.2, 0.2), seed
    )
    candidates.append({
        "candidate_id": "P3_IDENTIFIER_STRATIFIED", "split_protocol": "P3",
        "search_variant": "identifier_stratified", "split_rows": owner_split(rows, identifier_assignment, "camera"),
        "excluded": [], "feasibility_status": "PROVEN_FEASIBLE",
        "search_notes": "Camera identifier is assigned to one split; identifier is not a verified physical-camera ID.",
    })

    p4_search = search_joint_assignment(
        rows, "camera", budget, seed + 1000,
        owner_counts=(31, 10, 10), participant_counts=(34, 11, 11),
    )
    for variant in ("retention", "coverage"):
        found = p4_search[variant]
        split_rows, excluded = select_matching_records(
            rows, found["participant_assignment"], found["owner_assignment"], "camera"
        )
        candidates.append({
            "candidate_id": f"P4_{variant.upper()}", "split_protocol": "P4",
            "search_variant": variant, "split_rows": split_rows, "excluded": excluded,
            "feasibility_status": search_status(bool(found)),
            "search_notes": "Participant and camera-identifier co-assignment; mismatched records explicitly excluded.",
            "search": found,
        })
    return candidates, {
        "P1": {"budget": budget, "seed": seed, "owner_unit": "calibration_set"},
        "P4": {"budget": budget, "seed": seed + 1000, "owner_unit": "camera_identifier"},
    }


def _split_signature_rows(split_rows):
    return {
        split: unique_rows(rows, "full_calibration_signature")
        for split, rows in split_rows.items()
    }


def _range_text(rows, field):
    if not rows:
        return "empty"
    values = [float(row[field]) for row in rows]
    return f"{min(values):.9g}:{max(values):.9g}"


def participant_assignment_counts(split_rows, participant_assignment=None):
    active = {
        split: len({row["participant"] for row in split_rows[split]})
        for split in SPLITS
    }
    if participant_assignment is None:
        return {"assigned": None, "active": active}
    assigned = Counter(participant_assignment.values())
    return {
        "assigned": {split: assigned[split] for split in SPLITS},
        "active": active,
    }


def out_of_train_support_counts(split_rows, field):
    signatures = _split_signature_rows(split_rows)
    train_values = [float(row[field]) for row in signatures["train"]]
    if not train_values:
        return {"validation": len(signatures["validation"]), "test": len(signatures["test"])}
    low, high = min(train_values), max(train_values)
    return {
        split: sum(not low <= float(row[field]) <= high for row in signatures[split])
        for split in ("validation", "test")
    }


def candidate_metrics(candidate, full_edges_by_count, universe_signature_count):
    split_rows = candidate["split_rows"]
    validate_record_partition(split_rows, candidate["excluded"])
    overlaps = split_overlaps(split_rows)
    signature_rows = _split_signature_rows(split_rows)
    used_signatures = {
        row["full_calibration_signature"]
        for rows in split_rows.values() for row in rows
    }
    split_counts = {split: len(signature_rows[split]) for split in SPLITS}
    sensitivity = {}
    for bins, edges in full_edges_by_count.items():
        sensitivity[str(bins)] = {
            split: bin_support(histogram([float(row["fx"]) for row in signature_rows[split]], edges), required=1)
            for split in SPLITS
        }
    five = sensitivity["5"]
    hard_constraints = {
        "P0": ("participant_overlap",),
        "P1": ("participant_overlap", "exact_intrinsic_overlap", "exact_full_overlap"),
        "P2": ("exact_intrinsic_overlap", "exact_full_overlap"),
        "P3": ("camera_identifier_overlap",),
        "P4": ("participant_overlap", "exact_intrinsic_overlap", "exact_full_overlap", "camera_identifier_overlap"),
    }[candidate["split_protocol"]]
    hard_pass = all(overlaps[field] == 0 for field in hard_constraints)
    if not hard_pass:
        candidate["feasibility_status"] = "HARD_CONSTRAINT_FAILED"
    participant_counts = participant_assignment_counts(
        split_rows,
        candidate.get("search", {}).get("participant_assignment"),
    )
    active = participant_counts["active"]
    assigned = participant_counts["assigned"]
    out_of_support = out_of_train_support_counts(split_rows, "fx")
    minimum_active_evaluation = min(active["validation"], active["test"])
    return {
        "candidate_id": candidate["candidate_id"],
        "split_protocol": candidate["split_protocol"],
        "search_variant": candidate["search_variant"],
        "feasibility_status": candidate["feasibility_status"],
        "hard_constraints": ";".join(hard_constraints),
        "hard_constraints_pass": hard_pass,
        "total_calibration_signatures": universe_signature_count,
        "retained_calibration_signatures": len(used_signatures),
        "train_calibration_signatures": split_counts["train"],
        "validation_calibration_signatures": split_counts["validation"],
        "test_calibration_signatures": split_counts["test"],
        "excluded_calibration_signatures": universe_signature_count - len(used_signatures),
        "signature_retention_rate": len(used_signatures) / universe_signature_count,
        "used_camera_records": sum(len(rows) for rows in split_rows.values()),
        "excluded_camera_records": len(candidate["excluded"]),
        "record_retention_rate": sum(len(rows) for rows in split_rows.values()) / (
            sum(len(rows) for rows in split_rows.values()) + len(candidate["excluded"])
        ) if candidate["excluded"] else 1.0,
        "assigned_participant_definition": (
            "joint-search participant assignment before record exclusion"
            if assigned is not None else "not_applicable_participant_overlap_allowed"
        ),
        "assigned_train_participants": assigned["train"] if assigned is not None else "not_applicable",
        "assigned_validation_participants": assigned["validation"] if assigned is not None else "not_applicable",
        "assigned_test_participants": assigned["test"] if assigned is not None else "not_applicable",
        "active_participant_definition": "distinct participants with at least one retained camera record",
        "active_train_participants": active["train"],
        "active_validation_participants": active["validation"],
        "active_test_participants": active["test"],
        "train_participants": active["train"],
        "validation_participants": active["validation"],
        "test_participants": active["test"],
        "minimum_active_evaluation_participants": minimum_active_evaluation,
        **{
            f"active_evaluation_participants_at_least_{threshold}": minimum_active_evaluation >= threshold
            for threshold in (1, 3, 5, 10)
        },
        "train_sequences": len({row["sequence"] for row in split_rows["train"]}),
        "validation_sequences": len({row["sequence"] for row in split_rows["validation"]}),
        "test_sequences": len({row["sequence"] for row in split_rows["test"]}),
        "train_focal_coverage": _range_text(signature_rows["train"], "fx"),
        "validation_focal_coverage": _range_text(signature_rows["validation"], "fx"),
        "test_focal_coverage": _range_text(signature_rows["test"], "fx"),
        "validation_signatures_outside_train_fx_support": out_of_support["validation"],
        "test_signatures_outside_train_fx_support": out_of_support["test"],
        "minimum_bin_support": min(five["validation"]["minimum_bin_support"], five["test"]["minimum_bin_support"]),
        "empty_bin_count": five["validation"]["empty_bin_count"] + five["test"]["empty_bin_count"],
        "participant_overlap": overlaps["participant_overlap"],
        "sequence_overlap": overlaps["sequence_overlap"],
        "camera_key_overlap": overlaps["camera_key_overlap"],
        "exact_intrinsic_overlap": overlaps["exact_intrinsic_overlap"],
        "exact_full_overlap": overlaps["exact_full_overlap"],
        "camera_identifier_overlap": overlaps["camera_identifier_overlap"],
        "physical_camera_identity": "not_verifiable",
        "bin_sensitivity": sensitivity,
        "overlap_details": overlaps,
        "academic_limitations": candidate["search_notes"],
    }


def sampling_analysis(candidate, edges):
    rows = candidate["split_rows"]["train"]
    signatures = unique_rows(rows, "full_calibration_signature")
    signature_bin = {
        row["full_calibration_signature"]: assign_bin(float(row["fx"]), edges)
        for row in signatures
    }
    counts = histogram([float(row["fx"]) for row in signatures], edges)
    multiplicity = Counter(row["full_calibration_signature"] for row in rows)
    signature_rows = defaultdict(list)
    for row in rows:
        signature_rows[row["full_calibration_signature"]].append(row)
    results, weight_rows = [], []
    for protocol in SAMPLING_PROTOCOLS:
        if protocol == "S0":
            raw = {signature: float(count) for signature, count in multiplicity.items()}
            description = "camera-key proportional exposure"
            clipping = "none"
        elif protocol == "S1":
            raw = {signature: 1.0 for signature in multiplicity}
            description = "equal unique-calibration exposure"
            clipping = "none"
        elif protocol == "S2":
            factors = bin_weight_factors(counts, 1.0)
            raw = {signature: factors[bin_index] for signature, bin_index in signature_bin.items()}
            description = "full focal-bin balancing"
            clipping = "none"
        elif protocol == "S3":
            factors = bin_weight_factors(counts, 0.5)
            raw = {signature: factors[bin_index] for signature, bin_index in signature_bin.items()}
            description = "partial focal-bin balancing alpha=0.5"
            clipping = "max 3x uniform probability"
        else:
            threshold = max(10, math.ceil(max(counts) * 0.10))
            supported = {index for index, count in enumerate(counts) if count >= threshold}
            raw = {signature: (1.0 if bin_index in supported else 0.0) for signature, bin_index in signature_bin.items()}
            description = f"restricted support bins with at least {threshold} signatures"
            clipping = "unsupported bins receive zero design weight"
        if not any(raw.values()):
            probabilities = {key: 0.0 for key in raw}
        elif protocol == "S3":
            probabilities = clip_and_normalize(raw, max_multiple=3.0)
        else:
            total = sum(raw.values())
            probabilities = {key: value / total for key, value in raw.items()}
        bin_exposure = [0.0] * (len(edges) - 1)
        participant_exposure = Counter()
        identifier_exposure = Counter()
        for signature, probability in probabilities.items():
            bin_exposure[signature_bin[signature]] += probability
            signature_observations = signature_rows[signature]
            owners = Counter(row["participant"] for row in signature_observations)
            identifiers = Counter(row["camera"] for row in signature_observations)
            for participant, count in owners.items():
                participant_exposure[participant] += probability * count / sum(owners.values())
            for identifier, count in identifiers.items():
                identifier_exposure[identifier] += probability * count / sum(identifiers.values())
            weight_rows.append({
                "candidate_id": candidate["candidate_id"], "sampling_protocol": protocol,
                "calibration_signature_id": signature, "focal_bin": signature_bin[signature] + 1,
                "camera_key_multiplicity": multiplicity[signature], "raw_weight": raw[signature],
                "normalized_weight": probabilities[signature],
            })
        positive = [value for value in probabilities.values() if value > 0]
        uniform = 1.0 / len(probabilities) if probabilities else 0.0
        results.append({
            "candidate_id": candidate["candidate_id"], "sampling_protocol": protocol,
            "description": description, "signature_count": len(probabilities),
            "positive_weight_signature_count": len(positive), "weight_sum": sum(probabilities.values()),
            "effective_sample_size": effective_sample_size(list(probabilities.values())),
            "weight_concentration_max_over_uniform": max(positive) / uniform if positive and uniform else 0.0,
            "maximum_signature_probability": max(positive) if positive else 0.0,
            "bin_exposure": json.dumps(bin_exposure),
            "participant_exposure_max": max(participant_exposure.values(), default=0.0),
            "camera_identifier_exposure_max": max(identifier_exposure.values(), default=0.0),
            "clipping": clipping,
        })
    return results, weight_rows


def alpha_sensitivity_analysis(candidate, edges):
    signatures = unique_rows(candidate["split_rows"]["train"], "full_calibration_signature")
    signature_bin = {
        row["full_calibration_signature"]: assign_bin(float(row["fx"]), edges)
        for row in signatures
    }
    counts = histogram([float(row["fx"]) for row in signatures], edges)
    result = []
    for alpha in (0.0, 0.25, 0.5, 1.0):
        factors = bin_weight_factors(counts, alpha)
        raw = {signature: factors[index] for signature, index in signature_bin.items()}
        total = sum(raw.values())
        probabilities = {signature: value / total for signature, value in raw.items()}
        exposure = [0.0] * (len(edges) - 1)
        for signature, probability in probabilities.items():
            exposure[signature_bin[signature]] += probability
        uniform = 1.0 / len(probabilities) if probabilities else 0.0
        result.append({
            "candidate_id": candidate["candidate_id"],
            "alpha": alpha,
            "signature_count": len(probabilities),
            "effective_sample_size": effective_sample_size(list(probabilities.values())),
            "weight_concentration_max_over_uniform": (
                max(probabilities.values()) / uniform if probabilities and uniform else 0.0
            ),
            "bin_exposure": json.dumps(exposure),
            "clipping": "none; S3 separately reports alpha=0.5 with 3x clipping",
            "status": "sensitivity_candidate_not_finalized",
        })
    return result


def focal_coverage_rows(candidate, full_edges_by_count):
    result = []
    signature_rows = _split_signature_rows(candidate["split_rows"])
    for bins, edges in full_edges_by_count.items():
        for split in SPLITS:
            counts = histogram([float(row["fx"]) for row in signature_rows[split]], edges)
            participants = [set() for _ in range(bins)]
            identifiers = [set() for _ in range(bins)]
            for row in candidate["split_rows"][split]:
                index = assign_bin(float(row["fx"]), edges)
                participants[index].add(row["participant"])
                identifiers[index].add(row["camera"])
            for index in range(bins):
                result.append({
                    "candidate_id": candidate["candidate_id"], "split_protocol": candidate["split_protocol"],
                    "split": split, "focal_axis": "fx", "bin_method": "full_range_equal_width",
                    "bin_count": bins, "bin_index": index + 1, "lower_edge": edges[index],
                    "upper_edge": edges[index + 1], "unique_signature_count": counts[index],
                    "participant_count": len(participants[index]), "camera_identifier_count": len(identifiers[index]),
                })
    return result


def bin_definition_comparison(rows):
    signature_rows = unique_rows(rows, "full_calibration_signature")
    result = []
    for variable in ("fx", "fy", "nfx", "nfy", "hfov_deg", "vfov_deg"):
        values = [float(row[variable]) for row in signature_rows]
        low_q, high_q = (float(value) for value in np.quantile(np.asarray(values), [.05, .95]))
        restricted = [value for value in values if low_q <= value <= high_q]
        for bins in BIN_COUNTS:
            methods = {
                "equal_width_full_range": (values, equal_width_edges(values, bins), 0),
                "quantile_full_metadata": (values, quantile_edges(values, bins), 0),
                "equal_width_restricted_q05_q95": (
                    restricted, equal_width_edges(restricted, bins), len(values) - len(restricted)
                ),
            }
            for method, (source, edges, excluded) in methods.items():
                counts = histogram(source, edges)
                result.append({
                    "variable": variable, "method": method, "bin_count": bins,
                    "edges": json.dumps(edges), "counts": json.dumps(counts),
                    "minimum_bin_support": min(counts), "empty_bin_count": sum(count == 0 for count in counts),
                    "included_signatures": len(source), "excluded_signatures": excluded,
                    "support_min": min(source), "support_max": max(source),
                    "status": "candidate_only_not_finalized",
                })
    return result


def candidate_bin_sensitivity(coverage_rows, candidate_id):
    result = {}
    for bins in BIN_COUNTS:
        by_split = {}
        for split in SPLITS:
            counts = [
                int(row["unique_signature_count"])
                for row in coverage_rows
                if row["candidate_id"] == candidate_id
                and int(row["bin_count"]) == bins
                and row["split"] == split
            ]
            by_split[split] = {
                "counts": counts,
                "minimum": min(counts),
                "empty_bins": sum(count == 0 for count in counts),
                "minimum_requirement": {
                    str(required): all(count >= required for count in counts)
                    for required in (1, 2, 5, 10)
                },
            }
        result[str(bins)] = by_split
    return result


def legacy_leakage(candidate, demo_rows, split_dir):
    old = load_demo_candidate(DEFAULT_DEMO_MANIFEST, split_dir)["split_rows"]
    new_test = candidate["split_rows"]["test"]
    result = {}
    for old_split in SPLITS:
        old_rows = old[old_split]
        result[old_split] = {
            "participant_overlap": len({row["participant"] for row in old_rows} & {row["participant"] for row in new_test}),
            "exact_full_calibration_overlap": len({row["full_calibration_signature"] for row in old_rows} & {row["full_calibration_signature"] for row in new_test}),
            "camera_identifier_overlap": len({row["camera"] for row in old_rows} & {row["camera"] for row in new_test}),
        }
    return result


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def protected_paths(demo_manifest, split_dir):
    candidates = [
        demo_manifest,
        Path(split_dir) / "train.txt", Path(split_dir) / "val.txt", Path(split_dir) / "test.txt",
        Path(split_dir) / "split_report.json",
        HANDCALIB_ROOT / "results/02i0_gigahands_focal_coverage_feasibility.yaml",
        HANDCALIB_ROOT / "results/02i1_gigahands_full_focal_coverage.yaml",
        HANDCALIB_ROOT / "configs/02_anycalib_finetune.yaml",
        HANDCALIB_ROOT / "runs/02_anycalib_finetune/train/checkpoint_best.pt",
        HANDCALIB_ROOT / "runs/02_gigahands_full_focal_coverage/tables/gigahands_full_camera_metadata.csv",
    ]
    return [path for path in candidates if Path(path).exists()]


def _save_figure(fig, figures, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"{name}.{suffix}")
    plt.close(fig)


def plot_figures(output, full_rows, candidate_metrics_rows, coverage_rows, sampling_rows):
    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": .25, "font.size": 9})
    signatures = unique_rows(full_rows, "full_calibration_signature")

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist([float(row["fx"]) for row in full_rows], bins=35, density=True, color="0.8", edgecolor="0.5", label="camera-key weighted")
    axis.hist([float(row["fx"]) for row in signatures], bins=35, density=True, histtype="step", color="black", linewidth=1.5, label="unique signature")
    axis.set(xlabel="fx (px)", ylabel="Density", title="Full focal distribution")
    axis.legend()
    _save_figure(fig, figures, "01_full_focal_and_unique_calibration")

    participant_sets = defaultdict(set)
    for row in full_rows:
        participant_sets[row["participant"]].add(row["full_calibration_signature"])
    participants = sorted(participant_sets)
    matrix = np.zeros((len(participants), len(participants)))
    for i, left in enumerate(participants):
        for j, right in enumerate(participants):
            matrix[i, j] = len(participant_sets[left] & participant_sets[right])
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Greys", interpolation="nearest", aspect="auto")
    axis.set(xlabel="Participant index", ylabel="Participant index", title="Shared exact calibration signatures")
    fig.colorbar(image, ax=axis, label="Shared signatures")
    _save_figure(fig, figures, "02_participant_calibration_overlap")

    protocol_rows = [row for row in candidate_metrics_rows if row["split_protocol"] != "P0"]
    fig, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(protocol_rows))
    retained = [row["retained_calibration_signatures"] for row in protocol_rows]
    excluded = [row["excluded_calibration_signatures"] for row in protocol_rows]
    axis.bar(x, retained, color="0.65", edgecolor="black", label="retained")
    axis.bar(x, excluded, bottom=retained, color="white", edgecolor="black", hatch="//", label="excluded")
    axis.set_xticks(x, [row["candidate_id"] for row in protocol_rows], rotation=30, ha="right")
    axis.set(ylabel="Unique calibration signatures", title="Candidate retention")
    axis.legend()
    _save_figure(fig, figures, "03_candidate_retention")

    fig, axis = plt.subplots(figsize=(10, 4))
    markers = {"train": "o", "validation": "s", "test": "^"}
    for index, row in enumerate(protocol_rows):
        for offset, split in zip((-.2, 0, .2), SPLITS):
            text_range = row[f"{split}_focal_coverage"]
            if text_range == "empty":
                continue
            low, high = map(float, text_range.split(":"))
            axis.plot([index + offset, index + offset], [low, high], color="black", marker=markers[split], markevery=[0, 1], label=split if index == 0 else None)
    axis.set_xticks(range(len(protocol_rows)), [row["candidate_id"] for row in protocol_rows], rotation=30, ha="right")
    axis.set(ylabel="fx range (px)", title="Train/Validation/Test focal coverage")
    axis.legend()
    _save_figure(fig, figures, "04_split_focal_coverage")

    five_test = [row for row in coverage_rows if row["bin_count"] == 5 and row["split"] in ("validation", "test")]
    labels = sorted({row["candidate_id"] for row in five_test})
    minimum = []
    for label in labels:
        counts = [row["unique_signature_count"] for row in five_test if row["candidate_id"] == label]
        minimum.append(min(counts) if counts else 0)
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(range(len(labels)), minimum, color="0.7", edgecolor="black")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set(ylabel="Minimum unique signatures per eval bin", title="Five-bin evaluation support")
    _save_figure(fig, figures, "05_focal_bin_support")

    chosen_id = "P1_COVERAGE" if any(row["candidate_id"] == "P1_COVERAGE" for row in sampling_rows) else sampling_rows[0]["candidate_id"]
    chosen = [row for row in sampling_rows if row["candidate_id"] == chosen_id]
    fig, axis = plt.subplots(figsize=(8, 4))
    for row in chosen:
        exposure = json.loads(row["bin_exposure"])
        axis.plot(range(1, len(exposure) + 1), exposure, marker="o", label=row["sampling_protocol"])
    axis.set(xlabel="fx bin", ylabel="Expected exposure probability", title=f"Sampling exposure: {chosen_id}")
    axis.legend(ncol=3)
    _save_figure(fig, figures, "06_sampling_bin_exposure")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    protocols = [row["sampling_protocol"] for row in chosen]
    axes[0].bar(protocols, [row["effective_sample_size"] for row in chosen], color="0.7", edgecolor="black")
    axes[0].set(ylabel="Signature-level ESS", title="Effective sample size")
    axes[1].bar(protocols, [row["weight_concentration_max_over_uniform"] for row in chosen], color="white", edgecolor="black", hatch="//")
    axes[1].set(ylabel="Max weight / uniform", title="Weight concentration")
    _save_figure(fig, figures, "07_weight_concentration_and_ess")

    fig, axis = plt.subplots(figsize=(8, 5))
    for row in protocol_rows:
        axis.scatter(row["record_retention_rate"], row["minimum_bin_support"], s=35 + row["retained_calibration_signatures"] / 4, facecolors="white", edgecolors="black")
        axis.annotate(row["candidate_id"], (row["record_retention_rate"], row["minimum_bin_support"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set(xlabel="Camera-record retention rate", ylabel="Minimum eval-bin signatures", title="Independence, retention, and coverage trade-off")
    _save_figure(fig, figures, "08_independence_retention_coverage_tradeoff")


def compact_summary(summary):
    keys = (
        "experiment", "status", "analysis_scope", "dataset_scope", "source_metadata",
        "source_hashes", "repository_commit", "study_questions", "analysis_units",
        "independence_definitions", "candidate_split_protocols", "candidate_sampling_protocols",
        "focal_axis_candidates", "candidate_bin_definitions", "overlap_constraints",
        "optimization_objectives", "search_method", "search_budget", "feasibility_status",
        "candidate_summaries", "dataset_retention", "focal_coverage", "sampling_weight_analysis",
        "evaluation_metric_definitions", "legacy_checkpoint_leakage_risks", "RGB_requirements",
        "statistical_limitations", "recommended_candidates", "unresolved_decisions",
        "decision_status", "training_runs", "inference_runs", "original_files_preserved",
    )
    compact = {key: summary[key] for key in keys}
    definitions = dict(compact["candidate_bin_definitions"])
    definitions["comparison"] = {
        key: value for key, value in definitions["comparison"].items()
        if key.startswith("fx/")
    }
    compact["candidate_bin_definitions"] = definitions
    focal = {}
    for candidate_id, values in compact["focal_coverage"].items():
        item = {key: value for key, value in values.items() if key != "full_range_equal_width_sensitivity"}
        sensitivity = values["full_range_equal_width_sensitivity"]
        if isinstance(sensitivity, dict):
            item["full_range_equal_width_sensitivity"] = {
                bins: {
                    split: {
                        "counts": split_values["counts"],
                        "minimum": split_values["minimum"],
                        "empty_bins": split_values["empty_bins"],
                    }
                    for split, split_values in by_split.items()
                }
                for bins, by_split in sensitivity.items()
            }
        focal[candidate_id] = item
    compact["focal_coverage"] = focal
    compact["sampling_weight_analysis"] = {
        key: value for key, value in compact["sampling_weight_analysis"].items()
        if key.startswith("P1_COVERAGE/") or key == "execution_design"
    }
    compact["legacy_checkpoint_leakage_risks"] = {
        key: value for key, value in compact["legacy_checkpoint_leakage_risks"].items()
        if key in ("P1_COVERAGE", "P2_CALIBRATION_STRATIFIED", "P4_RETENTION")
    }
    return compact


def analysis_report(summary):
    candidates = summary["candidate_summaries"]
    lines = [
        "# 02-I.2 GigaHands focal 균형 학습·평가 프로토콜 설계", "",
        "## 1. 연구 목적", "", "GigaHands만 사용해 calibration 중복을 통제하고 focal coverage와 독립성을 함께 비교할 수 있는 metadata-only 후보를 설계했다.", "",
        "## 2. 기존 실험에서 발견된 문제", "", "기존 fine-tuning 성능은 Train-only constant prior와 비슷했고, camera-key 균형은 반복 calibration을 더 자주 노출했다.", "",
        "## 3. 연구 질문 및 가설", "", "기록 일부를 명시적으로 제외하면 참가자와 calibration이 독립인 후보를 만들 수 있는지, 그리고 부분 균형 sampling이 완전 균형보다 가중치 집중을 줄이는지 조사했다.", "",
        "## 4. 사용 데이터와 분석 범위", "", "GigaHands 236개 sequence, 11,756개 camera record, 602개 exact calibration만 사용했다. RGB, 학습, 추론은 사용하지 않았다.", "",
        "## 5. 독립 표본의 정의", "", "주 분석 단위는 exact full calibration signature다. camera_key, participant, camera identifier, 물리 카메라를 서로 다른 개념으로 유지했다. 물리 카메라는 not_verifiable이다.", "",
        "## 6. 분할 설계 방법", "", "P1은 participant와 calibration-set을 공동 배정하고 불일치 record를 제외한다. P2는 signature, P3는 camera identifier를 한 split에만 배정한다. P4는 participant와 camera identifier를 공동 배정한다.", "",
        "## 7. 탐색 조건과 변인 통제", "", f"고정 seed {summary['search_budget']['seed']}와 P1/P4 각각 {summary['search_budget']['iterations_per_joint_search']}회 탐색을 사용했다. 모델 성능은 목적 함수에 포함하지 않았다.", "",
        "## 8. 분할 후보 비교 결과", "",
        "|Candidate|Protocol|Signatures|Record retention|Assigned participants|Active participants|Eval min bin|Val/Test outside Train fx|Participant overlap|Calibration overlap|ID overlap|",
        "|---|---|---:|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"|{row['candidate_id']}|{row['split_protocol']}|{row['retained_calibration_signatures']}|"
            f"{row['record_retention_rate']:.3f}|{row['assigned_train_participants']}/{row['assigned_validation_participants']}/{row['assigned_test_participants']}|"
            f"{row['active_train_participants']}/{row['active_validation_participants']}/{row['active_test_participants']}|"
            f"{row['minimum_bin_support']}|{row['validation_signatures_outside_train_fx_support']}/{row['test_signatures_outside_train_fx_support']}|{row['participant_overlap']}|"
            f"{row['exact_full_overlap']}|{row['camera_identifier_overlap']}|"
        )
    lines += [
        "", "## 9. Focal 균형 sampling 후보", "", "S0 camera-key 비례, S1 signature 균등, S2 bin 완전 균형, S3 alpha=0.5 부분 균형과 3배 clipping, S4 제한 support를 비교했다. alpha=0, 0.25, 0.5, 1.0의 unclipped 민감도도 별도로 기록했다. 어느 방식도 확정하지 않았다.", "",
        "## 10. 평가 프로토콜", "", "Primary 오차는 기존 pair max relative focal error를 유지하고 camera-key별 frame median 후 camera-key 동일 가중 평균을 보고한다. M1 signature-equal, M2 사전 고정 bin-macro, M3 구간별 평균·중앙값·within-5%·표본 수·불확실성, M4 Train-only constant prior를 함께 보고한다.", "",
        "## 11. 데이터 손실과 실용성", "", "P2/P3는 metadata record를 보존하지만 participant 또는 sequence 독립성을 보장하지 않는다. P1/P4는 strict 조건을 위해 불일치 record를 버리므로 RGB 확보 비용과 보존율을 함께 봐야 한다.", "",
        "## 12. Calibration 중복 및 누출 위험", "", "602개 signature는 물리 카메라 602대를 뜻하지 않는다. 기존 checkpoint가 새 Test signature 또는 participant를 이미 보았을 수 있어 metadata overlap을 별도로 기록했다.", "",
        "## 13. 통계적 불확실성", "", "Frame을 독립 표본으로 세지 않는다. Calibration 또는 participant cluster bootstrap을 권장하며 1~2 signature bin은 표본 수와 기술 통계만 보고한다. 전체 fx 5-bin 양 끝은 각각 signature 2개뿐이므로 세 split 모두가 모든 bin을 포함하는 것은 산술적으로 불가능하다.", "",
        "## 14. 후속 학습 비교 실험 설계", "", "A0 pretrained, A1 기존 sampler, A2 signature 균형, A3 부분 bin 균형, A4 Train-only constant prior를 같은 optimizer-step budget과 평가 집합으로 비교한다.", "",
        "## 15. 필요한 RGB 영상 목록", "", "후보는 metadata-only다. full_hand_poses에는 RGB가 없으며 정확한 frame 수, 파일 크기, URL은 unknown으로 남겼다.", "",
        "## 16. 연구 해석", "", "전체 연결 성분은 모든 record 보존 분할을 막지만, record 제외를 허용한 후보의 존재까지 막지는 않는다. 이번 탐색은 그 차이를 수치화한다.", "",
        "## 17. 학술적 한계", "", "Heuristic 후보는 최적성을 증명하지 않는다. Metadata를 보고 설계한 평가는 metadata-designed evaluation이며 완전히 보지 않은 Test로 과장할 수 없다.", "",
        "## 18. 결론", "", "엄격한 독립성과 높은 보존율은 동시에 최대화되지 않는다. P1/P4는 독립성 후보, P2는 보존율 높은 calibration 평가 후보, S3는 과도한 반복을 제한하는 실용 후보다.", "",
        "## 19. 최종 결정이 필요한 항목", "", "독립성 수준, focal 축, bin 수, 최소 support, 허용 record 손실, 실제 학습에 포함할 sampling 방식을 연구자가 선택해야 한다.", "",
        "## 20. 다음 단계", "", "선택된 metadata 후보의 실제 RGB 존재와 decode 가능성을 먼저 검증한 뒤, split과 sampler를 별도 승인·고정하고 학습을 시작한다.", "",
    ]
    return "\n".join(lines)


def _flat_candidate_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in ("bin_sensitivity", "overlap_details")}


def _geo_state():
    path = HANDCALIB_ROOT / "GeoCalib"
    if not path.exists():
        return {"exists": False}
    result = {"exists": True}
    for name, args in (
        ("head", ["rev-parse", "HEAD"]),
        ("status", ["status", "--porcelain"]),
    ):
        process = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        result[name] = process.stdout.strip() if process.returncode == 0 else "not_a_git_repository"
    return result


def design_protocol(root, demo_manifest, split_dir, output, result_yaml, budget, seed, overwrite=False):
    output = prepare_output(output, overwrite)
    result_yaml = Path(result_yaml)
    if result_yaml.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result: {result_yaml}")
    candidate_dir = output / "candidates"
    validation_dir = output / "validation"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    protected = protected_paths(demo_manifest, split_dir)
    protected_before = {str(path.relative_to(REPOSITORY_ROOT)): full_audit.sha256_file(path) for path in protected}
    geo_before = _geo_state()
    full_rows, integrity = full_audit.read_full_metadata(root)
    source_aggregate_before = full_audit.aggregate_source_hash(integrity)
    demo_rows = full_audit.read_demo_manifest(demo_manifest)
    if len(full_rows) != 11756 or len({row["full_calibration_signature"] for row in full_rows}) != 602:
        raise RuntimeError("02-I.1 population reproduction failed")
    if len(demo_rows) != 245 or len({row["full_calibration_signature"] for row in demo_rows}) != 147:
        raise RuntimeError("02-I.0 population reproduction failed")

    full_signature_rows = unique_rows(full_rows, "full_calibration_signature")
    full_edges_by_count = {
        bins: equal_width_edges([float(row["fx"]) for row in full_signature_rows], bins)
        for bins in BIN_COUNTS
    }
    bin_definition_rows = bin_definition_comparison(full_rows)
    candidates, search_log = build_candidates(full_rows, demo_manifest, split_dir, budget, seed)
    metrics_rows = []
    coverage_rows = []
    sampling_rows = []
    alpha_sensitivity_rows = []
    weight_rows = []
    exclusion_rows = []
    overlap_rows = []
    participant_rows = []
    rgb_rows = []
    candidate_comparison = []
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    for candidate in candidates:
        universe = 147 if candidate["split_protocol"] == "P0" else 602
        metrics = candidate_metrics(candidate, full_edges_by_count, universe)
        metrics_rows.append(metrics)
        coverage_rows.extend(focal_coverage_rows(candidate, full_edges_by_count))
        current_sampling, current_weights = sampling_analysis(candidate, full_edges_by_count[5])
        sampling_rows.extend(current_sampling)
        weight_rows.extend(current_weights)
        alpha_sensitivity_rows.extend(alpha_sensitivity_analysis(candidate, full_edges_by_count[5]))
        overlap_rows.append({"candidate_id": candidate["candidate_id"], **metrics["overlap_details"]})
        candidate_export_rows = []
        for split in SPLITS:
            rows = candidate["split_rows"][split]
            participants = sorted({row["participant"] for row in rows})
            for participant in participants:
                participant_group = [row for row in rows if row["participant"] == participant]
                participant_rows.append({
                    "candidate_id": candidate["candidate_id"], "split": split,
                    "participant": participant, "camera_record_count": len(participant_group),
                    "sequence_count": len({row["sequence"] for row in participant_group}),
                    "unique_signature_count": len({row["full_calibration_signature"] for row in participant_group}),
                })
            for row in rows:
                focal_bin = assign_bin(float(row["fx"]), full_edges_by_count[5]) + 1
                rgb = rgb_requirement_row(candidate["candidate_id"], split, row, focal_bin)
                if candidate["split_protocol"] == "P0":
                    rgb.update({
                        "existing_rgb_status": "available_local_demo_rgb",
                        "download_required": False,
                        "frame_count": row.get("frame_count", "unknown"),
                        "selection_reason": "existing locked demo baseline",
                    })
                rgb_rows.append(rgb)
                candidate_export_rows.append(rgb)
        write_csv(
            candidate_dir / f"candidate_{candidate['candidate_id'].lower()}.csv",
            candidate_export_rows,
        )
        for row in candidate["excluded"]:
            exclusion_rows.append({
                "candidate_id": candidate["candidate_id"], "camera_key": row["camera_key"],
                "participant": row["participant"], "sequence": row["sequence"], "camera": row["camera"],
                "calibration_signature_id": row["full_calibration_signature"],
                "reason": row["exclusion_reason"],
            })
        for sampling in current_sampling:
            candidate_comparison.append({
                **_flat_candidate_metrics(metrics),
                "sampling_protocol": sampling["sampling_protocol"],
                "weight_concentration": sampling["weight_concentration_max_over_uniform"],
                "effective_sample_size": sampling["effective_sample_size"],
                "RGB_availability": "available_demo" if candidate["split_protocol"] == "P0" else "metadata_only_full_sequences",
            })

    eligible = [row for row in metrics_rows if row["split_protocol"] != "P0" and row["hard_constraints_pass"]]
    strict = max((row for row in eligible if row["split_protocol"] == "P4"), key=lambda row: (row["minimum_bin_support"], row["record_retention_rate"]))
    retention = max(eligible, key=lambda row: (row["record_retention_rate"], row["minimum_bin_support"]))
    coverage = max(eligible, key=lambda row: (row["minimum_bin_support"], row["record_retention_rate"]))
    widest = max(eligible, key=lambda row: (
        float(row["test_focal_coverage"].split(":")[1]) - float(row["test_focal_coverage"].split(":")[0]),
        row["minimum_bin_support"],
    ))
    practical = max((row for row in eligible if row["split_protocol"] == "P1"), key=lambda row: (row["minimum_bin_support"], row["record_retention_rate"]))
    recommended = {
        "strictest_independence": strict["candidate_id"],
        "highest_record_retention": retention["candidate_id"],
        "widest_test_focal_range": widest["candidate_id"],
        "best_evaluation_bin_support": coverage["candidate_id"],
        "practical_participant_independent_training": practical["candidate_id"],
        "practical_sampling_candidate": "S3 partial balance with alpha=0.5 and 3x clipping",
        "selection_status": "recommendations_only_not_finalized",
    }
    leakage = {
        candidate_id: legacy_leakage(candidate_map[candidate_id], demo_rows, split_dir)
        for candidate_id in sorted(candidate_map)
    }
    sampling_compact = {
        row["candidate_id"] + "/" + row["sampling_protocol"]: {
            "signature_count": row["signature_count"],
            "positive_weight_signature_count": row["positive_weight_signature_count"],
            "effective_sample_size": row["effective_sample_size"],
            "max_over_uniform": row["weight_concentration_max_over_uniform"],
            "bin_exposure": json.loads(row["bin_exposure"]),
        }
        for row in sampling_rows
    }
    for row in alpha_sensitivity_rows:
        sampling_compact[f"{row['candidate_id']}/ALPHA_{row['alpha']:.2f}"] = {
            "signature_count": row["signature_count"],
            "effective_sample_size": row["effective_sample_size"],
            "max_over_uniform": row["weight_concentration_max_over_uniform"],
            "bin_exposure": json.loads(row["bin_exposure"]),
            "clipping": row["clipping"],
            "status": row["status"],
        }
    sampling_compact["execution_design"] = {
        "stage_1": "sample exact calibration signature from the selected S0-S4 probability",
        "stage_2": "cycle or sample without replacement across available sequence-camera records for that signature",
        "stage_3": "sample RGB frames without replacement within a selected video when verified frame counts permit",
        "epoch_budget": "define by a fixed number of sampled frames or optimizer steps, not by exhausting metadata rows",
        "optimizer_step_budget_candidate": 31975,
        "optimizer_step_budget_status": "legacy-compatible candidate; not finalized for the new data scale",
        "duplicate_frame_control": "requires RGB inventory before implementation",
        "rare_signature_control": "S3 caps signature probability at 3x uniform; S2 remains an uncapped diagnostic",
        "seed_policy": {
            "split_search_seed": seed,
            "sampler_seed_candidate": seed + 100,
            "training_seed_candidates": [42, 43, 44],
            "training_repeats_status": "resource-dependent and not finalized",
        },
        "controlled_training_variables": [
            "architecture", "pretrained initialization", "RGB preprocessing",
            "canonical pinhole supervision", "pinhole model", "loss", "optimizer",
            "learning rate", "weight decay", "scheduler", "batch size", "precision",
            "optimizer steps", "validation interval", "checkpoint rule", "evaluation code",
        ],
    }
    summary = {
        "experiment": "02i2_gigahands_focal_balance_protocol_design",
        "status": "design_completed",
        "analysis_scope": "CPU metadata-only protocol design; no final split, bin, sampler, training, inference, RGB decode, or download",
        "dataset_scope": "GigaHands only: local demo plus full_hand_poses camera metadata",
        "source_metadata": {
            "full_root": str(Path(root).relative_to(HANDCALIB_ROOT)),
            "demo_manifest": str(Path(demo_manifest).relative_to(HANDCALIB_ROOT)),
            "full_camera_records": 11756, "full_exact_signatures": 602,
            "participants": 56, "sequences": 236,
        },
        "source_hashes": {"optim_params_aggregate_sha256": source_aggregate_before, **protected_before},
        "repository_commit": git_output("rev-parse", "HEAD"),
        "study_questions": [
            "Can exclusions yield participant and calibration independent candidates?",
            "How much record retention is traded for independence and focal support?",
            "How do S0-S4 alter exposure concentration and signature-level ESS?",
        ],
        "analysis_units": {
            "camera_record": 11756, "sequence_camera": 11756,
            "exact_intrinsic_signature": 602, "exact_full_signature": 602,
            "participant": 56, "sequence": 236, "camera_identifier": 51,
            "calibration_set": 12, "physical_camera": "not_verifiable",
        },
        "independence_definitions": {
            "P0": "existing demo baseline",
            "P1": "participant + exact intrinsic/full calibration disjoint",
            "P2": "exact intrinsic/full calibration disjoint; participant overlap allowed",
            "P3": "camera identifier disjoint; not physical-camera proof",
            "P4": "participant + exact intrinsic/full calibration + camera identifier disjoint",
        },
        "candidate_split_protocols": ["P0", "P1", "P2", "P3", "P4"],
        "candidate_sampling_protocols": {
            "S0": "camera-key proportional", "S1": "unique-calibration equal",
            "S2": "full focal-bin equal", "S3": "partial alpha=0.5 with 3x clip",
            "S4": "restricted-support equal",
        },
        "focal_axis_candidates": {
            "fx": "priority candidate for metric compatibility; not finalized",
            "nfx": "same ordering here because every source resolution is 1280x720",
            "hfov": "geometric monotonic transform of fx at fixed width",
            "fy_nfy_vfov": "mandatory secondary coverage checks; fx balance is not fy balance",
        },
        "candidate_bin_definitions": {
            "counts": list(BIN_COUNTS), "primary_diagnostic": "full-range equal-width fx",
            "alternatives": ["quantile from full metadata", "q05-q95 restricted-support equal-width"],
            "finalized": False, "metadata_designed_evaluation": True,
            "edges": {str(bins): edges for bins, edges in full_edges_by_count.items()},
            "comparison": {
                f"{row['variable']}/{row['method']}/{row['bin_count']}": {
                    "counts": json.loads(row["counts"]),
                    "minimum": row["minimum_bin_support"],
                    "empty_bins": row["empty_bin_count"],
                    "excluded_signatures": row["excluded_signatures"],
                }
                for row in bin_definition_rows
                if row["variable"] in ("fx", "nfx", "hfov_deg")
            },
        },
        "overlap_constraints": {
            "always": ["camera_key", "frame/video when RGB becomes available"],
            "protocol_specific": ["participant", "exact intrinsic", "exact full", "camera identifier"],
        },
        "optimization_objectives": [
            "hard constraints", "minimum split support", "evaluation focal-bin support",
            "train signatures", "participant/sequence diversity", "record retention",
        ],
        "search_method": "fixed-budget deterministic randomized co-assignment with exact post-hoc record-level validation",
        "search_budget": {"iterations_per_joint_search": budget, "seed": seed, "optimality_proven": False, **search_log},
        "feasibility_status": {row["candidate_id"]: row["feasibility_status"] for row in metrics_rows},
        "candidate_summaries": [_flat_candidate_metrics(row) for row in metrics_rows],
        "dataset_retention": {row["candidate_id"]: {"signature": row["signature_retention_rate"], "camera_record": row["record_retention_rate"]} for row in metrics_rows},
        "focal_coverage": {row["candidate_id"]: {
            "train": row["train_focal_coverage"],
            "validation": row["validation_focal_coverage"],
            "test": row["test_focal_coverage"],
            "minimum_eval_five_bin_support": row["minimum_bin_support"],
            "full_range_equal_width_sensitivity": (
                candidate_bin_sensitivity(coverage_rows, row["candidate_id"])
                if row["candidate_id"] in ("P1_COVERAGE", "P2_CALIBRATION_STRATIFIED", "P4_RETENTION")
                else "see candidates/focal_bin_coverage.csv"
            ),
        } for row in metrics_rows},
        "sampling_weight_analysis": sampling_compact,
        "evaluation_metric_definitions": {
            "primary": "err_max=max(abs(fx_pred-fx_gt)/fx_gt, abs(fy_pred-fy_gt)/fy_gt), camera-key component-wise frame median, existing compatibility",
            "camera_key_mean": "equal-weight mean across camera keys after within-camera-key frame aggregation",
            "M1": "equal-weight exact calibration signatures after combining repeated sequence-camera estimates within signature",
            "M2": "precommitted nonempty focal-bin macro mean",
            "M3": "per-bin mean, median, within-5%, counts and uncertainty",
            "M4": "Train-only mean/median constant focal prior; no Validation/Test GT fitting",
            "uncertainty": "calibration-signature or participant cluster resampling; frames are not independent units",
        },
        "legacy_checkpoint_leakage_risks": leakage,
        "RGB_requirements": {
            "full_hand_poses_rgb": "not_present", "candidate_rows": len(rgb_rows),
            "frame_counts": "unknown except existing demo", "download_size": "unknown",
            "download_urls": "unknown", "metadata_row_is_not_assumed_to_be_downloadable_video": True,
        },
        "statistical_limitations": [
            "heuristic optimality not proven", "metadata-designed evaluation selection",
            "physical camera identity not verifiable", "pseudoreplication from repeated calibration",
            "sparse-bin uncertainty", "participant and camera-identifier concentration",
            "RGB availability not verified", "performance improvement not tested",
        ],
        "recommended_candidates": recommended,
        "unresolved_decisions": [
            "independence priority", "primary focal axis", "bin count and boundaries",
            "minimum signatures per evaluation bin", "acceptable record retention",
            "sampling protocols to train", "RGB sequences to acquire", "training repeat count",
        ],
        "decision_status": "awaiting_researcher_selection",
        "training_runs": 0,
        "inference_runs": 0,
        "original_files_preserved": True,
    }

    write_csv(candidate_dir / "candidate_comparison.csv", candidate_comparison)
    write_csv(candidate_dir / "split_candidates.csv", [_flat_candidate_metrics(row) for row in metrics_rows])
    write_csv(candidate_dir / "exclusion_reasons.csv", exclusion_rows)
    write_csv(candidate_dir / "calibration_overlap.csv", overlap_rows)
    write_csv(candidate_dir / "participant_coverage.csv", participant_rows)
    write_csv(candidate_dir / "focal_bin_coverage.csv", coverage_rows)
    write_csv(candidate_dir / "bin_definition_comparison.csv", bin_definition_rows)
    write_csv(candidate_dir / "weighting_candidates.csv", weight_rows)
    write_csv(candidate_dir / "effective_sample_size.csv", sampling_rows)
    write_csv(candidate_dir / "alpha_sensitivity.csv", alpha_sensitivity_rows)
    write_csv(candidate_dir / "rgb_requirements.csv", rgb_rows)
    plot_figures(output, full_rows, metrics_rows, coverage_rows, sampling_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "analysis_report.md").write_text(analysis_report(summary), encoding="utf-8")
    (validation_dir / "search_log.json").write_text(json.dumps(search_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (validation_dir / "hard_constraint_checks.json").write_text(json.dumps({row["candidate_id"]: {"pass": row["hard_constraints_pass"], "overlaps": row["overlap_details"]} for row in metrics_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_yaml.parent.mkdir(parents=True, exist_ok=True)
    result_yaml.write_text(yaml.safe_dump(compact_summary(summary), allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    _, integrity_after = full_audit.read_full_metadata(root)
    source_aggregate_after = full_audit.aggregate_source_hash(integrity_after)
    protected_after = {str(path.relative_to(REPOSITORY_ROOT)): full_audit.sha256_file(path) for path in protected}
    geo_after = _geo_state()
    if source_aggregate_before != source_aggregate_after:
        raise RuntimeError("Original optim_params metadata changed")
    if protected_before != protected_after:
        raise RuntimeError("Protected research input changed")
    if geo_before != geo_after:
        raise RuntimeError("GeoCalib state changed")
    metadata = {
        "command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": summary["repository_commit"], "seed": seed, "search_budget": budget,
        "gpu_used": False, "training_runs": 0, "inference_runs": 0,
        "checkpoint_loaded": False, "rgb_decoded": False, "downloads": 0,
        "source_aggregate_sha256_before": source_aggregate_before,
        "source_aggregate_sha256_after": source_aggregate_after,
        "protected_sha256_before": protected_before, "protected_sha256_after": protected_after,
        "geocalib_before": geo_before, "geocalib_after": geo_after,
        "original_dataset_modified": False, "original_manifest_modified": False,
        "original_split_modified": False, "existing_checkpoints_modified": False,
        "existing_results_modified": False, "geocalib_modified": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-yaml", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--search-budget", type=int, default=DEFAULT_SEARCH_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.search_budget <= 0:
        raise SystemExit("--search-budget must be positive")
    summary = design_protocol(
        args.root.resolve(), args.demo_manifest.resolve(), args.split_dir.resolve(),
        args.output.resolve(), args.result_yaml.resolve(), args.search_budget, args.seed,
        overwrite=args.overwrite,
    )
    print(f"status={summary['status']}")
    print(f"decision_status={summary['decision_status']}")
    print(f"output={args.output.resolve()}")
    print("training_runs=0 inference_runs=0 final_split=false final_sampler=false")


if __name__ == "__main__":
    main()
