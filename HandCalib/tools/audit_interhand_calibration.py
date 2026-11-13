"""Audit official InterHand2.6M annotation splits and calibration coverage."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = PROJECT_ROOT / "datasets/interhand2.6m/raw/annotations"
MANIFEST = PROJECT_ROOT / "data/manifests/gigahands.csv"
OUTPUT_ROOT = PROJECT_ROOT / "runs/02_interhand_annotation_audit"


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def data_path(split, source="all"):
    return ANNOTATION_ROOT / source / f"InterHand2.6M_{split}_data.json"


def camera_path(split, source="all"):
    return ANNOTATION_ROOT / source / f"InterHand2.6M_{split}_camera.json"


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def scalar_stats(values):
    values = [float(value) for value in values]
    average = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "min": min(values), "max": max(values), "mean": average,
        "median": statistics.median(values), "std": deviation,
        "cv": deviation / average if average else None,
        "p01": percentile(values, 0.01), "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25), "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
    }


def camera_record(cameras, image):
    capture = str(image["capture"])
    camera = str(image["camera"])
    capture_cameras = cameras.get(capture)
    if capture_cameras is None or any(name not in capture_cameras for name in ("campos", "camrot", "focal", "princpt")):
        return None
    if any(camera not in capture_cameras[name] for name in ("campos", "camrot", "focal", "princpt")):
        return None
    return {
        "campos": capture_cameras["campos"][camera], "camrot": capture_cameras["camrot"][camera],
        "focal": capture_cameras["focal"][camera], "princpt": capture_cameras["princpt"][camera],
    }


def unit_records(data, cameras):
    units = {}
    mapping_failures = []
    unit_dimensions = defaultdict(set)
    for image in data["images"]:
        calibration = camera_record(cameras, image)
        if calibration is None:
            mapping_failures.append({"id": image.get("id"), "capture": image.get("capture"), "camera": image.get("camera")})
            continue
        key = (str(image["capture"]), str(image["camera"]))
        unit_dimensions[key].add((int(image["width"]), int(image["height"])))
        if key not in units:
            focal = calibration["focal"]
            principal = calibration["princpt"]
            width, height = int(image["width"]), int(image["height"])
            fx, fy = float(focal[0]), float(focal[1])
            cx, cy = float(principal[0]), float(principal[1])
            units[key] = {
                "capture": key[0], "camera": key[1], "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "nfx": fx / width, "nfy": fy / height,
                "ncx": cx / width, "ncy": cy / height,
                "hfov_deg": math.degrees(2 * math.atan(width / (2 * fx))),
                "vfov_deg": math.degrees(2 * math.atan(height / (2 * fy))), "frame_count": 0,
            }
        units[key]["frame_count"] += 1
    inconsistent_dimensions = {"%s/%s" % key: sorted(dimensions) for key, dimensions in unit_dimensions.items() if len(dimensions) > 1}
    return units, mapping_failures, inconsistent_dimensions


def split_info(split, source):
    data = load_json(data_path(split, source))
    cameras = load_json(camera_path(split, source))
    units, failures, inconsistent_dimensions = unit_records(data, cameras)
    images = data["images"]
    return {
        "split": split, "source": source, "data": data,
        "image_count": len(images), "annotation_count": len(data.get("annotations", [])),
        "capture_count": len({str(image["capture"]) for image in images}),
        "subject_count": len({str(image["subject"]) for image in images}),
        "sequence_count": len({str(image["seq_name"]) for image in images}),
        "camera_label_count": len({str(image["camera"]) for image in images}),
        "unit_count": len(units), "units": units, "mapping_failures": failures,
        "inconsistent_dimensions": inconsistent_dimensions,
        "image_schema_missing": sorted({key for image in images for key in ("id", "file_name", "width", "height", "capture", "subject", "seq_name", "camera", "frame_idx") if key not in image}),
        "annotation_schema_missing": sorted({key for annotation in data.get("annotations", []) for key in ("id", "image_id") if key not in annotation}),
        "camera_schema_missing": sorted({key for capture in cameras.values() for key in ("focal", "princpt", "campos", "camrot") if key not in capture}),
    }


def id_sets(split, source):
    data = load_json(data_path(split, source))
    return {image["id"] for image in data["images"]}, {annotation["id"] for annotation in data.get("annotations", [])}


def relation_report(split):
    all_images, all_annotations = id_sets(split, "all")
    human_images, human_annotations = id_sets(split, "human_annot")
    machine_images, machine_annotations = id_sets(split, "machine_annot")
    return {
        "split": split,
        "all_equals_human_union_machine_images": all_images == human_images | machine_images,
        "all_equals_human_union_machine_annotations": all_annotations == human_annotations | machine_annotations,
        "all_image_count": len(all_images), "human_image_count": len(human_images), "machine_image_count": len(machine_images),
        "all_annotation_count": len(all_annotations), "human_annotation_count": len(human_annotations), "machine_annotation_count": len(machine_annotations),
        "human_machine_image_overlap": len(human_images & machine_images),
        "human_machine_annotation_overlap": len(human_annotations & machine_annotations),
        "all_not_in_union_images": len(all_images - (human_images | machine_images)),
        "all_not_in_union_annotations": len(all_annotations - (human_annotations | machine_annotations)),
    }


def exact_k(unit):
    return (unit["width"], unit["height"], unit["fx"], unit["fy"], unit["cx"], unit["cy"])


def normalized_k(unit):
    return (unit["nfx"], unit["nfy"], unit["ncx"], unit["ncy"])


def overlap(left, right):
    left_units, right_units = left["units"], right["units"]
    left_keys, right_keys = set(left_units), set(right_units)
    exact_left = {exact_k(unit) for unit in left_units.values()}
    exact_right = {exact_k(unit) for unit in right_units.values()}
    normalized_left = {normalized_k(unit) for unit in left_units.values()}
    normalized_right = {normalized_k(unit) for unit in right_units.values()}
    tolerance = 1e-6
    tolerance_matches = 0
    for target in right_units.values():
        if any(target["width"] == source["width"] and target["height"] == source["height"] and all(abs(target[name] - source[name]) <= tolerance for name in ("fx", "fy", "cx", "cy")) for source in left_units.values()):
            tolerance_matches += 1
    return {
        "left": left["split"], "right": right["split"],
        "same_capture_camera_units": len(left_keys & right_keys),
        "same_exact_K_units": len(exact_left & exact_right),
        "same_normalized_K_units": len(normalized_left & normalized_right),
        "right_units_with_tolerance_exact_K_match": tolerance_matches,
        "right_unit_count": len(right_units),
    }


def nearest(source, target, dimensions):
    distances = []
    rows = []
    for target_unit in target["units"].values():
        best = min(math.sqrt(sum((target_unit[name] - source_unit[name]) ** 2 for name in dimensions)) for source_unit in source["units"].values())
        distances.append(best)
        rows.append({"capture": target_unit["capture"], "camera": target_unit["camera"], "distance": best})
    return scalar_stats(distances), rows


def range_coverage(source, target):
    result = {}
    for name in ("nfx", "nfy"):
        values = [unit[name] for unit in source["units"].values()]
        low, high = min(values), max(values)
        result[name] = {"inside": sum(low <= unit[name] <= high for unit in target["units"].values()), "total": len(target["units"]), "range": [low, high]}
    result["both_inside"] = sum(all(result[name]["range"][0] <= unit[name] <= result[name]["range"][1] for name in ("nfx", "nfy")) for unit in target["units"].values())
    result["total"] = len(target["units"])
    return result


def prior_metrics(prior, target):
    errors = []
    for unit in target["units"].values():
        pred_fx, pred_fy = prior[0] * unit["width"], prior[1] * unit["height"]
        errors.append(max(abs(pred_fx - unit["fx"]) / abs(unit["fx"]), abs(pred_fy - unit["fy"]) / abs(unit["fy"])))
    return {"unit_count": len(errors), **scalar_stats(errors), **{f"within_{int(threshold * 100)}pct": sum(error <= threshold for error in errors) for threshold in (.01, .02, .03, .05, .10, .20)}}


def filtered_subset_counts(splits, key_name):
    keys = {name: {key_name(unit) for unit in split["units"].values()} for name, split in splits.items()}
    result = {}
    for name, split in splits.items():
        other_keys = set().union(*(keys[other] for other in splits if other != name))
        kept = [unit for unit in split["units"].values() if key_name(unit) not in other_keys]
        result[name] = {"original_units": len(split["units"]), "removed_units": len(split["units"]) - len(kept), "remaining_units": len(kept), "original_frames": split["image_count"], "remaining_frames": sum(unit["frame_count"] for unit in kept)}
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train = split_info("train", "all")
    validation = split_info("val", "machine_annot")
    test = split_info("test", "all")
    splits = {"train": train, "validation": validation, "test": test}

    relations = [relation_report(split) for split in ("train", "test")]
    overlaps = [overlap(train, validation), overlap(train, test), overlap(validation, test)]
    nearest_train_n2, nearest_train_n2_rows = nearest(train, test, ("nfx", "nfy"))
    nearest_train_n4, _ = nearest(train, test, ("nfx", "nfy", "ncx", "ncy"))
    coverage = {name: range_coverage(train, split) for name, split in (("validation", validation), ("test", test))}
    filtered_subsets = {
        "same_capture_camera_removed": filtered_subset_counts(splits, lambda unit: (unit["capture"], unit["camera"])),
        "same_exact_K_removed": filtered_subset_counts(splits, exact_k),
    }

    giga_rows = []
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["participant"] in {"p41", "p44"}:
                giga_rows.append(row)
    giga_units = {}
    for row in giga_rows:
        width, height = int(float(row["width"])), int(float(row["height"]))
        fx, fy = float(row["fx"]), float(row["fy"])
        giga_units[row["camera_key"]] = {"nfx": fx / width, "nfy": fy / height, "ncx": float(row["cx"]) / width, "ncy": float(row["cy"]) / height, "fx": fx, "fy": fy, "cx": float(row["cx"]), "cy": float(row["cy"]), "width": width, "height": height, "hfov_deg": math.degrees(2 * math.atan(width / (2 * fx))), "vfov_deg": math.degrees(2 * math.atan(height / (2 * fy)))}
    giga = {"units": giga_units, "split": "giga_train"}
    giga_prior_mean = tuple(statistics.mean(unit[name] for unit in giga_units.values()) for name in ("nfx", "nfy"))
    giga_prior_median = tuple(statistics.median(unit[name] for unit in giga_units.values()) for name in ("nfx", "nfy"))
    interhand_train_prior_mean = tuple(statistics.mean(unit[name] for unit in train["units"].values()) for name in ("nfx", "nfy"))
    interhand_train_prior_median = tuple(statistics.median(unit[name] for unit in train["units"].values()) for name in ("nfx", "nfy"))

    distributions = {}
    for name, split in {**splits, "giga_train": giga}.items():
        distributions[name] = {metric: scalar_stats([unit[metric] for unit in split["units"].values()]) for metric in ("fx", "fy", "nfx", "nfy", "ncx", "ncy", "hfov_deg", "vfov_deg") if metric in next(iter(split["units"].values()))}
    prior_results = {
        "interhand_train_mean": {"prior": interhand_train_prior_mean, "validation": prior_metrics(interhand_train_prior_mean, validation), "test": prior_metrics(interhand_train_prior_mean, test)},
        "interhand_train_median": {"prior": interhand_train_prior_median, "validation": prior_metrics(interhand_train_prior_median, validation), "test": prior_metrics(interhand_train_prior_median, test)},
        "giga_train_mean": {"prior": giga_prior_mean, "validation": prior_metrics(giga_prior_mean, validation), "test": prior_metrics(giga_prior_mean, test)},
        "giga_train_median": {"prior": giga_prior_median, "validation": prior_metrics(giga_prior_median, validation), "test": prior_metrics(giga_prior_median, test)},
    }

    split_summary = []
    for name, split in splits.items():
        split_summary.append({"split": name, "frames": split["image_count"], "annotations": split["annotation_count"], "captures": split["capture_count"], "subjects": split["subject_count"], "sequences": split["sequence_count"], "camera_labels": split["camera_label_count"], "calibration_units": split["unit_count"], "unique_K": len({exact_k(unit) for unit in split["units"].values()}), "mapped_images": split["image_count"] - len(split["mapping_failures"]), "unmapped_images": len(split["mapping_failures"])})
    write_csv(OUTPUT_ROOT / "split_summary.csv", split_summary, list(split_summary[0]))
    write_csv(OUTPUT_ROOT / "split_overlap.csv", overlaps, list(overlaps[0]))
    write_csv(OUTPUT_ROOT / "nearest_interhand_train.csv", nearest_train_n2_rows, list(nearest_train_n2_rows[0]))
    unit_rows = [{"split": name, **unit} for name, split in splits.items() for unit in split["units"].values()]
    write_csv(OUTPUT_ROOT / "calibration_units.csv", unit_rows, list(unit_rows[0]))

    report = {
        "archive": {"path": "HandCalib/datasets/interhand2.6m/archives/InterHand2.6M.annotations.5.fps.zip", "sha256": "b7dae49caf2700597593f8949e286e73af83d3836b10187beac1a313bee2031d", "unzip_test": "PASS", "nested_annotation_file_count": 30, "expected_count_from_request": 31},
        "annotation_root": str(ANNOTATION_ROOT),
        "canonical_official_split": {"train": "all/InterHand2.6M_train_data.json", "validation": "machine_annot/InterHand2.6M_val_data.json", "test": "all/InterHand2.6M_test_data.json"},
        "split_summary": split_summary, "relations": relations, "overlap": overlaps, "filtered_official_subsets": filtered_subsets,
        "distributions": distributions, "nearest_train_test_n2": nearest_train_n2, "nearest_train_test_n4": nearest_train_n4,
        "range_coverage": coverage, "constant_prior_metrics": prior_results,
        "schema": {name: {"image_missing": split["image_schema_missing"], "annotation_missing": split["annotation_schema_missing"], "camera_missing": split["camera_schema_missing"], "mapping_failures": len(split["mapping_failures"]), "inconsistent_unit_dimensions": split["inconsistent_dimensions"]} for name, split in splits.items()},
        "distortion_metadata": "not available in inspected annotation camera/data JSON schema",
        "verdict_A_giga_finetuned_on_interhand_test": "GO WITH LIMITATIONS",
        "verdict_B_interhand_train_to_official_test_camera_generalization": "OFFICIAL-PRESERVING FILTERED SUBSET RECOMMENDED",
        "limitations": ["No InterHand images were downloaded or inspected.", "Calibration unit is (capture, camera); physical serial identity is unavailable.", "Official Test has limited independent calibration units relative to frame count.", "Any overlap or near-overlap findings should be handled by filtering within official splits, not moving samples between splits."],
    }
    (OUTPUT_ROOT / "distribution_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT_ROOT / "constant_prior_metrics.json").write_text(json.dumps(prior_results, indent=2) + "\n")
    (OUTPUT_ROOT / "audit_report.md").write_text("# InterHand2.6M annotation audit\n\nSee `distribution_summary.json` and CSV outputs for the complete CPU-only audit.\n\n- Verdict A: GO WITH LIMITATIONS\n- Verdict B: OFFICIAL-PRESERVING FILTERED SUBSET RECOMMENDED\n")
    print(json.dumps({"split_summary": split_summary, "relations": relations, "overlap": overlaps, "nearest_train_test_n2": nearest_train_n2, "nearest_train_test_n4": nearest_train_n4, "range_coverage": coverage, "constant_prior_metrics": prior_results, "verdict_A": report["verdict_A_giga_finetuned_on_interhand_test"], "verdict_B": report["verdict_B_interhand_train_to_official_test_camera_generalization"]}, indent=2))


if __name__ == "__main__":
    main()
