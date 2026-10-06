"""CPU-only consistency audit for the local InterHand2.6M annotations."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "datasets/interhand2.6m/raw/annotations"
ARCHIVE = ROOT / "datasets/interhand2.6m/archives/InterHand2.6M.annotations.5.fps.zip"
MANIFEST = ROOT / "data/manifests/gigahands.csv"
OUTPUT = ROOT / "runs/02_interhand_annotation_audit_completion"
SOURCES = (
    ("all", "train"),
    ("human_annot", "train"),
    ("machine_annot", "train"),
    ("machine_annot", "val"),
    ("all", "test"),
    ("human_annot", "test"),
    ("machine_annot", "test"),
)
CANONICAL = {"train": ("all", "train"), "validation": ("machine_annot", "val"), "test": ("all", "test")}


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def data_path(source, split):
    return ANNOTATIONS / source / ("InterHand2.6M_%s_data.json" % split)


def camera_path(source, split):
    return ANNOTATIONS / source / ("InterHand2.6M_%s_camera.json" % split)


def joint_path(source, split):
    return ANNOTATIONS / source / ("InterHand2.6M_%s_joint_3d.json" % split)


def stats(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": values[0], "median": statistics.median(values),
        "max": values[-1], "mean": statistics.mean(values),
    }


def projection(world, rotation, position, focal, principal):
    # This is the official convention: R @ (X_world - camera_position), then pinhole.
    camera = []
    for point in world:
        translated = [point[i] - position[i] for i in range(3)]
        camera.append([sum(rotation[row][col] * translated[col] for col in range(3)) for row in range(3)])
    projected = []
    for x, y, z in camera:
        if z == 0:
            projected.append([None, None])
        else:
            projected.append([focal[0] * x / z + principal[0], focal[1] * y / z + principal[1]])
    return camera, projected


def load_data(source, split):
    data = read_json(data_path(source, split))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids = [image.get("id") for image in images]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    image_set = set(image_ids)
    annotated_image_ids = [annotation.get("image_id") for annotation in annotations]
    annotated_set = set(annotated_image_ids)
    per_image = Counter(annotated_image_ids)
    return {
        "source": source, "split": split, "data": data, "images": images, "annotations": annotations,
        "image_ids": image_set, "annotation_ids": set(annotation_ids),
        "annotated_image_ids": annotated_set,
        "image_count": len(images), "unique_image_ids": len(image_set),
        "annotation_count": len(annotations), "unique_annotation_ids": len(set(annotation_ids)),
        "annotated_unique_images": len(annotated_set),
        "annotation_less_images": len(image_set - annotated_set),
        "images_with_multiple_annotations": sum(count > 1 for count in per_image.values()),
        "max_annotations_per_image": max(per_image.values(), default=0),
        "duplicate_image_ids": len(image_ids) - len(image_set),
        "duplicate_annotation_ids": len(annotation_ids) - len(set(annotation_ids)),
    }


def relation_report(split):
    human = load_data("human_annot", split)
    machine = load_data("machine_annot", split)
    all_data = load_data("all", split)
    image_union = human["image_ids"] | machine["image_ids"]
    annotation_union = human["annotation_ids"] | machine["annotation_ids"]
    return {
        "split": split,
        "images": {"human": len(human["image_ids"]), "machine": len(machine["image_ids"]),
                    "intersection": len(human["image_ids"] & machine["image_ids"]),
                    "union": len(image_union), "all": len(all_data["image_ids"]),
                    "all_equals_union": all_data["image_ids"] == image_union},
        "annotations": {"human": len(human["annotation_ids"]), "machine": len(machine["annotation_ids"]),
                         "intersection": len(human["annotation_ids"] & machine["annotation_ids"]),
                         "union": len(annotation_union), "all": len(all_data["annotation_ids"]),
                         "all_equals_union": all_data["annotation_ids"] == annotation_union},
    }


def unit_records(data, cameras):
    units = {}
    for image in data["images"]:
        capture, camera = str(image["capture"]), str(image["camera"])
        record = cameras[capture]
        focal, principal = record["focal"][camera], record["princpt"][camera]
        width, height = int(image["width"]), int(image["height"])
        key = (capture, camera)
        if key not in units:
            fx, fy = float(focal[0]), float(focal[1])
            cx, cy = float(principal[0]), float(principal[1])
            units[key] = {"capture": capture, "camera": camera, "width": width, "height": height,
                          "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                          "nfx": fx / width, "nfy": fy / height, "ncx": cx / width, "ncy": cy / height,
                          "hfov_deg": math.degrees(2 * math.atan(width / (2 * fx))),
                          "vfov_deg": math.degrees(2 * math.atan(height / (2 * fy))), "frame_count": 0}
        units[key]["frame_count"] += 1
    return units


def canonical_audit(name, source, split, sample_count=200):
    data = load_data(source, split)
    cameras = read_json(camera_path(source, split))
    joints = read_json(joint_path(source, split))
    units = unit_records(data, cameras)
    image_by_id = {image["id"]: image for image in data["images"]}
    resolutions = Counter((int(image["width"]), int(image["height"])) for image in data["images"])
    principal_outside = 0
    for unit in units.values():
        principal_outside += int(unit["cx"] < 0 or unit["cx"] > unit["width"] or unit["cy"] < 0 or unit["cy"] > unit["height"])
    samples = []
    reprojection_rows = []
    for image in sorted(data["images"], key=lambda value: value["id"])[:sample_count]:
        capture, camera, frame = str(image["capture"]), str(image["camera"]), str(image["frame_idx"])
        calibration = cameras[capture]
        joint = joints[capture][frame]
        cam, projected = projection(joint["world_coord"], calibration["camrot"][camera], calibration["campos"][camera], calibration["focal"][camera], calibration["princpt"][camera])
        finite = all(point[0] is not None and all(math.isfinite(value) for value in point) for point in projected)
        inside = sum(0 <= point[0] < image["width"] and 0 <= point[1] < image["height"] for point in projected if point[0] is not None)
        samples.append({"image_id": image["id"], "capture": capture, "camera": camera, "width": image["width"], "height": image["height"], "fx": calibration["focal"][camera][0], "fy": calibration["focal"][camera][1], "cx": calibration["princpt"][camera][0], "cy": calibration["princpt"][camera][1], "nfx": calibration["focal"][camera][0] / image["width"], "nfy": calibration["focal"][camera][1] / image["height"], "hfov_deg": math.degrees(2 * math.atan(image["width"] / (2 * calibration["focal"][camera][0]))), "vfov_deg": math.degrees(2 * math.atan(image["height"] / (2 * calibration["focal"][camera][1])))})
        reprojection_rows.append({"split": name, "image_id": image["id"], "finite_projected_joints": finite, "projected_joints": len(projected), "inside_image_joints": inside, "joint_valid_field_present": "joint_valid" in data["annotations"][0] if data["annotations"] else False, "direct_2d_ground_truth_present": False})
    values = {metric: stats([unit[metric] for unit in units.values()]) for metric in ("fx", "fy", "nfx", "nfy", "ncx", "ncy", "hfov_deg", "vfov_deg")}
    return {"name": name, "source": source, "split": split, "data": data, "units": units,
            "calibration_units": len(units), "resolutions": [{"width": width, "height": height, "count": count} for (width, height), count in sorted(resolutions.items())],
            "intrinsics_stats": values, "principal_point_outside_count": principal_outside,
            "samples": samples, "reprojection": reprojection_rows,
            "direct_2d_ground_truth_present": False,
            "reprojection_comparison": "unavailable: COCO annotation JSON has no 2D joint field; official loader derives joint_img from the same projection"}


def prior_metrics(prior, units):
    errors = []
    for unit in units.values():
        pred_fx, pred_fy = prior[0] * unit["width"], prior[1] * unit["height"]
        errors.append(max(abs(pred_fx - unit["fx"]) / abs(unit["fx"]), abs(pred_fy - unit["fy"]) / abs(unit["fy"])))
    return {"unit_count": len(errors), "mean_max_rel_f": statistics.mean(errors), "median_max_rel_f": statistics.median(errors), "within_5pct": sum(error <= .05 for error in errors), "within_20pct": sum(error <= .20 for error in errors)}


def exact_k(unit):
    return (unit["width"], unit["height"], unit["fx"], unit["fy"], unit["cx"], unit["cy"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    counts = []
    loaded = {}
    for source, split in SOURCES:
        info = load_data(source, split)
        loaded[(source, split)] = info
        counts.append({"source": source, "split": split, "images": info["image_count"], "unique_image_ids": info["unique_image_ids"], "annotations": info["annotation_count"], "unique_annotation_ids": info["unique_annotation_ids"], "annotated_unique_images": info["annotated_unique_images"], "annotation_less_images": info["annotation_less_images"], "images_with_multiple_annotations": info["images_with_multiple_annotations"], "max_annotations_per_image": info["max_annotations_per_image"]})
    relations = [relation_report(split) for split in ("train", "test")]
    audits = {name: canonical_audit(name, source, split) for name, (source, split) in CANONICAL.items()}
    train_units, test_units = audits["train"]["units"], audits["test"]["units"]
    same_units = len(set(train_units) & set(test_units))
    train_k, test_k = {exact_k(unit) for unit in train_units.values()}, {exact_k(unit) for unit in test_units.values()}
    overlaps = {"train_test_same_capture_camera": same_units, "train_test_exact_K": len(train_k & test_k), "validation_test_same_capture_camera": len(set(audits["validation"]["units"]) & set(test_units)), "validation_test_exact_K": len({exact_k(unit) for unit in audits["validation"]["units"].values()} & test_k)}
    with MANIFEST.open(newline="") as handle:
        giga_rows = [row for row in csv.DictReader(handle) if row["participant"] in {"p41", "p44"}]
    giga_units = {}
    for row in giga_rows:
        giga_units[row["camera_key"]] = row
    giga_rows = list(giga_units.values())
    giga_nfx = [float(row["fx"]) / float(row["width"]) for row in giga_rows]
    giga_nfy = [float(row["fy"]) / float(row["height"]) for row in giga_rows]
    priors = {"giga_mean": [statistics.mean(giga_nfx), statistics.mean(giga_nfy)], "giga_median": [statistics.median(giga_nfx), statistics.median(giga_nfy)]}
    diagnostics = {name: prior_metrics(prior, test_units) for name, prior in priors.items()}
    report = {
        "experiment": "02g1_interhand_audit_completion", "analysis_mode": "cpu_only_annotation_audit", "images_downloaded": False, "model_or_checkpoint_used": False, "inference_or_training_run": False,
        "archive": {"path": str(ARCHIVE.relative_to(ROOT.parent)), "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), "size_bytes": ARCHIVE.stat().st_size, "release_tag": "v1.0", "release_asset": "InterHand2.6M.annotations.5.fps.zip", "asset_uploaded_at": "2020-11-26T05:54:13Z", "provenance_note": "GitHub v1.0 release asset; official homepage describes a later 2021-03-22 v1.0 dataset release and publishes larger 5fps frame totals."},
        "official_public_counts": {"train_H_plus_M_frames": 1361062, "validation_M_frames": 380125, "test_H_plus_M_frames": 849160, "interpretation": "official 5fps frame totals, not annotation IDs or hand-instance counts; H+M is the official combined branch definition."},
        "count_breakdown": counts, "human_machine_relation": relations, "canonical_split": {name: "%s/%s" % pair for name, pair in CANONICAL.items()},
        "canonical_audits": {name: {key: value for key, value in audit.items() if key not in ("data", "units")} for name, audit in audits.items()},
        "overlap_recheck": overlaps, "giga_prior_diagnostic": {"equal_weight_units": len(giga_rows), "priors": priors, "results": diagnostics, "meaning": "constant GigaHands Train calibration prior applied to InterHand Test GT; NOT model inference."},
        "reprojection_convention": {"world2cam": "R @ (X_world - camera_position)", "cam2pixel": "[fx*x/z+cx, fy*y/z+cy]", "source": "official facebookresearch/InterHand2.6M data/InterHand2.6M/dataset.py and utils/transforms.py", "direct_error_available": False, "reason": "data.json stores bbox/joint_valid but no 2D joint coordinates; official loader derives joint_img using this projection."},
        "official_count_difference_classification": ["A: current local archive counts are JSON image/frame and annotation record counts, while public numbers are official frame totals", "B: H and M overlap in the local all union, so H+M is not a literal sum", "D: current asset upload predates the official homepage's 2021-03-22 v1.0 image release; exact cross-release equivalence is not established", "E: both are labelled 5fps, so 30fps alone does not explain the discrepancy"],
        "verdict_A": "GO WITH LIMITATIONS", "verdict_B": "OFFICIAL-PRESERVING FILTERED SUBSET RECOMMENDED",
        "limitations": ["No InterHand images were downloaded or inspected.", "Physical camera serial identity is unavailable; capture/camera means an annotation calibration unit.", "Direct 2D reprojection error cannot be computed from this archive because 2D joints are not stored.", "The current archive is provenance-identified but not proven byte-equivalent to the later homepage-linked v1.0 annotation release."],
    }
    (OUTPUT / "count_breakdown.json").write_text(json.dumps(counts, indent=2) + "\n")
    (OUTPUT / "human_machine_overlap.json").write_text(json.dumps(relations, indent=2) + "\n")
    (OUTPUT / "intrinsics_scale_audit.csv").write_text("split,width,height,count\n" + "\n".join("%s,%s,%s,%s" % (name, row["width"], row["height"], row["count"]) for name, audit in audits.items() for row in audit["resolutions"]) + "\n")
    write_rows = []
    for audit in audits.values():
        write_rows.extend(audit["reprojection"])
    fields = list(write_rows[0])
    with (OUTPUT / "reprojection_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(write_rows)
    (OUTPUT / "distribution_recheck.json").write_text(json.dumps({"canonical_audits": report["canonical_audits"], "overlap_recheck": overlaps, "giga_prior_diagnostic": report["giga_prior_diagnostic"]}, indent=2) + "\n")
    (OUTPUT / "audit_completion_report.md").write_text("# InterHand2.6M 02-G.1 audit completion\n\nCPU-only annotation and calibration audit. No images, model inference, training, or fine-tuning were run.\n\n- Verdict A: GO WITH LIMITATIONS\n- Verdict B: OFFICIAL-PRESERVING FILTERED SUBSET RECOMMENDED\n- Direct reprojection error: unavailable because data JSON has no stored 2D joint field.\n- Giga prior result is a constant-prior diagnostic, not AnyCalib model performance.\n")
    (ROOT / "results/02g1_interhand_audit_completion.yaml").write_text("# Generated summary for the CPU-only 02-G.1 audit.\n" + json.dumps(report, indent=2) + "\n")
    print(json.dumps({"count_breakdown": counts, "relations": relations, "canonical": {name: {"units": audit["calibration_units"], "resolutions": audit["resolutions"], "intrinsics": audit["intrinsics_stats"], "principal_outside": audit["principal_point_outside_count"]} for name, audit in audits.items()}, "overlap": overlaps, "diagnostics": diagnostics, "archive_sha256": report["archive"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
