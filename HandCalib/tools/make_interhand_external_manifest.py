"""Create the frozen, metadata-only InterHand external-test manifest."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/interhand2.6m/raw/annotations/all/InterHand2.6M_test_data.json"
DEFAULT_OUTPUT = ROOT / "data/manifests/interhand_external_test_v1.csv"
MAX_FRAMES_PER_UNIT = 16
FIELDS = ("dataset", "split", "image_id", "file_name", "capture", "camera", "calibration_unit", "seq_name", "frame_idx", "width", "height", "fx", "fy", "cx", "cy", "sample_order")


def stable_sample_positions(count, limit=MAX_FRAMES_PER_UNIT):
    """Choose midpoint quantiles with deterministic nearest-unused collision handling."""
    selected_count = min(limit, count)
    positions, used = [], set()
    for index in range(selected_count):
        target = round((index + 0.5) * count / selected_count - 0.5)
        target = max(0, min(count - 1, target))
        position = next(value for value in sorted(range(count), key=lambda value: (abs(value - target), value)) if value not in used)
        used.add(position)
        positions.append(position)
    return positions


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(data_path):
    with Path(data_path).open() as handle:
        data = json.load(handle)
    groups = defaultdict(list)
    for image in data["images"]:
        groups[(str(image["capture"]), str(image["camera"]))].append(image)
    rows = []
    for unit in sorted(groups):
        images = sorted(groups[unit], key=lambda image: (str(image["seq_name"]), int(image["frame_idx"]), int(image["id"])))
        for sample_order, position in enumerate(stable_sample_positions(len(images))):
            image = images[position]
            rows.append({"dataset": "InterHand2.6M", "split": "test", "image_id": image["id"], "file_name": image["file_name"], "capture": image["capture"], "camera": image["camera"], "calibration_unit": "%s/%s" % unit, "seq_name": image["seq_name"], "frame_idx": image["frame_idx"], "width": image["width"], "height": image["height"], "fx": "", "fy": "", "cx": "", "cy": "", "sample_order": sample_order})
    return rows, data


def add_intrinsics(rows, camera_path):
    with Path(camera_path).open() as handle:
        cameras = json.load(handle)
    for row in rows:
        camera = cameras[str(row["capture"])]
        focal, principal = camera["focal"][str(row["camera"])], camera["princpt"][str(row["camera"])]
        row.update({"fx": focal[0], "fy": focal[1], "cx": principal[0], "cy": principal[1]})


def write_manifest(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--camera", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    camera = args.camera or args.data.with_name(args.data.name.replace("_data.json", "_camera.json"))
    rows, data = load_rows(args.data)
    add_intrinsics(rows, camera)
    write_manifest(rows, args.output)
    units = {row["calibration_unit"] for row in rows}
    print(json.dumps({"manifest": str(args.output), "sha256": sha256_file(args.output), "source_sha256": sha256_file(args.data), "source_images": len(data["images"]), "selected_frames": len(rows), "calibration_units": len(units), "max_frames_per_unit": max((sum(row["calibration_unit"] == unit for row in rows) for unit in units), default=0), "sampling_rule": "sort (seq_name, frame_idx, image_id); choose round((i+0.5)*N/K-0.5), K=min(16,N), clip and nearest-unused collision resolution"}, indent=2))


if __name__ == "__main__":
    main()
