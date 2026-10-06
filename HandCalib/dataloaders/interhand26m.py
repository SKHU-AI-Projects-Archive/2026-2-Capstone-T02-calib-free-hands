"""InterHand2.6M frozen-manifest loader for external evaluation."""

import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not decode InterHand image: %s" % path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


class InterHand26MDataset(Dataset):
    """Read frozen metadata and source images; GT K is target metadata only."""

    def __init__(self, image_root, manifest_path, transform=None):
        self.image_root = Path(image_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.transform = transform
        if not self.manifest_path.exists():
            raise FileNotFoundError("InterHand manifest does not exist: %s" % self.manifest_path)
        with self.manifest_path.open(newline="") as handle:
            self.samples = [{"row": row} for row in csv.DictReader(handle)]
        required = {"dataset", "split", "image_id", "file_name", "capture", "camera", "calibration_unit", "seq_name", "frame_idx", "width", "height", "fx", "fy", "cx", "cy", "sample_order"}
        missing = required - set(self.samples[0]["row"]) if self.samples else required
        if missing:
            raise ValueError("InterHand manifest is missing columns: %s" % sorted(missing))
        self.pair_indices = {}
        for index, sample in enumerate(self.samples):
            row = sample["row"]
            if row["dataset"] != "InterHand2.6M" or row["split"] != "test":
                raise ValueError("Manifest contains a non-Official-Test sample")
            for key in ("width", "height", "frame_idx", "sample_order"):
                row[key] = int(row[key])
            for key in ("fx", "fy", "cx", "cy"):
                row[key] = float(row[key])
            row["image_id"] = int(row["image_id"])
            self.pair_indices.setdefault(row["calibration_unit"], []).append(index)
        self._validate_manifest()

    def _validate_manifest(self):
        image_ids = [sample["row"]["image_id"] for sample in self.samples]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("InterHand manifest contains duplicate image_id values")
        if len(self.pair_indices) != 360:
            raise ValueError("Expected 360 calibration units, found %d" % len(self.pair_indices))
        if any(len(indices) > 16 for indices in self.pair_indices.values()):
            raise ValueError("InterHand manifest exceeds 16 frames per calibration unit")
        for indices in self.pair_indices.values():
            signatures = {tuple(self.samples[index]["row"][key] for key in ("width", "height", "fx", "fy", "cx", "cy")) for index in indices}
            if len(signatures) != 1:
                raise ValueError("GT intrinsics are inconsistent within a calibration unit")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]["row"]
        image_path = self.image_root / "test" / row["file_name"]
        image = read_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "intrinsics": torch.tensor([row["fx"], row["fy"], row["cx"], row["cy"]], dtype=torch.float32), "meta": {"participant": "interhand", "sequence": row["seq_name"], "camera": row["camera"], "camera_key": row["calibration_unit"], "video_path": str(image_path), "video_name": row["seq_name"], "frame_index": row["frame_idx"], "width": row["width"], "height": row["height"], "image_id": row["image_id"], "dataset": row["dataset"], "split": row["split"]}}
