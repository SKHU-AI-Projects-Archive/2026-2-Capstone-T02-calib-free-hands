"""GigaHands의 RGB frame과 camera metadata를 읽는 기본 Dataset이다."""

from collections import defaultdict
import csv
from pathlib import Path
import random

import cv2
import torch
from torch.utils.data import Dataset, Sampler


def _resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    options = [root / candidate, root.parent.parent / candidate]
    if candidate.parts[:2] == ("datasets", "gigahands"):
        options.insert(0, root / Path(*candidate.parts[2:]))
    for option in options:
        if option.exists():
            return option
    return options[0]


def read_video_frame(path: str | Path, frame_index: int) -> torch.Tensor:
    """OpenCV로 한 frame을 읽어 RGB float tensor로 반환한다."""
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)


class GigaHandsDataset(Dataset):
    """Manifest와 participant split으로 GigaHands의 모든 frame을 참조한다."""

    def __init__(self, root, split_file, manifest_path=None, transform=None):
        self.root = Path(root).resolve()
        self.split_file = Path(split_file)
        self.transform = transform
        self.manifest_path = Path(manifest_path or self.root.parent.parent / "data/manifests/gigahands.csv").resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"GigaHands root does not exist: {self.root}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"GigaHands manifest does not exist: {self.manifest_path}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"GigaHands split does not exist: {self.split_file}")

        participants = {
            line.strip()
            for line in self.split_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.samples = []
        with self.manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["participant"] not in participants:
                    continue
                row["frame_count"] = int(row["frame_count"])
                for key in ("width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"):
                    row[key] = float(row[key])
                video_path = _resolve_path(row["video_path"], self.root)
                for frame_index in range(row["frame_count"]):
                    self.samples.append({"row": row, "video_path": video_path, "frame_index": frame_index})

        self.pair_indices = defaultdict(list)
        for index, sample in enumerate(self.samples):
            self.pair_indices[sample["row"]["camera_key"]].append(index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        row = sample["row"]
        image = read_video_frame(sample["video_path"], sample["frame_index"])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "intrinsics": torch.tensor([row["fx"], row["fy"], row["cx"], row["cy"]], dtype=torch.float32),
            "distortion": torch.tensor([row["k1"], row["k2"], row["p1"], row["p2"]], dtype=torch.float32),
            "meta": {
                "participant": row["participant"], "sequence": row["sequence"], "camera": row["camera"],
                "camera_key": row["camera_key"], "video_path": row["video_path"],
                "video_name": row["video_name"], "frame_index": sample["frame_index"],
                "width": int(row["width"]), "height": int(row["height"]),
            },
        }


class PairBalancedSampler(Sampler):
    """각 sequence-camera pair에서 동일한 frame 수를 비복원 추출한다."""

    def __init__(self, dataset, frames_per_pair=174, seed=42):
        self.dataset = dataset
        self.frames_per_pair = frames_per_pair
        self.seed = seed
        self.epoch = 0
        if any(len(indices) < frames_per_pair for indices in dataset.pair_indices.values()):
            raise ValueError("Every sequence-camera pair must contain enough frames")

    def __len__(self):
        return len(self.dataset.pair_indices) * self.frames_per_pair

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected = []
        for indices in self.dataset.pair_indices.values():
            selected.extend(rng.sample(indices, self.frames_per_pair))
        rng.shuffle(selected)
        return iter(selected)
