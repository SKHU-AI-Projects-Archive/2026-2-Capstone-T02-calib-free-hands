"""
GigaHands 데이터의 영상과 카메라 정보를 하나의 manifest로 정리합니다.

각 RGB 영상에 대해 participant, sequence, camera, 영상 경로와
fx, fy, cx, cy 등의 카메라 파라미터를 CSV 파일로 저장합니다.
원본 GigaHands 데이터는 수정하지 않습니다.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path


HANDCALIB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = HANDCALIB_ROOT / "datasets" / "gigahands"
DEFAULT_OUTPUT = HANDCALIB_ROOT / "data" / "manifests" / "gigahands.csv"
FIELDS = [
    "participant", "sequence", "camera", "camera_key", "video_path", "video_name",
    "frame_count", "width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
]


def load_camera_params(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("#"):
        raise ValueError(f"missing optim_params header: {path}")
    header = lines[0].lstrip("# ").split()
    rows = []
    for line in lines[1:]:
        if line.strip():
            values = line.split()
            if len(values) != len(header):
                raise ValueError(f"invalid camera row: {path}")
            rows.append(dict(zip(header, values)))
    return {row["cam_name"]: row for row in rows}


def probe_video(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    if stream.get("nb_frames") in (None, "N/A"):
        raise ValueError(f"frame count unavailable: {path}")
    return int(stream["nb_frames"]), int(stream["width"]), int(stream["height"])


def build_manifest(root=DEFAULT_ROOT, output=DEFAULT_OUTPUT):
    root = Path(root).resolve()
    output = Path(output)
    rows = []
    for sequence_dir in sorted(path for path in (root / "hand_pose").iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        participant = sequence.split("-", 1)[0]
        cameras = load_camera_params(sequence_dir / "optim_params.txt")
        for video in sorted((sequence_dir / "rgb_vid").glob("*/*.mp4")):
            camera = video.parent.name
            if camera not in cameras:
                raise ValueError(f"video has no camera parameters: {video}")
            params = cameras[camera]
            frame_count, width, height = probe_video(video)
            rows.append({
                "participant": participant,
                "sequence": sequence,
                "camera": camera,
                "camera_key": f"{sequence}/{camera}",
                "video_path": video.relative_to(HANDCALIB_ROOT).as_posix(),
                "video_name": video.name,
                "frame_count": frame_count,
                "width": int(params["width"]),
                "height": int(params["height"]),
                "fx": float(params["fx"]),
                "fy": float(params["fy"]),
                "cx": float(params["cx"]),
                "cy": float(params["cy"]),
                "k1": float(params["k1"]),
                "k2": float(params["k2"]),
                "p1": float(params["p1"]),
                "p2": float(params["p2"]),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build a GigaHands video/camera manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_manifest(args.root, args.output)
    unique_cameras = {row["camera_key"] for row in rows}
    print(f"wrote {args.output}: rows={len(rows)} unique_camera_key={len(unique_cameras)}")
    if len(rows) != 248 or len(unique_cameras) != 245:
        raise SystemExit("unexpected GigaHands manifest counts")


if __name__ == "__main__":
    main()
