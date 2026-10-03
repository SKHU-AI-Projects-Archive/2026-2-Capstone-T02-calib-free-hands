#!/usr/bin/env python3
"""Download and validate the official Brown IVL GigaHands demo archive."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


DEMO_URL = "https://g-ad09a0.56197.5898.data.globus.org/gigahands_demo_all.tar.gz"
HANDCALIB_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = HANDCALIB_ROOT / "datasets" / "gigahands"
DOWNLOAD_ROOT = HANDCALIB_ROOT / "data" / "downloads" / "gigahands"
ARCHIVE_PATH = DOWNLOAD_ROOT / "gigahands_demo_all.tar.gz"

EXPECTED_SEQUENCES = (
    "p36-tea-0010",
    "p41-boxing-0021",
    "p41-plant-0004",
    "p44-dog-0004",
    "p52-instrument-0034",
)
EXPECTED_RGB_COUNTS = {
    "p36-tea-0010": 49,
    "p41-boxing-0021": 49,
    "p41-plant-0004": 49,
    "p44-dog-0004": 52,
    "p52-instrument-0034": 49,
}
EXPECTED_PROFILE = {
    "sequence_count": 5,
    "sequences": EXPECTED_SEQUENCES,
    "rgb_mp4_total": 248,
    "rgb_mp4_by_sequence": EXPECTED_RGB_COUNTS,
    "optim_params_count": 5,
    "camera_rows": 245,
    "file_count": 1385,
}


def count_data_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows += 1
    return rows


def inspect_dataset(root: Path) -> dict[str, object]:
    hand_pose = root / "hand_pose"
    sequences = tuple(sorted(path.name for path in hand_pose.iterdir() if path.is_dir())) if hand_pose.is_dir() else ()
    rgb_mp4_by_sequence = {}
    for sequence in sequences:
        rgb_root = hand_pose / sequence / "rgb_vid"
        rgb_mp4_by_sequence[sequence] = sum(1 for path in rgb_root.rglob("*.mp4")) if rgb_root.is_dir() else 0

    optim_params = list(hand_pose.glob("*/optim_params.txt")) if hand_pose.is_dir() else []
    camera_rows = sum(count_data_rows(path) for path in optim_params)

    return {
        "sequence_count": len(sequences),
        "sequences": sequences,
        "rgb_mp4_total": sum(rgb_mp4_by_sequence.values()),
        "rgb_mp4_by_sequence": rgb_mp4_by_sequence,
        "optim_params_count": len(optim_params),
        "camera_rows": camera_rows,
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def profile_matches(profile: dict[str, object]) -> bool:
    return profile == EXPECTED_PROFILE


def format_profile(profile: dict[str, object]) -> str:
    return (
        f"sequences={profile['sequence_count']} {list(profile['sequences'])}; "
        f"rgb_mp4_total={profile['rgb_mp4_total']} "
        f"by_sequence={profile['rgb_mp4_by_sequence']}; "
        f"optim_params={profile['optim_params_count']}; "
        f"camera_rows={profile['camera_rows']}; files={profile['file_count']}"
    )


def check_existing_dataset(root: Path) -> bool:
    if not root.is_dir() or not any(root.iterdir()):
        return False
    profile = inspect_dataset(root)
    if profile_matches(profile):
        print(f"dataset already prepared: {root}")
        print(format_profile(profile))
        return True
    raise RuntimeError(
        f"existing dataset does not match the recorded profile; refusing to overwrite: {root}\n"
        f"observed: {format_profile(profile)}\n"
        f"expected: {format_profile(EXPECTED_PROFILE)}"
    )


def safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not supported: {member.name}")
            member_path = (destination / member.name).resolve()
            if os.path.commonpath((str(destination_root), str(member_path))) != str(destination_root):
                raise RuntimeError(f"unsafe archive member path: {member.name}")
        tar.extractall(destination)


def find_dataset_roots(extracted_root: Path) -> list[Path]:
    candidates = []
    for path in (extracted_root, *sorted(extracted_root.rglob("*"))):
        if path.is_dir() and (path / "hand_pose").is_dir() and (path / "object_pose").is_dir():
            candidates.append(path)
    return candidates


def download_archive(cache_root: Path = DOWNLOAD_ROOT, url: str = DEMO_URL) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / ARCHIVE_PATH.name
    if archive_path.is_file():
        print(f"using cached archive: {archive_path}")
        return archive_path

    partial = archive_path.with_name(f"{archive_path.name}.part")
    if partial.exists():
        partial.unlink()
    print(f"downloading: {url}")
    print(f"temporary archive: {partial}")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(archive_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return archive_path


def prepare_dataset(
    target: Path = DATASET_ROOT,
    cache_root: Path = DOWNLOAD_ROOT,
    url: str = DEMO_URL,
) -> None:
    if check_existing_dataset(target):
        return

    archive = download_archive(cache_root, url)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gigahands-extract-", dir=cache_root) as temporary:
        extracted_root = Path(temporary)
        safe_extract(archive, extracted_root)
        candidates = find_dataset_roots(extracted_root)
        if len(candidates) != 1:
            raise RuntimeError(f"expected exactly one dataset root, found {len(candidates)}: {candidates}")

        candidate = candidates[0]
        profile = inspect_dataset(candidate)
        if not profile_matches(profile):
            raise RuntimeError(
                "official archive profile differs from the recorded local profile; refusing to install it\n"
                f"observed: {format_profile(profile)}\n"
                f"expected: {format_profile(EXPECTED_PROFILE)}"
            )

        if target.exists():
            target.rmdir()
        shutil.move(str(candidate), str(target))
    print(f"dataset prepared: {target}")


def print_dry_run(target: Path = DATASET_ROOT, cache_root: Path = DOWNLOAD_ROOT, url: str = DEMO_URL) -> None:
    existing = target.is_dir() and any(target.iterdir())
    print(f"source URL: {url}")
    print(f"archive target: {cache_root / ARCHIVE_PATH.name}")
    print(f"dataset target: {target}")
    print(f"download cache: {cache_root}")
    if existing:
        print("expected action: inspect existing dataset; download/extraction will be skipped if its profile matches")
    else:
        print("expected action: download to .part, atomically rename, safely extract, validate, then install")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="inspect the existing dataset without downloading")
    parser.add_argument("--dry-run", action="store_true", help="print paths and actions without downloading")
    parser.add_argument("--target", type=Path, default=DATASET_ROOT, help="dataset installation directory")
    parser.add_argument("--cache-dir", type=Path, default=DOWNLOAD_ROOT, help="archive download directory")
    args = parser.parse_args()
    if args.check and args.dry_run:
        parser.error("--check and --dry-run are mutually exclusive")

    try:
        if args.dry_run:
            print_dry_run(args.target, args.cache_dir)
        elif args.check:
            if not args.target.is_dir() or not any(args.target.iterdir()):
                raise RuntimeError(f"dataset is absent or empty: {args.target}")
            profile = inspect_dataset(args.target)
            print(f"dataset profile: {args.target}")
            print(format_profile(profile))
            if not profile_matches(profile):
                raise RuntimeError("dataset profile does not match the recorded local reference")
            print("dataset check passed")
        else:
            prepare_dataset(args.target, args.cache_dir)
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
