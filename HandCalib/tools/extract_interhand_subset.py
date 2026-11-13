"""Extract only the frozen InterHand manifest images from multipart tar parts."""

import argparse
import io
import os
import tarfile
from pathlib import Path

import cv2

from verify_interhand_images import normalize_member_name, part_paths, read_manifest


def safe_target(root, member_name):
    path = Path(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("Unsafe archive member path: %s" % member_name)
    target = (Path(root) / path).resolve()
    if Path(root).resolve() not in target.parents:
        raise RuntimeError("Archive member escapes target directory: %s" % member_name)
    return target


class MultipartReader(io.RawIOBase):
    def __init__(self, paths):
        self.handles = [path.open("rb") for path in paths]
        self.index = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        while self.index < len(self.handles):
            count = self.handles[self.index].readinto(buffer)
            if count:
                return count
            self.handles[self.index].close()
            self.index += 1
        return 0

    def close(self):
        for handle in self.handles[self.index:]:
            handle.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = part_paths(args.archive_dir)
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("All 44 image archive parts must exist before extraction")
    manifest_rows = read_manifest(args.manifest)
    expected = {row["file_name"] for row in manifest_rows}
    dimensions = {row["file_name"]: (int(row["width"]), int(row["height"])) for row in manifest_rows}
    extracted_names = set()
    extracted = 0
    reused = 0
    with MultipartReader(paths) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            canonical_name = normalize_member_name(member.name)
            if canonical_name not in expected or canonical_name in extracted_names:
                continue
            target = safe_target(args.output_root, "test/" + canonical_name)
            if member.issym() or member.islnk() or not member.isfile():
                raise RuntimeError("Refusing non-regular selected archive member: %s" % member.name)
            if target.exists():
                image = cv2.imread(str(target), cv2.IMREAD_COLOR)
                expected_width, expected_height = dimensions[canonical_name]
                if image is None or image.shape[1] != expected_width or image.shape[0] != expected_height:
                    raise RuntimeError("Existing extracted image is invalid: %s" % target)
                reused += 1
                extracted_names.add(canonical_name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            with target.open("wb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            extracted += 1
            extracted_names.add(canonical_name)
    missing = sorted(expected - extracted_names)
    print("expected=%d extracted=%d reused=%d missing=%d unexpected=0" % (len(expected), extracted, reused, len(missing)))
    if missing:
        raise RuntimeError("Selected manifest members were not extracted: %s" % missing[:5])


if __name__ == "__main__":
    main()
