"""Extract only the frozen InterHand manifest images from multipart tar parts."""

import argparse
import io
import os
import tarfile
from pathlib import Path

from verify_interhand_images import candidate_names, part_paths, read_manifest


def without_test_prefix(path):
    return path[5:] if path.startswith("test/") else path


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
    desired = {candidate for row in read_manifest(args.manifest) for candidate in candidate_names("test/" + row["file_name"])}
    extracted = 0
    with MultipartReader(paths) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if member.name not in desired and without_test_prefix(member.name) not in desired:
                continue
            target = safe_target(args.output_root, "test/" + without_test_prefix(member.name))
            if member.issym() or member.islnk() or not member.isfile():
                raise RuntimeError("Refusing non-regular selected archive member: %s" % member.name)
            if target.exists():
                raise FileExistsError("Refusing to overwrite existing file: %s" % target)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            with target.open("wb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            extracted += 1
    print("extracted=%d" % extracted)


if __name__ == "__main__":
    main()
