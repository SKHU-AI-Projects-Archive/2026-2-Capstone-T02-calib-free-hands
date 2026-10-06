"""Verify the official InterHand 5fps multipart image archive and manifest mapping."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_BYTES = 81718036480
PARTS = ["part%s" % (chr(97 + first) + chr(97 + second)) for first in range(2) for second in range(26) if not (first == 1 and second > 17)]


def part_paths(archive_dir):
    archive_dir = Path(archive_dir)
    return [archive_dir / ("InterHand2.6M.images.5.fps.v1.0.tar.%s" % suffix) for suffix in PARTS]


def read_manifest(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_full_test_data(path):
    with Path(path).open() as handle:
        data = json.load(handle)
    return data["images"]


def candidate_names(file_name):
    return {file_name, "test/" + file_name, "images/test/" + file_name}


def stream_members(paths):
    process = subprocess.Popen(["tar", "-tf", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for path in paths:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    process.stdin.write(chunk)
        process.stdin.close()
        for line in process.stdout:
            yield line.decode().rstrip("\n")
        if process.wait() != 0:
            raise RuntimeError("tar member scan failed")
    finally:
        if process.poll() is None:
            process.kill()


def checksum_report(checksum_path, paths):
    checksum_path = Path(checksum_path)
    if not checksum_path.exists():
        return {"present": False, "checked": 0, "failed": [], "note": "CHECKSUM file is missing"}
    failed, checked = [], 0
    pattern = re.compile(r"^([0-9a-fA-F]{32}|[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\s+[* ]?(.+)$")
    by_name = {path.name: path for path in paths}
    for line in checksum_path.read_text(errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match or Path(match.group(2)).name not in by_name:
            continue
        expected, name = match.groups()
        algorithm = {32: "md5", 40: "sha1", 64: "sha256"}[len(expected)]
        digest_object = hashlib.new(algorithm)
        with by_name[Path(name).name].open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_object.update(chunk)
        digest = digest_object.hexdigest()
        checked += 1
        if digest.lower() != expected.lower():
            failed.append(name)
    return {"present": True, "checked": checked, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--full-test-data", type=Path, help="Canonical Official-Test data JSON for a separate full mapping gate.")
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--scan-members", action="store_true")
    args = parser.parse_args()
    paths = part_paths(args.archive_dir)
    missing = [str(path) for path in paths if not path.exists()]
    result = {"expected_part_count": 44, "present_part_count": len(paths) - len(missing), "expected_bytes": EXPECTED_BYTES, "present_bytes": sum(path.stat().st_size for path in paths if path.exists()), "missing_parts": missing}
    result["size_matches"] = not missing and result["present_bytes"] == EXPECTED_BYTES
    result["checksum"] = checksum_report(args.checksum or args.archive_dir / "InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM", paths)
    result["checksum"]["scope"] = "per-part checksums listed in CHECKSUM; reconstructed tar checksum is not assumed"
    if args.scan_members and not missing:
        members = set(stream_members(paths))
        rows = read_manifest(args.manifest)
        matched = sum(bool(candidate_names(row["file_name"]) & members) for row in rows)
        result.update({"manifest_rows": len(rows), "archive_members": len(members), "manifest_matches": matched, "manifest_missing": len(rows) - matched})
        if args.full_test_data:
            full_rows = read_full_test_data(args.full_test_data)
            full_matched = sum(bool(candidate_names(row["file_name"]) & members) for row in full_rows)
            result.update({"full_test_rows": len(full_rows), "full_test_matches": full_matched, "full_test_missing": len(full_rows) - full_matched})
    print(json.dumps(result, indent=2))
    if missing or not result["size_matches"] or result["checksum"]["failed"] or result.get("manifest_missing", 0) or result.get("full_test_missing", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
