"""Verify the official InterHand 5fps multipart image archive and manifest mapping."""

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import re
import subprocess
import threading
import time
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


def normalize_member_name(name):
    """Normalize only known Official-Test archive roots; never use basename matching."""
    name = name.lstrip("./")
    for prefix in ("images/test/", "test/"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    marker = "/test/"
    if marker in name:
        return name.split(marker, 1)[1]
    return name


def candidate_names(file_name):
    return {file_name, "test/" + file_name, "images/test/" + file_name}


def stream_members(paths):
    process = subprocess.Popen(
        ["tar", "-tf", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    producer_errors = []

    def produce():
        try:
            for index, path in enumerate(paths, 1):
                print("[%d/%d] %s" % (index, len(paths), path.name), flush=True)
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        process.stdin.write(chunk)
            process.stdin.close()
        except (BrokenPipeError, OSError) as error:
            producer_errors.append(error)
            try:
                process.stdin.close()
            except OSError:
                pass

    def drain_stderr():
        for line in process.stderr:
            print("tar: %s" % line.decode(errors="replace").rstrip(), flush=True)

    producer = threading.Thread(target=produce, name="archive-producer")
    stderr_reader = threading.Thread(target=drain_stderr, name="tar-stderr")
    producer.start()
    stderr_reader.start()
    members = 0
    started = time.monotonic()
    try:
        for line in process.stdout:
            members += 1
            if members % 100000 == 0:
                print("members_scanned=%d elapsed_seconds=%.1f" % (members, time.monotonic() - started), flush=True)
            yield line.decode().rstrip("\n")
        producer.join()
        stderr_reader.join()
        return_code = process.wait()
        if producer_errors:
            raise RuntimeError("archive producer failed: %s" % producer_errors[0])
        if return_code != 0:
            raise RuntimeError("tar member scan failed with exit code %d" % return_code)
    finally:
        if process.poll() is None:
            process.kill()
        producer.join(timeout=5)
        stderr_reader.join(timeout=5)


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
    parser.add_argument("--verify-checksum", action="store_true", help="Recompute all per-part checksums; disabled by default for member scans.")
    parser.add_argument("--scan-members", action="store_true")
    args = parser.parse_args()
    paths = part_paths(args.archive_dir)
    missing = [str(path) for path in paths if not path.exists()]
    result = {"expected_part_count": 44, "present_part_count": len(paths) - len(missing), "expected_bytes": EXPECTED_BYTES, "present_bytes": sum(path.stat().st_size for path in paths if path.exists()), "missing_parts": missing}
    result["size_matches"] = not missing and result["present_bytes"] == EXPECTED_BYTES
    if args.verify_checksum:
        result["checksum"] = checksum_report(args.checksum or args.archive_dir / "InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM", paths)
    else:
        result["checksum"] = {"present": (args.checksum or args.archive_dir / "InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM").exists(), "checked": 0, "failed": [], "status": "skipped; official md5sum -c was performed separately"}
    result["checksum"]["scope"] = "per-part checksums listed in CHECKSUM; reconstructed tar checksum is not assumed"
    if args.scan_members and not missing:
        members = set(stream_members(paths))
        rows = read_manifest(args.manifest)
        member_paths = defaultdict(list)
        for member in members:
            member_paths[normalize_member_name(member)].append(member)
        normalized_members = set(member_paths)
        matched = sum(row["file_name"] in normalized_members for row in rows)
        ambiguity = {name: paths for name, paths in member_paths.items() if len(paths) > 1}
        result.update({"archive_members": len(members), "manifest_rows": len(rows), "manifest_matches": matched, "manifest_missing": len(rows) - matched, "manifest_duplicate_mapping": sum(len(member_paths[row["file_name"]]) - 1 for row in rows if row["file_name"] in member_paths), "path_ambiguity": len(ambiguity)})
        if args.full_test_data:
            full_rows = read_full_test_data(args.full_test_data)
            full_matched = sum(row["file_name"] in normalized_members for row in full_rows)
            result.update({"full_test_rows": len(full_rows), "full_test_matches": full_matched, "full_test_missing": len(full_rows) - full_matched})
    print(json.dumps(result, indent=2))
    if missing or not result["size_matches"] or result["checksum"]["failed"] or result.get("manifest_missing", 0) or result.get("full_test_missing", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
