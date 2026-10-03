import argparse
from collections import Counter
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METADATA_EXTENSIONS = {".json", ".txt", ".csv", ".pkl", ".pickle", ".npz", ".npy", ".yaml", ".yml"}
OPTIM_NUMERIC_FIELDS = {
    "cam_id",
    "width",
    "height",
    "fx",
    "fy",
    "cx",
    "cy",
    "k1",
    "k2",
    "p1",
    "p2",
    "qvecw",
    "qvecx",
    "qvecy",
    "qvecz",
    "tvecx",
    "tvecy",
    "tvecz",
}


def inspect_gigahands(root):
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    extension_counts = Counter(path.suffix.lower() or "<no extension>" for path in files)
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    metadata_files = [
        path
        for path in files
        if path.suffix.lower() in METADATA_EXTENSIONS
        or any(token in path.name.lower() for token in ("annotation", "metadata", "param", "pose"))
    ]

    print(f"dataset root: {root}")
    print("top-level entries:")
    for child in sorted(root.iterdir()) if root.exists() else []:
        kind = "dir" if child.is_dir() else "file"
        print(f"  [{kind}] {child.name}")

    print("major subdirectories:")
    for child in sorted(path for path in root.rglob("*") if path.is_dir())[:50]:
        print(f"  {child.relative_to(root)}")

    print("file counts by extension:")
    for extension, count in sorted(extension_counts.items()):
        print(f"  {extension}: {count}")

    print(f"total files: {len(files)}")
    print(f"image files: {len(image_files)}")

    print("annotation/metadata-like files:")
    for path in sorted(metadata_files)[:100]:
        print(f"  {path.relative_to(root)}")
    if len(metadata_files) > 100:
        print(f"  ... {len(metadata_files) - 100} more")


def parse_optim_params(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = []
    rows = []
    malformed = []
    if lines and lines[0].startswith("#"):
        header = lines[0].lstrip("# ").split()

    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != len(header):
            malformed.append((line_number, "column_count", line))
            continue
        row = dict(zip(header, parts))
        numeric = {}
        for field in OPTIM_NUMERIC_FIELDS & set(header):
            try:
                numeric[field] = float(row[field])
            except ValueError:
                malformed.append((line_number, f"numeric:{field}", row[field]))
        row["_numeric"] = numeric
        row["_line_number"] = line_number
        rows.append(row)
    return header, rows, malformed


def summarize_values(values):
    if not values:
        return "n/a"
    return (
        f"min={min(values):.6g} max={max(values):.6g} "
        f"mean={statistics.fmean(values):.6g} median={statistics.median(values):.6g}"
    )


def probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return None, result.stderr.strip()
    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    duration = stream.get("duration") or fmt.get("duration")
    return {
        "width": int(stream["width"]) if stream.get("width") is not None else None,
        "height": int(stream["height"]) if stream.get("height") is not None else None,
        "nb_frames": stream.get("nb_frames"),
        "duration": float(duration) if duration not in (None, "N/A") else None,
        "size": int(fmt["size"]) if fmt.get("size") is not None else path.stat().st_size,
    }, None


def decode_representative_frames(path, width, height, duration):
    if not width or not height:
        return False, "missing_dimensions"
    if duration and duration > 0.2:
        positions = [0.0, duration / 2.0, max(duration - 0.1, 0.0)]
    else:
        positions = [0.0]
    expected_bytes = width * height * 3
    for position in positions:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{position:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(command, check=False, capture_output=True)
        if result.returncode != 0:
            return False, f"ffmpeg:{result.stderr.decode('utf-8', errors='replace').strip()}"
        if len(result.stdout) != expected_bytes:
            return False, f"decoded_bytes:{len(result.stdout)} expected:{expected_bytes}"
        if not result.stdout:
            return False, "empty_frame"
    return True, None


def finite_positive(value):
    return math.isfinite(value) and value > 0


def audit_optim_params(root):
    sequence_rows = {}
    print("optim_params audit:")
    for params_path in sorted((root / "hand_pose").glob("*/optim_params.txt")):
        sequence = params_path.parent.name
        header, rows, malformed = parse_optim_params(params_path)
        sequence_rows[sequence] = rows
        cam_ids = [row.get("cam_id") for row in rows]
        cam_names = [row.get("cam_name") for row in rows]
        duplicate_cam_ids = sorted(name for name, count in Counter(cam_ids).items() if count > 1)
        duplicate_cam_names = sorted(name for name, count in Counter(cam_names).items() if count > 1)
        nan_inf = 0
        non_positive_focal = 0
        invalid_size = 0
        cx_cy_warnings = 0
        for row in rows:
            numeric = row["_numeric"]
            values = [numeric.get(field) for field in ("width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")]
            nan_inf += sum(1 for value in values if value is not None and not math.isfinite(value))
            if not finite_positive(numeric.get("fx", 0.0)) or not finite_positive(numeric.get("fy", 0.0)):
                non_positive_focal += 1
            if not finite_positive(numeric.get("width", 0.0)) or not finite_positive(numeric.get("height", 0.0)):
                invalid_size += 1
            width = numeric.get("width")
            height = numeric.get("height")
            cx = numeric.get("cx")
            cy = numeric.get("cy")
            if width is not None and cx is not None and not (0 <= cx <= width):
                cx_cy_warnings += 1
            if height is not None and cy is not None and not (0 <= cy <= height):
                cx_cy_warnings += 1

        fx_values = [row["_numeric"]["fx"] for row in rows if "fx" in row["_numeric"]]
        fy_values = [row["_numeric"]["fy"] for row in rows if "fy" in row["_numeric"]]
        focal_abs_diff = [abs(row["_numeric"]["fx"] - row["_numeric"]["fy"]) for row in rows if "fx" in row["_numeric"] and "fy" in row["_numeric"]]
        focal_rel_diff = [
            abs(row["_numeric"]["fx"] - row["_numeric"]["fy"]) / max(abs(row["_numeric"]["fx"]), abs(row["_numeric"]["fy"]))
            for row in rows
            if "fx" in row["_numeric"] and "fy" in row["_numeric"]
        ]
        print(
            f"  {sequence}: rows={len(rows)} header_fields={len(header)} malformed={len(malformed)} "
            f"unique_cam_id={len(set(cam_ids))} duplicate_cam_id={len(duplicate_cam_ids)} "
            f"unique_cam_name={len(set(cam_names))} duplicate_cam_name={len(duplicate_cam_names)} "
            f"nan_or_inf={nan_inf} non_positive_fx_fy={non_positive_focal} "
            f"invalid_width_height={invalid_size} cx_cy_warnings={cx_cy_warnings}"
        )
        print(f"    fx {summarize_values(fx_values)}")
        print(f"    fy {summarize_values(fy_values)}")
        print(f"    abs_fx_fy_diff {summarize_values(focal_abs_diff)}")
        print(f"    rel_fx_fy_diff {summarize_values(focal_rel_diff)}")
        for line_number, reason, value in malformed[:5]:
            print(f"    malformed line={line_number} reason={reason} value={value}")

    all_rows = [row for rows in sequence_rows.values() for row in rows]
    all_fx = [row["_numeric"]["fx"] for row in all_rows if "fx" in row["_numeric"]]
    all_fy = [row["_numeric"]["fy"] for row in all_rows if "fy" in row["_numeric"]]
    all_abs = [abs(row["_numeric"]["fx"] - row["_numeric"]["fy"]) for row in all_rows if "fx" in row["_numeric"] and "fy" in row["_numeric"]]
    all_rel = [
        abs(row["_numeric"]["fx"] - row["_numeric"]["fy"]) / max(abs(row["_numeric"]["fx"]), abs(row["_numeric"]["fy"]))
        for row in all_rows
        if "fx" in row["_numeric"] and "fy" in row["_numeric"]
    ]
    print("  TOTAL:")
    print(f"    rows={len(all_rows)} fx {summarize_values(all_fx)}")
    print(f"    fy {summarize_values(all_fy)}")
    print(f"    abs_fx_fy_diff {summarize_values(all_abs)}")
    print(f"    rel_fx_fy_diff {summarize_values(all_rel)}")
    return sequence_rows


def audit_rgb_camera_mapping(root, sequence_rows, decode_frames=False):
    print("rgb camera mapping audit:")
    total = Counter()
    decode_failures = []
    resolution_mismatches = []
    missing_params = []
    for sequence, rows in sorted(sequence_rows.items()):
        row_by_cam = {row["cam_name"]: row for row in rows}
        videos = sorted((root / "hand_pose" / sequence / "rgb_vid").glob("*/*.mp4"))
        video_cam_names = {path.parent.name for path in videos}
        param_cam_names = set(row_by_cam)
        matched = 0
        resolution_match = 0
        sequence_missing_params = 0
        sequence_resolution_mismatches = 0
        decode_success = 0
        for video in videos:
            cam_name = video.parent.name
            row = row_by_cam.get(cam_name)
            if row is None:
                missing_params.append(str(video))
                sequence_missing_params += 1
                continue
            matched += 1
            metadata, error = probe_video(video)
            if error:
                resolution_mismatches.append((str(video), cam_name, f"probe_error:{error}", "n/a"))
                sequence_resolution_mismatches += 1
                continue
            param_width = int(row["_numeric"]["width"])
            param_height = int(row["_numeric"]["height"])
            video_resolution = (metadata["width"], metadata["height"])
            param_resolution = (param_width, param_height)
            if video_resolution == param_resolution:
                resolution_match += 1
            else:
                resolution_mismatches.append((str(video), cam_name, video_resolution, param_resolution))
                sequence_resolution_mismatches += 1
            if decode_frames:
                ok, reason = decode_representative_frames(video, metadata["width"], metadata["height"], metadata["duration"])
                if ok:
                    decode_success += 1
                else:
                    decode_failures.append((str(video), reason))

        param_without_rgb = sorted(param_cam_names - video_cam_names)
        missing_param_cams = sorted(video_cam_names - param_cam_names)
        print(
            f"  {sequence}: param_rows={len(rows)} unique_param_cams={len(param_cam_names)} "
            f"rgb_camera_dirs={len(video_cam_names)} rgb_mp4={len(videos)} matched={matched} "
            f"missing_param_for_rgb_cam={len(missing_param_cams)} param_cam_without_rgb={len(param_without_rgb)} "
            f"resolution_match={resolution_match} resolution_mismatch={sequence_resolution_mismatches} "
            f"decode_success={decode_success if decode_frames else 'skipped'}"
        )
        total["param_rows"] += len(rows)
        total["unique_param_cams_sum"] += len(param_cam_names)
        total["rgb_camera_dirs_sum"] += len(video_cam_names)
        total["rgb_mp4"] += len(videos)
        total["matched"] += matched
        total["resolution_match"] += resolution_match
        if decode_frames:
            total["decode_success"] += decode_success
    print(
        f"  TOTAL: param_rows={total['param_rows']} unique_param_cams_sum={total['unique_param_cams_sum']} "
        f"rgb_camera_dirs_sum={total['rgb_camera_dirs_sum']} rgb_mp4={total['rgb_mp4']} matched={total['matched']} "
        f"resolution_match={total['resolution_match']} resolution_mismatch={len(resolution_mismatches)} "
        f"missing_param={len(missing_params)} decode_failures={len(decode_failures) if decode_frames else 'skipped'}"
    )
    for item in resolution_mismatches[:20]:
        print(f"    resolution_mismatch video={item[0]} cam={item[1]} video={item[2]} param={item[3]}")
    for path, reason in decode_failures[:20]:
        print(f"    decode_failure video={path} reason={reason}")
    return resolution_mismatches, missing_params, decode_failures


def audit_multivideo_dirs(root):
    print("multi-video RGB camera directories:")
    for sequence_dir in sorted((root / "hand_pose").iterdir()):
        if not sequence_dir.is_dir():
            continue
        for camera_dir in sorted((sequence_dir / "rgb_vid").iterdir()):
            videos = sorted(camera_dir.glob("*.mp4"))
            if len(videos) <= 1:
                continue
            print(f"  {sequence_dir.name}/{camera_dir.name}: mp4_count={len(videos)}")
            for video in videos:
                metadata, error = probe_video(video)
                sidecar = video.with_suffix(".txt")
                first_lines = []
                if sidecar.exists():
                    first_lines = sidecar.read_text(encoding="utf-8").splitlines()[:3]
                print(
                    f"    {video.name}: size={video.stat().st_size} "
                    f"duration={metadata.get('duration') if metadata else 'probe_error'} "
                    f"frames={metadata.get('nb_frames') if metadata else error} sidecar={sidecar.name if sidecar.exists() else 'missing'}"
                )
                for line in first_lines:
                    print(f"      txt: {line}")


def audit_cross_sequence_cameras(sequence_rows):
    print("cross-sequence camera intrinsics:")
    by_cam = defaultdict(list)
    for sequence, rows in sequence_rows.items():
        for row in rows:
            by_cam[row["cam_name"]].append((sequence, row))
    sequence_names = sorted(sequence_rows)
    common = [cam for cam, items in by_cam.items() if len({seq for seq, _ in items}) == len(sequence_names)]
    partial = [cam for cam, items in by_cam.items() if len({seq for seq, _ in items}) < len(sequence_names)]
    exact_same = 0
    different = 0
    max_diffs = {field: 0.0 for field in ("width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")}
    example_diffs = []
    fields = tuple(max_diffs)
    for cam_name, items in sorted(by_cam.items()):
        values = []
        for sequence, row in items:
            values.append((sequence, tuple(row["_numeric"][field] for field in fields)))
        first = values[0][1]
        if all(value == first for _, value in values):
            exact_same += 1
        else:
            different += 1
            if len(example_diffs) < 5:
                example_diffs.append((cam_name, values))
        for index, field in enumerate(fields):
            field_values = [value[index] for _, value in values]
            max_diffs[field] = max(max_diffs[field], max(field_values) - min(field_values))
    print(f"  sequences={len(sequence_names)} cam_name_total={len(by_cam)} common_to_all_sequences={len(common)} partial={len(partial)}")
    print(f"  exact_same_intrinsics={exact_same} different_intrinsics={different}")
    print("  max_diffs=" + " ".join(f"{field}={value:.6g}" for field, value in max_diffs.items()))
    for cam_name, values in example_diffs:
        print(f"    differing_cam={cam_name}")
        for sequence, value in values:
            print(f"      {sequence}: {dict(zip(fields, value))}")


def audit_camera_data(root, decode_frames=False):
    root = Path(root)
    sequence_rows = audit_optim_params(root)
    audit_rgb_camera_mapping(root, sequence_rows, decode_frames=decode_frames)
    audit_multivideo_dirs(root)
    audit_cross_sequence_cameras(sequence_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the local GigaHands demo structure.")
    parser.add_argument("--root", default="datasets/gigahands", help="Dataset root to inspect.")
    parser.add_argument("--camera-audit", action="store_true", help="Run read-only RGB/camera consistency checks.")
    parser.add_argument("--decode-frames", action="store_true", help="Decode representative frames during camera audit.")
    return parser.parse_args()


def main():
    args = parse_args()
    inspect_gigahands(args.root)
    if args.camera_audit:
        audit_camera_data(args.root, decode_frames=args.decode_frames)


if __name__ == "__main__":
    main()
