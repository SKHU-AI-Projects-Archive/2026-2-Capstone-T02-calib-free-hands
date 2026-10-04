"""Frame prediction raw CSV에서 clip, pair, 전체 summary를 만든다."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import statistics


FRAME_FIELDS = [
    "participant", "sequence", "camera", "camera_key", "video_path", "video_name",
    "frame_index", "width", "height", "gt_fx", "gt_fy", "gt_cx", "gt_cy",
    "pred_fx", "pred_fy", "pred_cx", "pred_cy", "pred_width", "pred_height", "success", "rel_fx_error",
    "rel_fy_error", "max_rel_f_error", "max_rel_c_error",
]
PRED_FIELDS = ("pred_fx", "pred_fy", "pred_cx", "pred_cy")
ERROR_FIELDS = ("rel_fx_error", "rel_fy_error", "max_rel_f_error", "max_rel_c_error")


def _number(value):
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def write_frame_predictions(path, rows):
    """Frame prediction row를 gzip CSV로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FRAME_FIELDS})


def read_frame_predictions(path):
    """raw gzip CSV를 numeric field가 변환된 list로 읽는다."""
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("frame_index", "width", "height", "pred_width", "pred_height", "gt_fx", "gt_fy", "gt_cx", "gt_cy", *PRED_FIELDS, "rel_fx_error", "rel_fy_error", *ERROR_FIELDS):
            row[field] = int(row[field]) if field in ("frame_index", "width", "height", "pred_width", "pred_height") and row[field] else _number(row[field])
        row["success"] = str(row["success"]).lower() in {"1", "true", "yes"}
    return rows


def _stats(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    if not values:
        return {f"{field}_{name}": None for name in ("mean", "median", "variance", "std")}
    mean = statistics.fmean(values)
    variance = statistics.pvariance(values)
    return {
        f"{field}_mean": mean,
        f"{field}_median": statistics.median(values),
        f"{field}_variance": variance,
        f"{field}_std": variance**0.5,
    }


def _group_summary(rows, include_clip_count=False):
    successful = [row for row in rows if row["success"]]
    first = rows[0]
    summary = {
        "participant": first["participant"], "sequence": first["sequence"], "camera": first["camera"],
        "camera_key": first["camera_key"], "total_frames": len(rows),
        "successful_frames": len(successful), "success_rate": len(successful) / len(rows),
    }
    if include_clip_count:
        summary["clip_count"] = len({row["video_name"] for row in rows})
    for field in PRED_FIELDS:
        summary.update(_stats(successful, field))
    for field in ERROR_FIELDS:
        summary.update(_stats(successful, field))
    if include_clip_count:
        for field in PRED_FIELDS:
            summary[f"median_{field}"] = statistics.median([row[field] for row in successful if row[field] is not None]) if successful else None
        summary["pair_rel_fx_error"] = _pair_component_error(summary, "fx", first)
        summary["pair_rel_fy_error"] = _pair_component_error(summary, "fy", first)
        summary["pair_max_rel_f_error"] = max(
            value for value in (summary["pair_rel_fx_error"], summary["pair_rel_fy_error"])
            if value is not None
        ) if summary["pair_rel_fx_error"] is not None and summary["pair_rel_fy_error"] is not None else None
        summary["pair_max_rel_c_error"] = _pair_center_error(summary, first)
    return summary


def _pair_component_error(summary, axis, first):
    gt = first[f"gt_{axis}"]
    prediction = summary[f"median_pred_{axis}"]
    if prediction is None or gt is None:
        return None
    return abs(prediction - gt) / abs(gt)


def _pair_center_error(summary, first):
    if summary["median_pred_cx"] is None or first["gt_cx"] is None:
        return None
    return 2 * max(abs(summary["median_pred_cx"] - first["gt_cx"]) / first["width"], abs(summary["median_pred_cy"] - first["gt_cy"]) / first["height"])


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(raw_path, output_dir):
    """raw prediction에서 clip/pair CSV와 metrics.json을 생성한다."""
    rows = read_frame_predictions(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = defaultdict(list)
    pairs = defaultdict(list)
    for row in rows:
        clips[(row["camera_key"], row["video_name"])].append(row)
        pairs[row["camera_key"]].append(row)
    clip_rows = [_group_summary(group) for group in clips.values()]
    pair_rows = [_group_summary(group, include_clip_count=True) for group in pairs.values()]
    _write_csv(output_dir / "clip_summary.csv", clip_rows)
    _write_csv(output_dir / "pair_summary.csv", pair_rows)
    successful = [row for row in rows if row["success"]]
    frame_level = {"total_frames": len(rows), "successful_frames": len(successful), "success_rate": len(successful) / len(rows) if rows else 0}
    for field in ERROR_FIELDS:
        frame_level.update(_stats(successful, field))
    pair_level = {"pair_count": len(pair_rows)}
    valid_pairs = [row for row in pair_rows if row["pair_max_rel_f_error"] is not None]
    pair_level["valid_pair_count"] = len(valid_pairs)
    for field in ("pair_rel_fx_error", "pair_rel_fy_error", "pair_max_rel_f_error", "pair_max_rel_c_error"):
        values = [row[field] for row in pair_rows if row[field] is not None]
        pair_level[f"{field}_mean"] = statistics.fmean(values) if values else None
        pair_level[f"{field}_median"] = statistics.median(values) if values else None
    within_5 = [row for row in valid_pairs if row["pair_max_rel_f_error"] <= 0.05]
    pair_level["pair_focal_within_5pct_count"] = len(within_5)
    pair_level["pair_focal_within_5pct_rate"] = len(within_5) / len(valid_pairs) if valid_pairs else None
    stability_values = [row["pred_fx_std"] for row in pair_rows if row["pred_fx_std"] is not None]
    metrics = {
        "frame_level": frame_level,
        "pair_level": pair_level,
        "stability": {"pair_fx_std_mean": statistics.fmean(stability_values) if stability_values else None},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
