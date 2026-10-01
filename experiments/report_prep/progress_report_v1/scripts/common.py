"""Shared loading helpers for the report-preparation package.

Nothing here computes a new research result. Every value is read out of a
canonical raw/summary file produced by CAM-EXP-001..004.1.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[2]
EXP = REPO / "experiments"
RUNS = EXP / "runs"
MANIFESTS = EXP / "manifests"

R001 = RUNS / "CAM-EXP-001_gt_projection_validation"
R0011 = RUNS / "CAM-EXP-001_1_gigahands_bad_view_diagnosis"
R0012 = RUNS / "CAM-EXP-001_2_gigahands_frame_mapping_audit"
R0013 = RUNS / "CAM-EXP-001_3_gigahands_multiview_triangulation"
R002 = RUNS / "CAM-EXP-002_camera_focal_sensitivity"
R003 = RUNS / "CAM-EXP-003_single_frame_calibration_benchmark"
R0031 = RUNS / "CAM-EXP-003_1_distortion_aware_diagnostic"
R004 = RUNS / "CAM-EXP-004_static_camera_multiframe_aggregation"
R0041 = RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit"


def read_csv(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md_table(path, rows, columns=None, title=None, notes=None) -> None:
    """Same content as the CSV, rendered for a human reader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or list(rows[0].keys())
    out = []
    if title:
        out.append(f"# {title}\n")
    if notes:
        out.append(notes.strip() + "\n")
    out.append("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            v = "" if v is None else str(v)
            cells.append(v.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def pick(rows, **eq):
    """The single row matching all key=value constraints."""
    out = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in eq.items())]
    if len(out) != 1:
        raise LookupError(f"expected exactly 1 row for {eq}, got {len(out)}")
    return out[0]
