"""Shared paths and helpers for CAM-EXP-001.3."""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = RUN_DIR.parents[1]
REPO = EXPERIMENTS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.src.datasets.gigahands import takes  # noqa: E402

CAM_EXP_001 = EXPERIMENTS / "runs" / "CAM-EXP-001_gt_projection_validation"
CAM_EXP_0011 = EXPERIMENTS / "runs" / "CAM-EXP-001_1_gigahands_bad_view_diagnosis"
CAM_EXP_0012 = EXPERIMENTS / "runs" / "CAM-EXP-001_2_gigahands_frame_mapping_audit"
CENSUS_CSV = CAM_EXP_0012 / "results" / "raw" / "gigahands_demo_qc_census.csv.gz"
MANIFESTS = EXPERIMENTS / "manifests"

TAKES = takes()

# four-colour scheme, kept consistent with CAM-EXP-001.2 but with a different
# meaning for red/orange: those are now reconstructed from OTHER cameras only.
COLORS = {("left", "2d"): "#22c55e",     # green  - held-out LEFT 2D annotation
          ("left", "3d"): "#ef4444",     # red    - LOCO triangulated LEFT projection
          ("right", "2d"): "#3b82f6",    # blue   - held-out RIGHT 2D annotation
          ("right", "3d"): "#f97316"}    # orange - LOCO triangulated RIGHT projection
COLOR_LEGEND = ("green = held-out LEFT 2D annotation    blue = held-out RIGHT 2D annotation\n"
                "red = LEFT 3D triangulated from OTHER CAMERAS ONLY, projected in    "
                "orange = RIGHT, same")


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def write_csv(path: Path, rows: list, gzipped: bool | None = None) -> None:
    if not rows:
        return
    if gzipped is None:
        gzipped = str(path).endswith(".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and not k.startswith("_"):
                seen.add(k)
                keys.append(k)
    opener = ((lambda: gzip.open(path, "wt", newline="", encoding="utf-8")) if gzipped
              else (lambda: open(path, "w", newline="", encoding="utf-8")))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)")


def read_csv(path: Path) -> list:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
