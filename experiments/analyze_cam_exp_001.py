"""Derive every CAM-EXP-001 table and figure from the raw per-joint CSV.

Nothing here touches the datasets again: the raw CSV is the single source of
truth, so new tables, statistics and plots can be produced later without
re-reading any data. Run after run_cam_exp_001.py.
"""
from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments.src.metrics import summarize  # noqa: E402

RUN_DIR = HERE / "runs" / "CAM-EXP-001_gt_projection_validation"
RAW = RUN_DIR / "results" / "raw" / "reprojection_per_joint.csv.gz"
VIEW_KEYS = ("dataset", "subset", "sequence", "camera", "frame", "hand")
# Descriptive buckets used to show the shape of the distribution, never as
# a pass/fail criterion.
THRESHOLDS = (1, 2, 5, 10, 20, 50, 100, 200, 500)


def load():
    """Group comparable errors (valid joint, both GT and projection in frame)."""
    by_joint = defaultdict(list)
    by_view = defaultdict(list)
    by_dataset = defaultdict(list)
    with gzip.open(RAW, "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["valid"] != "1" or not r["error_px"]:
                continue
            if r["in_image"] != "1" or r["gt_in_image"] != "1":
                continue
            e = float(r["error_px"])
            ds = r["dataset"]
            by_dataset[ds].append(e)
            by_joint[(ds, int(r["joint"]))].append(e)
            by_view[tuple(r[k] for k in VIEW_KEYS)].append(e)
    return by_dataset, by_joint, by_view


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path.relative_to(HERE.parent).as_posix(), f"({len(rows)} rows)")


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW}; run run_cam_exp_001.py first")
    by_dataset, by_joint, by_view = load()

    write_csv(RUN_DIR / "tables" / "per_joint_summary.csv",
              [{"dataset": d, "joint": j, **summarize(e)}
               for (d, j), e in sorted(by_joint.items())])

    view_rows = [{**dict(zip(VIEW_KEYS, k)), **summarize(e)}
                 for k, e in sorted(by_view.items())]
    write_csv(RUN_DIR / "results" / "summary" / "view_level_summary.csv", view_rows)

    # View-level agreement: how many whole (camera, frame, hand) views agree.
    agg = defaultdict(list)
    for r in view_rows:
        if r["median_px"] != "":
            agg[r["dataset"]].append(float(r["median_px"]))
    dist_rows = []
    for ds, meds in sorted(agg.items()):
        m = np.array(meds)
        row = {"dataset": ds, "n_views": int(m.size),
               "view_median_of_medians": round(float(np.median(m)), 4)}
        for t in THRESHOLDS:
            row[f"frac_views_median_under_{t}px"] = round(float((m < t).mean()), 4)
        good = m[m < 50]
        row["n_views_under_50px"] = int(good.size)
        row["median_px_within_those_views"] = (round(float(np.median(good)), 4)
                                               if good.size else "")
        dist_rows.append(row)
    write_csv(RUN_DIR / "tables" / "view_agreement_summary.csv", dist_rows)

    joint_dist = []
    for ds, e in sorted(by_dataset.items()):
        a = np.array(e)
        row = {"dataset": ds, **summarize(e)}
        for t in THRESHOLDS:
            row[f"frac_joints_under_{t}px"] = round(float((a < t).mean()), 4)
        joint_dist.append(row)
    write_csv(RUN_DIR / "tables" / "error_distribution.csv", joint_dist)

    _plots(by_dataset, agg)


def _plots(by_dataset, agg) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable; skipping figures")
        return
    out_dir = RUN_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data, xlabel in (
        ("error_histogram", by_dataset, "per-joint reprojection error [px]"),
        ("view_median_histogram", agg, "per-view median reprojection error [px]"),
    ):
        items = [(k, np.asarray(v)) for k, v in sorted(data.items()) if len(v)]
        if not items:
            continue
        fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 3.6),
                                 squeeze=False)
        for ax, (ds, a) in zip(axes[0], items):
            pos = np.clip(a, 1e-3, None)
            ax.hist(pos, bins=np.logspace(-3, 4, 60), color="#3b7dd8")
            ax.set_xscale("log")
            ax.axvline(np.median(pos), color="#d8443b", ls="--",
                       label=f"median {np.median(a):.2f} px")
            ax.set_title(f"{ds} (n={a.size})", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("count", fontsize=8)
            ax.legend(fontsize=8)
        fig.suptitle("CAM-EXP-001 GT projection validation", fontsize=11)
        fig.tight_layout()
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("  wrote", p.relative_to(HERE.parent).as_posix())


if __name__ == "__main__":
    main()
