"""CAM-EXP-001 - GT projection validation.

Validates that the official GT camera + GT 3D joints of each dataset, when
interpreted with that dataset's own convention, reproject onto the dataset's
2D annotation. This is a dataset-loader / camera-convention validation, not
an evaluation of any calibration method.

Usage:  python run_cam_exp_001.py [--full]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments.src.datasets import assemblyhands, gigahands, hanco, interhand26m  # noqa: E402
from experiments.src.datasets.common import rel  # noqa: E402
from experiments.src.metrics import per_joint_rows, summarize  # noqa: E402
from experiments.src.metrics.reprojection import RAW_COLUMNS  # noqa: E402
from experiments.src.visualization import draw_overlay  # noqa: E402
from experiments.src.visualization.overlay import load_frame  # noqa: E402

RUN_DIR = HERE / "runs" / "CAM-EXP-001_gt_projection_validation"
INTERHAND_SET, INTERHAND_SPLIT = "human_annot", "test"

SMOKE = {"gigahands": dict(max_sequences=1, max_cameras=3, max_frames=3),
         "interhand26m": dict(max_images=40),
         "hanco": dict(max_sequences=1, max_frames=3),
         "assemblyhands": dict(max_images=20),
         "assemblyhands_eccv24": dict(max_images=20)}
FULL = {"gigahands": dict(max_sequences=None, max_cameras=None, max_frames=20),
        "interhand26m": dict(max_images=1500),
        "hanco": dict(max_sequences=None, max_frames=20),
        "assemblyhands": dict(max_images=400),
        "assemblyhands_eccv24": dict(max_images=1500)}


def git(*args, default=""):
    try:
        return subprocess.check_output(["git", *args], cwd=HERE.parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def setup_logging() -> logging.Logger:
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("cam-exp-001")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(RUN_DIR / "logs" / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def write_environment() -> dict:
    import cv2
    env = {
        "date": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "gpu_used": False,
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env


def build_sources(cfg: dict):
    """(name, iterator, overlay-capable) per dataset, skipping absent data."""
    out = []
    try:
        if gigahands.sequences():
            out.append(("gigahands", gigahands.iter_samples(**cfg["gigahands"]), True))
    except Exception:
        pass
    try:
        if interhand26m.available_sets():
            out.append(("interhand26m", interhand26m.iter_samples(
                INTERHAND_SET, INTERHAND_SPLIT, **cfg["interhand26m"]), False))
    except Exception:
        pass
    if hanco.tester_root().is_dir():
        out.append(("hanco", hanco.iter_samples(**cfg["hanco"]), True))
    ah_splits = assemblyhands.discover_splits()
    if "demo" in ah_splits:
        out.append(("assemblyhands", assemblyhands.iter_samples("demo", **cfg["assemblyhands"]), False))
    if "test-eccv2024" in ah_splits:
        out.append(("assemblyhands_eccv24", assemblyhands.iter_samples(
            "test-eccv2024", **cfg["assemblyhands_eccv24"]), False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run the wider sweep")
    ap.add_argument("--overlays", type=int, default=4, help="overlays per dataset")
    args = ap.parse_args()
    cfg = FULL if args.full else SMOKE

    for d in ("results/raw", "results/summary", "figures", "tables", "logs"):
        (RUN_DIR / d).mkdir(parents=True, exist_ok=True)
    log = setup_logging()
    env = write_environment()
    log.info("CAM-EXP-001 start (mode=%s) commit=%s branch=%s",
             "full" if args.full else "smoke", env["git_commit"][:8], env["git_branch"])

    config = {
        "experiment": "CAM-EXP-001_gt_projection_validation",
        "mode": "full" if args.full else "smoke",
        "purpose": ("validate dataset GT camera + GT 3D joint conventions by "
                    "reprojecting onto the dataset 2D annotation"),
        "interhand_annotation_set": f"{INTERHAND_SET}/{INTERHAND_SPLIT}",
        "sampling": cfg,
        "overlays_per_dataset": args.overlays,
        "apply_distortion": True,
        "conventions": {
            "gigahands": "X_cam = R(qvec_wxyz) @ X_world + tvec; OpenCV dist [k1,k2,p1,p2]; metres",
            "interhand26m": "X_cam = camrot @ (X_world - campos); millimetres; no distortion",
            "hanco": "X_cam = M[:3,:3] @ X_world + M[:3,3]; metres; images undistorted",
            "assemblyhands": "X_cam = R @ X_world + t from 3x4 extrinsics; millimetres; rectified images",
        },
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # gzip: the raw per-joint table is the durable artefact and must stay in
    # git, but uncompressed it is ~50 MB. pandas/csv read it transparently.
    raw_path = RUN_DIR / "results" / "raw" / "reprojection_per_joint.csv.gz"
    errs = defaultdict(list)          # dataset -> errors (valid joints)
    errs_inimg = defaultdict(list)    # dataset -> errors (valid AND projected in image)
    groups = defaultdict(list)        # (dataset, key, value) -> errors
    counts = defaultdict(lambda: defaultdict(int))
    overlays_done = defaultdict(int)
    overlay_cams = defaultdict(set)
    overlay_rows = []

    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for name, it, can_overlay in build_sources(cfg):
            log.info("[%s] streaming samples", name)
            n_samples = 0
            for s in it:
                n_samples += 1
                rows = list(per_joint_rows(s))
                writer.writerows(rows)
                counts[name]["joints_total"] += len(rows)
                sample_errs = []
                for r in rows:
                    if r["valid"] and r["error_px"] != "":
                        e = float(r["error_px"])
                        errs[name].append(e)
                        sample_errs.append(e)
                        if r["in_image"] and r["gt_in_image"] == 1:
                            errs_inimg[name].append(e)
                            groups[(name, "camera", r["camera"])].append(e)
                            groups[(name, "sequence", r["sequence"])].append(e)
                            groups[(name, "hand", r["hand"])].append(e)
                    counts[name]["joints_valid"] += int(bool(r["valid"]))
                    counts[name]["joints_in_image"] += int(bool(r["in_image"]))
                    if r["valid"]:
                        counts[name]["valid_in_image"] += int(bool(r["in_image"]))
                        counts[name]["valid_pos_depth"] += int(float(r["depth_mm_or_m"]) > 0)
                    counts[name]["joints_with_gt2d"] += int(r["error_px"] != "")
                    if r["valid"] and r["inside_bbox"] != "":
                        counts[name]["bbox_checked"] += 1
                        counts[name]["bbox_inside"] += int(r["inside_bbox"])
                if (can_overlay and s.image_path is not None
                        and overlays_done[name] < args.overlays
                        and s.camera_name not in overlay_cams[name]):
                    if _overlay(s, name, sample_errs, log, overlay_rows):
                        overlays_done[name] += 1
                        overlay_cams[name].add(s.camera_name)
            counts[name]["samples"] = n_samples
            log.info("[%s] %d samples, %d valid joints, %d with GT 2D",
                     name, n_samples, counts[name]["joints_valid"],
                     counts[name]["joints_with_gt2d"])

    _write_summaries(errs, errs_inimg, groups, counts, overlay_rows, log)
    log.info("CAM-EXP-001 done -> %s", rel(RUN_DIR))


def _overlay(s, name, sample_errs, log, overlay_rows) -> bool:
    try:
        idx = int(s.extra.get("video_frame", 0)) if s.image_path.suffix == ".mp4" else 0
        im = load_frame(s.image_path, idx)
        if im is None:
            return False
        uv, _ = s.camera.project(s.joints3d_world)
        me = float(np.mean(sample_errs)) if sample_errs else float("nan")
        im, uv, gt = _downscale(im, uv, s.joints2d_gt, max_width=800)
        seq = s.sequence.replace("/", "_")
        out = (RUN_DIR / "figures" / name /
               f"{seq}_{s.camera_name}_{s.frame}_{s.hand}.png")
        draw_overlay(
            im, uv, gt, s.valid,
            title_lines=[
                f"{s.dataset} {s.subset} {s.sequence}",
                f"cam={s.camera_name} frame={s.frame} hand={s.hand}",
                (f"mean reproj err={me:.2f} px" if sample_errs
                 else "no GT 2D annotation (projection only)"),
                "green=GT 2D  red=projected GT 3D",
            ],
            out_path=out)
        overlay_rows.append({"dataset": name, "sequence": s.sequence,
                             "camera": s.camera_name, "frame": s.frame, "hand": s.hand,
                             "mean_error_px": "" if not sample_errs else round(me, 4),
                             "figure": rel(out), "source_image": rel(s.image_path)})
        return True
    except Exception as exc:  # overlays are illustrative, never fatal
        log.warning("[%s] overlay failed: %s", name, exc)
        return False


def _downscale(im, uv, gt, max_width: int):
    """Shrink wide frames so the committed figures stay small."""
    import cv2
    h, w = im.shape[:2]
    if w <= max_width:
        return im, uv, gt
    k = max_width / w
    im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return im, uv * k, (None if gt is None else gt * k)


def _write_summaries(errs, errs_inimg, groups, counts, overlay_rows, log):
    ds_rows = []
    for name in counts:
        c = counts[name]
        base = {"dataset": name, "samples": c["samples"],
                "joints_total": c["joints_total"], "joints_valid": c["joints_valid"],
                "joints_in_image": c["joints_in_image"],
                "joints_with_gt2d": c["joints_with_gt2d"],
                "valid_in_image_rate": (round(c["valid_in_image"] / c["joints_valid"], 4)
                                        if c["joints_valid"] else ""),
                "valid_positive_depth_rate": (round(c["valid_pos_depth"] / c["joints_valid"], 4)
                                              if c["joints_valid"] else "")}
        all_s = {f"all_{k}": v for k, v in summarize(errs[name]).items()}
        frac = _fractions(errs_inimg[name])
        in_s = {f"inimg_{k}": v for k, v in summarize(errs_inimg[name]).items()}
        bbox = (round(c["bbox_inside"] / c["bbox_checked"], 4)
                if c.get("bbox_checked") else "")
        ds_rows.append({**base, **all_s, **in_s, **frac, "bbox_inside_rate": bbox,
                        "has_gt_2d": int(c["joints_with_gt2d"] > 0)})

    _csv(RUN_DIR / "results" / "summary" / "dataset_summary.csv", ds_rows)
    _csv(RUN_DIR / "tables" / "reprojection_summary.csv", ds_rows)

    grp_rows = [{"dataset": d, "group_kind": k, "group_value": v, **summarize(e)}
                for (d, k, v), e in sorted(groups.items())]
    _csv(RUN_DIR / "results" / "summary" / "reprojection_by_group.csv", grp_rows)
    if overlay_rows:
        _csv(RUN_DIR / "results" / "summary" / "overlay_index.csv", overlay_rows)
    for r in ds_rows:
        log.info("SUMMARY %s: valid=%s gt2d=%s median_inimg=%s p90=%s "
                 "in_img_rate=%s pos_depth=%s bbox_rate=%s",
                 r["dataset"], r["joints_valid"], r["joints_with_gt2d"],
                 r["inimg_median_px"], r["inimg_p90_px"],
                 r["valid_in_image_rate"], r["valid_positive_depth_rate"],
                 r["bbox_inside_rate"])


def _fractions(errors, thresholds=(1, 2, 5, 10, 20, 50)) -> dict:
    """Descriptive shape of the error distribution.

    These are reported to characterise the distribution, NOT as pass/fail
    thresholds - acceptable error differs per dataset annotation style.
    """
    a = np.asarray([e for e in errors if e != "" and np.isfinite(e)], dtype=float)
    if a.size == 0:
        return {f"inimg_frac_under_{t}px": "" for t in thresholds}
    return {f"inimg_frac_under_{t}px": round(float((a < t).mean()), 4) for t in thresholds}


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
