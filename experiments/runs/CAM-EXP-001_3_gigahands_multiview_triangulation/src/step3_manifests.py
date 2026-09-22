"""Step 3 - build the QC manifests that later experiments will read.

The dataset itself stays immutable: nothing is deleted or moved. These
manifests become the source of truth for which observations may be used, so a
loader can select ``qc_status == PASS_STRICT`` without knowing any of this
experiment's internals.

Three manifests are produced because they answer different questions:

* ``gigahands_demo_qc_v1``           - per observation (sequence, camera, frame, hand)
* ``gigahands_demo_bimanual_clean_v1`` - frames where BOTH hands are strict passes
* ``gigahands_demo_camera_benchmark_v1`` - RGB + calibration usability, which does
  not depend on whether the hand annotation is good
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone

from common import MANIFESTS, RUN_DIR, TAKES, read_csv, rel, write_csv
import verdicts

log = logging.getLogger("cam-exp-001.3")

QC_CSV = RUN_DIR / "results" / "raw" / "full_demo_qc.csv.gz"
TODAY = datetime.now(timezone.utc).date().isoformat()


def build(thr: dict, git_commit: str = "") -> dict:
    rows = read_csv(QC_CSV)
    take_by_name = {t.name: t for t in TAKES}

    # --- per-observation manifest -----------------------------------------
    by_frame = defaultdict(dict)
    for r in rows:
        by_frame[(r["sequence"], r["camera"], r["frame"])][r["hand"]] = r

    manifest = []
    for r in rows:
        partner = by_frame[(r["sequence"], r["camera"], r["frame"])].get(
            "right" if r["hand"] == "left" else "left")
        status = r["qc_status"]
        reason = r["reason"]
        # A strict pass whose partner hand is not a strict pass is still usable
        # on its own, but must not be treated as a bimanual observation.
        if status == verdicts.PASS_STRICT and (
                partner is None or partner["qc_status"] != verdicts.PASS_STRICT):
            status = verdicts.PASS_SINGLE_HAND
            reason = "this hand is clean; the other hand of the frame is not"
        take = take_by_name.get(r["sequence"])
        manifest.append({
            "sequence": r["sequence"], "take": r["take"], "camera": r["camera"],
            "frame": r["frame"], "hand": r["hand"],
            "chosen": r["chosen"], "annotation_present": r["annotation_present"],
            "zero_pattern": r["zero_pattern"],
            "triangulation_success": r["triangulation_success"],
            "n_available_cameras": r["n_available_cameras"],
            "n_inlier_cameras": r["n_inlier_cameras"],
            "triangulation_median_reprojection_px": r["triangulation_median_reprojection_px"],
            "triangulation_p90_reprojection_px": "",
            "max_ray_angle_deg": r["max_ray_angle_deg"],
            "provided3d_mpjpe_mm": r["provided3d_mpjpe_mm"],
            "provided3d_other_hand_mpjpe_mm": r["provided3d_other_hand_mpjpe_mm"],
            "heldout_same_hand_error_px": r["heldout_same_hand_error_px"],
            "heldout_other_hand_error_px": r["heldout_other_hand_error_px"],
            "identity_margin_px": r["identity_margin_px"],
            "case": r["case"],
            "qc_status": status,
            "reason": reason,
            "video_available": int(bool(take and take.video_path(r["camera"]))),
        })
    out_csv = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
    write_csv(out_csv, manifest)

    counts = Counter(m["qc_status"] for m in manifest)
    cases = Counter(m["case"] for m in manifest)

    # --- bimanual clean subset --------------------------------------------
    bim_by_frame = defaultdict(dict)
    for m in manifest:
        bim_by_frame[(m["sequence"], m["camera"], m["frame"])][m["hand"]] = m
    bimanual = []
    for (seq, cam, frame), hands in sorted(bim_by_frame.items()):
        L, R = hands.get("left"), hands.get("right")
        if not L or not R:
            continue
        if L["qc_status"] != verdicts.PASS_STRICT or R["qc_status"] != verdicts.PASS_STRICT:
            continue
        bimanual.append({
            "sequence": seq, "take": L["take"], "camera": cam, "frame": frame,
            "left_provided3d_mpjpe_mm": L["provided3d_mpjpe_mm"],
            "right_provided3d_mpjpe_mm": R["provided3d_mpjpe_mm"],
            "left_heldout_error_px": L["heldout_same_hand_error_px"],
            "right_heldout_error_px": R["heldout_same_hand_error_px"],
            "left_n_inlier_cameras": L["n_inlier_cameras"],
            "right_n_inlier_cameras": R["n_inlier_cameras"],
            "video_available": L["video_available"],
        })
    bim_csv = MANIFESTS / "gigahands_demo_bimanual_clean_v1.csv.gz"
    write_csv(bim_csv, bimanual)

    # --- camera benchmark usability ---------------------------------------
    # Deliberately independent of hand QC: a sample with a bad hand annotation
    # can still be a perfectly good RGB + calibration sample for camera work.
    cam_rows = []
    for take in TAKES:
        for cam_name in sorted(set(take.cameras_2d("left")) | set(take.cameras_2d("right"))):
            cam = take.cameras.get(cam_name)
            vid = take.video_path(cam_name)
            frames = [m for m in manifest
                      if m["sequence"] == take.name and m["camera"] == cam_name]
            strict = sum(1 for m in frames if m["qc_status"] == verdicts.PASS_STRICT)
            cam_rows.append({
                "sequence": take.name, "take": take.take, "camera": cam_name,
                "gt_camera_available": int(cam is not None),
                "rgb_video_available": int(vid is not None),
                "video_segment_matches_annotation": int(vid is not None),
                "n_chosen_frames": len(take.union_sorted),
                "n_observations": len(frames),
                "n_hand_pass_strict": strict,
                "hand_pass_strict_rate": round(strict / len(frames), 4) if frames else "",
                "usable_for_camera_benchmark": int(cam is not None and vid is not None),
                "note": ("hand annotation quality does not gate camera usability; "
                         "RGB + GT camera can be used even where the hand QC fails"),
            })
    cam_csv = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"
    write_csv(cam_csv, cam_rows)

    summary = {
        "manifest_version": "v1",
        "created": TODAY,
        "experiment": "CAM-EXP-001.3_gigahands_multiview_triangulation",
        "git_commit": git_commit,
        "source_dataset": "GigaHands demo_all (immutable; nothing deleted)",
        "n_observations": len(manifest),
        "qc_status_counts": dict(counts),
        "qc_status_shares": {k: round(v / len(manifest), 4) for k, v in counts.items()},
        "case_counts": dict(cases),
        "bimanual_clean_frames": len(bimanual),
        "camera_benchmark_rows": len(cam_rows),
        "n_cameras_usable_for_camera_benchmark":
            sum(r["usable_for_camera_benchmark"] for r in cam_rows),
        "thresholds": thr,
        "files": {"per_observation": rel(out_csv), "bimanual_clean": rel(bim_csv),
                  "camera_benchmark": rel(cam_csv)},
        "usage": ("filter qc_status == PASS_STRICT for core experiments; "
                  "PASS_SINGLE_HAND is usable for single-hand work; REVIEW must not "
                  "be used in core experiments; EXCLUDE is unusable. The dataset "
                  "files themselves are never modified."),
        "status_definitions": {
            "PASS_STRICT": "chosen, annotated, robustly triangulated with sufficient "
                           "geometry, identity confirmed, 2D and 3D consistent",
            "PASS_SINGLE_HAND": "this hand is strict-clean but the other hand of the "
                                "same frame is not",
            "REVIEW": "not clearly wrong but not clearly clean; excluded from core work",
            "EXCLUDE": "clearly unusable (no annotation, not chosen, confirmed identity "
                       "swap, or 2D matching no reconstruction)",
        },
    }
    (MANIFESTS / "gigahands_demo_qc_v1.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {rel(MANIFESTS / 'gigahands_demo_qc_v1.json')}")

    write_csv(RUN_DIR / "tables" / "qc_thresholds.csv",
              [{"threshold": k, "value": v} for k, v in thr.items()])
    return summary


def before_after(summary: dict) -> None:
    """Compare the CAM-EXP-001.2 census with this run's QC outcome."""
    from common import CAM_EXP_0012
    prev = read_csv(CAM_EXP_0012 / "results" / "raw" / "gigahands_demo_qc_census.csv.gz")
    prev_counts = Counter(r["status"] for r in prev)
    rows = [{"stage": "CAM-EXP-001.2 census (stride 20, heuristic)",
             "category": k, "n": v} for k, v in prev_counts.most_common()]
    rows += [{"stage": "CAM-EXP-001.3 QC (all chosen frames, triangulation)",
              "category": k, "n": v}
             for k, v in sorted(summary["qc_status_counts"].items())]
    rows += [{"stage": "CAM-EXP-001.3 adjudicated case",
              "category": k, "n": v}
             for k, v in sorted(summary["case_counts"].items())]
    rows.append({"stage": "note", "category":
                 "001.2 sampled every 20th frame and judged by heuristic; 001.3 covers "
                 "every chosen frame and judges by independent triangulation, so counts "
                 "are not directly comparable", "n": ""})
    write_csv(RUN_DIR / "tables" / "qc_before_after.csv", rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t = json.load(open(RUN_DIR / "results" / "summary" / "_thresholds.json"))["thresholds"]
    s = build(t)
    before_after(s)
    print(json.dumps({k: s[k] for k in ("n_observations", "qc_status_counts",
                                        "bimanual_clean_frames")}, indent=2))
