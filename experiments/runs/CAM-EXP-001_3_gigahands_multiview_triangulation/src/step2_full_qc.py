"""Step 2 - leave-one-camera-out QC over every chosen frame of the demo.

Scale strategy: RANSAC runs once per (sequence, frame, hand) over all usable
cameras, and each held-out camera is then judged against a hypothesis refitted
from the consensus inliers *excluding that camera*. See the note in ``loco.py``
for exactly what that does and does not guarantee; ``validate_fast_vs_strict``
measures the disagreement against the strict per-camera variant.

The pass is checkpointed per sequence so it can resume after an interruption.
No RGB is decoded here - this is numerical only.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict

import numpy as np

from common import RUN_DIR, TAKES, read_csv, rel, write_csv
import loco
import verdicts

log = logging.getLogger("cam-exp-001.3")

CKPT_DIR = RUN_DIR / "results" / "raw" / "_checkpoints"


def frames_for(take, stride: int = 1) -> list:
    return take.union_sorted[::stride]


CHUNK = 50          # frames per partial checkpoint


def run_take(take, thr: dict, stride: int, rng, resume: bool = True) -> list:
    """All (frame, camera, hand) QC rows for one take.

    Written out in chunks so an interrupted pass resumes from the last chunk
    rather than restarting the sequence.
    """
    rows = []
    t_px = thr["triangulation_inlier_px"]
    min_rec = thr["min_inliers_for_reconstruction"]
    cams = sorted(set(take.cameras_2d("left")) | set(take.cameras_2d("right")))
    frames = frames_for(take, stride)
    t_start = time.time()
    part_dir = CKPT_DIR / take.name
    part_dir.mkdir(parents=True, exist_ok=True)
    chunks = [frames[i:i + CHUNK] for i in range(0, len(frames), CHUNK)]
    done = 0
    for ci, chunk in enumerate(chunks):
        part = part_dir / f"part{ci:04d}.csv.gz"
        if resume and part.exists():
            rows.extend(read_csv(part))
            done += len(chunk)
            continue
        chunk_rows = _run_frames(take, chunk, thr, rng)
        write_csv(part, chunk_rows)
        rows.extend(chunk_rows)
        done += len(chunk)
        el = time.time() - t_start
        log.info("  [%s] %d/%d frames (%.1f s)", take.name, done, len(frames), el)
    return rows


def _run_frames(take, frames, thr: dict, rng) -> list:
    rows = []
    t_px = thr["triangulation_inlier_px"]
    min_rec = thr["min_inliers_for_reconstruction"]
    cams = sorted(set(take.cameras_2d("left")) | set(take.cameras_2d("right")))
    for frame in frames:
        ctx_by_hand, hyp_all = {}, {}
        for h in ("left", "right"):
            if frame in take.chosen[h]:
                hyp, ctx = loco.reconstruct_all(take, h, frame, t_px, min_rec, rng=rng)
            else:
                hyp, ctx = loco.HandHypothesis(
                    h, np.full((21, 3), np.nan), np.zeros(21, bool), 0, 0,
                    status="not_chosen"), None
            hyp_all[h], ctx_by_hand[h] = hyp, ctx
        for cam_name in cams:
            row = loco.adjudicate_fast(take, frame, ctx_by_hand, hyp_all,
                                       cam_name, min_rec)
            if row.get("status") != "ok":
                continue
            for h in ("left", "right"):
                case, why = verdicts.adjudicate_case(row, h, thr)
                st, why_st = verdicts.qc_status(row, h, case, thr)
                m = verdicts.identity_margin_px(row, h)
                row[f"{h}_case"] = case
                row[f"{h}_case_reason"] = why
                row[f"{h}_qc_status"] = st
                row[f"{h}_qc_reason"] = why_st
                row[f"{h}_identity_margin_px"] = round(m, 2) if np.isfinite(m) else ""
            rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    return rows


def main(thr: dict, stride: int = 1, resume: bool = True) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    all_rows = []
    for take in TAKES:
        ck = CKPT_DIR / f"{take.name}.csv.gz"
        if resume and ck.exists():
            rows = read_csv(ck)
            log.info("[%s] resumed %d rows from checkpoint", take.name, len(rows))
        else:
            t0 = time.time()
            rows = run_take(take, thr, stride, rng, resume=resume)
            write_csv(ck, rows)
            log.info("[%s] %d rows in %.1f s", take.name, len(rows), time.time() - t0)
        all_rows.extend(rows)

    write_csv(RUN_DIR / "results" / "raw" / "leave_one_camera_out.csv.gz", all_rows)
    _flatten_and_summarise(all_rows)
    return {"n_rows": len(all_rows)}


def _flatten_and_summarise(rows: list) -> None:
    """One record per (sequence, camera, frame, hand), plus summary tables."""
    flat = []
    for r in rows:
        for h in ("left", "right"):
            flat.append({
                "sequence": r["sequence"], "take": r["take"], "camera": r["camera"],
                "frame": r["frame"], "hand": h,
                "chosen": r.get(f"{h}_chosen", ""),
                "annotation_present": r.get(f"{h}_annotation_present", ""),
                "zero_pattern": r.get(f"{h}_zero_pattern", ""),
                "triangulation_success": int(r.get(f"{h}_tri_status") == "ok"),
                "n_available_cameras": r.get(f"{h}_n_available_cameras", ""),
                "n_inlier_cameras": r.get(f"{h}_n_inlier_cameras", ""),
                "n_ok_joints": r.get(f"{h}_n_ok_joints", ""),
                "max_ray_angle_deg": r.get(f"{h}_max_ray_angle_deg", ""),
                "triangulation_median_reprojection_px": r.get(f"{h}_tri_median_reproj_px", ""),
                "camera_in_consensus": r.get(f"{h}_camera_in_consensus", ""),
                "provided3d_mpjpe_mm": r.get(f"{h}_vs_provided_{h}_mpjpe_mm", ""),
                "provided3d_root_aligned_mpjpe_mm":
                    r.get(f"{h}_vs_provided_{h}_root_aligned_mpjpe_mm", ""),
                "provided3d_wrist_mm": r.get(f"{h}_vs_provided_{h}_wrist_mm", ""),
                "provided3d_other_hand_mpjpe_mm": r.get(
                    f"{h}_vs_provided_{'right' if h == 'left' else 'left'}_mpjpe_mm", ""),
                "heldout_same_hand_error_px": r.get("E_LL_px" if h == "left" else "E_RR_px", ""),
                "heldout_other_hand_error_px": r.get("E_LR_px" if h == "left" else "E_RL_px", ""),
                "heldout_same_hand_centroid_px": r.get("C_LL_px" if h == "left" else "C_RR_px", ""),
                "heldout_other_hand_centroid_px": r.get("C_LR_px" if h == "left" else "C_RL_px", ""),
                "identity_margin_px": r.get(f"{h}_identity_margin_px", ""),
                "case": r.get(f"{h}_case", ""),
                "case_reason": r.get(f"{h}_case_reason", ""),
                "qc_status": r.get(f"{h}_qc_status", ""),
                "reason": r.get(f"{h}_qc_reason", ""),
            })
    write_csv(RUN_DIR / "results" / "raw" / "full_demo_qc.csv.gz", flat)

    def tally(rows_, key, extra=None):
        c = Counter(r[key] for r in rows_)
        return [{key: k, "n": v, "share": round(v / max(len(rows_), 1), 4)}
                for k, v in c.most_common()]

    write_csv(RUN_DIR / "results" / "summary" / "identity_verdict_summary.csv",
              tally(flat, "case"))
    write_csv(RUN_DIR / "results" / "summary" / "qc_status_summary.csv",
              tally(flat, "qc_status"))

    for key, name in (("sequence", "per_sequence_summary"), ("camera", "per_camera_summary")):
        g = defaultdict(Counter)
        for r in flat:
            g[r[key]][r["qc_status"]] += 1
            g[r[key]]["_total"] += 1
            if r["case"].startswith(verdicts.CASE_SWAP):
                g[r[key]]["_swap"] += 1
        out = []
        for k, c in sorted(g.items()):
            tot = c["_total"]
            out.append({key: k, "n": tot,
                        "pass_strict": c[verdicts.PASS_STRICT],
                        "pass_single_hand": c[verdicts.PASS_SINGLE_HAND],
                        "review": c[verdicts.REVIEW], "exclude": c[verdicts.EXCLUDE],
                        "pass_strict_rate": round(c[verdicts.PASS_STRICT] / tot, 4),
                        "confirmed_swap": c["_swap"],
                        "swap_rate": round(c["_swap"] / tot, 4)})
        write_csv(RUN_DIR / "results" / "summary" / f"{name}.csv", out)

    tri = np.array([float(r["triangulation_median_reprojection_px"]) for r in flat
                    if r["triangulation_median_reprojection_px"] not in ("", None)])
    mp = np.array([float(r["provided3d_mpjpe_mm"]) for r in flat
                   if r["provided3d_mpjpe_mm"] not in ("", None)])
    mpx = np.array([float(r["provided3d_other_hand_mpjpe_mm"]) for r in flat
                    if r["provided3d_other_hand_mpjpe_mm"] not in ("", None)])

    def st(a, unit):
        if a.size == 0:
            return {}
        return {"n": int(a.size), "unit": unit,
                "median": round(float(np.median(a)), 4),
                "p90": round(float(np.percentile(a, 90)), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
                "p99": round(float(np.percentile(a, 99)), 4),
                "max": round(float(a.max()), 4)}

    write_csv(RUN_DIR / "results" / "summary" / "triangulation_summary.csv",
              [{"metric": "triangulation_median_reprojection_px", **st(tri, "px")},
               {"metric": "reconstruction_vs_provided_same_hand_mpjpe", **st(mp, "mm")},
               {"metric": "reconstruction_vs_provided_other_hand_mpjpe", **st(mpx, "mm")}])

    write_csv(RUN_DIR / "results" / "raw" / "provided3d_comparison.csv.gz",
              [{k: r[k] for k in ("sequence", "camera", "frame", "hand",
                                  "provided3d_mpjpe_mm",
                                  "provided3d_root_aligned_mpjpe_mm",
                                  "provided3d_wrist_mm",
                                  "provided3d_other_hand_mpjpe_mm",
                                  "triangulation_success", "n_inlier_cameras")}
               for r in flat])
    write_csv(RUN_DIR / "results" / "raw" / "identity_adjudication.csv.gz",
              [{k: r[k] for k in ("sequence", "camera", "frame", "hand", "case",
                                  "case_reason", "identity_margin_px",
                                  "heldout_same_hand_error_px",
                                  "heldout_other_hand_error_px",
                                  "heldout_same_hand_centroid_px",
                                  "heldout_other_hand_centroid_px",
                                  "provided3d_mpjpe_mm",
                                  "provided3d_other_hand_mpjpe_mm")}
               for r in flat])
    unresolved = [r for r in flat if r["case"] in
                  (verdicts.CASE_INSUFFICIENT, verdicts.CASE_BAD2D)
                  or r["case"].endswith("_UNVERIFIED_3D")]
    write_csv(RUN_DIR / "results" / "raw" / "unresolved_cases.csv.gz", unresolved[:5000])
    write_csv(RUN_DIR / "tables" / "unresolved_cases.csv",
              tally(unresolved, "case") if unresolved else [])
    write_csv(RUN_DIR / "tables" / "confirmed_swap_cases.csv.gz",
              [r for r in flat if r["case"].startswith(verdicts.CASE_SWAP)][:5000])
    write_csv(RUN_DIR / "tables" / "provided3d_suspect_cases.csv.gz",
              [r for r in flat if r["case"] == verdicts.CASE_PROVIDED3D][:5000])


def validate_fast_vs_strict(thr: dict, n: int = 25, rng=None) -> dict:
    """Quantify the disagreement between the fast and strict LOCO variants."""
    rng = rng or np.random.default_rng(1)
    take = TAKES[0]
    frames = take.intersect_sorted[::max(1, len(take.intersect_sorted) // n)][:n]
    cams = take.cameras_2d("left")
    t_px = thr["triangulation_inlier_px"]
    min_rec = thr["min_inliers_for_reconstruction"]
    diffs, agree = [], []
    for f in frames:
        cam_name = cams[int(rng.integers(len(cams)))]
        ctx_by_hand, hyp_all = {}, {}
        for h in ("left", "right"):
            hyp, ctx = loco.reconstruct_all(take, h, f, t_px, min_rec, rng=rng)
            hyp_all[h], ctx_by_hand[h] = hyp, ctx
        fast = loco.adjudicate_fast(take, f, ctx_by_hand, hyp_all, cam_name, min_rec)
        strict = loco.adjudicate(take, f, cam_name, t_px, min_rec, rng=rng)
        for key in ("E_LL_px", "E_RR_px"):
            a, b = fast.get(key), strict.get(key)
            if a not in ("", None) and b not in ("", None):
                diffs.append(abs(float(a) - float(b)))
        for h in ("left", "right"):
            ca, _ = verdicts.adjudicate_case(fast, h, thr)
            cb, _ = verdicts.adjudicate_case(strict, h, thr)
            agree.append(ca == cb)
    d = np.asarray(diffs)
    out = {"n_compared_frames": len(frames), "n_metric_pairs": int(d.size),
           "median_abs_diff_px": round(float(np.median(d)), 4) if d.size else "",
           "p95_abs_diff_px": round(float(np.percentile(d, 95)), 4) if d.size else "",
           "max_abs_diff_px": round(float(d.max()), 4) if d.size else "",
           "case_agreement_rate": round(float(np.mean(agree)), 4) if agree else ""}
    write_csv(RUN_DIR / "results" / "summary" / "fast_vs_strict_validation.csv", [out])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t = json.load(open(RUN_DIR / "results" / "summary" / "_thresholds.json"))["thresholds"]
    print(json.dumps(main(t), indent=2))
