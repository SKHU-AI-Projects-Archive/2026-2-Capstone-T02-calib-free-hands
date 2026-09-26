"""Build the manual hand-QC audit package: sample, render, and wait for a human.

This closes the standing blocker HAND_QC_MANUAL_VALIDATION_PENDING. The sample
is stratified on target-independent properties only - sequence, anatomical side,
existing QC label, physical camera, temporal position and the existing
reconstruction-quality bins. No calibration result, no focal error and no solver
output takes any part in choosing these cases.

Each case gets one audit panel:
  * the dataset-provided 2D observation drawn on the RGB frame, and
  * the OTHER-CAMERA-ONLY reconstruction reprojected into this camera using the
    provided camera parameters.

That reprojection uses provided calibration and is AUDIT VISUALISATION ONLY. It
is never an input to the CAM-006 focal solver.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMERA_MANIFEST, FRAMES64, LOCO_MIN_INLIER_CAMERAS,  # noqa: E402
                    LOCO_THRESHOLD_PX, MANIFESTS, QCDIR, QC_MANIFEST, REPO,
                    SEED, fnum, loco_src_on_path, read_csv, sha256, write_csv,
                    write_json)

N_CASES = 200
PER_SHEET = 12
QC_CSV = MANIFESTS / "cam_exp_006_manual_hand_qc_v1.csv"

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
               (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def quality_bin(v: float) -> str:
    """Bins from the EXISTING QC metric only (triangulation reprojection px)."""
    if not np.isfinite(v):
        return "na"
    if v < 3.0:
        return "q1_low"
    if v < 5.0:
        return "q2_mid"
    return "q3_high"


def temporal_bin(frame: int, n_frames: int) -> str:
    if n_frames <= 1:
        return "t_single"
    r = frame / max(n_frames - 1, 1)
    return "t_early" if r < 1 / 3 else ("t_mid" if r < 2 / 3 else "t_late")


def sample_cases(rng):
    qc = read_csv(QC_MANIFEST)
    cam_ok = {(r["sequence"], r["camera"]) for r in read_csv(CAMERA_MANIFEST)
              if r["usable_for_camera_benchmark"] == "1"}
    frames64 = defaultdict(list)
    for r in read_csv(FRAMES64):
        frames64[(r["sequence"], r["camera"])].append(int(r["frame"]))
    maxframe = defaultdict(int)
    for r in qc:
        maxframe[r["sequence"]] = max(maxframe[r["sequence"]], int(r["frame"]))

    pool = []
    for r in qc:
        key = (r["sequence"], r["camera"])
        if key not in cam_ok:
            continue
        if r["qc_status"] not in ("PASS_STRICT", "PASS_SINGLE_HAND"):
            continue
        if r["chosen"] != "1" or r["zero_pattern"] == "1":
            continue
        if int(r["frame"]) not in set(frames64.get(key, [])):
            continue          # keep the audit on the frozen frame grid
        pool.append({
            "sequence": r["sequence"], "camera": r["camera"],
            "frame": int(r["frame"]), "hand_side": r["hand"],
            "existing_qc_label": r["qc_status"],
            "tri_reproj_px": fnum(r["triangulation_median_reprojection_px"]),
            "n_inlier_cameras": r["n_inlier_cameras"],
            "case_label": r["case"],
        })
    for p in pool:
        p["stratum"] = "|".join([
            p["sequence"], p["hand_side"], p["existing_qc_label"],
            quality_bin(p["tri_reproj_px"]),
            temporal_bin(p["frame"], maxframe[p["sequence"]] + 1)])

    by_stratum = defaultdict(list)
    for p in pool:
        by_stratum[p["stratum"]].append(p)
    strata = sorted(by_stratum)
    for s in strata:
        idx = rng.permutation(len(by_stratum[s]))
        by_stratum[s] = [by_stratum[s][i] for i in idx]

    # round-robin over strata, and spread physical cameras as we go
    chosen, seen_cam = [], defaultdict(int)
    i = 0
    while len(chosen) < N_CASES and any(i < len(by_stratum[s]) for s in strata):
        for s in strata:
            if len(chosen) >= N_CASES or i >= len(by_stratum[s]):
                continue
            cands = by_stratum[s][i:i + 6]
            cands.sort(key=lambda c: seen_cam[c["camera"]])
            pick = cands[0]
            if any((c["sequence"], c["camera"], c["frame"], c["hand_side"])
                   == (pick["sequence"], pick["camera"], pick["frame"],
                       pick["hand_side"]) for c in chosen):
                continue
            chosen.append(pick)
            seen_cam[pick["camera"]] += 1
        i += 1
    return chosen[:N_CASES], len(pool), len(strata)


def draw_hand(img, uv, colour, thickness=2, radius=4):
    for a, b in CONNECTIONS:
        if np.all(np.isfinite(uv[a])) and np.all(np.isfinite(uv[b])):
            cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                     colour, thickness, cv2.LINE_AA)
    for j in range(len(uv)):
        if np.all(np.isfinite(uv[j])):
            cv2.circle(img, tuple(uv[j].astype(int)), radius + (2 if j == 0 else 0),
                       colour, -1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    args = ap.parse_args()

    loco_src_on_path()
    import loco
    from experiments.src.datasets.gigahands import takes

    rng = np.random.default_rng(SEED)
    cases, pool_n, n_strata = sample_cases(rng)
    print(f"pool {pool_n} observations, {n_strata} strata -> {len(cases)} cases")

    take_by_name = {t.name: t for t in takes()}
    QCDIR.mkdir(parents=True, exist_ok=True)
    panels = QCDIR / "panels"
    panels.mkdir(exist_ok=True)

    rows = []
    todo = cases if not args.count else cases[args.start:args.start + args.count]
    by_video = defaultdict(list)
    for k, c in enumerate(cases):
        c["case_id"] = f"C{k:03d}"
    for c in (todo if args.count else cases):
        by_video[(c["sequence"], c["camera"])].append(c)

    for (seq, cam), cs in sorted(by_video.items()):
        take = take_by_name[seq]
        vid = take.video_path(cam)
        camera = take.cameras[cam]
        want = {c["frame"]: c for c in cs}
        cap = cv2.VideoCapture(str(vid))
        i, hi = 0, max(want)
        while i <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                c = want[i]
                hyp = loco.reconstruct(take, c["hand_side"], i, exclude_camera=cam,
                                       threshold_px=LOCO_THRESHOLD_PX,
                                       min_inlier_cameras=LOCO_MIN_INLIER_CAMERAS)
                img = frame.copy()
                g2d = take.joints2d(c["hand_side"], cam, i)
                uv_obs = (np.asarray(g2d, float)[:, :2]
                          if g2d is not None else np.full((21, 2), np.nan))
                draw_hand(img, uv_obs, (0, 255, 255))     # provided 2D: yellow
                uv_proj = np.full((21, 2), np.nan)
                if hyp.ok:
                    uv_proj = loco.project_hypothesis(camera, hyp)
                    draw_hand(img, uv_proj, (255, 0, 255), 2, 3)   # LOCO: magenta
                cv2.putText(img, f"{c['case_id']}  {seq}  {cam}  f{i}  "
                                 f"{c['hand_side'].upper()}  {c['existing_qc_label']}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "yellow = provided 2D observation   "
                                 "magenta = other-camera-only 3D reprojected "
                                 "(AUDIT VISUALISATION ONLY)",
                            (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1, cv2.LINE_AA)
                out = panels / f"{c['case_id']}.png"
                cv2.imwrite(str(out), img)
                d = float(np.nanmedian(np.linalg.norm(uv_obs - uv_proj, axis=1))) \
                    if hyp.ok else float("nan")
                rows.append({
                    "case_id": c["case_id"], "sequence": seq, "camera": cam,
                    "frame": i, "hand_side": c["hand_side"],
                    "existing_qc_label": c["existing_qc_label"],
                    "existing_case_label": c["case_label"],
                    "stratum": c["stratum"],
                    "rgb_source": f"{take.name} / {cam} video",
                    "2d_source": "dataset-provided 2D observation",
                    "reference_3d_source": "OTHER_CAMERA_ONLY_REFERENCE_3D "
                                           "(this camera excluded)",
                    "loco_status": hyp.status, "loco_ok": int(bool(hyp.ok)),
                    "loco_n_inlier_cameras": hyp.n_inlier_cameras,
                    "loco_reproj_to_this_camera_px": round(d, 3)
                    if np.isfinite(d) else "",
                    "audit_image": f"manual_qc/panels/{c['case_id']}.png",
                    # human-filled columns, blank on purpose
                    "RGB_MATCH": "", "2D_ON_CORRECT_HAND": "",
                    "HAND_SIDE_CORRECT": "", "MAJOR_2D_FAILURE": "",
                    "MAJOR_OCCLUSION": "",
                    "LOCO_3D_REPROJECTION_PLAUSIBLE": "",
                    "NOT_REVIEWABLE": "", "NOTES": "",
                })
            i += 1
        cap.release()
        print(f"  {seq}/{cam}: {len(cs)} panels", flush=True)

    rows.sort(key=lambda r: r["case_id"])
    write_csv(QC_CSV, rows)

    # contact sheets
    sheets = QCDIR / "contact_sheets"
    sheets.mkdir(exist_ok=True)
    cell_w, cell_h = 640, 400
    for s0 in range(0, len(rows), PER_SHEET):
        chunk = rows[s0:s0 + PER_SHEET]
        cols, rows_n = 3, int(np.ceil(len(chunk) / 3))
        sheet = np.zeros((rows_n * cell_h, cols * cell_w, 3), np.uint8)
        for k, r in enumerate(chunk):
            img = cv2.imread(str(QCDIR / "panels" / f"{r['case_id']}.png"))
            if img is None:
                continue
            h, w = img.shape[:2]
            sc = min(cell_w / w, cell_h / h)
            im = cv2.resize(img, (int(w * sc), int(h * sc)))
            y0, x0 = (k // cols) * cell_h, (k % cols) * cell_w
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.imwrite(str(sheets / f"sheet_{s0 // PER_SHEET:02d}.png"), sheet)

    write_json(QCDIR / "_sampling_meta.json", {
        "n_cases": len(rows), "pool_size": pool_n, "n_strata": n_strata,
        "seed": SEED,
        "strata_definition": "sequence | anatomical side | existing QC label | "
                             "existing triangulation-reprojection bin | temporal "
                             "position bin",
        "selection_rule": "round-robin over strata, preferring the least-used "
                          "physical camera among the next six candidates; no "
                          "duplicate (sequence, camera, frame, hand)",
        "target_independence": "no calibration result, focal error or solver "
                               "output took any part in the selection",
        "audit_reprojection": "the magenta overlay uses provided camera "
                              "parameters and is AUDIT VISUALISATION ONLY; it is "
                              "never an input to the CAM-006 focal solver",
        "manifest_sha256": sha256(QC_CSV),
    })
    print(f"\nwrote {QC_CSV}")
    print(f"panels: {panels}")
    print(f"contact sheets: {sheets}  ({(len(rows) + PER_SHEET - 1) // PER_SHEET})")


if __name__ == "__main__":
    main()
