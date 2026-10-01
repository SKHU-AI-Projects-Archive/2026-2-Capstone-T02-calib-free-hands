"""Assemble the full 200-case manual-QC manifest and contact sheets.

The panels were rendered in chunks; this recomputes the deterministic case list,
recomputes the (cheap, video-free) LOCO metrics for every case, and writes one
manifest plus the complete set of contact sheets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_manual_qc as B  # noqa: E402
from common import (LOCO_MIN_INLIER_CAMERAS, LOCO_THRESHOLD_PX, QCDIR, SEED,  # noqa: E402
                    loco_src_on_path, sha256, write_csv, write_json)


def main() -> None:
    loco_src_on_path()
    import loco
    from experiments.src.datasets.gigahands import takes

    rng = np.random.default_rng(SEED)
    cases, pool_n, n_strata = B.sample_cases(rng)
    for k, c in enumerate(cases):
        c["case_id"] = f"C{k:03d}"
    take_by_name = {t.name: t for t in takes()}

    rows = []
    for c in cases:
        take = take_by_name[c["sequence"]]
        cam = c["camera"]
        hyp = loco.reconstruct(take, c["hand_side"], c["frame"], exclude_camera=cam,
                               threshold_px=LOCO_THRESHOLD_PX,
                               min_inlier_cameras=LOCO_MIN_INLIER_CAMERAS)
        g2d = take.joints2d(c["hand_side"], cam, c["frame"])
        uv_obs = (np.asarray(g2d, float)[:, :2] if g2d is not None
                  else np.full((21, 2), np.nan))
        d = float("nan")
        if hyp.ok:
            uv_proj = loco.project_hypothesis(take.cameras[cam], hyp)
            d = float(np.nanmedian(np.linalg.norm(uv_obs - uv_proj, axis=1)))
        panel = QCDIR / "panels" / f"{c['case_id']}.png"
        rows.append({
            "case_id": c["case_id"], "sequence": c["sequence"], "camera": cam,
            "frame": c["frame"], "hand_side": c["hand_side"],
            "existing_qc_label": c["existing_qc_label"],
            "existing_case_label": c["case_label"], "stratum": c["stratum"],
            "rgb_source": f"{c['sequence']} / {cam} video",
            "2d_source": "dataset-provided 2D observation",
            "reference_3d_source": "OTHER_CAMERA_ONLY_REFERENCE_3D "
                                   "(this camera excluded)",
            "loco_status": hyp.status, "loco_ok": int(bool(hyp.ok)),
            "loco_n_inlier_cameras": hyp.n_inlier_cameras,
            "loco_reproj_to_this_camera_px": round(d, 3) if np.isfinite(d) else "",
            "audit_image": f"manual_qc/panels/{c['case_id']}.png",
            "panel_exists": int(panel.exists()),
            "RGB_MATCH": "", "2D_ON_CORRECT_HAND": "", "HAND_SIDE_CORRECT": "",
            "MAJOR_2D_FAILURE": "", "MAJOR_OCCLUSION": "",
            "LOCO_3D_REPROJECTION_PLAUSIBLE": "", "NOT_REVIEWABLE": "",
            "NOTES": "",
        })
    write_csv(B.QC_CSV, rows)

    sheets = QCDIR / "contact_sheets"
    for old in sheets.glob("sheet_*.png"):
        old.unlink()
    sheets.mkdir(exist_ok=True)
    cell_w, cell_h = 640, 400
    n_sheets = 0
    for s0 in range(0, len(rows), B.PER_SHEET):
        chunk = rows[s0:s0 + B.PER_SHEET]
        cols = 3
        rows_n = int(np.ceil(len(chunk) / cols))
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
        cv2.imwrite(str(sheets / f"sheet_{s0 // B.PER_SHEET:02d}.png"), sheet)
        n_sheets += 1

    missing = [r["case_id"] for r in rows if not r["panel_exists"]]
    write_json(QCDIR / "_sampling_meta.json", {
        "n_cases": len(rows), "pool_size": pool_n, "n_strata": n_strata,
        "seed": SEED, "n_contact_sheets": n_sheets,
        "panels_missing": missing,
        "strata_definition": "sequence | anatomical side | existing QC label | "
                             "existing triangulation-reprojection bin | temporal "
                             "position bin",
        "selection_rule": "round-robin over strata, preferring the least-used "
                          "physical camera among the next six candidates",
        "target_independence": "no calibration result, focal error or solver "
                               "output took any part in the selection",
        "audit_reprojection": "the magenta overlay uses provided camera "
                              "parameters and is AUDIT VISUALISATION ONLY; never "
                              "an input to the CAM-006 focal solver",
        "manifest_sha256": sha256(B.QC_CSV),
    })
    print(f"{len(rows)} cases, {n_sheets} contact sheets, "
          f"{len(missing)} panels missing")
    ok = sum(r["loco_ok"] for r in rows)
    d = np.array([float(r["loco_reproj_to_this_camera_px"]) for r in rows
                  if r["loco_reproj_to_this_camera_px"] != ""])
    print(f"LOCO reconstruction ok: {ok}/{len(rows)}; reprojection to the "
          f"held-out camera median {np.median(d):.2f} px, p90 "
          f"{np.percentile(d, 90):.2f} px")


if __name__ == "__main__":
    main()
