"""PHASE I — ablations, controls, frame-count curve and mechanism diagnostics."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (RAW, SUM, TAB, fnum, med, read_csv, read_json,  # noqa: E402
                    write_csv, write_json)


def per_unit(rows, method):
    return {(r["sequence"], r["camera"]): fnum(r["err_pct"])
            for r in rows if r["method"] == method
            and np.isfinite(fnum(r["err_pct"]))}


def main():
    res = read_csv(RAW / "focal_unit_results.csv.gz")
    common = [tuple(r.values()) for r in read_csv(TAB / "common_set.csv")]
    common = set((r["sequence"], r["camera"])
                 for r in read_csv(TAB / "common_set.csv"))

    M = {m: per_unit(res, m) for m in
         ("M0_SCENE_ONLY", "M1_SCENE_GENERIC_INDEPENDENT",
          "M2_SCENE_SHARED_NO_GENERIC", "M3_SCENE_SHARED_GENERIC_FULL")}

    abl = []
    for a, b, label in (
            ("M3_SCENE_SHARED_GENERIC_FULL", "M2_SCENE_SHARED_NO_GENERIC",
             "generic prior contribution (M3 - M2)"),
            ("M3_SCENE_SHARED_GENERIC_FULL", "M1_SCENE_GENERIC_INDEPENDENT",
             "shared sequence anatomy contribution (M3 - M1)"),
            ("M3_SCENE_SHARED_GENERIC_FULL", "M0_SCENE_ONLY",
             "PRIMARY (M3 - M0)")):
        ks = [k for k in common if k in M[a] and k in M[b]]
        va, vb = [M[a][k] for k in ks], [M[b][k] for k in ks]
        abl.append({"comparison": label, "n": len(ks),
                    "median_A": med(va), "median_B": med(vb),
                    "median_paired_gain_pp": med([y - x for x, y
                                                  in zip(va, vb)]),
                    "relative_reduction_pct":
                        100.0 * (med(vb) - med(va)) / med(vb)
                        if med(vb) else np.nan})
    write_csv(SUM / "ablation_summary.csv", abl)
    write_csv(TAB / "ablation_table.csv", abl)

    # ---- hand-only diagnostic and mechanism magnitudes
    curves = [r for r in read_csv(RAW / "candidate_score_curves.csv.gz")
              if r.get("status") == "ok"]
    ref = {(r["sequence"], r["camera"]): fnum(r["f_ref"]) for r in res
           if np.isfinite(fnum(r.get("f_ref", "")))}
    hrows = []
    for r in curves:
        if r["method"] != "M3_SCENE_SHARED_GENERIC_FULL":
            continue
        k = (r["sequence"], r["camera"])
        fr = ref.get(k)
        fh = fnum(r["f_hand_only"])
        hrows.append({"sequence": k[0], "camera": k[1],
                      "f_hand_only": fh,
                      "err_pct": (100.0 * abs(fh - fr) / fr
                                  if fr and np.isfinite(fh) else np.nan),
                      "hand_boundary": fnum(r["hand_boundary"]),
                      "hand_score_range": fnum(r["hand_score_range"]),
                      "norm_a": fnum(r["norm_a"]),
                      "norm_delta_L": fnum(r["norm_delta_L"]),
                      "norm_delta_R": fnum(r["norm_delta_R"])})
    write_csv(SUM / "hand_only_summary.csv", hrows)

    diag = {
        "H_ONLY_median_err_pct": med([r["err_pct"] for r in hrows]),
        "H_ONLY_role": "diagnostic, NOT a deployment baseline",
        "hand_boundary_rate": float(np.mean([r["hand_boundary"]
                                             for r in hrows])) if hrows
        else np.nan,
        "median_hand_score_range": med([r["hand_score_range"] for r in hrows]),
        "anatomy_magnitudes": {
            "median_norm_a": med([r["norm_a"] for r in hrows]),
            "median_norm_delta_L": med([r["norm_delta_L"] for r in hrows]),
            "median_norm_delta_R": med([r["norm_delta_R"] for r in hrows]),
            "reading": "||a|| near 0 would mean p_seq collapsed onto the "
                       "generic prior; ||delta|| >> ||a|| would mean the "
                       "shared anatomy is effectively inactive",
        },
    }
    write_json(SUM / "mechanism_diagnostics.json", diag)

    # ---- wrong-bone control, if the control curves exist
    cpath = RAW / "control_wrong_bone_curves.csv.gz"
    ctrl = []
    if cpath.exists():
        cc = [r for r in read_csv(cpath) if r.get("status") == "ok"
              and r["method"] == "M3_SCENE_SHARED_GENERIC_FULL"]
        for r in cc:
            k = (r["sequence"], r["camera"])
            fr = ref.get(k)
            fh = fnum(r["f_hand_only"])
            ctrl.append({"sequence": k[0], "camera": k[1],
                         "control": "C_WRONG_BONE",
                         "f_hand_only": fh,
                         "err_pct": (100.0 * abs(fh - fr) / fr
                                     if fr and np.isfinite(fh) else np.nan),
                         "hand_score_range": fnum(r["hand_score_range"]),
                         "hand_boundary": fnum(r["hand_boundary"])})
        base_err = diag["H_ONLY_median_err_pct"]
        write_csv(SUM / "controls_summary.csv", [
            {"control": "C0_CORRECT_MAPPING",
             "median_hand_only_err_pct": base_err,
             "median_hand_score_range": diag["median_hand_score_range"],
             "boundary_rate": diag["hand_boundary_rate"]},
            {"control": "C_WRONG_BONE",
             "median_hand_only_err_pct": med([r["err_pct"] for r in ctrl]),
             "median_hand_score_range": med([r["hand_score_range"]
                                             for r in ctrl]),
             "boundary_rate": float(np.mean([r["hand_boundary"]
                                             for r in ctrl])) if ctrl
             else np.nan}])
        write_csv(RAW / "control_results.csv.gz", ctrl)
    write_csv(TAB / "controls_table.csv", ctrl or
              [{"control": "C_WRONG_BONE", "status": "not_run"}])

    for a in abl:
        print("%-46s n=%-4s A %.3f  B %.3f  gain %+.3f pp  rel %+.1f%%"
              % (a["comparison"], a["n"], a["median_A"], a["median_B"],
                 a["median_paired_gain_pp"], a["relative_reduction_pct"]))
    print("H_ONLY median err %.3f%%  boundary %.2f"
          % (diag["H_ONLY_median_err_pct"], diag["hand_boundary_rate"]))
    print("anatomy:", {k: round(v, 4) for k, v in
                       diag["anatomy_magnitudes"].items()
                       if isinstance(v, float)})


if __name__ == "__main__":
    main()
