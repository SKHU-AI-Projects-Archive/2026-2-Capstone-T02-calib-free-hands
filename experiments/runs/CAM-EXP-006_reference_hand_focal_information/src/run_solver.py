"""Run the frozen focal-profile solver over every view, frame count and control.

Conditions:
  MAIN              the reference 3D as built
  JOINT_PERMUTATION the 21 reference joints permuted before the solve
  WRONG_POSE        the reference 3D of the same hand in a temporally distant
                    frame of the same view
  PLANARIZED        the reference 3D projected onto its own best-fit plane

Every condition uses the identical solver, the identical frames and the
identical 2D. GT focal is not read anywhere in this file.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import focal_profile_solver as S  # noqa: E402
from common import CACHE, MANIFESTS, RAW, SEED, read_csv, write_csv  # noqa: E402

FRAMES = MANIFESTS / "cam_exp_006_reference_hand_frames_v1.csv.gz"
NPZ = CACHE / "reference_3d_v1.npz"
OUT = RAW / "focal_profile_solutions.csv.gz"
COUNTS = (1, 2, 4, 8, 16)
CONDITIONS = ("MAIN", "JOINT_PERMUTATION", "WRONG_POSE", "PLANARIZED")


def planarize(xyz):
    c = xyz.mean(0)
    u, s, vt = np.linalg.svd(xyz - c, full_matrices=False)
    n = vt[2]
    d = (xyz - c) @ n
    return xyz - np.outer(d, n)


def main() -> None:
    z = np.load(NPZ)
    rows = read_csv(FRAMES)

    by_view = defaultdict(list)
    for r in rows:
        by_view[(r["sequence"], r["camera"])].append(
            (int(r["grid_pos"]), int(r["frame"])))

    # image size per view, from the frozen prediction table (metadata only)
    size = {}
    for r in read_csv(RAW.parent.parent.parent /
                      "CAM-EXP-004_1_pre_cam005_robustness_audit" / "results" /
                      "raw" / "extended_frame_predictions.csv.gz"):
        size[(r["sequence"], r["camera"])] = (int(r["image_width"]),
                                              int(r["image_height"]))

    rng = np.random.default_rng(SEED)
    out = []
    for vi, ((seq, cam), fr) in enumerate(sorted(by_view.items()), 1):
        fr.sort()
        order = [f for _, f in fr]
        w, h = size[(seq, cam)]

        # gather usable hands per frame once
        per_frame = {}
        for f in order:
            hands = []
            for hand in ("left", "right"):
                k = f"{seq}|{cam}|{f}|{hand}"
                use = z[k + "|use"]
                if use.sum() < 12:
                    continue
                hands.append((hand, z[k + "|xyz"][use].astype(float),
                              z[k + "|uv"][use].astype(float), use))
            per_frame[f] = hands

        # each (condition, frame, hand) profile is computed exactly once and
        # reused across the nested frame counts
        grid = S.gamma_grid()
        prof = defaultdict(list)
        for f in order:
            for hand, xyz, uv, use in per_frame[f]:
                for cond in CONDITIONS:
                    U = uv
                    if cond == "MAIN":
                        X = xyz
                    elif cond == "JOINT_PERMUTATION":
                        X = xyz[rng.permutation(len(xyz))]
                    elif cond == "PLANARIZED":
                        X = planarize(xyz)
                    else:   # WRONG_POSE: same hand, temporally distant frame
                        alt = order[(order.index(f) + len(order) // 2)
                                    % len(order)]
                        k2 = f"{seq}|{cam}|{alt}|{hand}"
                        use2 = z[k2 + "|use"]
                        both = use & use2
                        if both.sum() < 12:
                            continue
                        X = z[k2 + "|xyz"][both].astype(float)
                        U = z[f"{seq}|{cam}|{f}|{hand}|uv"][both].astype(float)
                    if len(X) != len(U):
                        continue
                    prof[(cond, f)].append(S.profile_one(X, U, w, h, grid))

        for cond in CONDITIONS:
            for n in COUNTS:
                ps = [p for f in order[:n] for p in prof.get((cond, f), [])]
                if not ps:
                    r = {"gamma": np.nan, "status": "NO_CASES",
                         "flat_profile": 0, "boundary_solution": 0,
                         "curvature": np.nan, "depth_of_minimum": np.nan,
                         "n_hands": 0}
                else:
                    r = S.pick(S.combine(ps), grid)
                    r["n_hands"] = len(ps)
                    r["focal_px"] = r["gamma"] * max(w, h)
                out.append({
                    "sequence": seq, "camera": cam, "condition": cond,
                    "n_frames": n, "image_width": w, "image_height": h,
                    "n_hand_observations": r["n_hands"],
                    "gamma_hat": r["gamma"],
                    "focal_hat_px": r.get("focal_px", np.nan),
                    "min_objective_px": float(np.exp(
                        np.nanmin(S.combine(ps)))) if r["n_hands"] else np.nan,
                    "depth_of_minimum_log": r["depth_of_minimum"],
                    "curvature": r["curvature"],
                    "flat_profile": r["flat_profile"],
                    "boundary_solution": r["boundary_solution"],
                    "solver_status": r["status"],
                })
        if vi % 10 == 0:
            print(f"  {vi}/{len(by_view)} views", flush=True)

    write_csv(OUT, out)
    print(f"wrote {len(out)} solutions to {OUT.name}")


if __name__ == "__main__":
    main()
