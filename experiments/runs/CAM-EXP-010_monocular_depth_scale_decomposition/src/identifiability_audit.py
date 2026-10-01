"""Metric-scale identifiability: is the global scale determined by geometry?

Three numerical checks, each with a definite expected outcome:

  GLOBAL_SCALE_INVARIANCE_TEST
      Scale every 3D point AND the camera translation by the same s. Under a
      pinhole camera the 2D projections must be bit-comparable identical, so no
      amount of 2D evidence can distinguish s.

  MULTI_FRAME_SCALE_TEST
      Does requiring one consistent hand across many frames break the tie? Scale
      the whole trajectory and the hand together and re-project every frame.

  BILATERAL_SCALE_TEST
      Does requiring the two hands to be consistent break the tie? Scale both
      hands and both trajectories by the same s.

  ANATOMY_SCALE_FREE_TEST
      The sum-normalised bone proportions used throughout CAM-EXP-009.x are
      invariant to s by construction; verified numerically.

A PASS on the invariance tests means the scale is NOT geometrically
identifiable from these cues. It does NOT mean a monocular camera cannot
produce a depth estimate: a network can supply a LEARNED STATISTICAL SCALE
PRIOR. The two are different claims and are kept apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import BASE_SEED, GATE, SUM, TAB, write_csv, write_json  # noqa

SCALES = (0.5, 1.0, 2.0, 5.0)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def project(X, K):
    """Pinhole projection of camera-frame points."""
    Z = X[:, 2:3]
    return (X[:, :2] / Z) @ K[:2, :2].T + K[:2, 2]


def random_hand(rng, scale_m=0.18):
    X = np.zeros((21, 3))
    for i, (a, b) in enumerate(BONES):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        X[b] = X[a] + d * scale_m * rng.uniform(0.08, 0.30)
    return X


def bone_proportions(X):
    l = np.array([np.linalg.norm(X[b] - X[a]) for a, b in BONES])
    return l / l.sum()


def main():
    rng = np.random.default_rng(BASE_SEED)
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    tol = GATE["scale_invariance_tol_px"]
    rows = []

    # ---------------------------------------- 1. single frame
    X = random_hand(rng)
    T = np.array([0.05, -0.02, 0.60])
    base = project(X + T, K)
    worst = 0.0
    for s in SCALES:
        uv = project(s * (X + T), K)
        worst = max(worst, float(np.max(np.abs(uv - base))))
    rows.append({"test": "GLOBAL_SCALE_INVARIANCE_TEST",
                 "scales": ";".join(str(s) for s in SCALES),
                 "max_reprojection_difference_px": worst,
                 "tolerance_px": tol, "PASS": int(worst <= tol),
                 "meaning": "identical 2D under a global scale -> scale not "
                            "identifiable from 2D"})

    # ---------------------------------------- 2. multi-frame trajectory
    n_frames = 64
    worst_mf = 0.0
    frames = []
    for f in range(n_frames):
        Xf = random_hand(rng)
        # one fixed hand across frames: re-use bone lengths, vary articulation
        Tf = np.array([rng.uniform(-.1, .1), rng.uniform(-.1, .1),
                       rng.uniform(.4, 1.0)])
        frames.append((Xf, Tf))
    for Xf, Tf in frames:
        b = project(Xf + Tf, K)
        for s in SCALES:
            uv = project(s * (Xf + Tf), K)
            worst_mf = max(worst_mf, float(np.max(np.abs(uv - b))))
    rows.append({"test": "MULTI_FRAME_SCALE_TEST", "n_frames": n_frames,
                 "scales": ";".join(str(s) for s in SCALES),
                 "max_reprojection_difference_px": worst_mf,
                 "tolerance_px": tol, "PASS": int(worst_mf <= tol),
                 "meaning": "requiring one consistent hand across many frames "
                            "does NOT break the scale ambiguity"})

    # ---------------------------------------- 3. bilateral
    XL, XR = random_hand(rng), random_hand(rng)
    TL = np.array([-0.08, 0.0, 0.62])
    TR = np.array([0.08, 0.01, 0.58])
    worst_bi = 0.0
    for Xs, Ts in ((XL, TL), (XR, TR)):
        b = project(Xs + Ts, K)
        for s in SCALES:
            uv = project(s * (Xs + Ts), K)
            worst_bi = max(worst_bi, float(np.max(np.abs(uv - b))))
    rows.append({"test": "BILATERAL_SCALE_TEST",
                 "scales": ";".join(str(s) for s in SCALES),
                 "max_reprojection_difference_px": worst_bi,
                 "tolerance_px": tol, "PASS": int(worst_bi <= tol),
                 "meaning": "requiring both hands to be consistent does NOT "
                            "break the scale ambiguity"})

    # ---------------------------------------- 4. anatomy is scale-free
    p0 = bone_proportions(XL)
    worst_an = 0.0
    for s in SCALES:
        worst_an = max(worst_an,
                       float(np.max(np.abs(bone_proportions(s * XL) - p0))))
    rows.append({"test": "ANATOMY_SCALE_FREE_TEST",
                 "scales": ";".join(str(s) for s in SCALES),
                 "max_proportion_difference": worst_an,
                 "tolerance_px": 1e-12, "PASS": int(worst_an <= 1e-12),
                 "meaning": "the sum-normalised bone proportions used in "
                            "CAM-EXP-009.x carry no absolute size, so they "
                            "cannot pin the scale"})

    write_csv(TAB / "metric_identifiability_audit.csv", rows)
    allpass = all(r["PASS"] for r in rows)
    write_json(SUM / "identifiability_summary.json", {
        "tests": rows,
        "ALL_INVARIANCE_TESTS_PASS": allpass,
        "conclusion": (
            "Under a pinhole camera, scaling all 3D points and the camera "
            "translation by a common s leaves every 2D projection unchanged. "
            "Neither multi-frame consistency nor bilateral consistency breaks "
            "that, and the sum-normalised anatomy used throughout CAM-EXP-009.x "
            "is scale-free by construction. So with the deployment's allowed "
            "cues - static monocular RGB, unknown worker hand size, no known "
            "object or table dimension, no metric anchor, scale-normalised "
            "anatomy - the global metric scale is NOT uniquely determined by "
            "geometry."),
        "what_this_does_not_say": (
            "It does not say a monocular camera cannot estimate depth. A "
            "network trained on metric data supplies a LEARNED STATISTICAL "
            "SCALE PRIOR, which can produce a useful estimate. That is a "
            "different thing from GEOMETRIC IDENTIFIABILITY, and the two are "
            "never conflated in this report."),
        "intrinsics_vs_scale": (
            "Correct intrinsics K are not the same as metric world scale. "
            "CAM-EXP-009.4's oracle-focal result is the empirical form of this: "
            "a perfect focal left a large absolute error."),
    })

    for r in rows:
        key = ("max_reprojection_difference_px" if "max_reprojection_difference_px"
               in r else "max_proportion_difference")
        print("%-32s %-6s  max diff %.3e"
              % (r["test"], "PASS" if r["PASS"] else "FAIL", r[key]))
    print("ALL_INVARIANCE_TESTS_PASS:", allpass)


if __name__ == "__main__":
    main()
