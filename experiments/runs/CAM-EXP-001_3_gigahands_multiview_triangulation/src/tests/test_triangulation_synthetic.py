"""Synthetic regression tests for robust multi-view triangulation.

These run on generated data with a known answer, so they validate the
implementation before it is trusted on GigaHands. No dataset is read.

Run:  PYTHONPATH=<repo root> python src/tests/test_triangulation_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent))

from experiments.src.geometry import Camera  # noqa: E402
from experiments.src.geometry.triangulation import (  # noqa: E402
    normalize_observation, robust_triangulate, triangulate_hand)

FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_rig(n_cams: int = 8, radius: float = 1.2, dist=(0.0, 0.0, 0.0, 0.0),
             spread_deg: float = 360.0) -> list:
    """Cameras on an arc looking at the origin."""
    cams = []
    for i in range(n_cams):
        a = np.radians(spread_deg * i / max(n_cams, 1))
        c = np.array([radius * np.cos(a), radius * np.sin(a), 0.35])
        fwd = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        R = np.stack([right, true_up, fwd])       # world -> camera
        t = -R @ c
        K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
        cams.append(Camera(K=K, R=R, t=t, dist=np.array(dist), width=1280,
                           height=720, name=f"cam{i}"))
    return cams


def make_hand(rng, n_joints: int = 21) -> np.ndarray:
    return rng.normal(0, 0.05, size=(n_joints, 3))


def observe(cams, X, noise_px=0.0, rng=None):
    uv = []
    for c in cams:
        p, _ = c.project(np.asarray(X).reshape(-1, 3))
        if noise_px and rng is not None:
            p = p + rng.normal(0, noise_px, size=p.shape)
        uv.append(p)
    return np.asarray(uv)


# ---------------------------------------------------------------------------
def test_undistort_inverts_project():
    """normalize_observation must invert Camera.project including distortion."""
    cams = make_rig(3, dist=(-0.39, 0.13, 0.003, -0.0007))
    rng = np.random.default_rng(0)
    X = make_hand(rng)
    for cam in cams:
        uv, _ = cam.project(X)
        xn = normalize_observation(cam, uv)
        Xc = (cam.R @ X.T).T + cam.t
        expect = Xc[:, :2] / Xc[:, 2:3]
        err = float(np.max(np.abs(xn - expect)))
        check(f"undistort inverts project [{cam.name}]", err < 1e-4, f"max err={err:.2e}")


def test_noiseless_recovery():
    cams = make_rig(8)
    rng = np.random.default_rng(1)
    X = make_hand(rng)
    uv = observe(cams, X)
    res = triangulate_hand(cams, uv.transpose(0, 1, 2), threshold_px=2.0)
    errs = [np.linalg.norm(r.point - X[j]) for j, r in enumerate(res) if r.ok]
    check("noiseless: all joints reconstructed", len(errs) == len(X), f"{len(errs)}/{len(X)}")
    check("noiseless: error < 1e-6 m", max(errs) < 1e-6, f"max={max(errs):.2e} m")


def test_noise_recovery():
    cams = make_rig(8)
    rng = np.random.default_rng(2)
    X = make_hand(rng)
    uv = observe(cams, X, noise_px=1.5, rng=rng)
    res = triangulate_hand(cams, uv, threshold_px=8.0)
    errs = np.array([np.linalg.norm(r.point - X[j]) for j, r in enumerate(res) if r.ok])
    check("1.5px noise: all joints ok", len(errs) == len(X), f"{len(errs)}/{len(X)}")
    check("1.5px noise: median 3D error < 3 mm", np.median(errs) < 0.003,
          f"median={np.median(errs) * 1000:.2f} mm")


def test_with_distortion():
    cams = make_rig(8, dist=(-0.39, 0.13, 0.003, -0.0007))
    rng = np.random.default_rng(3)
    X = make_hand(rng)
    uv = observe(cams, X, noise_px=1.0, rng=rng)
    res = triangulate_hand(cams, uv, threshold_px=8.0)
    errs = np.array([np.linalg.norm(r.point - X[j]) for j, r in enumerate(res) if r.ok])
    check("distortion: all joints ok", len(errs) == len(X), f"{len(errs)}/{len(X)}")
    check("distortion: median 3D error < 3 mm", np.median(errs) < 0.003,
          f"median={np.median(errs) * 1000:.2f} mm")


def test_single_swapped_camera():
    """One camera reports the other hand; it must be rejected as an outlier."""
    cams = make_rig(8)
    rng = np.random.default_rng(4)
    XL = make_hand(rng)
    XR = XL + np.array([0.30, 0.0, 0.0])
    uv = observe(cams, XL, noise_px=1.0, rng=rng)
    uv_wrong = observe([cams[3]], XR)[0]
    uv[3] = uv_wrong                                   # camera 3 annotates the wrong hand
    res = triangulate_hand(cams, uv, threshold_px=8.0)
    errs = np.array([np.linalg.norm(r.point - XL[j]) for j, r in enumerate(res) if r.ok])
    excluded = np.mean([not r.inliers[3] for r in res if r.ok])
    check("swapped camera: reconstruction still correct", np.median(errs) < 0.003,
          f"median={np.median(errs) * 1000:.2f} mm")
    check("swapped camera: outlier rejected", excluded > 0.9,
          f"excluded in {excluded:.0%} of joints")


def test_twenty_percent_outliers():
    cams = make_rig(10)
    rng = np.random.default_rng(5)
    X = make_hand(rng)
    uv = observe(cams, X, noise_px=1.0, rng=rng)
    for k in (2, 7):                                   # 20 % gross outliers
        uv[k] += rng.normal(0, 200, size=uv[k].shape)
    res = triangulate_hand(cams, uv, threshold_px=8.0)
    errs = np.array([np.linalg.norm(r.point - X[j]) for j, r in enumerate(res) if r.ok])
    check("20% outliers: median 3D error < 3 mm", np.median(errs) < 0.003,
          f"median={np.median(errs) * 1000:.2f} mm")


def test_missing_cameras():
    cams = make_rig(8)
    rng = np.random.default_rng(6)
    X = make_hand(rng)
    uv = observe(cams, X, noise_px=1.0, rng=rng)
    valid = np.ones((len(cams), len(X)), bool)
    valid[:5] = False                                  # only 3 cameras left
    res = triangulate_hand(cams, uv, valid, threshold_px=8.0, min_inliers=3)
    errs = np.array([np.linalg.norm(r.point - X[j]) for j, r in enumerate(res) if r.ok])
    check("3 cameras: still reconstructs", len(errs) > 0, f"{len(errs)} joints")
    check("3 cameras: median 3D error < 5 mm", np.median(errs) < 0.005,
          f"median={np.median(errs) * 1000:.2f} mm")


def test_insufficient_observations():
    cams = make_rig(8)
    rng = np.random.default_rng(7)
    X = make_hand(rng)
    uv = observe(cams, X)
    valid = np.zeros((len(cams), len(X)), bool)
    valid[0] = True                                    # a single view
    res = triangulate_hand(cams, uv, valid, threshold_px=8.0)
    check("single view reports insufficient_observations",
          all(r.status == "insufficient_observations" for r in res))


def test_low_baseline_flagged():
    """Nearly co-located cameras must be reported with a small ray angle."""
    cams = make_rig(4, spread_deg=4.0)
    rng = np.random.default_rng(8)
    X = make_hand(rng)
    uv = observe(cams, X, noise_px=1.0, rng=rng)
    res = triangulate_hand(cams, uv, threshold_px=8.0)
    angles = [r.max_ray_angle_deg for r in res if r.ok]
    wide = make_rig(4, spread_deg=270.0)
    uvw = observe(wide, X, noise_px=1.0, rng=rng)
    resw = triangulate_hand(wide, uvw, threshold_px=8.0)
    anglesw = [r.max_ray_angle_deg for r in resw if r.ok]
    check("low baseline yields small ray angle",
          angles and np.median(angles) < 15.0, f"median={np.median(angles):.1f} deg")
    check("wide baseline yields large ray angle",
          anglesw and np.median(anglesw) > 60.0, f"median={np.median(anglesw):.1f} deg")


def test_no_provided_3d_used():
    """The module must depend only on cameras and 2D observations.

    Checked structurally rather than by text search: docstrings legitimately
    discuss dataset-provided 3D, so only executable code and imports count.
    """
    import ast
    import inspect
    from experiments.src.geometry import triangulation as T

    tree = ast.parse(inspect.getsource(T))
    for node in ast.walk(tree):                      # drop docstrings
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:]
    code = ast.unparse(tree).lower()
    banned = ["keypoints_3d", "mano", "chosen_frames", "datasets"]
    hits = [b for b in banned if b in code]
    check("no dataset-3D reference in executable code", not hits, f"found {hits}")

    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    bad_imports = [m for m in mods if "dataset" in m or "gigahands" in m]
    check("imports nothing from dataset loaders", not bad_imports, f"{sorted(mods)}")


def main() -> int:
    print("CAM-EXP-001.3 synthetic triangulation tests")
    for fn in (test_undistort_inverts_project, test_noiseless_recovery,
               test_noise_recovery, test_with_distortion,
               test_single_swapped_camera, test_twenty_percent_outliers,
               test_missing_cameras, test_insufficient_observations,
               test_low_baseline_flagged, test_no_provided_3d_used):
        print(f"\n{fn.__name__}:")
        fn()
    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
