"""Leave-one-camera-out reconstruction and identity adjudication.

Non-circularity contract (enforced by the call order below and by
``src/tests/test_no_circularity.py``):

  1. ``reconstruct()`` receives only 2D observations and camera calibration.
     It never reads dataset-provided 3D, and it never uses any earlier
     experiment's verdict to pick or reject a camera.
  2. The held-out camera is removed from the observation set *before*
     reconstruction, so its annotation cannot influence the hypothesis it is
     later tested against.
  3. Dataset-provided 3D is read only in ``compare_to_provided()``, which runs
     after a hypothesis is frozen, and its result never feeds back into
     inlier selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.src.datasets.gigahands import HANDS, is_zero_2d
from experiments.src.geometry.triangulation import (max_ray_angle,
                                                    triangulate_hand_fast)

CONF_THRESHOLD = 0.5
MIN_JOINTS_PER_VIEW = 8


@dataclass
class HandHypothesis:
    """A 3D hand reconstructed from 2D observations alone."""

    hand: str
    points: np.ndarray                       # (21, 3), NaN where a joint failed
    ok_joints: np.ndarray                    # (21,) bool
    n_available_cameras: int = 0
    n_inlier_cameras: int = 0
    median_reproj_px: float = float("nan")
    p90_reproj_px: float = float("nan")
    max_ray_angle_deg: float = 0.0
    inlier_cameras: list = field(default_factory=list)
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.ok_joints.any())

    def centroid(self):
        if not self.ok_joints.any():
            return None
        return np.nanmean(self.points[self.ok_joints], axis=0)


def gather_observations(take, hand: str, frame: int, exclude_camera: str | None):
    """Collect usable 2D observations for one hand, minus the held-out camera.

    Usability is judged from the 2D record and the calibration only: the row
    must exist, must not be the all-(0,0) pattern, must have enough confident
    joints, and the camera must be calibrated.
    """
    cams, uv, valid, names = [], [], [], []
    for cam_name in take.cameras_2d(hand):
        if exclude_camera is not None and cam_name == exclude_camera:
            continue
        cam = take.cameras.get(cam_name)
        if cam is None:
            continue
        g = take.joints2d(hand, cam_name, frame)
        if g is None or is_zero_2d(g).all():
            continue
        v = (g[:, 2] >= CONF_THRESHOLD) & ~is_zero_2d(g)
        if int(v.sum()) < MIN_JOINTS_PER_VIEW:
            continue
        cams.append(cam)
        uv.append(g[:, :2])
        valid.append(v)
        names.append(cam_name)
    if not cams:
        return [], np.zeros((0, 21, 2)), np.zeros((0, 21), bool), []
    return cams, np.asarray(uv), np.asarray(valid), names


def reconstruct(take, hand: str, frame: int, exclude_camera: str | None,
                threshold_px: float, min_inlier_cameras: int,
                rng=None) -> HandHypothesis:
    """Robustly triangulate one hand from every camera except the held-out one."""
    cams, uv, valid, names = gather_observations(take, hand, frame, exclude_camera)
    n_avail = len(cams)
    pts = np.full((21, 3), np.nan)
    okj = np.zeros(21, bool)
    if n_avail < 2:
        return HandHypothesis(hand, pts, okj, n_avail, 0,
                              status="insufficient_observations")
    res = triangulate_hand_fast(cams, uv, valid, threshold_px=threshold_px,
                                min_inliers=min_inlier_cameras, rng=rng)
    meds, p90s, angles, inlier_counts = [], [], [], []
    inlier_union = np.zeros(n_avail, bool)
    for j, r in enumerate(res):
        if r.ok:
            pts[j] = r.point
            okj[j] = True
            meds.append(r.median_residual())
            p90s.append(r.p90_residual())
            angles.append(r.max_ray_angle_deg)
            inlier_counts.append(r.n_inliers)
            inlier_union |= r.inliers
    if not okj.any():
        return HandHypothesis(hand, pts, okj, n_avail, 0,
                              status="insufficient_inliers")
    return HandHypothesis(
        hand=hand, points=pts, ok_joints=okj, n_available_cameras=n_avail,
        n_inlier_cameras=int(np.median(inlier_counts)),
        median_reproj_px=float(np.nanmedian(meds)),
        p90_reproj_px=float(np.nanmedian(p90s)),
        max_ray_angle_deg=float(np.nanmedian(angles)),
        inlier_cameras=[names[k] for k in np.where(inlier_union)[0]],
        status="ok")


def project_hypothesis(cam, hyp: HandHypothesis):
    """Project a frozen hypothesis into the held-out camera (pixels)."""
    uv = np.full((21, 2), np.nan)
    if not hyp.ok:
        return uv
    idx = np.where(hyp.ok_joints)[0]
    p, z = cam.project(hyp.points[idx])
    p[z <= 0] = np.nan
    uv[idx] = p
    return uv


def annotation_distance(uv_pred: np.ndarray, g2d: np.ndarray,
                        conf_threshold: float = CONF_THRESHOLD):
    """Median per-joint distance between a projection and a 2D annotation."""
    if g2d is None or uv_pred is None:
        return float("nan"), 0
    m = ((g2d[:, 2] >= conf_threshold) & ~is_zero_2d(g2d)
         & np.isfinite(uv_pred).all(1))
    if int(m.sum()) < 5:
        return float("nan"), int(m.sum())
    return float(np.median(np.linalg.norm(uv_pred[m] - g2d[m, :2], axis=1))), int(m.sum())


def centroid_distance(uv_pred: np.ndarray, g2d: np.ndarray,
                      conf_threshold: float = CONF_THRESHOLD) -> float:
    """Centroid distance - the fair instrument for identity questions.

    Left and right skeletons use mirrored joint ordering, so a true identity
    swap keeps a large per-joint residual even when the annotation sits exactly
    on the other hand. Centroids are invariant to that re-ordering.
    """
    if g2d is None or uv_pred is None:
        return float("nan")
    m = ((g2d[:, 2] >= conf_threshold) & ~is_zero_2d(g2d)
         & np.isfinite(uv_pred).all(1))
    if int(m.sum()) < 5:
        return float("nan")
    return float(np.linalg.norm(uv_pred[m].mean(0) - g2d[m, :2].mean(0)))


def compare_to_provided(hyp: HandHypothesis, provided: np.ndarray | None) -> dict:
    """Compare a FROZEN hypothesis with the dataset-provided 3D.

    Called only after reconstruction; nothing here feeds back into the
    hypothesis. Distances are reported in millimetres (GigaHands 3D is metres).
    """
    out = {"mpjpe_mm": float("nan"), "root_aligned_mpjpe_mm": float("nan"),
           "wrist_mm": float("nan"), "centroid_mm": float("nan"), "n_joints": 0}
    if provided is None or not hyp.ok:
        return out
    m = hyp.ok_joints & np.isfinite(provided).all(1)
    if int(m.sum()) < 5:
        return out
    A, B = hyp.points[m], provided[m]
    d = np.linalg.norm(A - B, axis=1) * 1000.0
    out["mpjpe_mm"] = float(np.mean(d))
    out["n_joints"] = int(m.sum())
    out["centroid_mm"] = float(np.linalg.norm(A.mean(0) - B.mean(0)) * 1000.0)
    # root alignment on joint 0 (wrist) isolates pose error from placement
    if hyp.ok_joints[0] and np.isfinite(provided[0]).all():
        out["wrist_mm"] = float(np.linalg.norm(hyp.points[0] - provided[0]) * 1000.0)
        dr = np.linalg.norm((A - hyp.points[0]) - (B - provided[0]), axis=1) * 1000.0
        out["root_aligned_mpjpe_mm"] = float(np.mean(dr))
    else:
        dr = np.linalg.norm((A - A.mean(0)) - (B - B.mean(0)), axis=1) * 1000.0
        out["root_aligned_mpjpe_mm"] = float(np.mean(dr))
    return out


def adjudicate(take, frame: int, held_out: str, threshold_px: float,
               min_inlier_cameras: int, rng=None) -> dict:
    """Full leave-one-camera-out test for both hands at one (frame, camera).

    Produces the four cross distances E_LL, E_LR, E_RR, E_RL plus the frozen
    hypotheses' comparison against the provided 3D.
    """
    cam = take.cameras.get(held_out)
    row: dict = {"sequence": take.name, "take": take.take, "camera": held_out,
                 "frame": frame}
    if cam is None:
        row["status"] = "no_calibration"
        return row

    # --- reconstruction stage: 2D + calibration only ----------------------
    hyps = {h: reconstruct(take, h, frame, held_out, threshold_px,
                           min_inlier_cameras, rng=rng) for h in HANDS}
    proj = {h: project_hypothesis(cam, hyps[h]) for h in HANDS}
    ann = {h: take.joints2d(h, held_out, frame) for h in HANDS}

    for h in HANDS:
        hy = hyps[h]
        row[f"{h}_tri_status"] = hy.status
        row[f"{h}_n_available_cameras"] = hy.n_available_cameras
        row[f"{h}_n_inlier_cameras"] = hy.n_inlier_cameras
        row[f"{h}_n_ok_joints"] = int(hy.ok_joints.sum())
        row[f"{h}_tri_median_reproj_px"] = _r(hy.median_reproj_px)
        row[f"{h}_tri_p90_reproj_px"] = _r(hy.p90_reproj_px)
        row[f"{h}_max_ray_angle_deg"] = _r(hy.max_ray_angle_deg)
        g = ann[h]
        row[f"{h}_annotation_present"] = int(g is not None)
        row[f"{h}_zero_pattern"] = int(g is not None and bool(is_zero_2d(g).all()))
        row[f"{h}_chosen"] = int(frame in take.chosen[h])

    # --- four cross distances (per-joint and centroid) --------------------
    e_ll, n_ll = annotation_distance(proj["left"], ann["left"])
    e_lr, _ = annotation_distance(proj["right"], ann["left"])
    e_rr, n_rr = annotation_distance(proj["right"], ann["right"])
    e_rl, _ = annotation_distance(proj["left"], ann["right"])
    row.update({"E_LL_px": _r(e_ll), "E_LR_px": _r(e_lr),
                "E_RR_px": _r(e_rr), "E_RL_px": _r(e_rl),
                "n_joints_LL": n_ll, "n_joints_RR": n_rr})
    row.update({"C_LL_px": _r(centroid_distance(proj["left"], ann["left"])),
                "C_LR_px": _r(centroid_distance(proj["right"], ann["left"])),
                "C_RR_px": _r(centroid_distance(proj["right"], ann["right"])),
                "C_RL_px": _r(centroid_distance(proj["left"], ann["right"]))})

    # --- evaluation stage: provided 3D read only now ----------------------
    for h in HANDS:
        other = "right" if h == "left" else "left"
        same = compare_to_provided(hyps[h], take.joints3d(h, frame))
        cross = compare_to_provided(hyps[h], take.joints3d(other, frame))
        for k, v in same.items():
            row[f"{h}_vs_provided_{h}_{k}"] = _r(v)
        row[f"{h}_vs_provided_{other}_mpjpe_mm"] = _r(cross["mpjpe_mm"])
        row[f"{h}_vs_provided_{other}_centroid_mm"] = _r(cross["centroid_mm"])
    row["status"] = "ok"
    row["_hyps"] = hyps
    row["_proj"] = proj
    row["_ann"] = ann
    return row


def _r(v, nd: int = 4):
    try:
        return round(float(v), nd) if np.isfinite(v) else ""
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Scalable variant used for the full-demo pass.
#
# The strict variant above re-runs RANSAC once per held-out camera, which is
# far too slow for every (frame, camera, hand) in the demo. The variant below
# runs RANSAC once per (frame, hand) over all cameras to obtain a consensus
# inlier set S, then, for each held-out camera C, refits the point from
# S \ {C} before projecting into C.
#
# What this does and does not guarantee:
#   * the hypothesis a camera is judged against is always FITTED without that
#     camera's observations;
#   * for a camera that disagrees with the consensus - exactly the cameras this
#     experiment is about - C is not in S at all, so the hypothesis is entirely
#     independent of it;
#   * the only coupling is that C's observation could have participated in
#     RANSAC hypothesis generation for a camera that already agrees, which can
#     confirm a good camera but cannot manufacture a swap.
# `validate_fast_against_strict` measures the residual disagreement between the
# two variants on the smoke set.
# ---------------------------------------------------------------------------

def reconstruct_all(take, hand: str, frame: int, threshold_px: float,
                    min_inlier_cameras: int, rng=None):
    """RANSAC once over every usable camera. Returns (hypothesis, ctx)."""
    cams, uv, valid, names = gather_observations(take, hand, frame, None)
    if len(cams) < 2:
        return (HandHypothesis(hand, np.full((21, 3), np.nan), np.zeros(21, bool),
                               len(cams), 0, status="insufficient_observations"), None)
    res = triangulate_hand_fast(cams, uv, valid, threshold_px=threshold_px,
                                min_inliers=min_inlier_cameras, rng=rng)
    pts = np.full((21, 3), np.nan)
    okj = np.zeros(21, bool)
    meds, angles, counts = [], [], []
    inlier_union = np.zeros(len(cams), bool)
    for j, r in enumerate(res):
        if r.ok:
            pts[j], okj[j] = r.point, True
            meds.append(r.median_residual())
            angles.append(r.max_ray_angle_deg)
            counts.append(r.n_inliers)
            inlier_union |= r.inliers
    status = "ok" if okj.any() else "insufficient_inliers"
    hyp = HandHypothesis(hand, pts, okj, len(cams),
                         int(np.median(counts)) if counts else 0,
                         float(np.nanmedian(meds)) if meds else float("nan"),
                         float("nan"),
                         float(np.nanmedian(angles)) if angles else 0.0,
                         [names[k] for k in np.where(inlier_union)[0]], status)
    # Cache the normalised observations and projection matrices: refit_excluding
    # is called once per held-out camera, and re-running cv2.undistortPoints for
    # every joint of every camera would dominate the full-demo pass.
    from experiments.src.geometry.triangulation import (normalize_observation,
                                                        projection_matrix)
    xn_all = np.full((len(cams), 21, 2), np.nan)
    for i, c in enumerate(cams):
        xn_all[i] = normalize_observation(c, uv[i])
    ctx = {"cams": cams, "uv": uv, "valid": valid, "names": names,
           "per_joint": res, "index": {n: i for i, n in enumerate(names)},
           "xn": xn_all, "P": [projection_matrix(c) for c in cams],
           # geometric conditioning of the consensus camera set, computed once:
           # dropping a single camera changes it negligibly, and recomputing it
           # per joint per held-out camera dominated the full-demo runtime.
           "ray_angle": float(hyp.max_ray_angle_deg)}
    return hyp, ctx


def refit_excluding(ctx, hand: str, exclude_camera: str,
                    min_inlier_cameras: int) -> HandHypothesis:
    """Refit each joint from its consensus inliers minus the held-out camera."""
    from experiments.src.geometry.triangulation import triangulate_dlt
    if ctx is None:
        return HandHypothesis(hand, np.full((21, 3), np.nan), np.zeros(21, bool),
                              0, 0, status="insufficient_observations")
    cams, uv, names = ctx["cams"], ctx["uv"], ctx["names"]
    drop = ctx["index"].get(exclude_camera, -1)
    pts = np.full((21, 3), np.nan)
    okj = np.zeros(21, bool)
    counts = []
    for j, r in enumerate(ctx["per_joint"]):
        idx = [k for k in np.where(r.inliers)[0] if k != drop]
        if len(idx) < max(2, min_inlier_cameras):
            continue
        X = triangulate_dlt([ctx["P"][k] for k in idx],
                            [ctx["xn"][k, j] for k in idx])
        if X is None:
            continue
        pts[j], okj[j] = X, True
        counts.append(len(idx))
    if not okj.any():
        return HandHypothesis(hand, pts, okj, len(cams), 0,
                              status="insufficient_inliers")
    return HandHypothesis(hand, pts, okj, len(cams), int(np.median(counts)),
                          float("nan"), float("nan"),
                          float(ctx.get("ray_angle", 0.0)),
                          [names[k] for k in range(len(names)) if k != drop], "ok")


def adjudicate_fast(take, frame: int, ctx_by_hand: dict, hyp_all: dict,
                    held_out: str, min_inlier_cameras: int) -> dict:
    """Leave-one-camera-out row built from a precomputed consensus."""
    cam = take.cameras.get(held_out)
    row = {"sequence": take.name, "take": take.take, "camera": held_out,
           "frame": frame}
    if cam is None:
        row["status"] = "no_calibration"
        return row
    hyps = {h: refit_excluding(ctx_by_hand.get(h), h, held_out, min_inlier_cameras)
            for h in HANDS}
    proj = {h: project_hypothesis(cam, hyps[h]) for h in HANDS}
    ann = {h: take.joints2d(h, held_out, frame) for h in HANDS}

    for h in HANDS:
        hy, ha = hyps[h], hyp_all.get(h)
        g = ann[h]
        row[f"{h}_tri_status"] = hy.status
        row[f"{h}_n_available_cameras"] = hy.n_available_cameras
        row[f"{h}_n_inlier_cameras"] = hy.n_inlier_cameras
        row[f"{h}_n_ok_joints"] = int(hy.ok_joints.sum())
        row[f"{h}_tri_median_reproj_px"] = _r(ha.median_reproj_px if ha else np.nan)
        row[f"{h}_max_ray_angle_deg"] = _r(hy.max_ray_angle_deg)
        row[f"{h}_annotation_present"] = int(g is not None)
        row[f"{h}_zero_pattern"] = int(g is not None and bool(is_zero_2d(g).all()))
        row[f"{h}_chosen"] = int(frame in take.chosen[h])
        row[f"{h}_camera_in_consensus"] = int(
            bool(ha and held_out in (ha.inlier_cameras or [])))

    e_ll, n_ll = annotation_distance(proj["left"], ann["left"])
    e_lr, _ = annotation_distance(proj["right"], ann["left"])
    e_rr, n_rr = annotation_distance(proj["right"], ann["right"])
    e_rl, _ = annotation_distance(proj["left"], ann["right"])
    row.update({"E_LL_px": _r(e_ll), "E_LR_px": _r(e_lr), "E_RR_px": _r(e_rr),
                "E_RL_px": _r(e_rl), "n_joints_LL": n_ll, "n_joints_RR": n_rr,
                "C_LL_px": _r(centroid_distance(proj["left"], ann["left"])),
                "C_LR_px": _r(centroid_distance(proj["right"], ann["left"])),
                "C_RR_px": _r(centroid_distance(proj["right"], ann["right"])),
                "C_RL_px": _r(centroid_distance(proj["left"], ann["right"]))})

    for h in HANDS:                       # provided 3D read only now
        other = "right" if h == "left" else "left"
        same = compare_to_provided(hyps[h], take.joints3d(h, frame))
        cross = compare_to_provided(hyps[h], take.joints3d(other, frame))
        for k, v in same.items():
            row[f"{h}_vs_provided_{h}_{k}"] = _r(v)
        row[f"{h}_vs_provided_{other}_mpjpe_mm"] = _r(cross["mpjpe_mm"])
        row[f"{h}_vs_provided_{other}_centroid_mm"] = _r(cross["centroid_mm"])
    row["status"] = "ok"
    row["_hyps"] = hyps
    row["_proj"] = proj
    row["_ann"] = ann
    return row
