"""Case adjudication and QC status assignment.

Two separate decisions are made here and kept distinct on purpose:

* ``adjudicate_case`` asks *what happened* at one (frame, held-out camera):
  is a per-camera 2D identity swap supported, is the provided 3D suspect, is
  the 2D simply bad, or is the evidence insufficient?
* ``qc_status`` asks *may this observation be used* in later experiments. It is
  deliberately conservative: anything that is not clearly clean becomes REVIEW
  rather than PASS, and only clearly unusable records are EXCLUDEd.

Nothing here is allowed to upgrade a record on the strength of an earlier
experiment's opinion; the inputs are the measurements from this run only.
"""
from __future__ import annotations

import numpy as np

# case labels
CASE_SWAP = "CONFIRMED_2D_HAND_IDENTITY_SWAP"
CASE_PROVIDED3D = "SUSPECT_PROVIDED_3D_IDENTITY_ERROR"
CASE_BAD2D = "BAD_2D_GEOMETRY"
CASE_INSUFFICIENT = "UNRESOLVED_INSUFFICIENT_GEOMETRY"
CASE_GOOD = "MULTIVIEW_CONFIRMED_GOOD"
CASE_NO_ANNOT = "NO_USABLE_2D_ANNOTATION"
CASE_NOT_CHOSEN = "NOT_CHOSEN_FRAME"

# QC statuses
PASS_STRICT = "PASS_STRICT"
PASS_SINGLE_HAND = "PASS_SINGLE_HAND"
REVIEW = "REVIEW"
EXCLUDE = "EXCLUDE"


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def reconstruction_is_trustworthy(row: dict, hand: str, thr: dict) -> tuple:
    """Is the hypothesis for this hand good enough to judge anything with?"""
    if row.get(f"{hand}_tri_status") != "ok":
        return False, f"triangulation status={row.get(f'{hand}_tri_status')}"
    n_in = _f(row.get(f"{hand}_n_inlier_cameras"))
    ang = _f(row.get(f"{hand}_max_ray_angle_deg"))
    n_ok = _f(row.get(f"{hand}_n_ok_joints"))
    if not np.isfinite(n_in) or n_in < thr["qc_min_inlier_cameras"]:
        return False, f"only {n_in:.0f} inlier cameras"
    if not np.isfinite(ang) or ang < thr["min_ray_angle_deg"]:
        return False, f"ray angle {ang:.0f} deg below {thr['min_ray_angle_deg']}"
    if not np.isfinite(n_ok) or n_ok < 15:
        return False, f"only {n_ok:.0f} joints reconstructed"
    return True, ""


def adjudicate_case(row: dict, hand: str, thr: dict) -> tuple:
    """Return (case_label, reason) for one hand at one held-out camera."""
    other = "right" if hand == "left" else "left"
    if not row.get(f"{hand}_chosen"):
        return CASE_NOT_CHOSEN, "frame is not in chosen_frames for this hand"
    if row.get(f"{hand}_zero_pattern") == 1 or not row.get(f"{hand}_annotation_present"):
        return CASE_NO_ANNOT, "held-out 2D is absent or the all-(0,0) pattern"

    ok_self, why_self = reconstruction_is_trustworthy(row, hand, thr)
    ok_other, _ = reconstruction_is_trustworthy(row, other, thr)
    if not ok_self:
        return CASE_INSUFFICIENT, why_self

    same = _f(row.get("E_LL_px" if hand == "left" else "E_RR_px"))
    cross = _f(row.get("E_LR_px" if hand == "left" else "E_RL_px"))
    c_same = _f(row.get("C_LL_px" if hand == "left" else "C_RR_px"))
    c_cross = _f(row.get("C_LR_px" if hand == "left" else "C_RL_px"))
    mp_same = _f(row.get(f"{hand}_vs_provided_{hand}_mpjpe_mm"))
    mp_cross = _f(row.get(f"{hand}_vs_provided_{other}_mpjpe_mm"))

    # Does our independent reconstruction match the provided 3D of the SAME
    # hand, or of the OTHER hand? This is what separates a 2D problem from a
    # provided-3D identity problem.
    provided_agrees = np.isfinite(mp_same) and mp_same <= thr["provided3d_mpjpe_mm"]
    provided_swapped = (np.isfinite(mp_cross) and np.isfinite(mp_same)
                        and mp_cross < mp_same / thr["identity_margin_factor"]
                        and mp_cross <= thr["provided3d_mpjpe_mm"])
    if provided_swapped:
        return (CASE_PROVIDED3D,
                f"reconstruction matches provided {other} 3D ({mp_cross:.1f} mm) far "
                f"better than provided {hand} 3D ({mp_same:.1f} mm)")

    if not np.isfinite(same):
        return CASE_INSUFFICIENT, "held-out annotation not comparable"

    # Identity swap: the held-out annotation sits on the other hand's
    # reconstruction, by a clear margin, while the reconstruction itself is
    # confirmed against the provided 3D of its own hand.
    swap_by_centroid = (np.isfinite(c_same) and np.isfinite(c_cross)
                        and c_cross * thr["identity_margin_factor"] < c_same
                        and (c_same - c_cross) >= thr["identity_min_gap_px"])
    if swap_by_centroid and ok_other:
        if provided_agrees:
            return (CASE_SWAP,
                    f"held-out {hand} annotation is {c_cross:.0f} px from the {other} "
                    f"reconstruction but {c_same:.0f} px from its own; reconstruction "
                    f"agrees with provided {hand} 3D ({mp_same:.1f} mm)")
        return (CASE_SWAP + "_UNVERIFIED_3D",
                f"annotation sits on the {other} reconstruction "
                f"({c_cross:.0f} vs {c_same:.0f} px) but provided 3D agreement is "
                f"{mp_same:.1f} mm")

    if same <= thr["same_hand_reprojection_px"]:
        return CASE_GOOD, f"held-out annotation agrees with the reconstruction ({same:.1f} px)"

    return (CASE_BAD2D,
            f"held-out annotation matches neither reconstruction "
            f"(same {same:.0f} px, other {cross:.0f} px)")


def qc_status(row: dict, hand: str, case: str, thr: dict) -> tuple:
    """Map a case to a usability decision for one observation."""
    if case == CASE_NOT_CHOSEN:
        return EXCLUDE, "no valid 3D pose for this hand on this frame"
    if case == CASE_NO_ANNOT:
        return EXCLUDE, "2D annotation absent or all-(0,0)"
    if case.startswith(CASE_SWAP):
        return EXCLUDE, "per-camera 2D hand identity swap"
    if case == CASE_BAD2D:
        return EXCLUDE, "2D annotation matches no reconstruction"
    if case == CASE_PROVIDED3D:
        return REVIEW, "provided 3D identity is suspect; needs manual confirmation"
    if case == CASE_INSUFFICIENT:
        return REVIEW, "insufficient multi-view geometry to decide"
    if case == CASE_GOOD:
        same = _f(row.get("E_LL_px" if hand == "left" else "E_RR_px"))
        mp = _f(row.get(f"{hand}_vs_provided_{hand}_mpjpe_mm"))
        if not np.isfinite(mp) or mp > thr["provided3d_mpjpe_mm"]:
            return REVIEW, f"2D agrees but provided-3D deviation is {mp:.1f} mm"
        if not np.isfinite(same) or same > thr["same_hand_reprojection_px"]:
            return REVIEW, "2D agreement outside the clean-control range"
        return PASS_STRICT, "multi-view consistent in 2D and 3D"
    return REVIEW, f"unhandled case {case}"


def identity_margin_px(row: dict, hand: str) -> float:
    """How much better the correct hand explains the annotation (px).

    Positive means the same-hand reconstruction is the better explanation.
    """
    c_same = _f(row.get("C_LL_px" if hand == "left" else "C_RR_px"))
    c_cross = _f(row.get("C_LR_px" if hand == "left" else "C_RL_px"))
    if not (np.isfinite(c_same) and np.isfinite(c_cross)):
        return float("nan")
    return float(c_cross - c_same)
