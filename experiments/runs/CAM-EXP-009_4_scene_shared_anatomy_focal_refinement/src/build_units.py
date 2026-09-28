"""PHASE A — eligible units, scene frames, hand frames, folds, spec freeze.

Nothing here reads the reference focal: scene predictions come through
`read_scene_predictions()`, which drops every `gt_*` column at load.

Eligibility is frozen before any focal result. No unit is ever removed for
having a large focal error, an odd hand score, a boundary solution or a method
that underperforms — those are results.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BONES, FRAME_COUNTS, FUSION_LAMBDA_GRID, GATE,  # noqa: E402
                    LAMBDA_GENERIC_GRID, LAMBDA_SIDE_GRID, MANIFESTS, METHODS,
                    MIN_FRAMES_PER_SIDE, N_HAND_FRAMES, N_OUTER_FOLDS,
                    N_SCENE_FRAMES, PARTICIPANT_OF, Q_MAX, Q_MIN, Q_N, QC,
                    SCENE_MODEL_KEY, SCENE_SRC, SUM, TAB, hash_bucket,
                    read_csv, read_scene_predictions, sha256, stable_seed,
                    write_csv, write_json)

QC_PASS = ("PASS_STRICT", "PASS_SINGLE_HAND")


def pick(frames, n):
    """Deterministic, target-independent: evenly spaced by frame index."""
    f = sorted(set(int(x) for x in frames))
    if len(f) <= n:
        return f
    idx = np.linspace(0, len(f) - 1, n)
    return [f[int(round(i))] for i in idx]


def main():
    # ---------------------------------------------------- scene availability
    scene = read_scene_predictions()
    assert not any(k.startswith("gt_") for k in scene[0]), \
        "reference focal leaked into the scene reader"
    scene_ok = [r for r in scene if r.get("success") in ("1", "True", "true")]
    by_unit_scene = collections.defaultdict(list)
    for r in scene_ok:
        by_unit_scene[(r["sequence"], r["camera"])].append(int(r["frame"]))

    # ---------------------------------------------------- hand availability
    qc = read_csv(QC)
    hand = [r for r in qc if r["qc_status"] in QC_PASS
            and r["triangulation_success"] == "1"
            and r.get("video_available") in ("1", "True", "true")]
    by_unit_hand = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in hand:
        by_unit_hand[(r["sequence"], r["camera"])][r["hand"]].append(
            int(r["frame"]))

    # ---------------------------------------------------- eligible units
    units, scene_rows, hand_rows, split_rows = [], [], [], []
    for key in sorted(set(by_unit_scene) | set(by_unit_hand)):
        seq, cam = key
        sf_all = sorted(by_unit_scene.get(key, []))
        hl = sorted(by_unit_hand.get(key, {}).get("left", []))
        hr = sorted(by_unit_hand.get(key, {}).get("right", []))
        reasons = []
        if len(sf_all) < 8:
            reasons.append("insufficient_scene_frames")
        if len(hl) < MIN_FRAMES_PER_SIDE:
            reasons.append("insufficient_left_frames")
        if len(hr) < MIN_FRAMES_PER_SIDE:
            reasons.append("insufficient_right_frames")
        eligible = not reasons
        fold = hash_bucket(cam, N_OUTER_FOLDS)          # split by CAMERA
        units.append({
            "sequence": seq, "camera": cam,
            "participant": PARTICIPANT_OF[seq],
            "outer_fold": fold,
            "n_scene_frames_available": len(sf_all),
            "n_left_available": len(hl), "n_right_available": len(hr),
            "eligible": int(eligible),
            "ineligible_reason": ";".join(reasons),
        })
        if not eligible:
            continue
        for n in FRAME_COUNTS:
            for f in pick(sf_all, n):
                scene_rows.append({"sequence": seq, "camera": cam,
                                   "n_set": n, "frame": f})
        for side, fr in (("left", hl), ("right", hr)):
            sel = pick(fr, N_HAND_FRAMES)
            for f in sel:
                # deterministic FIT/EVAL split, independent of fit quality
                half = hash_bucket("%s|%s|%s|%d" % (seq, cam, side, f), 2)
                hand_rows.append({"sequence": seq, "camera": cam,
                                  "hand": side, "frame": f})
                split_rows.append({"sequence": seq, "camera": cam,
                                   "hand": side, "frame": f,
                                   "split": "FIT" if half == 0 else "EVAL"})

    elig = [u for u in units if u["eligible"]]
    u_p = MANIFESTS / "cam_exp_0094_units_v1.csv.gz"
    s_p = MANIFESTS / "cam_exp_0094_scene_frames_v1.csv.gz"
    h_p = MANIFESTS / "cam_exp_0094_hand_frames_v1.csv.gz"
    sp_p = MANIFESTS / "cam_exp_0094_hand_split_v1.csv.gz"
    f_p = MANIFESTS / "cam_exp_0094_outer_camera_folds_v1.csv"
    write_csv(u_p, units)
    write_csv(s_p, scene_rows)
    write_csv(h_p, hand_rows)
    write_csv(sp_p, split_rows)

    cams = sorted(set(u["camera"] for u in elig))
    write_csv(f_p, [{"camera": c, "outer_fold": hash_bucket(c, N_OUTER_FOLDS),
                     "n_units": sum(1 for u in elig if u["camera"] == c)}
                    for c in cams])

    # ---------------------------------------------------- grids + spec
    lg = MANIFESTS / "cam_exp_0094_fusion_lambda_grid_v1.json"
    fg = MANIFESTS / "cam_exp_0094_focal_search_grid_v1.json"
    write_json(lg, {"fusion_lambda_grid": list(FUSION_LAMBDA_GRID),
                    "lambda_0_is_scene_only_sanity": True,
                    "selected_on": "TRAIN physical cameras only; the TEST "
                                   "reference focal is never used",
                    "frozen_before_real_focal_results": True})
    write_json(fg, {"q_min": Q_MIN, "q_max": Q_MAX, "q_n": Q_N,
                    "anchored_on": "the unit's own scene-only focal estimate",
                    "boundary_is_reported_not_clipped": True,
                    "frozen_before_real_focal_results": True})

    ctrl = MANIFESTS / "cam_exp_0094_controls_spec_v1.json"
    write_json(ctrl, {
        "C_WRONG_BONE": {
            "definition": "right-hand bone indices permuted WITHIN each finger "
                          "before the hand score",
            "why_within_finger": "a global shuffle would also destroy the "
                                 "coarse bone-size ordering",
            "frozen_before_results": True},
        "RIG_MEDIAN_TRAIN_ONLY": {
            "definition": "median reference focal of the TRAIN physical "
                          "cameras, applied to the TEST camera",
            "role": "DATASET_CONFOUND_DIAGNOSTIC, not a deployment method",
            "uses_test_focal": False},
        "REF_FOCAL_ORACLE": {
            "definition": "the TEST reference focal itself",
            "role": "UPPER_DIAGNOSTIC for absolute-3D only, not an estimator"},
        "H_ONLY": {"definition": "hand score only, no scene term",
                   "role": "diagnostic, not a deployment baseline"},
    })

    spec = MANIFESTS / "cam_exp_0094_method_spec_v1.json"
    write_json(spec, {
        "experiment": "CAM-EXP-009.4",
        "title": "Sequence-Level Scene + Shared-Hand-Anatomy Focal Refinement",
        "created_after_cam00931_results": True,
        "created_before_cam0094_real_focal_results": True,
        "primary_question": "Does adding sequence-specific shared hand anatomy "
                            "and a generic anatomical prior to a scene-based "
                            "sequence focal estimate improve focal error, "
                            "absolute wrist/root 3D error and absolute MPJPE, "
                            "versus scene-only?",
        "primary_comparison": "M3_SCENE_SHARED_GENERIC_FULL vs M0_SCENE_ONLY",
        "methods": METHODS,
        "model": {
            "p_seq": "softmax(log(p0 + 1e-8) + a)",
            "p_L": "softmax(log(p_seq + 1e-8) + delta_L)",
            "p_R": "softmax(log(p_seq + 1e-8) + delta_R)",
            "p0": "neutral MANO bone proportions "
                  "(cam_exp_0094_generic_mano_prior_v1.json)",
            "absolute_hand_size_used": False,
            "participant_template_carried_over": False,
            "p_seq_refit_per_unit": True,
        },
        "scene_branch": {
            "source": str(SCENE_SRC.relative_to(SCENE_SRC.parents[4])),
            "model_key": SCENE_MODEL_KEY,
            "aggregation": "exp(median(log f_i)) over the unit's scene frames",
            "focal_scalar": "sqrt(fx*fy), unchanged from CAM-EXP-003/005/008",
            "nuisance": "the scene estimator's own principal point is used; "
                        "no GT cx/cy, k1/k2 or fy/fx ratio enters inference",
        },
        "hand_branch": {
            "observations": "target-camera monocular WiLoR: keypoints_2d "
                            "(original pixels) and root-relative keypoints_3d",
            "bone_lengths_from_network": "DISCARDED - only unit directions are "
                                         "used",
            "topology": [list(b) for b in BONES],
            "solver": "CAM-EXP-009.1 convention: UNDISTORT_ONCE_INTERNAL, "
                      "%d alternating iterations, deterministic init" % 32,
            "held_out": "deterministic FIT/EVAL frame split by SHA-256 bucket",
        },
        "frames": {
            "scene_primary": N_SCENE_FRAMES, "hand_primary": N_HAND_FRAMES,
            "nested_counts": list(FRAME_COUNTS),
            "min_frames_per_side": MIN_FRAMES_PER_SIDE,
            "selection": "evenly spaced by frame index; never by performance",
        },
        "regularisers": {"lambda_generic_grid": list(LAMBDA_GENERIC_GRID),
                         "lambda_side_grid": list(LAMBDA_SIDE_GRID),
                         "selected_on": "SYNTHETIC gate only, never on the "
                                        "real reference focal"},
        "evaluation": {
            "outer_split_unit": "PHYSICAL CAMERA",
            "n_outer_folds": N_OUTER_FOLDS,
            "statistics_cluster": "PHYSICAL CAMERA",
            "bootstrap": 10000,
            "focal_metric": "100*|f_pred - f_ref|/f_ref",
        },
        "gate": GATE,
        "units": {"total": len(units), "eligible": len(elig),
                  "cameras": len(cams),
                  "participants": len(set(u["participant"] for u in elig)),
                  "sequences": len(set(u["sequence"] for u in elig))},
        "never_used_in_inference": [
            "reference / GT focal", "GT principal point", "GT distortion",
            "GT fy/fx ratio", "absolute average hand size",
            "known object or table dimensions", "EXIF",
            "participant anatomy from another recording",
            "target-camera reference 3D",
            "the FINAL_CONFIRMATORY_HOLDOUT"],
    })

    hashes = {p.name: sha256(p) for p in (spec, u_p, s_p, h_p, sp_p, f_p, lg,
                                          fg, ctrl)}
    write_json(SUM / "manifest_hashes.json", hashes)

    fold_n = collections.Counter(u["outer_fold"] for u in elig)
    write_csv(TAB / "outer_fold_summary.csv",
              [{"outer_fold": k, "n_units": v,
                "n_cameras": len(set(u["camera"] for u in elig
                                     if u["outer_fold"] == k))}
               for k, v in sorted(fold_n.items())])
    write_csv(SUM / "unit_eligibility_summary.csv",
              [{"metric": "units_total", "value": len(units)},
               {"metric": "units_eligible", "value": len(elig)},
               {"metric": "physical_cameras", "value": len(cams)},
               {"metric": "sequences",
                "value": len(set(u["sequence"] for u in elig))},
               {"metric": "participants",
                "value": len(set(u["participant"] for u in elig))}])

    print("units total %d, eligible %d" % (len(units), len(elig)))
    print("cameras %d, sequences %d, participants %d"
          % (len(cams), len(set(u["sequence"] for u in elig)),
             len(set(u["participant"] for u in elig))))
    print("ineligible reasons:",
          collections.Counter(u["ineligible_reason"] for u in units
                              if not u["eligible"]))
    print("outer folds:", dict(sorted(fold_n.items())))
    print("hand frame rows %d, scene frame rows %d"
          % (len(hand_rows), len(scene_rows)))
    for k, v in hashes.items():
        print("  %-46s %s" % (k, v[:16]))


if __name__ == "__main__":
    main()
