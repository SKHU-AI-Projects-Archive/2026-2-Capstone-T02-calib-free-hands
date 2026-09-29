"""PHASE A — the exact per-frame manifest, frozen before any focal result.

One row per (sequence, camera, frame) for every usable RGB video. This is the
artefact that answers "which frames exactly did you use?" down to the frame
index.

Frame index convention: 0-based video frame index, matching the GigaHands
loader (verified in CAM-EXP-001.2: 2D row index, 3D row index and RGB frame
index are the same 0-based index).

Reference focal is not read here.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (CAMERA_BENCHMARK, DISPLAY_SEQUENCE, LAMBDA_GRID,  # noqa
                    MANIFESTS, MIN_HAND_FRAMES_PER_SIDE, MIN_SCENE_COVERAGE,
                    MODELS, N_OUTER_FOLDS, PARTICIPANT_OF, Q_MAX, Q_MIN, Q_N,
                    QC, REPO, SUM, TAB, hash_bucket, read_csv, sha256,
                    usable_cameras, write_csv, write_json)


def main():
    cams = usable_cameras()
    vfc = {(r["sequence"], r["camera"]): int(r["video_frame_count"])
           for r in read_csv(SUM / "video_frame_counts.csv")}

    qc = read_csv(QC)
    hand_ok = defaultdict(lambda: defaultdict(set))
    for r in qc:
        if r["qc_status"] in ("PASS_STRICT", "PASS_SINGLE_HAND") \
                and r["triangulation_success"] == "1":
            hand_ok[(r["sequence"], r["camera"])][r["hand"]].add(
                int(r["frame"]))

    rows = []
    for (seq, cam), info in sorted(cams.items()):
        n = vfc.get((seq, cam), 0)
        usable = info["usable"] and n > 0
        hl = hand_ok[(seq, cam)].get("left", set())
        hr = hand_ok[(seq, cam)].get("right", set())
        if not usable:
            rows.append({
                "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
                "participant": PARTICIPANT_OF.get(seq, ""), "camera": cam,
                "frame": "", "video_frame_count": n,
                "rgb_segment_matches_annotation":
                    info["video_segment_matches_annotation"],
                "eligible_rgb": 0, "scene_input": 0,
                "hand_left_available": 0, "hand_right_available": 0,
                "hand_any_available": 0,
                "exclusion_reason": info["note"] or "not usable for benchmark",
            })
            continue
        for f in range(n):                      # 0-based, every frame
            rows.append({
                "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
                "participant": PARTICIPANT_OF.get(seq, ""), "camera": cam,
                "frame": f, "video_frame_count": n,
                "rgb_segment_matches_annotation":
                    info["video_segment_matches_annotation"],
                "eligible_rgb": 1, "scene_input": 1,
                "hand_left_available": int(f in hl),
                "hand_right_available": int(f in hr),
                "hand_any_available": int(f in hl or f in hr),
                "exclusion_reason": "",
            })

    man = MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz"
    write_csv(man, rows)

    scene_rows = [r for r in rows if r["scene_input"] == 1]
    vids = sorted(set((r["sequence"], r["camera"]) for r in scene_rows))
    folds = {c: hash_bucket(c, N_OUTER_FOLDS)
             for c in sorted(set(v[1] for v in vids))}
    fold_p = MANIFESTS / "cam_exp_011_outer_camera_folds_v1.csv"
    write_csv(fold_p, [{"camera": c, "outer_fold": f,
                        "n_videos": sum(1 for v in vids if v[1] == c)}
                       for c, f in sorted(folds.items())])

    cam_p = MANIFESTS / "cam_exp_011_usable_cameras_v1.csv"
    write_csv(cam_p, [{"sequence": s, "camera": c,
                       "usable": int(cams[(s, c)]["usable"]),
                       "note": cams[(s, c)]["note"]}
                      for (s, c) in sorted(cams)])

    hand_p = MANIFESTS / "cam_exp_011_hand_coverage_rule_v1.json"
    write_json(hand_p, {
        "min_hand_frames_per_side": MIN_HAND_FRAMES_PER_SIDE,
        "source": "inherited from CAM-EXP-009.4; not re-chosen here",
        "hand_availability_source": "frozen CAM-EXP-001.3 QC "
                                    "(PASS_STRICT / PASS_SINGLE_HAND, "
                                    "triangulation_success=1)",
        "hand_quality_does_not_filter_scene_frames": True,
        "frozen_before_focal_results": True,
    })
    cs_p = MANIFESTS / "cam_exp_011_common_set_rule_v1.json"
    write_json(cs_p, {
        "min_scene_coverage_fraction": MIN_SCENE_COVERAGE,
        "THREE_MODEL_COMMON_SCENE_FRAMES": "frames where all three scene "
                                           "models returned a valid focal",
        "GLOBAL_6_CONDITION_COMMON_VIDEO_SET": "videos with a result in all "
                                               "six conditions",
        "primary_reported_result": "STRICT_COMMON_FRAME_RESULT",
        "secondary": "MODEL_NATIVE_ALLFRAME_RESULT",
        "no_performance_based_exclusion": True,
        "frozen_before_focal_results": True,
    })
    lam_p = MANIFESTS / "cam_exp_011_lambda_grid_v1.json"
    write_json(lam_p, {"lambda_grid": list(LAMBDA_GRID),
                       "selected_on": "TRAIN physical cameras only, per base "
                                      "model; the tuning procedure is "
                                      "identical for all three models",
                       "test_focal_used": False,
                       "frozen_before_focal_results": True})

    spec_p = MANIFESTS / "cam_exp_011_method_spec_v1.json"
    write_json(spec_p, {
        "experiment": "CAM-EXP-011",
        "title": "All-Frame Three-Model Scene-vs-Hand Paired Focal Benchmark",
        "created_before_cam011_focal_results": True,
        "question": "On the same videos, does adding whole-video hand "
                    "structure to an existing scene focal estimator reduce "
                    "focal error?",
        "models": MODELS,
        "conditions": ["SCENE_ONLY", "SCENE_PLUS_HAND"],
        "frame_policy": "ALL usable RGB frames; no 8/16/32/64 sampling in the "
                        "primary result",
        "historical_context": "CAM-EXP-003 used 8 uniformly spaced frames and "
                              "CAM-EXP-004.1 extended to 64 and found scene "
                              "aggregation largely saturated. Neither is "
                              "treated as an error; this run answers the "
                              "whole-video question directly.",
        "evaluation_unit": "sequence-camera video; one video, one focal, one "
                           "vote. Frames are repeated observations inside a "
                           "video, never independent samples.",
        "aggregation": "exp(median(log f_i)) over a video's valid frames",
        "scalar_focal": "sqrt(fx*fy), unchanged from CAM-EXP-003/005/008",
        "hand_correction": "generic hand anatomy correction, equivalent to "
                           "CAM-EXP-009.4's M1_SCENE_GENERIC_INDEPENDENT: "
                           "neutral MANO prior, 20 connected bones, absolute "
                           "size removed, WiLoR 2D observations and unit bone "
                           "directions, network bone LENGTHS discarded, "
                           "LEFT/RIGHT anatomy independent",
        "shared_anatomy_M3_used": False,
        "why_not_M3": "CAM-EXP-009.4 measured the shared-anatomy term's "
                      "contribution as exactly 0.000 pp under the tested "
                      "parameterisation",
        "hand_correction_identical_across_models": True,
        "only_difference_between_models": "the scene focal evidence",
        "evaluation": {"outer_split_unit": "PHYSICAL CAMERA",
                       "n_outer_folds": N_OUTER_FOLDS,
                       "cluster": "PHYSICAL CAMERA",
                       "bootstrap": 10000,
                       "primary_metric": "median relative focal error (%)",
                       "mean_is_not_primary": True},
        "candidate_grid": {"q_min": Q_MIN, "q_max": Q_MAX, "q_n": Q_N},
        "counts": {
            "sequences": 5,
            "usable_videos": len(vids),
            "scene_input_frames": len(scene_rows),
            "expected_historical_total": 52445,
            "measured_total": len(scene_rows),
        },
        "never_used": ["reference focal for frame selection",
                       "reference focal for hand availability",
                       "reference focal for model success filtering",
                       "test focal for lambda selection",
                       "reference 3D inside the hand correction",
                       "result-based camera or unit exclusion"],
    })

    hashes = {p.name: sha256(p) for p in (spec_p, man, fold_p, cam_p, hand_p,
                                          cs_p, lam_p)}
    write_json(SUM / "manifest_hashes.json", hashes)

    per_seq = defaultdict(lambda: [0, 0, 0, 0])
    for r in scene_rows:
        a = per_seq[r["sequence"]]
        a[0] += 1
        a[1] += r["hand_left_available"]
        a[2] += r["hand_right_available"]
        a[3] += r["hand_any_available"]
    print("manifest rows (scene input): %d over %d videos"
          % (len(scene_rows), len(vids)))
    for s, a in sorted(per_seq.items()):
        print("  %-12s scene %6d   handL %5d  handR %5d  handAny %5d"
              % (DISPLAY_SEQUENCE.get(s, s), a[0], a[1], a[2], a[3]))
    print("outer folds:", {k: sum(1 for c in folds.values() if c == k)
                           for k in range(N_OUTER_FOLDS)})
    for k, v in hashes.items():
        print("  %-46s %s" % (k, v[:16]))


if __name__ == "__main__":
    main()
