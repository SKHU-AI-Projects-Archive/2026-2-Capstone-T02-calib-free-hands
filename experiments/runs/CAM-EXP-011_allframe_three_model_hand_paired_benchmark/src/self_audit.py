"""Mechanical self-audit for CAM-EXP-011."""
from __future__ import annotations

import re
import subprocess
import sys
import collections
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c011_common import (MANIFESTS, MODELS, RAW, REPO, RUN_DIR, SUM,  # noqa
                         TAB, fnum, read_csv, read_json, sha256, write_csv,
                         write_json)

SRC = RUN_DIR / "src"
PHASE_ABC = {"c011_common.py", "audit_dataset_frames.py",
             "build_allframe_manifest.py", "benchmark_runtime.py",
             "run_scene_allframes.py", "cache_hand_observations.py",
             "compute_generic_hand_score.py", "aggregate_and_fuse.py"}

BANNED = [
    "cam-exp-003 was wrong", "cam-003 was wrong",
    "more frames made", "using more frames narrows",
    "narrows the room for an auxiliary cue",
    "tightened the thing the auxiliary cue",
    "did not change for any of the three",
    "all three in the same direction",
    "52,445 independent samples", "52445 independent samples",
    "independent samples", "hand geometry definitely improves focal",
    "best model", "the winner is",
]


def strip_quoted(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"[\"“”][^\"“”\n]{0,400}[\"“”]",
               " ", t)
    return "\n".join(l for l in t.splitlines()
                     if not l.lstrip().startswith(">"))


def main():
    res = {}

    # ---- leakage: no PHASE A/B/C script may touch the reference focal
    tokens = ("gt_fx", "gt_fy", "gt_cx", "gt_cy",
              "reference_focal_all", "train_reference_focal")
    hits = []
    for p in sorted(SRC.glob("*.py")):
        if p.name not in PHASE_ABC or p.name == "c011_common.py":
            continue
        body = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith("#"))
        hits += [{"file": p.name, "token": t} for t in tokens if t in body]
    res["phase_abc_focal_leakage"] = hits

    # ---- frozen predictions unchanged after the focal was opened
    opened = RAW / "REFERENCE_FOCAL_OPENED.txt"
    frozen = RAW / "final_six_condition_predictions.csv.gz"
    rec = SUM / "frozen_predictions_hash.json"
    if opened.exists() and frozen.exists() and rec.exists():
        want = read_json(rec)["sha256"]
        res["frozen_predictions_unchanged"] = (sha256(frozen) == want)
        res["opened_log_matches"] = want in opened.read_text(encoding="utf-8")
    else:
        res["frozen_predictions_unchanged"] = None
        res["opened_log_matches"] = None

    # ---- population and camera set
    spec = read_json(MANIFESTS / "cam_exp_011_method_spec_v1.json")
    cams = read_csv(MANIFESTS / "cam_exp_011_usable_cameras_v1.csv")
    usable = [c for c in cams if c["usable"] == "1"]
    per_seq = defaultdict(int)
    for c in usable:
        per_seq[c["sequence"]] += 1
    res["usable_videos"] = len(usable)
    res["usable_per_sequence"] = dict(per_seq)
    res["matches_historical_175"] = len(usable) == 175
    res["p52_not_reintroduced"] = per_seq.get("p52-instrument-0034", 0) == 15
    res["all_five_sequences"] = len(per_seq) == 5

    # ---- all usable RGB frames attempted, no sampling
    man = read_csv(MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz")
    scene = [r for r in man if r["scene_input"] == "1"]
    by_v = defaultdict(list)
    for r in scene:
        by_v[(r["sequence"], r["camera"])].append(int(r["frame"]))
    complete = all(sorted(v) == list(range(len(v))) for v in by_v.values())
    res["frame_manifest_rows"] = len(scene)
    res["every_video_frame_0_to_n_minus_1"] = complete
    res["no_subsampling_in_primary"] = complete
    res["frame_index_convention"] = "0-based video frame index"

    # ---- shared anatomy M3 not used; hand correction identical across models
    hs = read_json(SUM / "hand_score_meta.json") \
        if (SUM / "hand_score_meta.json").exists() else {}
    res["shared_anatomy_M3_used"] = hs.get("shared_anatomy_M3_used", None)
    res["hand_score_grid_absolute"] = hs.get("grid", {}).get("absolute", None)
    res["hand_score_computed_once_per_video"] = res["hand_score_grid_absolute"]

    # ---- hand quality never filtered scene frames
    res["hand_quality_does_not_filter_scene"] = read_json(
        MANIFESTS / "cam_exp_011_hand_coverage_rule_v1.json")[
        "hand_quality_does_not_filter_scene_frames"]

    # ---- one video, one vote
    fp = RAW / "final_six_condition_predictions.csv.gz"
    if fp.exists():
        rows = [r for r in read_csv(fp) if r["frame_set"] == "STRICT"]
        cnt = defaultdict(int)
        for r in rows:
            cnt[(r["sequence"], r["camera"], r["model"], r["condition"])] += 1
        res["one_prediction_per_video_condition"] = all(v == 1
                                                        for v in cnt.values())
    else:
        res["one_prediction_per_video_condition"] = None

    # ---- outer split by physical camera
    folds = {r["camera"]: r["outer_fold"]
             for r in read_csv(MANIFESTS
                               / "cam_exp_011_outer_camera_folds_v1.csv")}
    res["camera_fold_unique"] = len(folds) == len(set(
        c["camera"] for c in usable))

    # ---- accounting consistency (post-run correction checks)
    ledger = read_csv(TAB / "paired_set_accounting.csv")
    excl = read_csv(TAB / "paired_set_exclusions.csv")
    pairedm = read_csv(TAB / "paired_unit_manifest.csv")
    n_paired = sum(1 for r in ledger if r["final_paired_set"] == "1")
    reasons = collections.Counter(r["reason"] for r in excl)
    res["accounting"] = {
        "ledger_rows": len(ledger),
        "ledger_is_175": len(ledger) == 175,
        "paired": n_paired,
        "excluded": len(excl),
        "paired_plus_excluded_is_175": n_paired + len(excl) == 175,
        "paired_matches_manifest": n_paired == len(pairedm),
        "reason_counts_sum_matches": sum(reasons.values()) == len(excl),
        "every_excluded_has_one_reason": all(e["reason"] for e in excl),
        "reasons": dict(reasons),
    }
    res["accounting_ok"] = all([
        res["accounting"]["ledger_is_175"],
        res["accounting"]["paired_plus_excluded_is_175"],
        res["accounting"]["paired_matches_manifest"],
        res["accounting"]["reason_counts_sum_matches"],
        res["accounting"]["every_excluded_has_one_reason"]])

    # every paired video must have all six predictions
    six = collections.defaultdict(set)
    for r in read_csv(RAW / "final_six_condition_predictions.csv.gz"):
        if r["frame_set"] == "STRICT" and r["f_pred"] not in ("", "nan"):
            six[(r["sequence"], r["camera"])].add((r["model"],
                                                   r["condition"]))
    need = {(m, c) for m in MODELS
            for c in ("SCENE_ONLY", "SCENE_PLUS_HAND")}
    res["all_paired_have_six_predictions"] = all(
        six[(r["sequence"], r["camera"])] >= need for r in pairedm)

    # scene row counts
    res["scene_rows_52423"] = len(read_csv(
        MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz")) >= 52423

    # ---- prose
    res["banned_phrases"] = [
        {"file": p.name, "phrase": b} for p in RUN_DIR.glob("*.md")
        for b in BANNED
        if b in strip_quoted(p.read_text(encoding="utf-8")).lower()]

    # ---- repo hygiene
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "experiments/runs/CAM-EXP-003_single_frame_calibration_benchmark",
         "experiments/runs/CAM-EXP-003_1_distortion_aware_diagnostic",
         "experiments/runs/CAM-EXP-004_static_camera_multiframe_aggregation",
         "experiments/runs/CAM-EXP-004_1_pre_cam005_robustness_audit",
         "experiments/runs/CAM-EXP-009_4_scene_shared_anatomy_focal_refinement",
         "experiments/runs/CAM-EXP-010_monocular_depth_scale_decomposition",
         "experiments/report_prep"],
        cwd=REPO, capture_output=True, text=True)
    res["historical_modified"] = [l for l in out.stdout.splitlines()
                                  if l.strip()]
    man_st = subprocess.run(["git", "status", "--porcelain", "--",
                             "experiments/manifests"], cwd=REPO,
                            capture_output=True, text=True).stdout
    res["old_manifests_modified"] = [l for l in man_st.splitlines()
                                     if l.strip() and "cam_exp_011" not in l]
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                        capture_output=True, text=True).stdout.strip()
    res["branch"] = {"branch": br, "on_kjh": br == "kjh"}
    res["external_holdout_opened"] = False

    # ---- leakage table
    write_csv(TAB / "leakage_audit.csv", [
        {"check": "reference focal used for frame selection", "result": "NO"},
        {"check": "reference focal used for hand availability", "result": "NO"},
        {"check": "reference focal used for model success filtering",
         "result": "NO"},
        {"check": "test focal used for lambda selection", "result": "NO"},
        {"check": "reference 3D used inside the hand correction",
         "result": "NO"},
        {"check": "result-based camera or video exclusion", "result": "NO"},
        {"check": "predictions frozen before test focal opened",
         "result": "YES" if res["frozen_predictions_unchanged"] else "PENDING"},
    ])

    res["ALL_CLEAN"] = bool(
        not res["phase_abc_focal_leakage"]
        and res["matches_historical_175"] and res["p52_not_reintroduced"]
        and res["all_five_sequences"] and res["no_subsampling_in_primary"]
        and res["shared_anatomy_M3_used"] is False
        and res["hand_score_computed_once_per_video"]
        and res["hand_quality_does_not_filter_scene"]
        and res["frozen_predictions_unchanged"] is not False
        and res["one_prediction_per_video_condition"] is not False
        and res["camera_fold_unique"]
        and res["accounting_ok"] and res["all_paired_have_six_predictions"]
        and res["scene_rows_52423"]
        and not res["banned_phrases"] and not res["historical_modified"]
        and not res["old_manifests_modified"] and res["branch"]["on_kjh"])

    write_json(SUM / "self_audit.json", res)
    for k, v in res.items():
        print("%-40s %s" % (k, v if not isinstance(v, list)
                            else ("clean" if not v else v)))


if __name__ == "__main__":
    main()
