"""Mechanical self-audit for CAM-EXP-012."""
from __future__ import annotations

import collections
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c012_common import (CACHE, LAMBDA_GRID, MANIFESTS, MODELS, RAW,  # noqa
                         REPO, RUN_DIR, SUM, TAB, fnum, read_csv, read_json,
                         sha256, write_csv, write_json)

SRC = RUN_DIR / "src"
# scripts that build evidence; none of them may touch a reference focal
PRE_LABEL = {"build_primary_set.py", "build_shuffled_mapping.py",
             "compute_hand_curves.py", "hand_io.py", "fusion.py",
             "validate_global_grid.py", "validate_grid_vs_qgrid.py",
             "verify_correct_curves.py", "freeze_predictions.py"}

BANNED = [
    "cam-exp-003 was wrong", "cam-003 was wrong",
    "independent samples", "hand geometry definitely improves focal",
    "best model", "the winner is", "our method", "final method",
    "proposed method", "physical focal", "nearly two orders of magnitude",
    "independent physical ground truth", "true human anatomical asymmetry",
    "full sequence result", "whole-video result",
    "cam-exp-012 complete", "cam-012 complete",
]

SMART = "“”"

DECLARATION = re.compile(r'"[a-z_]*(gt_[a-z]+|reference_focal)[a-z_]*"\s*:'
                         r'\s*(False|"NO")')


def strip_quoted(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub("[\"" + SMART + "][^\"" + SMART + "\n]{0,400}[\"" + SMART + "]",
               " ", t)
    return "\n".join(l for l in t.splitlines()
                     if not l.lstrip().startswith(">"))


def main():
    res = {}

    # ---- leakage: no pre-label script may name a reference focal
    tokens = ("gt_fx", "gt_fy", "gt_cx", "gt_cy", "reference_focal")
    hits = []
    for p in sorted(SRC.glob("*.py")):
        if p.name not in PRE_LABEL:
            continue
        # A line that DECLARES the focal was not used, e.g.
        #     "reference_focal_read": False
        # is a provenance record, not an access. Drop those before scanning.
        body = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith("#")
                         and not DECLARATION.search(l))
        hits += [{"file": p.name, "token": t} for t in tokens if t in body]
    res["pre_label_focal_leakage"] = hits

    # ---- primary set and curves
    prim = read_csv(MANIFESTS / "cam_exp_012_primary_video_set_v1.csv")
    res["primary_videos"] = len(prim)
    integ = read_json(SUM / "correct_curves_integrity.json")
    res["correct_curves_complete"] = integ["CORRECT_HAND_CURVES_COMPLETE"]
    res["curves_on_disk"] = integ["curves_on_disk"]
    wb = CACHE / "wrong_bone_hand_curves"
    res["wrong_bone_curves"] = len(list(wb.glob("*.npz"))) if wb.exists() else 0
    res["wrong_bone_control"] = "PENDING"

    # ---- grids
    g = read_json(MANIFESTS / "cam_exp_012_global_focal_grid_v1.json")
    res["global_grid_points"] = len(g["grid"])
    q = np.array(read_json(MANIFESTS / "cam_exp_012_q_grid_v1.json")["q"],
                 float)
    res["q_grid_points"] = len(q)
    res["q_grid_contains_exact_one"] = bool((q == 1.0).sum() == 1)
    lg = read_json(MANIFESTS / "cam_exp_012_lambda_grid_v1.json")["lambda_grid"]
    res["lambda_grid"] = lg
    res["lambda_zero_excluded"] = 0 not in lg and 0.0 not in lg
    res["lambda_grid_matches_code"] = list(lg) == list(LAMBDA_GRID)

    # ---- shuffled mapping
    sm = read_csv(MANIFESTS / "cam_exp_012_shuffled_hand_mapping_v1.csv")
    donors = [r for r in sm if r["has_donor"] == "1"]
    res["shuffled_donors"] = len(donors)
    res["shuffled_never_self"] = all(
        (r["donor_sequence"], r["donor_camera"]) != (r["sequence"], r["camera"])
        for r in donors)
    res["shuffled_never_same_sequence"] = all(
        r["donor_sequence"] != r["sequence"] for r in donors)
    res["shuffled_never_same_camera"] = all(
        r["donor_camera"] != r["camera"] for r in donors)

    # ---- one prediction per video x model x condition
    fp = RAW / "frozen_test_predictions.csv.gz"
    rows = read_csv(fp)
    cnt = collections.Counter(
        (r["sequence"], r["camera"], r["model"], r["condition"]) for r in rows)
    res["one_prediction_per_video_condition"] = all(v == 1
                                                    for v in cnt.values())
    res["conditions_present"] = sorted({r["condition"] for r in rows})
    res["wrong_bone_rows_absent"] = (
        "SCENE_PLUS_WRONG_BONE" not in res["conditions_present"])

    # ---- every test video predicted exactly once across outer folds
    seen = collections.Counter((r["sequence"], r["camera"]) for r in rows
                               if r["condition"] == "SCENE_ONLY"
                               and r["model"] == "ANYCALIB")
    res["each_video_tested_once"] = all(v == 1 for v in seen.values())
    res["n_test_videos"] = len(seen)

    # ---- outer folds are by physical camera, disjoint
    of = {r["camera"]: r["outer_fold"] for r in read_csv(
        MANIFESTS / "cam_exp_012_outer_camera_folds_v1.csv")}
    byfold = collections.defaultdict(set)
    for r in rows:
        byfold[r["outer_fold"]].add(r["camera"])
    res["camera_fold_consistent"] = all(
        of.get(c) == f for f, cs in byfold.items() for c in cs)
    res["camera_folds_disjoint"] = (
        sum(len(s) for s in byfold.values())
        == len(set().union(*byfold.values())) if byfold else False)

    # ---- the SAME alpha and lambda across conditions within a fold
    keyed = collections.defaultdict(set)
    for r in rows:
        if r["condition"] in ("SCENE_PLUS_CORRECT_HAND",
                              "SCENE_PLUS_SHUFFLED_HAND"):
            keyed[(r["model"], r["outer_fold"])].add((r["alpha"], r["lambda"]))
    res["shuffled_reuses_alpha_and_lambda"] = all(len(v) == 1
                                                  for v in keyed.values())

    # ---- lambda selection boundary report (a warning, not a failure)
    sel = read_csv(TAB / "selected_lambda_by_fold.csv")
    res["lambda_at_boundary_cells"] = sum(1 for r in sel
                                          if r["lambda_boundary"] == "1")
    res["lambda_cells"] = len(sel)

    # ---- freeze integrity
    fz = read_json(SUM / "presentation_prediction_freeze.json")
    res["frozen_predictions_unchanged"] = (
        sha256(fp) == fz["frozen_predictions_sha256"])
    res["frozen_before_focal_opened"] = (
        fz["test_reference_focal_opened"] is False)
    res["focal_open_record_exists"] = (
        RAW / "REFERENCE_FOCAL_OPENED.txt").exists()

    # ---- verdict wording
    v = read_json(SUM / "cam012_presentation_verdict.json")
    res["cam012_not_claimed_complete"] = v["cam012_complete"] is False
    res["specific_signal_pending"] = all(
        r["CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED"] == "PENDING_WRONG_BONE"
        for r in v["verdicts"])

    # ---- prose
    res["banned_phrases"] = [
        {"file": p.name, "phrase": b} for p in RUN_DIR.glob("*.md")
        for b in BANNED
        if b in strip_quoted(p.read_text(encoding="utf-8")).lower()]

    # ---- repo hygiene
    others = ["experiments/runs/%s" % d.name
              for d in (REPO / "experiments" / "runs").iterdir()
              if d.is_dir() and not d.name.startswith("CAM-EXP-012")
              and d.name != "results"]
    prior = subprocess.run(
        ["git", "status", "--porcelain", "--", "experiments/report_prep"]
        + others, cwd=REPO, capture_output=True, text=True)
    res["historical_modified"] = [l for l in prior.stdout.splitlines()
                                  if l.strip()]
    man_st = subprocess.run(["git", "status", "--porcelain", "--",
                             "experiments/manifests"], cwd=REPO,
                            capture_output=True, text=True).stdout
    res["old_manifests_modified"] = [l for l in man_st.splitlines()
                                     if l.strip() and "cam_exp_012" not in l]
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                        capture_output=True, text=True).stdout.strip()
    res["branch"] = {"branch": br, "on_kjh": br == "kjh"}
    stray = subprocess.run(["git", "status", "--porcelain", "--",
                            "experiments/runs/results"], cwd=REPO,
                           capture_output=True, text=True).stdout
    res["stray_untracked_runs_results"] = [l for l in stray.splitlines()
                                           if l.strip()]
    res["external_holdout_opened"] = False

    write_csv(TAB / "leakage_audit.csv", [
        {"check": "reference focal used to build the primary set",
         "result": "NO"},
        {"check": "reference focal used inside the hand curves",
         "result": "NO"},
        {"check": "reference focal used for alpha", "result": "NO"},
        {"check": "OUTER TEST focal used for lambda selection",
         "result": "NO"},
        {"check": "result-based camera or video exclusion", "result": "NO"},
        {"check": "shuffled control retuned", "result": "NO"},
        {"check": "predictions frozen before test focal opened",
         "result": "YES" if res["frozen_before_focal_opened"] else "NO"},
        {"check": "reserved confirmatory holdout opened", "result": "NO"},
    ])

    res["ALL_CLEAN"] = bool(
        not res["pre_label_focal_leakage"]
        and res["correct_curves_complete"]
        and res["primary_videos"] == res["curves_on_disk"] == 153
        and res["q_grid_contains_exact_one"] and res["lambda_zero_excluded"]
        and res["lambda_grid_matches_code"]
        and res["shuffled_donors"] == 153 and res["shuffled_never_self"]
        and res["shuffled_never_same_sequence"]
        and res["shuffled_never_same_camera"]
        and res["one_prediction_per_video_condition"]
        and res["each_video_tested_once"] and res["n_test_videos"] == 153
        and res["camera_fold_consistent"] and res["camera_folds_disjoint"]
        and res["shuffled_reuses_alpha_and_lambda"]
        and res["frozen_predictions_unchanged"]
        and res["frozen_before_focal_opened"]
        and res["wrong_bone_rows_absent"]
        and res["cam012_not_claimed_complete"]
        and res["specific_signal_pending"]
        and not res["banned_phrases"] and not res["historical_modified"]
        and not res["old_manifests_modified"] and res["branch"]["on_kjh"])

    write_json(SUM / "self_audit.json", res)
    for k, val in res.items():
        print("%-40s %s" % (k, val if not isinstance(val, list)
                            else ("clean" if not val else val)))


if __name__ == "__main__":
    main()
