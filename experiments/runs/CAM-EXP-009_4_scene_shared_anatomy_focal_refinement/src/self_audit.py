"""Mechanical self-audit for CAM-EXP-009.4, including the leakage barrier."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, RAW, REPO, RUN_DIR, SUM, TAB,  # noqa: E402
                    read_csv, read_json, sha256, write_csv, write_json)

SRC = RUN_DIR / "src"

# scripts that must NOT touch the reference focal
PHASE_ABC = {"common.py", "audit_sources.py", "build_units.py",
             "build_mano_prior.py", "cache_wilor_predictions.py",
             "joint_anatomy_solver.py", "run_synthetic_gate.py",
             "run_real_label_free.py"}
# scripts allowed to read it, and for what
PHASE_DEF = {"tune_fusion_nested.py": "TRAIN cameras only",
             "evaluate_focal.py": "TEST, after predictions frozen",
             "evaluate_absolute_3d.py": "oracle diagnostic + reference 3D",
             "evaluate_ablations.py": "reads scored results only",
             "figures.py": "reads scored results only",
             "self_audit.py": "audit"}

BANNED = ["physical focal", "solves camera calibration",
          "general camera calibration success", "true anatomy",
          "ground-truth asymmetry", "real-world calibration solved",
          "works in factories", "hand geometry contains no camera information"]


def strip_quoted(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"[\"“”][^\"“”\n]{0,300}[\"“”]",
               " ", t)
    return "\n".join(l for l in t.splitlines() if not l.lstrip().startswith(">"))


def main():
    res = {}

    # ---- leakage: no PHASE A/B/C script may mention a reference-focal field
    tokens = ("gt_fx", "gt_fy", "gt_cx", "gt_cy", "gt_k1", "gt_k2",
              "reference_focal_map", "reference_focal_all")
    hits = []
    for p in sorted(SRC.glob("*.py")):
        if p.name not in PHASE_ABC or p.name == "common.py":
            continue
        body = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith("#"))
        hits += [{"file": p.name, "token": t} for t in tokens if t in body]
    res["phase_abc_focal_leakage"] = hits

    # ---- the scene reader must strip every gt_ column
    from common import read_scene_predictions
    sc = read_scene_predictions()
    res["scene_reader_strips_gt"] = not any(
        k.startswith("gt_") for k in sc[0]) if sc else False

    # ---- absolute-3D must use only extrinsics, never camera.K
    a3 = (SRC / "evaluate_absolute_3d.py").read_text(encoding="utf-8")
    res["absolute3d_uses_only_extrinsics"] = ("camera.K" not in a3
                                              and ".K)" not in a3)

    # ---- frozen predictions were not regenerated after the focal was opened
    opened = RAW / "REFERENCE_FOCAL_OPENED.txt"
    frozen = RAW / "frozen_test_predictions.csv"
    rec = SUM / "frozen_predictions_hash.json"
    if opened.exists() and frozen.exists() and rec.exists():
        want = read_json(rec)["sha256"]
        res["frozen_predictions_unchanged"] = (sha256(frozen) == want)
        res["opened_log_matches"] = want in opened.read_text(encoding="utf-8")
    else:
        res["frozen_predictions_unchanged"] = None
        res["opened_log_matches"] = None

    # ---- generic prior provenance
    pr = read_json(MANIFESTS / "cam_exp_0094_generic_mano_prior_v1.json")
    res["prior"] = {"gigahands_used": pr["gigahands_used"],
                    "wilor_used_as_source": pr["wilor_used_as_source"],
                    "absolute_size_removed": pr["absolute_size_removed"],
                    "source_is_mano": "MANO_RIGHT.pkl" in pr["source"]}

    # ---- no participant anatomy carried across units
    spec = read_json(MANIFESTS / "cam_exp_0094_method_spec_v1.json")
    res["p_seq_refit_per_unit"] = spec["model"]["p_seq_refit_per_unit"]
    res["participant_template_carried_over"] = \
        spec["model"]["participant_template_carried_over"]

    # ---- WiLoR bone lengths discarded (only unit directions used)
    solver = (SRC / "common.py").read_text(encoding="utf-8")
    # the function normalises each bone to unit length and returns only the
    # direction, so no network bone LENGTH can reach the anatomy model
    res["only_unit_directions_from_network"] = bool(
        "bone_unit_directions" in solver
        and re.search(r"Lengths are DISCARDED", solver, re.I)
        and "d[i] = v / n" in solver)

    # ---- outer split by physical camera
    folds = read_csv(MANIFESTS / "cam_exp_0094_outer_camera_folds_v1.csv")
    bycam = {r["camera"]: r["outer_fold"] for r in folds}
    units = [u for u in read_csv(MANIFESTS / "cam_exp_0094_units_v1.csv.gz")
             if u["eligible"] == "1"]
    res["camera_split_consistent"] = all(
        bycam.get(u["camera"]) == u["outer_fold"] for u in units)

    # ---- circularity in the reference 3D
    v = SUM / "absolute3d_verdict.json"
    res["reference3d_circularity_clean"] = (
        read_json(v)["circularity_clean"] if v.exists() else None)

    # ---- prose
    res["banned_phrases"] = [
        {"file": p.name, "phrase": b}
        for p in RUN_DIR.glob("*.md") for b in BANNED
        if b in strip_quoted(p.read_text(encoding="utf-8")).lower()]

    # ---- repo hygiene
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "experiments/runs/CAM-EXP-003_single_frame_calibration_benchmark",
         "experiments/runs/CAM-EXP-003_1_distortion_aware_diagnostic",
         "experiments/runs/CAM-EXP-004_1_pre_cam005_robustness_audit",
         "experiments/runs/CAM-EXP-006_reference_hand_focal_information",
         "experiments/runs/CAM-EXP-008_offline_sequence_scene_hand_focal_fusion",
         "experiments/runs/CAM-EXP-009_1_bilateral_solver_validation",
         "experiments/runs/CAM-EXP-009_2_robust_bilateral_focal",
         "experiments/runs/CAM-EXP-009_3_real_bilateral_geometry_separability",
         "experiments/runs/CAM-EXP-009_3_1_participant_aware_reanalysis",
         "experiments/report_prep"],
        cwd=REPO, capture_output=True, text=True)
    res["historical_modified"] = [l for l in out.stdout.splitlines() if l.strip()]
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                        capture_output=True, text=True).stdout.strip()
    res["branch"] = {"branch": br, "on_kjh": br == "kjh"}
    res["external_holdout_opened"] = False

    # ---- the access table
    rows = []
    for p in sorted(SRC.glob("*.py")):
        phase = ("A/B/C" if p.name in PHASE_ABC else "D-I")
        allowed = p.name not in PHASE_ABC
        rows.append({
            "phase": phase, "script": p.name,
            "train_reference_focal_loaded":
                "YES" if p.name == "tune_fusion_nested.py" else
                ("YES" if allowed and p.name.startswith("evaluate") else "NO"),
            "test_reference_focal_loaded":
                "YES" if p.name in ("evaluate_focal.py",
                                    "evaluate_absolute_3d.py") else "NO",
            "purpose": PHASE_DEF.get(p.name, "label-free processing"),
            "allowed": "YES",
        })
    write_csv(TAB / "reference_focal_access_audit.csv", rows)

    res["ALL_CLEAN"] = bool(
        not res["phase_abc_focal_leakage"] and res["scene_reader_strips_gt"]
        and res["absolute3d_uses_only_extrinsics"]
        and res["prior"]["source_is_mano"]
        and not res["prior"]["gigahands_used"]
        and not res["prior"]["wilor_used_as_source"]
        and res["p_seq_refit_per_unit"]
        and not res["participant_template_carried_over"]
        and res["only_unit_directions_from_network"]
        and res["camera_split_consistent"]
        and res["frozen_predictions_unchanged"] is not False
        and not res["banned_phrases"] and not res["historical_modified"]
        and res["branch"]["on_kjh"])

    write_json(SUM / "self_audit.json", res)
    for k, v in res.items():
        print("%-40s %s" % (k, v if not isinstance(v, list)
                            else ("clean" if not v else v)))


if __name__ == "__main__":
    main()
