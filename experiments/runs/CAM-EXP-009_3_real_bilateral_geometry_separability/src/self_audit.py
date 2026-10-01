"""Mechanical self-audit for CAM-EXP-009.3.

Prose checks strip quoted spans first: earlier runs in this programme produced
false positives by matching a sentence that DECLARES a phrase is not used.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FOCAL_BEARING_MANIFESTS, MANIFESTS, REPO,  # noqa: E402
                    RUN_DIR, SUM, read_json, sha256, write_json)

SRC = RUN_DIR / "src"

BANNED = [
    "true human asymmetry", "ground-truth anatomical asymmetry",
    "ground truth anatomical asymmetry", "physical bone asymmetry",
    "physical bone truth", "actual human left-right difference",
    "true anatomy", "human biometric identity", "ground-truth asymmetry",
    "physical focal", "focal improvement",
    "real-world calibration solved", "works in factories",
]


def strip_quoted(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"[\"“”][^\"“”\n]{0,300}"
                  r"[\"“”]", " ", text)
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith(">"))


def check_prose():
    hits = []
    for p in list(RUN_DIR.glob("*.md")) + list(RUN_DIR.glob("notes/*.md")):
        body = strip_quoted(p.read_text(encoding="utf-8")).lower()
        for b in BANNED:
            if b in body:
                hits.append({"file": p.name, "phrase": b})
    return hits


def check_focal_free():
    """No focal-bearing manifest, no calibrator, no candidate grid."""
    bad = []
    tokens = ("gt_fx", "gt_fy", "anycalib", "AnyCalib", "geocalib", "GeoCalib",
              "focal_grid", "candidate_focal", "PIPELINE_BASELINE_FOCAL",
              "GT_EFFECTIVE_FOCAL")
    exempt = {"common.py", "self_audit.py", "audit_sources.py"}
    for p in sorted(SRC.glob("*.py")):
        if p.name in exempt:
            continue
        body = "\n".join(ln for ln in
                         p.read_text(encoding="utf-8").splitlines()
                         if not ln.strip().startswith("#"))
        # a "never_used" list DECLARES these are absent; that is the opposite
        # of a use, and it must not be scored as contamination
        body = re.sub(r'"never_used":\s*\[[^\]]*\]', " ", body)
        for t in tokens + FOCAL_BEARING_MANIFESTS:
            if t in body:
                bad.append({"file": p.name, "token": t})
    return bad


def check_topology():
    from common import BONES
    forbidden = {(4, 8), (4, 12), (8, 20), (4, 20), (8, 12)}
    bad = [b for b in BONES if tuple(sorted(b)) in forbidden]
    parents = {}
    for a, b in BONES:
        parents[b] = a
    connected = len(BONES) == 20 and len(set(parents)) == 20 and 0 not in parents
    return {"n_bones": len(BONES), "forbidden_pairs_used": bad,
            "all_connected_single_parent": connected}


def check_circularity():
    p = SUM / "reference_geometry_audit.json"
    if not p.exists():
        return {"available": False}
    d = read_json(p)
    return {"available": True,
            "target_camera_in_reference_count":
                d["target_camera_in_reference_count"],
            "clean": d["circularity_clean"]}


def check_history_untouched():
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "experiments/runs/CAM-EXP-001_3_gigahands_multiview_triangulation",
         "experiments/runs/CAM-EXP-009_bilateral_hand_geometry_focal",
         "experiments/runs/CAM-EXP-009_1_bilateral_solver_validation",
         "experiments/runs/CAM-EXP-009_2_robust_bilateral_focal",
         "experiments/report_prep"],
        cwd=REPO, capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def check_manifest_hashes():
    rec = read_json(SUM / "manifest_hashes.json")
    return {k: (rec[k] == sha256(MANIFESTS / k)) for k in rec}


def check_branch():
    b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       cwd=REPO, capture_output=True, text=True).stdout.strip()
    main = subprocess.run(["git", "rev-parse", "main"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    return {"branch": b, "on_kjh": b == "kjh", "main_sha": main}


def main():
    res = {
        "banned_phrases": check_prose(),
        "focal_contamination": check_focal_free(),
        "topology": check_topology(),
        "circularity": check_circularity(),
        "historical_runs_modified": check_history_untouched(),
        "manifest_hashes_match": check_manifest_hashes(),
        "branch": check_branch(),
    }
    res["ALL_CLEAN"] = (
        not res["banned_phrases"] and not res["focal_contamination"]
        and not res["topology"]["forbidden_pairs_used"]
        and res["topology"]["all_connected_single_parent"]
        and res["circularity"].get("clean", False)
        and not res["historical_runs_modified"]
        and all(res["manifest_hashes_match"].values())
        and res["branch"]["on_kjh"])
    write_json(SUM / "self_audit.json", res)
    for k, v in res.items():
        print("%-28s %s" % (k, v if not isinstance(v, list)
                            else ("clean" if not v else v)))


if __name__ == "__main__":
    main()
