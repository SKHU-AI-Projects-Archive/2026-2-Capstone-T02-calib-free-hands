"""Mechanical self-audit for CAM-EXP-009.3.1."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C093, MANIFESTS, PARTICIPANT_OF, PARTICIPANTS,  # noqa: E402
                    RAW, REPO, RUN_DIR, SUM, read_csv, read_json, sha256,
                    write_json)

SRC = RUN_DIR / "src"
BANNED = ["true anatomy", "ground-truth asymmetry", "physical bone truth",
          "human biometric identity", "physical focal", "focal improvement",
          "real-world calibration solved", "works in factories",
          "human hand anatomy changes between sessions"]


def strip_quoted(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"[\"“”][^\"“”\n]{0,300}"
                  r"[\"“”]", " ", text)
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith(">"))


def main():
    res = {}

    res["banned_phrases"] = [
        {"file": p.name, "phrase": b}
        for p in list(RUN_DIR.glob("*.md"))
        for b in BANNED
        if b in strip_quoted(p.read_text(encoding="utf-8")).lower()]

    # focal independence
    tokens = ("gt_fx", "gt_fy", "anycalib", "AnyCalib", "geocalib", "GeoCalib",
              "focal_grid", "candidate_focal")
    hits = []
    for p in sorted(SRC.glob("*.py")):
        if p.name in ("common.py", "self_audit.py"):
            continue
        body = "\n".join(ln for ln in p.read_text(encoding="utf-8").splitlines()
                         if not ln.strip().startswith("#"))
        hits += [{"file": p.name, "token": t} for t in tokens if t in body]
    res["focal_contamination"] = hits

    # identity source is official, not MANO
    prov = read_json(SUM / "identity_provenance_summary.json")
    res["identity"] = {
        "status": prov["STATUS"],
        "source_is_official": prov["official_source_url"].startswith("https://"),
        "mano_excluded": any("MANO" in x for x in
                             prov["identity_source_excludes"]),
        "p41_two_sessions_same_participant":
            PARTICIPANT_OF["p41-boxing-0021"] == PARTICIPANT_OF["p41-plant-0004"],
    }

    # no reconstruction rerun; source artefacts intact
    src_audit = read_json(SUM / "source_artifact_audit.json")
    res["source_reuse"] = {
        "reference_reconstruction_rerun":
            src_audit["reference_reconstruction_rerun"],
        "all_facts_ok": src_audit["ALL_OK"],
        "source_run_clean": not src_audit["source_run_working_tree_dirty"],
    }
    res["v1_reproduced"] = read_json(
        SUM / "cam0093_v1_reproduction.json")["ALL_REPRODUCED"]

    # cross-participant pairs must never contain p41 <-> p41
    cross = read_csv(RAW / "cross_participant_distances.csv.gz")
    res["cross_participant_excludes_same_participant"] = not [
        r for r in cross
        if r["left_participant"] == r["right_participant"]]

    # permutation uses 4 blocks / 24 permutations
    perm = read_json(SUM / "participant_permutation_summary.json")
    res["permutation"] = {
        "blocks": perm["n_participant_blocks"],
        "n_permutations": perm["n_permutations"],
        "p41_same_block": perm["p41_sessions_move_as_one_block"],
        "ok": perm["n_participant_blocks"] == 4
        and perm["n_permutations"] == 24,
    }

    # historical runs untouched
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         str(C093.relative_to(REPO)),
         "experiments/runs/CAM-EXP-009_1_bilateral_solver_validation",
         "experiments/runs/CAM-EXP-009_2_robust_bilateral_focal",
         "experiments/report_prep"],
        cwd=REPO, capture_output=True, text=True)
    res["historical_modified"] = [ln for ln in out.stdout.splitlines()
                                  if ln.strip()]

    # old v1 manifests untouched
    v1m = subprocess.run(
        ["git", "status", "--porcelain", "--", "experiments/manifests"],
        cwd=REPO, capture_output=True, text=True).stdout
    res["cam0093_manifests_modified"] = [
        ln for ln in v1m.splitlines() if "cam_exp_0093_" in ln]

    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=REPO, capture_output=True, text=True).stdout.strip()
    res["branch"] = {"branch": br, "on_kjh": br == "kjh"}

    res["underpowered_declared"] = "PARTICIPANT_REANALYSIS_UNDERPOWERED" in \
        read_json(SUM / "cam00931_verdict.json")["TAGS"]
    res["labeled_post_result"] = read_json(
        SUM / "cam00931_verdict.json")["created_after_cam0093_results"]

    res["ALL_CLEAN"] = bool(
        not res["banned_phrases"] and not res["focal_contamination"]
        and res["identity"]["status"] == "OFFICIAL_PARTICIPANT_MAPPING_VERIFIED"
        and res["identity"]["mano_excluded"]
        and res["identity"]["p41_two_sessions_same_participant"]
        and not res["source_reuse"]["reference_reconstruction_rerun"]
        and res["source_reuse"]["all_facts_ok"] and res["v1_reproduced"]
        and res["cross_participant_excludes_same_participant"]
        and res["permutation"]["ok"] and not res["historical_modified"]
        and not res["cam0093_manifests_modified"]
        and res["branch"]["on_kjh"] and res["underpowered_declared"]
        and res["labeled_post_result"])

    write_json(SUM / "self_audit.json", res)
    for k, val in res.items():
        print("%-42s %s" % (k, val if not isinstance(val, list)
                            else ("clean" if not val else val)))


if __name__ == "__main__":
    main()
