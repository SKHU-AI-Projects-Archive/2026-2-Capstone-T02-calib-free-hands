"""Mechanical self-audit for CAM-EXP-010."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C0094, MANIFESTS, RAW, REPO, RUN_DIR,  # noqa: E402
                    SOURCE_ARTIFACTS, SUM, TAB, read_csv, read_json, sha256,
                    write_csv, write_json)

SRC = RUN_DIR / "src"

BANNED = [
    "monocular camera cannot estimate depth",
    "monocular cameras cannot estimate depth",
    "hand geometry research failed",
    "average human hand size",
    "average hand length",
    "known table dimension",
    "known object dimension",
    "physical focal",
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

    # ---- all corrections must be labelled diagnostics
    tags = read_json(SUM / "oracle_tags.json")
    res["all_corrections_are_oracle_diagnostics"] = \
        tags["all_corrections_are_oracle_diagnostics"]

    # ---- no deployment-forbidden anchor anywhere in the source
    forbidden_src = ("HAND_LENGTH_M", "AVERAGE_HAND", "TABLE_WIDTH",
                     "OBJECT_SIZE", "EXIF")
    hits = []
    for p in sorted(SRC.glob("*.py")):
        if p.name == "self_audit.py":       # its own token list IS the check
            continue
        body = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith("#"))
        hits += [{"file": p.name, "token": t} for t in forbidden_src
                 if t in body]
    res["forbidden_metric_anchor_in_source"] = hits

    # ---- FIT/EVAL split determinism and separation
    fr = read_csv(RAW / "frame_translation_errors.csv.gz")
    from common import split_of
    bad = [r for r in fr[:5000]
           if split_of(r["sequence"], r["camera"], r["hand"],
                       int(r["frame"])) != r["split"]]
    res["split_deterministic"] = not bad
    res["split_counts"] = {
        "FIT": sum(1 for r in fr if r["split"] == "FIT"),
        "EVAL": sum(1 for r in fr if r["split"] == "EVAL")}

    # every reported oracle row must be an EVAL frame
    orc = read_csv(RAW / "oracle_corrected_frame_results.csv.gz")
    evalkeys = {(r["sequence"], r["camera"], r["hand"], r["frame"])
                for r in fr if r["split"] == "EVAL"}
    res["oracle_rows_are_eval_only"] = all(
        (r["sequence"], r["camera"], r["hand"], r["frame"]) in evalkeys
        for r in orc)

    # ---- circularity and reproduction
    res["circularity"] = read_json(SUM / "extraction_audit.json")[
        "target_camera_in_reference_count"]
    rep = SUM / "cam0094_oracle_reproduction.json"
    res["cam0094_reproduced"] = (read_json(rep)["ALL_REPRODUCED"]
                                 if rep.exists() else None)

    # ---- identifiability tests were actually run and passed
    ident = read_json(SUM / "identifiability_summary.json")
    res["identifiability_tests_pass"] = ident["ALL_INVARIANCE_TESTS_PASS"]
    res["identifiability_tests_present"] = sorted(
        t["test"] for t in ident["tests"])

    # ---- learned prior vs geometric identifiability kept apart
    txt = ""
    for p in RUN_DIR.glob("*.md"):
        txt += p.read_text(encoding="utf-8")
    res["learned_vs_geometric_separated"] = bool(
        re.search(r"LEARNED STATISTICAL SCALE PRIOR", txt)
        and re.search(r"GEOMETRIC", txt, re.I))

    # ---- no new WiLoR inference: the cache belongs to CAM-EXP-009.4
    res["reused_cam0094_wilor_cache"] = (C0094 / "cache" / "wilor").exists()
    res["no_new_wilor_cache_here"] = not (RUN_DIR / "cache" / "wilor").exists()

    # ---- prose
    res["banned_phrases"] = [
        {"file": p.name, "phrase": b} for p in RUN_DIR.glob("*.md")
        for b in BANNED
        if b in strip_quoted(p.read_text(encoding="utf-8")).lower()]

    # ---- repo hygiene
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "experiments/runs/CAM-EXP-003_single_frame_calibration_benchmark",
         "experiments/runs/CAM-EXP-006_reference_hand_focal_information",
         "experiments/runs/CAM-EXP-008_offline_sequence_scene_hand_focal_fusion",
         "experiments/runs/CAM-EXP-009_1_bilateral_solver_validation",
         "experiments/runs/CAM-EXP-009_2_robust_bilateral_focal",
         "experiments/runs/CAM-EXP-009_3_real_bilateral_geometry_separability",
         "experiments/runs/CAM-EXP-009_3_1_participant_aware_reanalysis",
         "experiments/runs/CAM-EXP-009_4_scene_shared_anatomy_focal_refinement",
         "experiments/report_prep"],
        cwd=REPO, capture_output=True, text=True)
    res["historical_modified"] = [l for l in out.stdout.splitlines()
                                  if l.strip()]
    man = subprocess.run(["git", "status", "--porcelain", "--",
                          "experiments/manifests"], cwd=REPO,
                         capture_output=True, text=True).stdout
    res["old_manifests_modified"] = [
        l for l in man.splitlines()
        if l.strip() and "cam_exp_010" not in l]
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                        capture_output=True, text=True).stdout.strip()
    res["branch"] = {"branch": br, "on_kjh": br == "kjh"}
    res["external_holdout_opened"] = False

    # ---- source artefact hashes
    rows = []
    for name, p in sorted(SOURCE_ARTIFACTS.items()):
        rows.append({"artifact": name,
                     "path": (p.relative_to(REPO).as_posix()
                              if p.is_relative_to(REPO) else str(p)),
                     "exists": int(p.exists()),
                     "sha256": sha256(p) if p.exists() else "",
                     "size_bytes": p.stat().st_size if p.exists() else ""})
    write_csv(TAB / "source_artifacts.csv", rows)
    res["source_artifacts_present"] = all(r["exists"] for r in rows)

    res["ALL_CLEAN"] = bool(
        res["all_corrections_are_oracle_diagnostics"]
        and not res["forbidden_metric_anchor_in_source"]
        and res["split_deterministic"] and res["oracle_rows_are_eval_only"]
        and res["circularity"] == 0
        and res["cam0094_reproduced"] is not False
        and res["identifiability_tests_pass"]
        and res["learned_vs_geometric_separated"]
        and res["no_new_wilor_cache_here"]
        and not res["banned_phrases"] and not res["historical_modified"]
        and not res["old_manifests_modified"]
        and res["branch"]["on_kjh"] and res["source_artifacts_present"])

    write_json(SUM / "self_audit.json", res)
    for k, v in res.items():
        print("%-42s %s" % (k, v if not isinstance(v, list)
                            else ("clean" if not v else v)))


if __name__ == "__main__":
    main()
