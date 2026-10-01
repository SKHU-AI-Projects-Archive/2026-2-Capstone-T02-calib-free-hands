"""Mechanical self-audit for CAM-EXP-009.2.

Checks that can be made by a machine are made by a machine. The prose checks
strip quoted spans first, because earlier runs in this programme produced false
positives by matching a sentence that *declares a phrase is not used*.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (C91, MANIFESTS, RUN_DIR, SUM, read_csv,  # noqa: E402
                        read_json, sha256, write_json)

SRC = RUN_DIR / "src"

BANNED = [
    "physical focal", "independent physical ground truth", "our method",
    "final method", "proposed method", "nearly two orders of magnitude",
    "full sequence result", "whole-video result",
    "true human anatomical asymmetry",
]


def strip_quoted(text):
    """Remove quoted spans and fenced code, where a phrase may be DISCUSSED."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"[\"“”][^\"“”\n]{0,200}"
                  r"[\"“”]", " ", text)
    text = "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith(">"))
    return text


def check_prose():
    hits = []
    for p in list(RUN_DIR.glob("*.md")) + list(RUN_DIR.glob("notes/*.md")):
        body = strip_quoted(p.read_text(encoding="utf-8")).lower()
        for b in BANNED:
            if b in body:
                hits.append({"file": p.name, "phrase": b})
    return hits


def check_no_gt_focal_input():
    """AST check: no call passes f_true / lL_true / lR_true into the solver."""
    bad = []
    solver_calls = {"fit_bones", "score_fixed_shape", "build", "_profile",
                    "run_trial"}
    oracle_ok = {"run_synth_controls.py"}          # C4, declared oracle
    for p in SRC.glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = (n.func.attr if isinstance(n.func, ast.Attribute)
                  else getattr(n.func, "id", None))
            if fn not in solver_calls:
                continue
            for a in list(n.args) + [k.value for k in n.keywords]:
                src = ast.unparse(a)
                if re.search(r"f_true|lL_true|lR_true|GT_|_gt\b", src):
                    if p.name in oracle_ok:
                        continue
                    bad.append({"file": p.name, "line": n.lineno,
                                "call": fn, "arg": src})
    return bad


def check_dof():
    """D10/D5 must not be used; they force equal bone lengths in a group."""
    bad = []
    for p in SRC.glob("*.py"):
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"dof\s*=\s*[\"']D(10|5)[\"']", ln):
                bad.append({"file": p.name, "line": i, "text": ln.strip()})
    return bad


def check_solver_unmodified():
    """The CAM-EXP-009.1 solver must be untouched in the working tree."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         str(C91.relative_to(C91.parents[2]))],
        cwd=C91.parents[2], capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def check_seeds_disjoint():
    d = read_csv(MANIFESTS / "cam_exp_0092_synthetic_dev_trials_v1.csv.gz")
    t = read_csv(MANIFESTS / "cam_exp_0092_synthetic_test_trials_v1.csv.gz")
    inter = set(r["seed"] for r in d) & set(r["seed"] for r in t)
    return sorted(inter)


def check_manifest_hashes():
    rec = read_json(SUM / "manifest_hashes.json")
    return {k: (rec[k] == sha256(MANIFESTS / k)) for k in rec}


def main():
    res = {
        "banned_phrases": check_prose(),
        "gt_focal_passed_to_solver": check_no_gt_focal_input(),
        "forbidden_dof_used": check_dof(),
        "cam0091_solver_modified": check_solver_unmodified(),
        "dev_test_seed_overlap": check_seeds_disjoint(),
        "manifest_hashes_match": check_manifest_hashes(),
    }
    res["ALL_CLEAN"] = (
        not res["banned_phrases"] and not res["gt_focal_passed_to_solver"]
        and not res["forbidden_dof_used"] and not res["cam0091_solver_modified"]
        and not res["dev_test_seed_overlap"]
        and all(res["manifest_hashes_match"].values()))
    write_json(SUM / "self_audit.json", res)
    for k, v in res.items():
        print("%-32s %s" % (k, v if not isinstance(v, list) else
                            ("clean" if not v else v)))


if __name__ == "__main__":
    main()
