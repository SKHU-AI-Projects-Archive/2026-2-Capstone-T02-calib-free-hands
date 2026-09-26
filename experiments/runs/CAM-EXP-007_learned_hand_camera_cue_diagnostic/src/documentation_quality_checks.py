"""Documentation quality checks for CAM-EXP-007 provenance and scope.

Checks the DOCUMENTS and the numerical integrity of the run. Runs no analysis
and changes no result.

Usage:  python src/documentation_quality_checks.py [baseline_hashes.json]
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, RUN_DIR, TAB, write_csv  # noqa: E402

DOCS = sorted(RUN_DIR.glob("*.md"))
BASELINE = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# A quoted span is how these documents FORBID a phrasing - they quote the
# banned wording in order to rule it out. Spans can cross line breaks, so the
# substitution runs on the whole document before splitting into units.
QUOTED = re.compile(r'"[^"]*"', re.S)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def doc_units(strip_quoted=False):
    """Yield (filename, first line, unwrapped paragraph or table row).

    These documents are hard-wrapped Markdown, so a statement and the label
    qualifying it often sit on different physical lines. Scanning line by line
    produces false positives, so checks run on unwrapped paragraphs.
    """
    for d in DOCS:
        text = d.read_text(encoding="utf-8")
        if strip_quoted:
            text = QUOTED.sub(
                lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
        buf, start = [], 1
        for i, line in enumerate(text.splitlines(), 1):
            is_table = line.lstrip().startswith("|")
            if not line.strip() or is_table:
                if buf:
                    yield d.name, start, " ".join(buf)
                    buf = []
                if is_table:
                    yield d.name, i, line
                continue
            if not buf:
                start = i
            buf.append(line.strip())
        if buf:
            yield d.name, start, " ".join(buf)


def main() -> int:
    res = []

    def add(qid, desc, hits, detail=""):
        res.append({"check": qid, "description": desc,
                    "status": "PASS" if not hits else "FAIL",
                    "violations": len(hits),
                    "detail": detail or "; ".join(f"{a}:{b}"
                                                  for a, b, _ in hits[:4])})

    txt = "\n".join(d.read_text(encoding="utf-8") for d in DOCS)
    # hard-wrapped Markdown puts newlines mid-sentence, so whole-phrase checks
    # run on a whitespace-normalised copy
    flat = re.sub(r"\s+", " ", txt)

    # Q1 - no claim that the scene fix preceded the target read
    bad = [(d, i, u) for d, i, u in doc_units(strip_quoted=True)
           if re.search(r"(correction|fix|corrected|mismatch)[^.]{0,120}"
                        r"before the target was read", u, re.I)
           or re.search(r"before the target was read[^.]{0,120}"
                        r"(correction|fix|corrected|mismatch)", u, re.I)]
    bad = [b for b in bad if not re.search(r"not described as|was not|"
                                           r"deliberately \*\*not\*\*", b[2],
                                           re.I)]
    add("Q1", "no claim that the scene fix preceded the target read", bad)

    # Q2 - the correction is recorded with both facts
    ok = ("IMPLEMENTATION_CORRECTION_BEFORE_PERFORMANCE_INSPECTION" in txt
          and re.search(r"target had already been loaded", flat, re.I)
          and re.search(r"no model performance metric had been inspected|"
                        r"no performance metric had been inspected", flat,
                        re.I))
    add("Q2", "correction recorded as target-loaded=YES, "
              "performance-inspected=NO",
        [] if ok else [("docs", 0, "missing one of the two facts")])

    # Q3 - no claim that BH-FDR was performed
    bad = []
    for d, i, u in doc_units(strip_quoted=True):
        if re.search(r"\bBH[- ]FDR\b|FDR[- ]adjusted|"
                     r"multiple[- ]comparison correction (was )?applied|"
                     r"multiplicity correction applied", u, re.I):
            if not re.search(r"\bno\b|never|not applied|removed|claimed",
                             u, re.I):
                bad.append((d, i, u))
    src = (RUN_DIR / "src" / "feature_associations.py").read_text(
        encoding="utf-8")
    implements_fdr = bool(re.search(r"fdr|benjamini|p_adj", src, re.I)
                          and "docstring" not in src[:400].lower())
    add("Q3", "no unsupported BH-FDR claim in docs or code", bad,
        f"implementation_present={implements_fdr}")

    # Q4 - no exclusive 'dataset, not model' attribution
    bad = [(d, i, u) for d, i, u in doc_units(strip_quoted=True)
           if re.search(r"dataset,? not the model|"
                        r"the model is not the limitation|"
                        r"only remaining problem is the dataset|"
                        r"binding (problem|obstacle) is the dataset|"
                        r"whatever the cue is", u, re.I)]
    add("Q4", "no exclusive dataset-not-model attribution", bad)

    # Q5 - S2 not claimed as a positive hand signal
    bad = []
    for d, i, u in doc_units(strip_quoted=True):
        if re.search(r"\bS2\b", u) and re.search(
                r"latent signal appears|hand (cue|signal) (is )?(detected|"
                r"found|present)|evidence of a (detected )?hand", u, re.I):
            if not re.search(r"\bnot\b", u, re.I):
                bad.append((d, i, u))
    add("Q5", "S2 stress subset not claimed as a positive hand signal", bad)

    # Q6 - CAM-007.1 framed as not-justified-now, not impossible-forever
    has_soft = bool(re.search(r"not justified by the present evidence", flat,
                              re.I))
    bad = [(d, i, u) for d, i, u in doc_units(strip_quoted=True)
           if re.search(r"cam-?007\.1 (is )?impossible|"
                        r"never (build|attempt)|permanently excluded", u,
                        re.I)]
    if not has_soft:
        bad.append(("docs", 0, "missing 'not justified by the present "
                               "evidence' framing"))
    add("Q6", "CAM-007.1 = not justified now, not impossible forever", bad)

    # Q7-Q10 - numerical integrity
    if BASELINE and BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        groups = [("Q7_raw", "raw", RUN_DIR / "results" / "raw"),
                  ("Q7_summary", "summary", RUN_DIR / "results" / "summary"),
                  ("Q7_figures", "figures", RUN_DIR / "figures"),
                  ("Q7_tables", "tables", RUN_DIR / "tables")]
        for qid, key, root in groups:
            old = base[key]
            new = {str(p.relative_to(root)).replace("\\", "/"): sha(p)
                   for p in sorted(root.rglob("*")) if p.is_file()}
            changed = [k for k in old if k not in new or new[k] != old[k]]
            added = [k for k in new if k not in old]
            add(qid, f"CAM-007 {key}: no pre-existing file changed",
                [(qid, 0, k) for k in changed],
                f"changed={len(changed)} added={len(added)}"
                + (f" ({', '.join(added)})" if added else ""))
        old = base["manifests"]
        new = {p.name: sha(p)
               for p in sorted(MANIFESTS.glob("cam_exp_007_*"))}
        changed = [k for k in old if k not in new or new[k] != old[k]]
        add("Q8", "CAM-007 frozen manifests byte-identical",
            [("m", 0, k) for k in changed], f"changed={len(changed)}")
        old = base["historical"]
        repo = RUN_DIR.parents[2]          # baseline paths are repo-relative
        changed = [k for k in old
                   if not (repo / k).exists() or sha(repo / k) != old[k]]
        add("Q9", "historical CAM-001..006.1 results/figures/tables "
                  "byte-identical",
            [("h", 0, k) for k in changed[:4]],
            f"checked={len(old)} changed={len(changed)}")
    else:
        add("Q7-Q9", "numerical integrity (baseline not supplied)",
            [("baseline", 0, "missing")], "pass baseline json as argv[1]")

    # Q10 - report_prep untouched
    st = subprocess.run(["git", "status", "--short",
                         "experiments/report_prep/"],
                        capture_output=True, text=True,
                        cwd=str(RUN_DIR.parents[2])).stdout.strip()
    add("Q10", "report_prep untouched",
        [] if st == "" else [("report_prep", 0, st[:60])])

    # Q11 - executable code of feature_associations.py unchanged
    old_src = subprocess.run(
        ["git", "show", "HEAD:experiments/runs/"
         "CAM-EXP-007_learned_hand_camera_cue_diagnostic/src/"
         "feature_associations.py"],
        capture_output=True, text=True,
        cwd=str(RUN_DIR.parents[2])).stdout

    def strip_doc(s):
        t = ast.parse(s)
        for n in ast.walk(t):
            if isinstance(n, (ast.Module, ast.FunctionDef,
                              ast.AsyncFunctionDef, ast.ClassDef)):
                if (n.body and isinstance(n.body[0], ast.Expr)
                        and isinstance(n.body[0].value, ast.Constant)
                        and isinstance(n.body[0].value.value, str)):
                    n.body = n.body[1:]
        return ast.dump(t)

    same = bool(old_src) and strip_doc(old_src) == strip_doc(src)
    add("Q11", "feature_associations.py executable code unchanged "
               "(docstring only)",
        [] if same else [("src", 0, "executable code differs")])

    TAB.mkdir(parents=True, exist_ok=True)
    write_csv(TAB / "documentation_quality_checks.csv", res)
    n = sum(1 for r in res if r["status"] == "PASS")
    for r in res:
        print(f"  [{'ok  ' if r['status'] == 'PASS' else 'FAIL'}] "
              f"{r['check']:12s} {r['description']}"
              + (f"   -> {r['detail']}" if r["detail"] else ""))
    print(f"\n{n}/{len(res)} checks passed")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
