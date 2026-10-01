"""Documentation quality checks for CAM-EXP-006.1 provenance classification.

These check the DOCUMENTS, not the science. They enforce that:
  * no additive causal decomposition wording is used,
  * post-hoc diagnostics are never labelled pre-registered,
  * the B-vs-C cutoff is not described as pre-registered,
  * every numerical artefact is byte-identical to its pre-edit state,
  * CAM-007 is not universally ruled out.

Usage:  python src/provenance_quality_checks.py [baseline_hashes.json]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, R006, RUN_DIR, TAB, write_csv  # noqa: E402

DOCS = sorted(RUN_DIR.glob("*.md"))
BASELINE = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# A quoted span is how these documents FORBID a phrasing: they quote the banned
# wording in order to rule it out. Such spans must be removed before scanning,
# or the prohibition itself gets flagged. Spans can cross line breaks, so the
# substitution runs on the whole document before it is split into lines.
QUOTED = re.compile(r'"[^"]*"', re.S)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def doc_lines(strip_quoted=False):
    """Yield (filename, line number, text)."""
    for d in DOCS:
        text = d.read_text(encoding="utf-8")
        if strip_quoted:
            # keep the line structure so numbers stay meaningful
            text = QUOTED.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
        for i, line in enumerate(text.splitlines(), 1):
            yield d.name, i, line


def doc_units(strip_quoted=False):
    """Yield (filename, first line number, unwrapped paragraph or table row).

    These documents are hard-wrapped Markdown, so a statement and the label
    that qualifies it routinely sit on different physical lines. Scanning line
    by line therefore produces false positives - for instance a sentence
    naming the frozen noiseless sweep whose POST_HOC label wraps onto the next
    line. Checks run on unwrapped paragraphs instead; table rows are already
    one line each and are yielded individually.
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
    results = []

    def add(qid, desc, hits, detail=""):
        results.append({"check": qid, "description": desc,
                        "status": "PASS" if not hits else "FAIL",
                        "violations": len(hits),
                        "detail": detail or "; ".join(
                            f"{a}:{b}" for a, b, _ in hits[:4])})

    # ---- Q1: additive causal decomposition wording ----------------------
    pats = [
        r"\b\d{1,3}\s*%\s*of the (error|failure) was\b",
        r"distortion caused (exactly )?\w+\s*%",
        r"\bcaused exactly\b",
        r"\bexplains \d{1,3}\s*% of the (error|failure)\b",
        r"weak perspective caused the",
        r"reference[- ]3D error caused",
        r"\bdistortion is the whole\b",
        r"\bcontributes nothing\b",
    ]
    hits = [(d, i, l) for d, i, l in doc_units(strip_quoted=True)
            if any(re.search(p, l, re.I) for p in pats)]
    add("Q1", "no additive causal decomposition wording", hits)

    # ---- Q2 / Q3: post-hoc diagnostics never called pre-registered ------
    def posthoc_mislabel(topic_pat):
        """Flag a line only if it calls the post-hoc analysis pre-registered.

        A line may legitimately contain "frozen" or "pre-registered" while
        referring to a DIFFERENT, genuinely frozen analysis - for instance the
        noiseless distance sweep named alongside its post-hoc noise column. A
        line that also carries an explicit post-hoc label is correct, not a
        violation.
        """
        out = []
        for d, i, l in doc_units():
            if not re.search(topic_pat, l, re.I):
                continue
            if re.search(r"POST[_ -]?HOC", l, re.I):
                continue
            if re.search(r"not (pre-?registered|frozen|confirmatory)|was not|"
                         r"were not|rather than|added after", l, re.I):
                continue
            if re.search(r"pre-?registered|frozen|confirmatory", l, re.I):
                out.append((d, i, l))
        return out

    add("Q2", "distance x noise diagnostic not labelled pre-registered",
        posthoc_mislabel(r"distance\s*[x×]\s*1?\s*px|distance \+ ?noise|"
                         r"1 px noise (sweep|column)"))
    add("Q3", "reference-3D perturbation not labelled pre-registered",
        posthoc_mislabel(r"reference-3D (perturbation|noise) sweep"))
    add("Q3b", "REAL_REGIME_MATCHED not labelled pre-registered",
        posthoc_mislabel(r"REAL_REGIME_MATCHED|4\.12\s*[x×]"))

    # ---- Q4: B-vs-C cutoff not described as pre-registered --------------
    frozen = json.loads(
        (MANIFESTS / "cam_exp_0061_validation_spec_v1.json").read_text(
            encoding="utf-8"))
    blob = json.dumps(frozen).lower()
    cutoff_in_spec = any(k in blob for k in
                         ("r5_usable", "restored", "identifiable >=", "<= 10"))
    hits = []
    if not cutoff_in_spec:
        for d, i, l in doc_units():
            if re.search(r"pre-?registered verdict|"
                         r"verdict b was pre-?registered|"
                         r"pre-?registered b-vs-c", l, re.I):
                hits.append((d, i, l))
    add("Q4", "B-vs-C cutoff not presented as pre-registered", hits,
        f"cutoff_present_in_frozen_spec={cutoff_in_spec}")

    txt = "\n".join(d.read_text(encoding="utf-8") for d in DOCS)
    add("Q4b", "POST_HOC_INTERPRETIVE_VERDICT label present",
        [] if "POST_HOC_INTERPRETIVE_VERDICT" in txt
        else [("docs", 0, "missing")])

    # ---- Q5-Q8: numerical integrity against the pre-edit baseline -------
    if BASELINE and BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        groups = [
            ("Q6_CAM006_raw", "CAM006_raw", R006 / "results" / "raw"),
            ("Q6_CAM006_summary", "CAM006_summary",
             R006 / "results" / "summary"),
            ("Q6_CAM006_figures", "CAM006_figures", R006 / "figures"),
            ("Q7_CAM0061_raw", "CAM0061_raw", RUN_DIR / "results" / "raw"),
            ("Q7_CAM0061_summary", "CAM0061_summary",
             RUN_DIR / "results" / "summary"),
            ("Q7_CAM0061_figures", "CAM0061_figures", RUN_DIR / "figures"),
            ("Q5_CAM0061_tables", "CAM0061_tables", RUN_DIR / "tables"),
        ]
        for qid, key, root in groups:
            old = base[key]
            new = {str(p.relative_to(root)).replace("\\", "/"): sha(p)
                   for p in sorted(root.rglob("*")) if p.is_file()}
            changed = [k for k in old if k not in new or new[k] != old[k]]
            added = [k for k in new if k not in old]
            # a NEW documentation table is allowed; a CHANGED number is not
            add(qid, f"{root.parent.name}/{root.name}: no pre-existing file "
                     f"changed", [(qid, 0, k) for k in changed],
                f"changed={len(changed)} added={len(added)}"
                + (f" ({', '.join(added)})" if added else ""))
        old = base["manifests"]
        new = {p.name: sha(p) for p in sorted(MANIFESTS.glob("cam_exp_006*"))}
        changed = [k for k in old if k not in new or new[k] != old[k]]
        add("Q8", "frozen manifests byte-identical",
            [("manifests", 0, k) for k in changed], f"changed={len(changed)}")
    else:
        add("Q5-Q8", "numerical integrity (baseline not supplied)",
            [("baseline", 0, "missing")], "pass baseline json as argv[1]")

    # ---- Q9: CAM-007 not universally ruled out --------------------------
    bad = [(d, i, l) for d, i, l in doc_units(strip_quoted=True)
           if re.search(r"cam-?007 (is )?(impossible|ruled out)|"
                        r"predicted hand (is )?useless|"
                        r"learned hand cue cannot work|"
                        r"all learned hand-derived .* (impossible|ruled out)",
                        l, re.I)
           and not re.search(r"not ruled out|does not|do not|never", l, re.I)]
    if "REDESIGN" not in txt:
        bad.append(("docs", 0, "REDESIGN recommendation missing"))
    add("Q9", "CAM-007 = REDESIGN, not universally ruled out", bad)

    TAB.mkdir(parents=True, exist_ok=True)
    write_csv(TAB / "provenance_quality_checks.csv", results)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    for r in results:
        mark = "ok  " if r["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {r['check']:20s} {r['description']}"
              + (f"   -> {r['detail']}" if r["detail"] else ""))
    print(f"\n{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
