"""Automated checks on the report-preparation package itself.

These are the checks that would otherwise be done by eye and forgotten: do the
sources exist, do the figures match their data files, and has a dangerous phrase
crept in.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, PKG, REPO, RUNS, read_csv, read_json  # noqa: E402

FAIL, WARN, OK = [], [], []

# the checker's own previous output quotes every phrase it looks for
SELF = {"quality_check_report.json"}


def scannable_files():
    for p in PKG.rglob("*"):
        if (p.is_file() and p.suffix.lower() in (".md", ".csv", ".json")
                and p.name not in SELF):
            yield p


def check(cond, name, detail=""):
    (OK if cond else FAIL).append(f"{name}{(' - ' + detail) if detail else ''}")


def warn(cond, name, detail=""):
    (OK if cond else WARN).append(f"{name}{(' - ' + detail) if detail else ''}")


# A --------------------------------------------------- every source file exists
def check_sources():
    num = read_json(PKG / "report_numbers.json")["numbers"]
    missing = sorted({v["source_file"] for v in num.values()
                      if not (REPO / v["source_file"]).exists()})
    check(not missing, "A. every report_numbers source file exists",
          f"missing: {missing}")


# B ------------------------------- main results agree with the canonical source
def check_main_results():
    num = read_json(PKG / "report_numbers.json")["numbers"]
    rows = read_csv(PKG / "tables" / "report_main_results.csv")
    # spot-check the numbers that appear as bare decimals in the value strings
    probes = {
        "cam0013_recon_vs_provided3d_same_hand_median_mm": None,
        "cam002_focal_5pct_root_shift_median_mm": None,
        "cam003_anycalib_pinhole_focal_err_median": None,
        "cam0031_anycalib_gen_radial_focal_err_median": None,
        "cam0041_anycalib_gen_N64_focal_err_median": None,
        "cam004_anycalib_gen_bias_fraction_pct": None,
    }
    blob = " ".join(r["value"] for r in rows)
    for k in probes:
        val = num[k]["value"]
        found = str(val) in blob or f"{val:g}" in blob
        check(found, f"B. {k} appears verbatim in report_main_results",
              f"value {val} not found")


# C ------------------------------ figure CSVs agree with the canonical source
def check_figure_data():
    d = PKG / "figures" / "main" / "data"
    fig04 = read_csv(d / "Fig04.csv")
    src = read_csv(RUNS / "CAM-EXP-002_camera_focal_sensitivity" / "results"
                   / "summary" / "focal_sensitivity_summary.csv")
    bad = []
    for r in fig04:
        s = next(x for x in src
                 if float(x["focal_error_percent"]) == float(r["focal_error_percent"]))
        if float(s["incremental_signed_dz_median_mm"]) != float(r["median_signed_dz_mm"]):
            bad.append(r["focal_error_percent"])
    check(not bad, "C1. Fig04.csv matches CAM-EXP-002 summary", f"rows {bad}")

    fig06 = read_csv(d / "Fig06.csv")
    src6 = read_csv(RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit" / "results"
                    / "summary" / "frame_count_8_16_32_64.csv")
    bad = []
    for r in fig06:
        if int(r["N"]) < 8:
            continue
        s = [x for x in src6 if x["estimator"] == r["estimator"]
             and int(x["N"]) == int(r["N"])]
        # Fig06.csv is rounded to 3 decimals for readability
        if s and abs(float(s[0]["median_rel_err_pct"])
                     - float(r["median_rel_err_pct"])) > 5e-4:
            bad.append((r["estimator"], r["N"]))
    check(not bad, "C2. Fig06.csv N>=8 matches the CAM-EXP-004.1 summary "
                   "(same run, same aggregation)", f"{bad}")

    for n in ("Fig01", "Fig03", "Fig04", "Fig05", "Fig06", "Fig07"):
        check((d / f"{n}.csv").exists(), f"C3. {n}.csv exists")


# D --------------------------------------- no "n = 1400" as an independent count
def check_pseudo_replication():   # noqa: E302  (uses _scannable_text, defined below)
    pat = re.compile(r"(n\s*=\s*1400|1400\s+independent|N\s*=\s*1400)", re.I)
    hits = []
    for p in scannable_files():
        # the same exclusion as the wording sweep: a column or a "do not write"
        # block whose job is to quote the bad phrasing is not a violation
        txt = _scannable_text(p)
        for m in pat.finditer(txt):
            ctx = txt[max(0, m.start() - 200):m.end() + 120]
            # our own warnings against this phrasing are not violations
            if re.search(r"\bnot\b|never|repeated observation|do not write",
                         ctx, re.I):
                continue
            hits.append(f"{p.relative_to(PKG)}: {ctx}")
    check(not hits, "D. no '1400' presented as an independent sample count",
          "; ".join(hits[:3]))


# E -------------------------------------------------- dangerous wording sweep
DANGEROUS = {
    r"\bGT\s*2D\b": "call it provided 2D annotation",
    r"ground[- ]truth 3D|ground truth 3D": "call it provided 3D",
    r"official sentinel": "call it an observed invalid all-zero pattern",
    r"\bfinal method\b|\bour method\b|proposed method": "E2 is a frozen "
                                                        "exploratory baseline",
    r"8 frames are always enough": "state the evaluated range instead",
    r"full camera calibration sensitivity": "CAM-EXP-002 is focal-only",
}
ALLOWED_CONTEXT = ("wording_to_avoid", "Do not write", "**Do not write:**",
                   "avoid", "never", "NOT ", "not a", "is not")


AVOID_COLUMNS = ("wording_to_avoid", "what_to_avoid")


def _scannable_text(p: Path) -> str:
    """Text that is ASSERTED, excluding fields whose job is to quote bad wording."""
    if p.suffix.lower() == ".csv":
        try:
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
        except Exception:
            return p.read_text(encoding="utf-8", errors="ignore")
        return chr(10).join(str(v) for r in rows for k, v in r.items()
                            if k not in AVOID_COLUMNS)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() == ".md":
        # drop the "wording to avoid" column of a rendered table row
        out = []
        for line in txt.splitlines():
            if line.count("|") >= 6 and "wording" not in line.lower():
                cells = line.split("|")
                out.append("|".join(c for i, c in enumerate(cells)
                                    if i != 7))
            else:
                out.append(line)
        return chr(10).join(out)
    return txt


def check_wording():
    hits = []
    for p in scannable_files():
        txt = _scannable_text(p)
        for pat, advice in DANGEROUS.items():
            for m in re.finditer(pat, txt, re.I):
                ctx = txt[max(0, m.start() - 160):m.end() + 60]
                if any(a.lower() in ctx.lower() for a in ALLOWED_CONTEXT):
                    continue    # it is being quoted as a phrase to avoid
                hits.append(f"{p.relative_to(PKG)}: '{m.group(0)}' -> {advice}")
    check(not hits, "E. no dangerous phrasing used assertively",
          "; ".join(hits[:5]))


# F ------------------------------------------ E2 never quoted as 6.14 % alone
def check_e2():
    bad = []
    for p in scannable_files():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"6\.14", txt):
            window = txt[max(0, m.start() - 400):m.end() + 400]
            if "6.46" in window or "5.73" in window or "selection" in window.lower():
                continue
            bad.append(f"{p.relative_to(PKG)}: {window[350:470]}")
    check(not bad, "F. 6.14 % never appears without its re-run / LOSO context",
          "; ".join(bad[:3]))


# G --------------------------------- GeoCalib numbers carry the reproducibility note
def check_geocalib_caveat():
    files = ["tables/report_main_results.md", "tables/claim_evidence_ledger.md",
             "tables/report_limitations.md",
             "source_map/report_wording_guardrails.md", "evidence_index.md"]
    missing = []
    for f in files:
        p = PKG / f
        if not p.exists():
            missing.append(f + " (file absent)")
            continue
        t = p.read_text(encoding="utf-8", errors="ignore").lower()
        if "geocalib" in t and not any(k in t for k in
                                       ("run-to-run", "run to run", "0.32",
                                        "0.35", "nondetermin", "실행 간")):
            missing.append(f)
    check(not missing, "G. every file mentioning GeoCalib carries the "
                       "reproducibility caveat", f"{missing}")


# H ------------------------------------------- AnyCam provenance is consistent
def check_anycam():
    commit = "e609cc8a9e4ee8f78cf2ce39ebeb86b35e82d10d"
    man = read_json(MANIFESTS / "external_model_provenance_v1.json")
    a = man["models"]["AnyCam"]
    check(a.get("exact_commit") == commit,
          "H1. manifest records the AnyCam commit", str(a.get("exact_commit")))
    check(a.get("execution_status", {}).get("END_TO_END_INFERENCE")
          == "BLOCKED_ON_WINDOWS",
          "H2. manifest records that end-to-end inference did not run")
    tab = read_csv(RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit" / "tables"
                   / "external_model_provenance.csv")
    row = next(r for r in tab if r["model"] == "AnyCam")
    check(row["exact_commit"] == commit,
          "H3. the run table agrees with the manifest", row["exact_commit"])
    stale = []
    for p in list(RUNS.rglob("*.json")) + list(RUNS.rglob("*.csv")) + \
            list(RUNS.rglob("*.md")) + list(MANIFESTS.rglob("*.json")):
        if not p.is_file():
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        if "AnyCam" not in t or "NOT_INSTALLED" not in t:
            continue
        # a NOT_INSTALLED inside the recorded correction history is the audit
        # trail, not a stale value
        if p.name == "external_model_provenance_v1.json":
            man2 = json.loads(t)
            if "NOT_INSTALLED" in json.dumps(man2.get("models", {})):
                stale.append(str(p.relative_to(REPO)))
            continue
        stale.append(str(p.relative_to(REPO)))
    check(not stale, "H4. no file still records AnyCam as NOT_INSTALLED",
          f"{stale}")


# I ----------------------------------- historical raw results were not modified
def check_no_raw_edits():
    import subprocess
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    touched = [l[3:].strip() for l in out.splitlines() if l.strip()]
    raw = [t for t in touched
           if "/results/raw/" in t or "/results/summary/" in t]
    check(not raw, "I. no historical raw/summary result file was modified",
          f"{raw}")


def main() -> None:
    check_sources()
    check_main_results()
    check_figure_data()
    check_pseudo_replication()
    check_wording()
    check_e2()
    check_geocalib_caveat()
    check_anycam()
    check_no_raw_edits()

    print(f"PASS  {len(OK)}")
    for w in WARN:
        print(f"WARN  {w}")
    for f in FAIL:
        print(f"FAIL  {f}")
    (PKG / "quality_check_report.json").write_text(json.dumps(
        {"passed": OK, "warnings": WARN, "failed": FAIL,
         "status": "PASS" if not FAIL else "FAIL"}, indent=2), encoding="utf-8")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
