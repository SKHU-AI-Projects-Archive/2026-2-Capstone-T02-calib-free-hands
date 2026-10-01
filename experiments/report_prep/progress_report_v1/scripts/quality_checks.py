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
        # In a rendered table, the column whose job is to quote bad wording is
        # not an assertion. Find it from the header row and drop that cell from
        # every row of the table.
        out, drop = [], None
        for line in txt.splitlines():
            cells = line.split("|")
            if len(cells) < 4:
                out.append(line)
                drop = None
                continue
            header = [c.strip().lower() for c in cells]
            if any(h in ("wording to avoid", "wording_to_avoid",
                         "what to avoid") for h in header):
                drop = next(i for i, h in enumerate(header)
                            if h in ("wording to avoid", "wording_to_avoid",
                                     "what to avoid"))
                out.append(line)
                continue
            if drop is not None:
                # Cells can contain escaped pipes, so the columns of a data row
                # cannot be split reliably. The same content is scanned from the
                # CSV twin of this table with the avoid column removed, so the
                # whole row is skipped here rather than mis-sliced.
                continue
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


# J ------------------------- CAM-EXP-002: pool and evaluated sample are distinct
def check_cam002_usage():
    num = read_json(PKG / "report_numbers.json")["numbers"]
    usage = read_csv(PKG / "tables" / "report_dataset_usage.csv")
    row = next(r for r in usage if r["experiment"] == "CAM-EXP-002")

    pool = num["gigahands_bimanual_clean_frames"]["value"]
    attempted = num["cam002_frames_attempted"]["value"]
    hands = num["cam002_hands_evaluated"]["value"]

    check(str(row.get("eligible_pool_frames")) == str(pool),
          "J1. CAM-EXP-002 row records the eligible pool separately",
          f"got {row.get('eligible_pool_frames')!r}")
    check(str(row.get("actual_frames_attempted")) == str(attempted)
          and str(row.get("actual_hands_evaluated")) == str(hands),
          "J2. CAM-EXP-002 row records the actual evaluated sample",
          f"attempted {row.get('actual_frames_attempted')!r}, "
          f"hands {row.get('actual_hands_evaluated')!r}")
    check(str(row.get("n_frames_input")) != str(pool),
          "J3. the eligible pool is NOT presented as the experiment's input "
          "frame count", f"n_frames_input = {row.get('n_frames_input')!r}")
    # the summary tables' own n_hands must agree with the raw per-hand file
    summ = read_csv(RUNS / "CAM-EXP-002_camera_focal_sensitivity" / "results"
                    / "summary" / "baseline_vs_gt_focal.csv")
    check(all(int(r["n_hands"]) == hands for r in summ),
          "J4. evaluated hands match the CAM-EXP-002 summary tables",
          f"raw {hands} vs summary {[r['n_hands'] for r in summ]}")

    # no artifact may pair the pool size with wording that implies it was used
    bad = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"16,?413", t):
            ctx = t[max(0, m.start() - 600):m.end() + 600]
            if re.search(r"pool|draws? from|eligible|not what|not the",
                         ctx, re.I):
                continue
            bad.append(f"{p.relative_to(PKG)}: {ctx[180:320]}")
    check(not bad, "J5. the 16,413-frame pool is never presented as the "
                   "evaluated sample", "; ".join(bad[:3]))


# K ------------------------------ bias decomposition unit is the view, not the camera
def check_bias_unit():
    bad = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"98\.0?8?1?\d*\s*%|98\.1", t):
            ctx = t[max(0, m.start() - 300):m.end() + 300]
            if not re.search(r"physical[- ]camera bias|per-camera bias|camera bias",
                             ctx, re.I):
                continue
            # a sentence whose job is to forbid or to document the correction of
            # that phrasing is not an assertion of it
            if re.search(r"never|not a physical|wrong unit|implies|do not|"
                         r"corrected|avoid|instead of", ctx, re.I):
                continue
            bad.append(f"{p.relative_to(PKG)}: {ctx[260:400]}")
    check(not bad, "K1. the bias-decomposition figure is never described as a "
                   "physical-camera bias", "; ".join(bad[:3]))

    loose = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"per-camera bias|camera-specific bias", t, re.I):
            ctx = t[max(0, m.start() - 250):m.end() + 250]
            if re.search(r"do not|avoid|not a physical|never|wrong unit|"
                         r"implies|corrected|instead of|rather than", ctx, re.I):
                continue
            loose.append(f"{p.relative_to(PKG)}: {m.group(0)}")
    check(not loose, "K2. no bare 'per-camera bias' phrasing is asserted",
          "; ".join(loose[:5]))

    prin = (PKG / "tables" / "common_evaluation_principles.md").read_text(
        encoding="utf-8")
    check("decomposition" in prin.lower() and "resampling" in prin.lower(),
          "K3. the principles table separates the decomposition unit from the "
          "resampling unit")


# L ------------------------------ Experiment 1 tail is not claimed as fully explained
def check_exp1_tail():
    bad = []
    pats = [r"tail is caused by", r"all (large-error )?(cases|failures) (are|were) "
            r"explain", r"explained the (whole|entire) tail",
            r"identified the cause of all"]
    for p in scannable_files():
        t = _scannable_text(p)
        for pat in pats:
            for m in re.finditer(pat, t, re.I):
                bad.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not bad, "L1. the Experiment 1 tail is never claimed as fully "
                   "explained", "; ".join(bad[:5]))

    num = read_json(PKG / "report_numbers.json")["numbers"]
    check("cam0013_identity_unresolved_insufficient_geometry_n" in num,
          "L2. the unresolved-geometry count is in report_numbers")
    ei = (PKG / "evidence_index.md").read_text(encoding="utf-8")
    check("UNRESOLVED_INSUFFICIENT_GEOMETRY" in ei and "BAD_2D_GEOMETRY" in ei,
          "L3. the evidence index names the unexplained classes")


# M ------------------------------------ Fig03 / Fig04 distinguish pool from sample
def check_figure_labels():
    d = PKG / "figures" / "main" / "data"
    f3 = {r["quantity"] for r in read_csv(d / "Fig03.csv")}
    check({"cam002_frames_attempted", "cam002_hands_evaluated"} <= f3,
          "M1. Fig03 data carries the actual evaluated sample, not only the pool")
    f4 = read_csv(d / "Fig04.csv")
    check(all("evaluated_sample_note" in r and "eligible pool" in
              r["evaluated_sample_note"] for r in f4),
          "M2. Fig04 data records that the sample was drawn from a pool")


# N ------------------------------- focal terminology and independence wording
def check_focal_terminology():
    bad = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"physical focal|actual focal|true focal|"
                             r"physical_focal", t, re.I):
            ctx = t[max(0, m.start() - 300):m.end() + 300]
            # our own prohibitions and the correction history are not assertions
            if re.search(r"do not|never|wrong term|wrong for|avoid|"
                         r"corrected|instead|rather than|not an optical|"
                         r"was the wrong|not a physical|it is not|must not", ctx, re.I):
                continue
            bad.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not bad, "N1. 'physical focal' is not used as assertive terminology",
          "; ".join(bad[:5]))

    num = read_json(PKG / "report_numbers.json")["numbers"]
    check("cam002_gigahands_physical_focal_median_px" not in num
          and "cam002_gt_effective_focal_median_px" in num,
          "N1b. the focal key is renamed to GT_EFFECTIVE_FOCAL with no alias")

    # every artifact that puts the two focal numbers together must say what each
    # one is
    missing = []
    for p in scannable_files():
        t = _scannable_text(p)
        if "5000" not in t and "5,000" not in t:
            continue
        if "922" not in t:
            continue
        has_pipeline = re.search(r"PIPELINE_BASELINE_FOCAL|focal convention|"
                                 r"virtual focal", t, re.I)
        has_reference = re.search(r"GT_EFFECTIVE_FOCAL|dataset-provided reference "
                                 r"focal|dataset-provided camera intrinsic", t, re.I)
        if not (has_pipeline and has_reference):
            missing.append(str(p.relative_to(PKG)))
    check(not missing, "N2. wherever 5000 px and 922 px appear together, both "
                       "semantic roles are named", f"{missing}")

    unphysical = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"wrong focal|incorrect focal|unphysical focal|"
                             r"pipeline is broken|physical mismatch", t, re.I):
            ctx = t[max(0, m.start() - 250):m.end() + 250]
            if re.search(r"do not|never|avoid|not claim|it never claimed",
                         ctx, re.I):
                continue
            unphysical.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not unphysical, "N2b. 5000 px is never asserted to be a wrong or "
                          "unphysical camera intrinsic", "; ".join(unphysical[:5]))


def check_e2_independence():
    bad = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"independent (re-?run|run|replication|test|"
                             r"validation|dataset)", t, re.I):
            ctx = t[max(0, m.start() - 300):m.end() + 300]
            if re.search(r"do not|never|not an independent|is not|needs the "
                         r"sealed|rejected|avoid|genuinely independent|"
                         r"nothing here is|is NOT an", ctx, re.I):
                continue
            bad.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not bad, "N3. E2's 6.46 % is never called an independent run or test",
          "; ".join(bad[:5]))

    missing = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"6\.46", t):
            ctx = t[max(0, m.start() - 500):m.end() + 500]
            if re.search(r"same[- ](175[- ])?(benchmark[- ])?views|same-data|"
                         r"same data|separate (execution|rerun)|same-views|"
                         r"benchmark executions", ctx, re.I):
                continue
            missing.append(f"{p.relative_to(PKG)}: {ctx[460:620]}")
    check(not missing, "N4. 6.46 % always carries the 'same benchmark views' "
                       "context", "; ".join(missing[:3]))


def check_tolerance_wording():
    bad = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"[+±]/?-?\s*0\.3\d\s*(pp|percentage)|"
                             r"run-to-run tolerance|tolerance of about", t, re.I):
            ctx = t[max(0, m.start() - 300):m.end() + 300]
            if re.search(r"do not|never|avoid|not a statistical|observed "
                         r"difference|corrected", ctx, re.I):
                continue
            bad.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not bad, "N5. no '+/-0.3x pp tolerance' is stated as an uncertainty "
                   "interval", "; ".join(bad[:5]))

    # the two GeoCalib measurements must be recorded separately
    num = read_json(PKG / "report_numbers.json")["numbers"]
    has_repeat = all(f"determinism_geocalib_distorted_radial_{k}" in num
                     for k in ("median_spread_pct", "p90_spread_pct",
                               "max_spread_pct"))
    has_runs = ("e2_or_model_rerun_geocalib_distorted_cam004_median" in num
                and "e2_or_model_rerun_geocalib_distorted_cam0041_median" in num)
    check(has_repeat and has_runs,
          "N6. the repeat-spread diagnostic and the two-execution difference "
          "are stored as separate numbers",
          f"repeat={has_repeat} runs={has_runs}")


# O -------------------- CAM-EXP-002 magnitude and camera-vs-hand-model wording
def check_cam002_magnitude():
    bad = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"orders of magnitude", t, re.I):
            ctx = t[max(0, m.start() - 300):m.end() + 300]
            # correction-history prose quotes the retired phrase in order to
            # record that it was retired; that is not an assertion of it
            if re.search(r"do not|never|avoid|would be|wording_to_avoid|"
                         r"almost three times|recorded|forbids|now reads|"
                         r"retired|is ~100|phrase implies", ctx, re.I):
                continue
            bad.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not bad, "O1. the CAM-EXP-002 improvement is never called 'orders of "
                   "magnitude'", "; ".join(bad[:5]))

    dicho = []
    for p in scannable_files():
        t = _scannable_text(p)
        for m in re.finditer(r"camera,? not the hand model|camera,? not hand|"
                             r"first thing to fix|hand model is not the problem",
                             t, re.I):
            ctx = t[max(0, m.start() - 300):m.end() + 300]
            if re.search(r"do not|never|avoid|wording_to_avoid|nothing in this|"
                         r"exonerat|now reads|retired|said|recorded",
                         ctx, re.I):
                continue
            dicho.append(f"{p.relative_to(PKG)}: '{m.group(0)}'")
    check(not dicho, "O2. no dichotomy that clears the hand-pose model",
          "; ".join(dicho[:5]))

    num = read_json(PKG / "report_numbers.json")["numbers"]
    base = num["cam002_baseline_root_error_median_mm"]["value"]
    ref = num["cam002_gt_effective_focal_root_error_median_mm"]["value"]
    factor = num["cam002_root_error_reduction_factor"]["value"]
    check(abs(factor - base / ref) < 0.01,
          "O3a. the stored reduction factor equals the ratio of the two medians",
          f"{factor} vs {base / ref:.4f}")

    missing = []
    for p in scannable_files():
        t = _scannable_text(p)
        # any artifact quoting the factor must quote it consistently
        for m in re.finditer(r"(\d+\.?\d*)\s*x lower|~\s*(\d+\.?\d*)x", t):
            ctx = t[max(0, m.start() - 200):m.end() + 200]
            # only factors describing the CAM-EXP-002 focal comparison, keyed on
            # its two medians rather than on the generic word "focal", which
            # also appears in neighbouring rows of the same CSV
            if not re.search(r"2881|78\.54", ctx):
                continue
            val = float(m.group(1) or m.group(2))
            if abs(val - factor) > 0.5:
                missing.append(f"{p.relative_to(PKG)}: {val} vs {factor}")
    check(not missing, "O3b. every quoted improvement factor matches the "
                       "canonical one", "; ".join(missing[:3]))

    # the residual under the reference focal must be stated where the claim is
    res = []
    for f in ("evidence_index.md", "tables/hypothesis_result_evidence.csv",
              "tables/claim_evidence_ledger.csv"):
        t = _scannable_text(PKG / f)
        if "78.54" in t and "34.404" not in t:
            res.append(f)
    check(not res, "O4. wherever the reference-focal result is claimed, the "
                   "residual error is stated too", f"{res}")


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
    check_cam002_usage()
    check_bias_unit()
    check_exp1_tail()
    check_figure_labels()
    check_focal_terminology()
    check_e2_independence()
    check_tolerance_wording()
    check_cam002_magnitude()

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
