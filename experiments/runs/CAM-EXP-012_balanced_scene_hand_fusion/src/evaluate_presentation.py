"""PHASE G/H — open the TEST reference focal and build the presentation tables.

This is the FIRST script permitted to read the test reference focal. It refuses
to run unless the predictions have already been frozen and hashed with
`test_reference_focal_opened = false`.

Wrong-bone is deferred, so `CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED` stays
PENDING regardless of what the numbers say.
"""
from __future__ import annotations

import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import (DISPLAY_SEQUENCE, MANIFESTS, MODELS, RAW,  # noqa
                         REPO, SCENE_SRC_64, SUM, TAB, fnum, med,
                         paired_cluster_bootstrap, pct,
                         paired_cluster_bootstrap as pcb, read_csv,
                         scalar_focal, sha256, write_csv, write_json)

FROZEN = RAW / "frozen_test_predictions.csv.gz"
TIE_TOL = 1e-9


def reference_focal_all():
    out = {}
    for r in read_csv(SCENE_SRC_64):
        k = (r["sequence"], r["camera"])
        if k in out:
            continue
        fx, fy = fnum(r.get("gt_fx")), fnum(r.get("gt_fy"))
        if np.isfinite(fx) and np.isfinite(fy):
            out[k] = scalar_focal(fx, fy)
    return out


def main():
    freeze = SUM / "presentation_prediction_freeze.json"
    if not freeze.exists():
        raise SystemExit("predictions must be frozen before the focal is opened")
    fz = read_json_safe(freeze)
    if fz.get("test_reference_focal_opened") is not False:
        print("warning: freeze record does not assert a closed focal")
    if sha256(FROZEN) != fz["frozen_predictions_sha256"]:
        raise SystemExit("frozen predictions changed since the freeze record")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    _op = RAW / "REFERENCE_FOCAL_OPENED.txt"
    if not _op.exists():
        _op.write_text(
        "REFERENCE FOCAL OPENED\ntimestamp_utc: %s\ngit_head: %s\n"
        "frozen_predictions_sha256: %s\nfreeze_record_sha256: %s\n"
        "wrong_bone_control: PENDING (deferred, see amendment)\n"
        "After this point no prediction may be regenerated.\n"
        % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), head,
               sha256(FROZEN), sha256(freeze)), encoding="utf-8")

    ref = reference_focal_all()
    rows = read_csv(FROZEN)
    scored = []
    for r in rows:
        k = (r["sequence"], r["camera"])
        g, f = ref.get(k), fnum(r["f_pred"])
        if g is None or not np.isfinite(g):
            continue
        scored.append({**r, "f_ref": g,
                       "error_pct": (100.0 * abs(f - g) / g
                                     if np.isfinite(f) else np.nan)})
    write_csv(RAW / "unit_results.csv.gz", scored)

    idx = {(r["sequence"], r["camera"], r["model"], r["condition"]): r
           for r in scored}
    CONDS = ["SCENE_ONLY", "SCENE_PLUS_CORRECT_HAND",
             "SCENE_PLUS_SHUFFLED_HAND", "CORRECT_HAND_LAMBDA1"]
    have = defaultdict(set)
    for r in scored:
        if np.isfinite(fnum(r["error_pct"])):
            have[(r["model"], r["condition"])].add(
                (r["sequence"], r["camera"]))
    need = [(m, c) for m in MODELS
            for c in ("SCENE_ONLY", "SCENE_PLUS_CORRECT_HAND",
                      "SCENE_PLUS_SHUFFLED_HAND")]
    common = set.intersection(*[have[k] for k in need]) if all(
        have[k] for k in need) else set()
    common = sorted(common)
    write_csv(TAB / "presentation_paired_set.csv",
              [{"sequence": s, "camera": c} for s, c in common])

    main_tbl, ctrl_tbl, mov_tbl, verdicts, paired_rows = [], [], [], [], []
    for m in MODELS:
        E = {c: [fnum(idx[(s, cam, m, c)]["error_pct"])
                 for (s, cam) in common if (s, cam, m, c) in idx]
             for c in CONDS if all((s, cam, m, c) in idx
                                   for (s, cam) in common)}
        off, on = E["SCENE_ONLY"], E["SCENE_PLUS_CORRECT_HAND"]
        sh = E.get("SCENE_PLUS_SHUFFLED_HAND", [])
        gains = [a - b for a, b in zip(off, on)]
        cams = [c for (_s, c) in common]
        g_med, lo, hi = pcb(gains, cams)
        m_off, m_on = med(off), med(on)
        win = 100.0 * np.mean([b < a - TIE_TOL for a, b in zip(off, on)])
        loss = 100.0 * np.mean([b > a + TIE_TOL for a, b in zip(off, on)])
        tie = 100.0 - win - loss
        main_tbl.append({
            "Model": MODELS[m]["display"],
            "Scene_only_median_error_pct": round(m_off, 3),
            "Scene_plus_correct_hand_median_error_pct": round(m_on, 3),
            "Gain_of_medians_pp": round(m_off - m_on, 3),
            "Paired_median_gain_pp": round(g_med, 3),
            "Paired_mean_gain_pp": round(float(np.mean(gains)), 3),
            "Relative_gain_pct": round(100.0 * (m_off - m_on) / m_off, 2)
            if m_off else "",
            "Win_rate_pct": round(win, 1),
            "Tie_rate_pct": round(tie, 1), "Loss_rate_pct": round(loss, 1),
            "CI95_low": round(lo, 3), "CI95_high": round(hi, 3),
            "N_videos": len(common)})
        paired_rows.append({"model": m, "median_gain_pp": g_med,
                            "mean_gain_pp": float(np.mean(gains)),
                            "ci95_low": lo, "ci95_high": hi,
                            "win_pct": win, "tie_pct": tie,
                            "loss_pct": loss, "n": len(common)})

        adv = ([a - b for a, b in zip(sh, on)] if sh else [])
        a_med, a_lo, a_hi = pcb(adv, cams) if adv else (np.nan,) * 3
        ctrl_tbl.append({
            "Model": MODELS[m]["display"],
            "Scene": round(m_off, 3), "Correct_Hand": round(m_on, 3),
            "Shuffled_Hand": round(med(sh), 3) if sh else "",
            "Wrong_Bone": "PENDING",
            "Correct_vs_Shuffled_advantage_pp": round(a_med, 3)
            if np.isfinite(a_med) else "",
            "CI95_low": round(a_lo, 3) if np.isfinite(a_lo) else "",
            "CI95_high": round(a_hi, 3) if np.isfinite(a_hi) else "",
            "N": len(common)})

        sh_pct = []
        for (s, cam) in common:
            a = idx[(s, cam, m, "SCENE_ONLY")]
            b = idx[(s, cam, m, "SCENE_PLUS_CORRECT_HAND")]
            f0, f1 = fnum(a["f_pred"]), fnum(b["f_pred"])
            if np.isfinite(f0) and f0 > 0 and np.isfinite(f1):
                sh_pct.append(100.0 * abs(f1 - f0) / f0)
        sh_pct = np.array(sh_pct)
        mov_tbl.append({
            "Model": MODELS[m]["display"],
            "Median_shift_pct": round(float(np.median(sh_pct)), 4),
            "moved_gt0_pct": round(100.0 * np.mean(sh_pct > 1e-9), 1),
            "ge_0.25_pct": round(100.0 * np.mean(sh_pct >= 0.25), 1),
            "ge_0.5_pct": round(100.0 * np.mean(sh_pct >= 0.5), 1),
            "ge_1_pct": round(100.0 * np.mean(sh_pct >= 1.0), 1),
            "ge_2_pct": round(100.0 * np.mean(sh_pct >= 2.0), 1)})

        gain_ok = bool(m_on < m_off and g_med > 0 and lo > 0)
        shuf_ok = bool(np.isfinite(a_lo) and a_med > 0 and a_lo > 0)
        verdicts.append({
            "Model": MODELS[m]["display"],
            "HAND_ON_NUMERICAL_GAIN_SUPPORTED": gain_ok,
            "CORRECT_VS_SHUFFLED_SIGNAL_SUPPORTED": shuf_ok,
            "CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED": "PENDING_WRONG_BONE"})

    write_csv(TAB / "presentation_main_table_early.csv", main_tbl)
    write_csv(TAB / "presentation_control_table_early.csv", ctrl_tbl)
    write_csv(TAB / "focal_movement_table.csv", mov_tbl)
    write_csv(TAB / "presentation_verdicts_early.csv", verdicts)
    write_csv(SUM / "paired_gain_summary.csv", paired_rows)

    per_seq = []
    for seq in sorted({s for (s, _c) in common}):
        vs = [(s, c) for (s, c) in common if s == seq]
        for m in MODELS:
            o = [fnum(idx[(s, c, m, "SCENE_ONLY")]["error_pct"]) for s, c in vs]
            h = [fnum(idx[(s, c, m, "SCENE_PLUS_CORRECT_HAND")]["error_pct"])
                 for s, c in vs]
            per_seq.append({"Sequence": DISPLAY_SEQUENCE.get(seq, seq),
                            "Model": MODELS[m]["display"],
                            "Scene": round(med(o), 3),
                            "Correct": round(med(h), 3),
                            "Gain": round(med(o) - med(h), 3), "N": len(vs)})
    write_csv(TAB / "presentation_per_sequence_early.csv", per_seq)

    write_json(SUM / "cam012_presentation_verdict.json", {
        "status": "PRIMARY PRESENTATION ANALYSIS COMPLETE; "
                  "FULL CONTROL ANALYSIS PENDING",
        "cam012_complete": False,
        "wrong_bone_control": "PENDING",
        "n_paired_videos": len(common),
        "main_table": main_tbl, "control_table": ctrl_tbl,
        "movement_table": mov_tbl, "verdicts": verdicts,
        "paired": paired_rows,
    })

    print("\nPAIRED SET: %d videos\n" % len(common))
    print("%-20s %12s %14s %9s %9s %10s" % ("Model", "Scene only",
                                            "Scene+Hand", "Gain pp", "Win %",
                                            "CI95"))
    for r in main_tbl:
        print("%-20s %11.3f%% %13.3f%% %+9.3f %8.1f%%  [%.3f, %.3f]"
              % (r["Model"], r["Scene_only_median_error_pct"],
                 r["Scene_plus_correct_hand_median_error_pct"],
                 r["Paired_mean_gain_pp"], r["Win_rate_pct"], r["CI95_low"],
                 r["CI95_high"]))
    print("\n%-20s %9s %9s %9s %12s" % ("Model", "Scene", "Correct",
                                        "Shuffled", "Corr adv pp"))
    for r in ctrl_tbl:
        print("%-20s %9.3f %9.3f %9s %12s  CI[%s, %s]"
              % (r["Model"], r["Scene"], r["Correct_Hand"],
                 r["Shuffled_Hand"], r["Correct_vs_Shuffled_advantage_pp"],
                 r["CI95_low"], r["CI95_high"]))
    print("\n%-20s %8s %9s %8s %8s %8s" % ("Model", "med shift", ">0", ">=0.5",
                                           ">=1", ">=2"))
    for r in mov_tbl:
        print("%-20s %8.4f %8.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (r["Model"], r["Median_shift_pct"], r["moved_gt0_pct"],
                 r["ge_0.5_pct"], r["ge_1_pct"], r["ge_2_pct"]))
    print()
    for v in verdicts:
        print("%-20s gain=%s  vs_shuffled=%s  specific=%s"
              % (v["Model"], v["HAND_ON_NUMERICAL_GAIN_SUPPORTED"],
                 v["CORRECT_VS_SHUFFLED_SIGNAL_SUPPORTED"],
                 v["CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED"]))


def read_json_safe(p):
    import json
    return json.loads(Path(p).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
