"""PHASE F/G — open the TEST reference focal and produce the presentation tables.

First script permitted to read the test reference focal. It logs the opening
with the git HEAD and the hashes of the frozen predictions, the frame manifest
and the lambda selection. No prediction is regenerated afterwards.

PRIMARY metric: median relative focal error per sequence-camera video. One
video, one focal, one vote — frames are never counted as independent samples.
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
from c011_common import (DISPLAY_SEQUENCE, MANIFESTS, MODELS, RAW,  # noqa
                         REPO, SCENE_SRC_64, SUM, TAB, fnum, med,
                         paired_cluster_bootstrap, pct, read_csv,
                         scalar_focal, sha256, write_csv, write_json)

FROZEN = RAW / "final_six_condition_predictions.csv.gz"
PRIMARY_SET = "STRICT"


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


def open_log():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    txt = ("REFERENCE FOCAL OPENED\n"
           "timestamp_utc: %s\ngit_head: %s\n"
           "frozen_predictions_sha256: %s\n"
           "frame_manifest_sha256: %s\n"
           "lambda_selection_sha256: %s\n"
           "After this point no prediction may be regenerated.\n"
           % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), head,
              sha256(FROZEN),
              sha256(MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz"),
              sha256(RAW / "fold_lambda_selection.csv")))
    (RAW / "REFERENCE_FOCAL_OPENED.txt").write_text(txt, encoding="utf-8")
    print(txt)


def stats(errs):
    return {"n": len(errs), "median_error_pct": med(errs),
            "mean_error_pct": float(np.mean(errs)) if errs else np.nan,
            "p75": pct(errs, 75), "p90": pct(errs, 90),
            "within5_pct": 100.0 * np.mean([e <= 5 for e in errs])
            if errs else np.nan,
            "within10_pct": 100.0 * np.mean([e <= 10 for e in errs])
            if errs else np.nan}


def main():
    open_log()
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
                                     if np.isfinite(f) else np.nan),
                       "signed_error_pct": (100.0 * (f - g) / g
                                            if np.isfinite(f) else np.nan),
                       "signed_log_bias_pct":
                           (100.0 * np.log(f / g) if np.isfinite(f) and f > 0
                            else np.nan)})
    write_csv(RAW / "scored_predictions.csv.gz", scored)

    # ---------------- GLOBAL six-condition common video set
    have = defaultdict(set)
    for r in scored:
        if r["frame_set"] != PRIMARY_SET or not np.isfinite(r["error_pct"]):
            continue
        have[(r["model"], r["condition"])].add((r["sequence"], r["camera"]))
    need = [(m, c) for m in MODELS for c in ("SCENE_ONLY", "SCENE_PLUS_HAND")]
    common = set.intersection(*[have[k] for k in need]) if all(
        have[k] for k in need) else set()
    write_csv(TAB / "paired_unit_manifest.csv",
              [{"sequence": s, "camera": c} for s, c in sorted(common)])

    # ---------------- primary results
    idx = {(r["sequence"], r["camera"], r["model"], r["condition"],
            r["frame_set"]): r for r in scored}
    primary, main_tbl, paired_rows = [], [], []
    for m in MODELS:
        per = {}
        for cond in ("SCENE_ONLY", "SCENE_PLUS_HAND"):
            errs = [idx[(s, c, m, cond, PRIMARY_SET)]["error_pct"]
                    for (s, c) in sorted(common)]
            per[cond] = errs
            primary.append({"model": m, "display": MODELS[m]["display"],
                            "condition": cond, "frame_set": PRIMARY_SET,
                            **stats(errs)})
        off, on = per["SCENE_ONLY"], per["SCENE_PLUS_HAND"]
        gains = [a - b for a, b in zip(off, on)]
        cams = [c for (s, c) in sorted(common)]
        ptv, lo, hi = paired_cluster_bootstrap(gains, cams)
        m_off, m_on = med(off), med(on)
        win = 100.0 * np.mean([b < a for a, b in zip(off, on)]) if off else np.nan
        shift = med([abs(fnum(idx[(s, c, m, "SCENE_PLUS_HAND",
                                   PRIMARY_SET)]["f_pred"])
                         - fnum(idx[(s, c, m, "SCENE_PLUS_HAND",
                                     PRIMARY_SET)]["f_scene"]))
                     / max(fnum(idx[(s, c, m, "SCENE_PLUS_HAND",
                                     PRIMARY_SET)]["f_scene"]), 1e-9) * 100.0
                     for (s, c) in sorted(common)])
        paired_rows.append({
            "model": m, "display": MODELS[m]["display"],
            "n_videos": len(common),
            "scene_only_median_pct": m_off, "scene_plus_hand_median_pct": m_on,
            "gain_pp": m_off - m_on,
            "relative_reduction_pct": (100.0 * (m_off - m_on) / m_off
                                       if m_off else np.nan),
            "paired_median_gain_pp": ptv, "ci95_low": lo, "ci95_high": hi,
            "win_rate_pct": win,
            "median_hand_focal_shift_pct": shift})
        main_tbl.append({
            "Model": MODELS[m]["display"],
            "Scene only": round(m_off, 2), "Scene + Hand": round(m_on, 2),
            "Gain": round(m_off - m_on, 2),
            "Relative improvement": round(
                100.0 * (m_off - m_on) / m_off, 1) if m_off else "",
            "N videos": len(common)})
    write_csv(SUM / "primary_results.csv", primary)
    write_csv(SUM / "paired_gain_summary.csv", paired_rows)
    write_csv(TAB / "presentation_main_table.csv", main_tbl)

    # ---------------- NATIVE (secondary) sanity
    native = []
    for m in MODELS:
        for cond in ("SCENE_ONLY", "SCENE_PLUS_HAND"):
            errs = [idx[k]["error_pct"] for k in idx
                    if k[2] == m and k[3] == cond and k[4] == "NATIVE"
                    and np.isfinite(idx[k]["error_pct"])]
            native.append({"model": m, "condition": cond,
                           "frame_set": "NATIVE", **stats(errs)})
    write_csv(SUM / "model_native_allframe_results.csv", native)

    # ---------------- per sequence
    per_seq = []
    for seq in sorted(set(s for (s, c) in common)):
        vids = [(s, c) for (s, c) in common if s == seq]
        for m in MODELS:
            for cond in ("SCENE_ONLY", "SCENE_PLUS_HAND"):
                errs = [idx[(s, c, m, cond, PRIMARY_SET)]["error_pct"]
                        for (s, c) in vids]
                sb = [idx[(s, c, m, cond, PRIMARY_SET)]["signed_error_pct"]
                      for (s, c) in vids]
                sp = [fnum(idx[(s, c, m, cond,
                                PRIMARY_SET)]["within_video_spread_pct"])
                      for (s, c) in vids]
                per_seq.append({
                    "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
                    "model": MODELS[m]["display"], "condition": cond,
                    "n_video_units": len(vids), **stats(errs),
                    "median_signed_bias_pct": med(sb),
                    "median_frame_spread_pct": med(sp)})
    write_csv(SUM / "per_sequence_results.csv", per_seq)
    write_csv(TAB / "per_sequence_results.csv", per_seq)

    # ---------------- accuracy / precision (scene-only)
    ap = []
    for m in MODELS:
        errs = [idx[(s, c, m, "SCENE_ONLY", PRIMARY_SET)]["error_pct"]
                for (s, c) in sorted(common)]
        sb = [idx[(s, c, m, "SCENE_ONLY", PRIMARY_SET)]["signed_log_bias_pct"]
              for (s, c) in sorted(common)]
        sp = [fnum(idx[(s, c, m, "SCENE_ONLY",
                        PRIMARY_SET)]["within_video_spread_pct"])
              for (s, c) in sorted(common)]
        ap.append({"Model": MODELS[m]["display"],
                   "median_abs_error_pct": med(errs),
                   "median_signed_bias_pct": med(sb),
                   "median_within_video_spread_pct": med(sp),
                   "bias_direction": ("over-estimates" if med(sb) > 0
                                      else "under-estimates")})
    for i, r in enumerate(sorted(ap, key=lambda z: z[
            "median_within_video_spread_pct"]), 1):
        r["precision_rank"] = i
    write_csv(SUM / "accuracy_precision_summary.csv", ap)
    write_csv(TAB / "presentation_accuracy_precision.csv", ap)

    same_dir = len({r["gain_pp"] > 0 for r in paired_rows}) == 1
    write_json(SUM / "cam011_verdict.json", {
        "n_paired_videos": len(common),
        "primary_frame_set": PRIMARY_SET,
        "main_table": main_tbl,
        "paired": paired_rows,
        "accuracy_precision": ap,
        "all_three_same_direction": same_dir,
        "all_three_improved": all(r["gain_pp"] > 0 for r in paired_rows),
        "note": "one video = one vote; frames are repeated observations inside "
                "a video, not independent samples",
    })

    print("\nGLOBAL paired video set: %d" % len(common))
    print("%-22s %12s %14s %8s %10s %8s"
          % ("Model", "Scene only", "Scene + Hand", "Gain", "win-rate", "CI"))
    for r in paired_rows:
        print("%-22s %11.2f%% %13.2f%% %+7.2f %9.1f%%  [%.2f, %.2f]"
              % (r["display"], r["scene_only_median_pct"],
                 r["scene_plus_hand_median_pct"], r["gain_pp"],
                 r["win_rate_pct"], r["ci95_low"], r["ci95_high"]))
    print("\nall three same direction:", same_dir)


if __name__ == "__main__":
    main()
