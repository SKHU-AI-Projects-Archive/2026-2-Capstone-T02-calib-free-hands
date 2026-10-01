"""Post-run accounting audit for CAM-EXP-011. READ-ONLY on frozen artifacts.

This reruns no inference. It reads the existing prediction caches, the hand
observation table, the hand-score cache and the frozen six-condition
predictions, and produces an exact per-video ledger so that 175 -> 151 is
fully explained with mutually exclusive reasons.

It also resolves the 152-vs-153 discrepancy between an intermediate progress
log and the final index, by counting the actual artefacts rather than trusting
either log.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (CACHE, DISPLAY_SEQUENCE, MANIFESTS,  # noqa: E402
                         MIN_HAND_FRAMES_PER_SIDE, MIN_SCENE_COVERAGE, MODELS,
                         RAW, SUM, TAB, fnum, read_csv, write_csv, write_json)


def main():
    # ---------------- the 175 usable videos
    cams = read_csv(MANIFESTS / "cam_exp_011_usable_cameras_v1.csv")
    videos = [(r["sequence"], r["camera"]) for r in cams if r["usable"] == "1"]
    videos = sorted(set(videos))

    # ---------------- scene frames per video
    man = [r for r in read_csv(MANIFESTS
                               / "cam_exp_011_allframe_manifest_v1.csv.gz")
           if r["scene_input"] == "1"]
    scene_n = Counter((r["sequence"], r["camera"]) for r in man)

    # ---------------- per-model valid frames
    valid = {}
    for m, fn in (("ANYCALIB", "anycalib_allframe_predictions.csv.gz"),
                  ("GEOCALIB", "geocalib_allframe_predictions.csv.gz"),
                  ("PERSPECTIVE_FIELDS",
                   "perspective_fields_allframe_predictions.csv.gz")):
        c = Counter()
        for r in read_csv(RAW / fn):
            if r["valid"] == "1":
                c[(r["sequence"], r["camera"])] += 1
        valid[m] = c

    # ---------------- hand observations (frame level)
    hl, hr, hb, ha = Counter(), Counter(), Counter(), Counter()
    for r in read_csv(RAW / "hand_observations.csv.gz"):
        k = (r["sequence"], r["camera"])
        L = r["left_available"] == "1"
        R = r["right_available"] == "1"
        hl[k] += int(L)
        hr[k] += int(R)
        hb[k] += int(L and R)
        ha[k] += int(r["hand_any_available"] == "1")

    # ---------------- hand-score cache: what actually exists on disk
    hs_dir = CACHE / "hand_scores"
    cache_exists, cache_eligible, cache_nl, cache_nr = {}, {}, {}, {}
    for p in sorted(hs_dir.glob("*.npz")):
        seq, cam = p.stem.split("__", 1)
        d = np.load(p)
        k = (seq, cam)
        cache_exists[k] = True
        cache_eligible[k] = int(d["eligible"]) == 1
        cache_nl[k] = int(d["n_left"])
        cache_nr[k] = int(d["n_right"])

    # ---------------- frozen six-condition predictions
    six = read_csv(RAW / "final_six_condition_predictions.csv.gz")
    have = defaultdict(set)
    for r in six:
        if r["frame_set"] != "STRICT":
            continue
        if np.isfinite(fnum(r["f_pred"])):
            have[(r["sequence"], r["camera"])].add((r["model"],
                                                    r["condition"]))
    need = {(m, c) for m in MODELS
            for c in ("SCENE_ONLY", "SCENE_PLUS_HAND")}

    # ---------------- the paired set actually used by evaluate.py
    paired = set(read_csv(TAB / "paired_unit_manifest.csv")
                 and [(r["sequence"], r["camera"])
                      for r in read_csv(TAB / "paired_unit_manifest.csv")])

    # ---------------- ledger
    rows, exclusions = [], []
    for k in videos:
        seq, cam = k
        n_scene = scene_n[k]
        cov = {m: valid[m][k] / max(n_scene, 1) for m in MODELS}
        cov_ok = all(c >= MIN_SCENE_COVERAGE for c in cov.values())
        in_cache = k in cache_exists
        elig = bool(cache_eligible.get(k, False))
        nl = cache_nl.get(k, hl[k])
        nr = cache_nr.get(k, hr[k])
        six_ok = have[k] >= need
        final = k in paired

        # PRIMARY exclusion reason - mutually exclusive, first match wins
        reason = ""
        if not final:
            if ha[k] == 0:
                reason = "NO_USABLE_HAND"
            elif not in_cache:
                reason = "HAND_SCORE_NOT_ATTEMPTED"
            elif not elig:
                if nl < MIN_HAND_FRAMES_PER_SIDE and \
                        nr < MIN_HAND_FRAMES_PER_SIDE:
                    reason = "INSUFFICIENT_BOTH_SIDES"
                elif nl < MIN_HAND_FRAMES_PER_SIDE:
                    reason = "INSUFFICIENT_LEFT_HAND_FRAMES"
                else:
                    reason = "INSUFFICIENT_RIGHT_HAND_FRAMES"
            elif not cov_ok:
                reason = "SCENE_COVERAGE_FAILED"
            elif not six_ok:
                # Observed cause: the hand score lives on a fixed absolute
                # grid (400-2400 px). When a scene estimate is extreme its
                # candidate window f_scene * [0.5, 1.5] falls entirely outside
                # that grid, the interpolation returns NaN, and that model's
                # "+ Hand" cell is missing. The video is then dropped from the
                # six-condition set for ALL models equally, so the comparison
                # stays paired.
                reason = "HAND_GRID_DOES_NOT_COVER_CANDIDATE_WINDOW"
            else:
                reason = "UNRESOLVED_SEE_AUDIT"
            exclusions.append({
                "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
                "camera": cam, "reason": reason,
                "left_frames": nl, "right_frames": nr,
                "any_hand_frames": ha[k],
                "hand_score_cache_exists": int(in_cache),
                "hand_score_eligible": int(elig)})

        rows.append({
            "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
            "camera": cam, "usable_rgb_video": 1,
            "scene_frame_count": n_scene,
            "anycalib_valid_frames": valid["ANYCALIB"][k],
            "geocalib_valid_frames": valid["GEOCALIB"][k],
            "pf_valid_frames": valid["PERSPECTIVE_FIELDS"][k],
            "left_hand_available_frames": hl[k],
            "right_hand_available_frames": hr[k],
            "both_hand_available_frames": hb[k],
            "any_hand_available_frames": ha[k],
            "hand_score_cache_exists": int(in_cache),
            "hand_score_computable": int(elig),
            "hand_score_n_left": nl, "hand_score_n_right": nr,
            "hand_coverage_rule_pass": int(elig),
            "anycalib_scene_aggregate_exists": int(valid["ANYCALIB"][k] > 0),
            "geocalib_scene_aggregate_exists": int(valid["GEOCALIB"][k] > 0),
            "pf_scene_aggregate_exists":
                int(valid["PERSPECTIVE_FIELDS"][k] > 0),
            "scene_coverage_rule_pass": int(cov_ok),
            "six_condition_prediction_exists": int(six_ok),
            "final_paired_set": int(final),
            "primary_exclusion_reason": reason,
            "secondary_notes": "",
        })

    write_csv(TAB / "paired_set_accounting.csv", rows)
    write_csv(TAB / "paired_set_exclusions.csv", exclusions)

    counts = Counter(e["reason"] for e in exclusions)
    write_csv(TAB / "presentation_paired_exclusion_summary.csv",
              [{"Reason": k, "N videos": v}
               for k, v in sorted(counts.items(), key=lambda z: -z[1])]
              + [{"Reason": "TOTAL EXCLUDED", "N videos": len(exclusions)},
                 {"Reason": "FINAL PAIRED", "N videos": len(paired)},
                 {"Reason": "TOTAL USABLE VIDEOS", "N videos": len(videos)}])

    # ---------------- per-sequence coverage
    per_seq = []
    for seq in sorted(set(s for s, _ in videos)):
        tot = sum(1 for s, _ in videos if s == seq)
        pr = sum(1 for (s, c) in paired if s == seq)
        br = Counter(e["reason"] for e in exclusions if e["sequence"] == seq)
        per_seq.append({
            "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
            "total_usable_videos": tot, "paired_videos": pr,
            "excluded_videos": tot - pr,
            "paired_rate_pct": round(100.0 * pr / max(tot, 1), 1),
            "primary_exclusion_breakdown": "; ".join(
                "%s=%d" % (k, v) for k, v in sorted(br.items())) or "-"})
    write_csv(TAB / "paired_coverage_by_sequence.csv", per_seq)

    # ---------------- 152 vs 153 resolution, from artefacts not logs
    idx = read_csv(RAW / "hand_score_index.csv.gz")
    idx_elig = sum(1 for r in idx if r["eligible"] == "1")
    shard_files = sorted(RAW.glob("hand_score_index_shard*.csv.gz"))
    shard_rows = sum(len(read_csv(p)) for p in shard_files)
    audit = {
        "total_usable_videos": len(videos),
        "videos_with_any_hand_frame": sum(1 for k in videos if ha[k] > 0),
        "videos_with_no_usable_hand": sum(1 for k in videos if ha[k] == 0),
        "hand_score_cache_files_on_disk": len(cache_exists),
        "hand_score_cache_eligible": sum(1 for k in cache_eligible
                                         if cache_eligible[k]),
        "hand_score_index_rows": len(idx),
        "hand_score_index_eligible": idx_elig,
        "shard_index_files": len(shard_files),
        "shard_index_rows_total": shard_rows,
        "coverage_rule_pass": sum(r["hand_coverage_rule_pass"] for r in rows),
        "scene_coverage_rule_pass": sum(r["scene_coverage_rule_pass"]
                                        for r in rows),
        "six_condition_complete": sum(r["six_condition_prediction_exists"]
                                      for r in rows),
        "final_paired": len(paired),
        "excluded": len(exclusions),
        "exclusion_counts": dict(counts),
        "checks": {
            "paired_plus_excluded_equals_175":
                len(paired) + len(exclusions) == len(videos),
            "exclusion_reason_counts_sum_equals_excluded":
                sum(counts.values()) == len(exclusions),
            "every_excluded_has_one_reason":
                all(e["reason"] for e in exclusions),
        },
    }
    write_json(SUM / "hand_score_accounting_audit.json", audit)

    print("usable videos %d | paired %d | excluded %d"
          % (len(videos), len(paired), len(exclusions)))
    print("exclusion reasons:")
    for k, v in sorted(counts.items(), key=lambda z: -z[1]):
        print("   %-34s %d" % (k, v))
    print("checks:", audit["checks"])
    print("\nhand-score accounting:")
    for k in ("videos_with_any_hand_frame", "videos_with_no_usable_hand",
              "hand_score_cache_files_on_disk", "hand_score_cache_eligible",
              "hand_score_index_rows", "hand_score_index_eligible",
              "shard_index_rows_total", "coverage_rule_pass"):
        print("   %-38s %s" % (k, audit[k]))


if __name__ == "__main__":
    main()
