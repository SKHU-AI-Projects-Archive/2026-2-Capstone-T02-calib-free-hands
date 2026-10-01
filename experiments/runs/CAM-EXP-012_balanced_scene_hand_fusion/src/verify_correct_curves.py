"""PHASE A — integrity audit of the correct-hand curves. Read-only, cheap.

Completion requires ALL of:
  A  every expected primary video has a curve
  B  count == frozen primary video count
  C  every curve is readable and has finite values
  D  all 8 shard logs reached their final line
  E  no duplicates, no missing videos
  F  cross-fit metadata present and consistent with the fold rule
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import (CACHE, MANIFESTS, MIN_OBS_PER_FOLD_PER_SIDE,  # noqa
                         RAW, SUM, read_csv, read_json, write_json)


def main():
    prim = read_csv(MANIFESTS / "cam_exp_012_primary_video_set_v1.csv")
    expected = {(r["sequence"], r["camera"]) for r in prim}
    d = CACHE / "correct_hand_curves"
    files = sorted(d.glob("*.npz"))
    got = [tuple(p.stem.split("__", 1)) for p in files]

    dup = len(got) != len(set(got))
    missing = sorted(expected - set(got))
    extra = sorted(set(got) - expected)

    bad, meta_bad, n_ok = [], [], 0
    grid = np.array(read_json(MANIFESTS
                              / "cam_exp_012_global_focal_grid_v1.json")
                    ["grid"], float)
    for p in files:
        try:
            z = np.load(p)
            s = z["score"]
            f = z["f"]
            if not np.isfinite(s).any():
                bad.append(p.stem + ":all_nan")
                continue
            if len(f) != len(grid) or not np.allclose(f, grid, rtol=1e-12):
                bad.append(p.stem + ":grid_mismatch")
                continue
            need = ("n_A_left", "n_B_left", "n_A_right", "n_B_right")
            if not all(k in z for k in need):
                meta_bad.append(p.stem + ":missing_metadata")
                continue
            if min(int(z[k]) for k in need) < MIN_OBS_PER_FOLD_PER_SIDE:
                meta_bad.append(p.stem + ":fold_below_minimum")
                continue
            n_ok += 1
        except Exception as exc:
            bad.append("%s:%s" % (p.stem, type(exc).__name__))

    logs = sorted(RAW.glob("_curves_correct_shard*.log"))
    done_logs = 0
    for L in logs:
        t = L.read_text(encoding="utf-8", errors="ignore")
        if "videos written" in t:
            done_logs += 1

    complete = bool(
        not missing and not extra and not dup and not bad and not meta_bad
        and len(got) == len(expected) and done_logs == len(logs)
        and len(logs) == 8)

    out = {
        "expected_primary_videos": len(expected),
        "curves_on_disk": len(got),
        "curves_valid": n_ok,
        "missing": [list(x) for x in missing[:20]],
        "n_missing": len(missing),
        "extra": [list(x) for x in extra],
        "duplicates": dup,
        "unreadable_or_bad": bad[:20],
        "metadata_problems": meta_bad[:20],
        "shard_logs_found": len(logs),
        "shard_logs_finished": done_logs,
        "checks": {
            "A_all_expected_present": not missing,
            "B_count_matches": len(got) == len(expected),
            "C_all_readable_and_finite": not bad,
            "D_all_shards_finished": done_logs == len(logs) == 8,
            "E_no_duplicates_no_extra": not dup and not extra,
            "F_crossfit_metadata_ok": not meta_bad,
        },
        "CORRECT_HAND_CURVES_COMPLETE": complete,
    }
    write_json(SUM / "correct_curves_integrity.json", out)
    print("expected %d | on disk %d | valid %d | shards finished %d/%d"
          % (len(expected), len(got), n_ok, done_logs, len(logs)))
    for k, v in out["checks"].items():
        print("   %-32s %s" % (k, v))
    print("CORRECT_HAND_CURVES_COMPLETE:", complete)
    if not complete and len(got) < len(expected):
        print("   still running: %d remaining" % (len(expected) - len(got)))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
