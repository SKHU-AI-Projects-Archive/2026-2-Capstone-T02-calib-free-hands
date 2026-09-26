"""Per-feature association between explicit hand features and the bias target.

Descriptive supplement. Effect size with a physical-camera cluster bootstrap is
the headline; BH-FDR is reported alongside but does not drive any decision.
Latent dimensions are NOT enumerated individually.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, SUM, VIEW_TARGETS, corr_cluster_ci, fnum, read_csv, write_csv
from run_probes import assign_group


def main() -> None:
    hand = {(r["sequence"], r["camera"]): r
            for r in read_csv(RAW / "view_hand_features.csv.gz")}
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(VIEW_TARGETS)}
    views = [v for v in sorted(tgt) if v in hand
             and hand[v]["hand_feature_eligible"] == "1"]
    y = np.array([fnum(tgt[v]["anycalib_signed_log_bias"]) for v in views])
    cams = [tgt[v]["physical_camera_id"] for v in views]
    cols = [c for c in next(iter(hand.values())) if assign_group(c)]

    rows = []
    for c in cols:
        x = np.array([fnum(hand[v][c]) for v in views])
        rho, lo, hi, n = corr_cluster_ci(x, y, cams, "spearman")
        rows.append({"feature": c, "group": assign_group(c), "n_views": n,
                     "spearman_vs_signed_bias": round(rho, 4),
                     "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                     "abs_rho": abs(rho) if np.isfinite(rho) else 0.0,
                     "ci_excludes_zero": int(np.isfinite(lo) and np.isfinite(hi)
                                             and (lo > 0 or hi < 0))})
    rows.sort(key=lambda r: -r["abs_rho"])
    for r in rows:
        r.pop("abs_rho")
    write_csv(SUM / "explicit_feature_associations.csv", rows)
    sig = [r for r in rows if r["ci_excludes_zero"]]
    print(f"features {len(rows)}, CI excludes zero: {len(sig)}")
    print("top 5 by |rho| (pre-registered features only):")
    for r in rows[:5]:
        print(f"  {r['feature']:38s} {r['group']}  rho {r['spearman_vs_signed_bias']:+.3f} "
              f"CI [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] "
              f"{'*' if r['ci_excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
