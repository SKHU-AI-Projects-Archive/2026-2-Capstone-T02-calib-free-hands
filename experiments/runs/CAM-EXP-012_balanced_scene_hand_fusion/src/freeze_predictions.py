"""PHASE F — freeze and hash the TEST predictions while the focal is still closed.

Refuses to run if the reference focal has already been opened.
"""
from __future__ import annotations

import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import RAW, REPO, SUM, read_csv, sha256, write_json  # noqa

FROZEN = RAW / "frozen_test_predictions.csv.gz"


def main():
    opened = RAW / "REFERENCE_FOCAL_OPENED.txt"
    if opened.exists():
        raise SystemExit("reference focal already opened; cannot re-freeze")
    if not FROZEN.exists():
        raise SystemExit("no predictions to freeze; run run_nested_cv.py first")

    rows = read_csv(FROZEN)
    by_cond = Counter(r["condition"] for r in rows)
    by_model = Counter(r["model"] for r in rows)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    rec = {
        "frozen_predictions_file": FROZEN.name,
        "frozen_predictions_sha256": sha256(FROZEN),
        "n_rows": len(rows),
        "n_videos": len({(r["sequence"], r["camera"]) for r in rows}),
        "rows_by_condition": dict(by_cond),
        "rows_by_model": dict(by_model),
        "test_reference_focal_opened": False,
        "wrong_bone_control": "PENDING",
        "git_head": head,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Predictions frozen before any OUTER TEST reference focal "
                 "was read. evaluate_presentation.py verifies this hash "
                 "before opening the focal."),
    }
    write_json(SUM / "presentation_prediction_freeze.json", rec)
    print("frozen %d rows over %d videos" % (rec["n_rows"], rec["n_videos"]))
    for k, v in sorted(by_cond.items()):
        print("   %-28s %d" % (k, v))
    print("sha256", rec["frozen_predictions_sha256"])


if __name__ == "__main__":
    main()
