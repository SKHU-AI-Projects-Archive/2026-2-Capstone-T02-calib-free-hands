"""Integrity audit of the CAM-EXP-009.3 artefacts this run reuses.

Nothing is recomputed. The 14,015 reference-3D reconstructions are NOT re-run.
The facts CAM-EXP-009.3 recorded are re-checked against the actual files; where
a file disagrees with the recorded number, the FILE is reported and the run
stops rather than forcing the remembered value.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C093, C093_COMMIT, EXPECTED, MANIFESTS,  # noqa: E402
                    REPO, SOURCE_ARTIFACTS, SUM, TAB, read_csv, read_json,
                    sha256, write_csv, write_json)


def main():
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", str(C093.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(C093.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True).stdout.strip()

    rows, missing = [], []
    for name, p in sorted(SOURCE_ARTIFACTS.items()):
        if not p.exists():
            missing.append(name)
            continue
        n = ""
        if name.endswith(".csv.gz") or name.endswith(".csv"):
            n = len(read_csv(p))
        rows.append({
            "artifact": name,
            "path": str(p.relative_to(REPO)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
            "rows": n,
            "role": "source of truth, reused unmodified",
        })
    write_csv(TAB / "source_artifact_hashes.csv", rows)

    # re-check the recorded facts against the files
    geo = read_json(SOURCE_ARTIFACTS["reference_geometry_audit.json"])
    fv = read_csv(SOURCE_ARTIFACTS["frame_bone_vectors.csv.gz"])
    cols = list(fv[0]) if fv else []
    nbones = sum(1 for c in cols if c.startswith("z") and c[1:].isdigit())

    checks = [
        {"fact": "reconstructions_attempted",
         "expected": EXPECTED["reconstructions_attempted"],
         "actual": geo["reconstructions_attempted"]},
        {"fact": "frame_vectors", "expected": EXPECTED["frame_vectors"],
         "actual": len(fv)},
        {"fact": "target_camera_in_reference_count",
         "expected": EXPECTED["target_camera_in_reference_count"],
         "actual": geo["target_camera_in_reference_count"]},
        {"fact": "n_bones_in_representation", "expected": EXPECTED["n_bones"],
         "actual": nbones},
    ]
    for c in checks:
        c["match"] = int(c["expected"] == c["actual"])

    # absolute scale removed? sum of exp(z) must be 1 per row
    import numpy as np
    sums = []
    for r in fv[:500]:
        z = np.array([float(r["z%02d" % b]) for b in range(nbones)])
        sums.append(float(np.exp(z).sum()))
    scale_ok = bool(np.allclose(sums, 1.0, atol=1e-6))
    checks.append({"fact": "absolute_scale_removed_sum_p_eq_1",
                   "expected": 1, "actual": int(scale_ok),
                   "match": int(scale_ok)})

    # frame cap
    import collections
    per_unit = collections.Counter(
        (r["sequence"], r["camera"], r["hand"]) for r in fv)
    cap_ok = max(per_unit.values()) <= EXPECTED["max_frames_per_unit"]
    checks.append({"fact": "max_frames_per_unit<=48",
                   "expected": EXPECTED["max_frames_per_unit"],
                   "actual": max(per_unit.values()), "match": int(cap_ok)})

    write_csv(SUM / "source_artifact_checks.csv", checks)
    all_ok = all(c["match"] for c in checks) and not missing

    spec = {
        "source_run": str(C093.relative_to(REPO)),
        "source_run_last_commit": commit,
        "expected_commit_prefix": C093_COMMIT,
        "commit_matches_expected": commit.startswith(C093_COMMIT[:7])
        or C093_COMMIT.startswith(commit[:7]),
        "source_run_working_tree_dirty": bool(dirty),
        "reference_reconstruction_rerun": False,
        "artifacts": rows,
        "missing_artifacts": missing,
        "fact_checks": checks,
        "ALL_OK": all_ok,
    }
    write_json(MANIFESTS / "cam_exp_00931_source_artifacts_v1.json", spec)
    write_json(SUM / "source_artifact_audit.json", spec)

    print("source run commit:", commit, "| dirty:", bool(dirty))
    for c in checks:
        print("  %-38s expected %-8s actual %-8s %s"
              % (c["fact"], c["expected"], c["actual"],
                 "OK" if c["match"] else "MISMATCH"))
    print("missing:", missing or "none")
    if not all_ok:
        raise SystemExit("SOURCE ARTIFACT AUDIT FAILED")


if __name__ == "__main__":
    main()
