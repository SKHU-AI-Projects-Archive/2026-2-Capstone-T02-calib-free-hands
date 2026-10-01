"""PHASE A — record every external artefact this run depends on.

Writes `tables/source_audit.csv`. Reference focal is not read here: the scene
cache is opened through the focal-blind reader only for a row count.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C0091, C0093, C013, MANIFESTS, QC, REPO,  # noqa: E402
                    RUNS, SCENE_MODEL_KEY, SCENE_SRC, SUM, TAB, read_csv,
                    read_scene_predictions, sha256, write_csv, write_json)


def commit_for(path: Path) -> str:
    try:
        rel = path.relative_to(REPO).as_posix()
    except ValueError:
        return ""
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", rel],
                          cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def main():
    rows = []

    def add(path, role, note=""):
        p = Path(path)
        if not p.exists():
            rows.append({"artifact": p.name, "path": str(p), "exists": 0,
                         "role": role, "note": "MISSING"})
            return
        n = ""
        if p.suffix in (".gz", ".csv") and "csv" in p.name:
            try:
                n = len(read_csv(p))
            except Exception:
                n = ""
        rows.append({"artifact": p.name,
                     "path": p.relative_to(REPO).as_posix()
                     if REPO in p.parents or p.is_relative_to(REPO) else str(p),
                     "exists": 1, "size_bytes": p.stat().st_size,
                     "sha256": sha256(p), "rows": n,
                     "last_commit": commit_for(p),
                     "role": role, "note": note})

    add(SCENE_SRC, "scene focal predictions (%s)" % SCENE_MODEL_KEY,
        "contains gt_* columns; read ONLY through read_scene_predictions(), "
        "which strips them")
    add(QC, "hand frame eligibility (frozen CAM-EXP-001.3 QC)",
        "no focal column")
    add(REPO / "mano_data" / "MANO_RIGHT.pkl", "generic anatomy prior source",
        "neutral shape, rest pose; absolute size discarded")
    add(REPO / "models" / "anyhand_wilor.ckpt", "hand network checkpoint", "")
    add(REPO / "models" / "model_config_wilor.yaml", "hand network config", "")
    add(REPO / "models" / "detector.pt", "hand detector", "")
    add(C013 / "src" / "loco.py",
        "OTHER_CAMERA_ONLY_REFERENCE_3D reconstruction", "reused unchanged")
    add(REPO / "rgb_predictor.py", "WiLoR wrapper (field definitions audited)",
        "keypoints_2d in original pixels; keypoints_3d root-relative")
    add(REPO / "demo" / "hand_topology.py", "21-joint order and bone set",
        "OpenPose order, 20 connected bones")
    add(RUNS / "CAM-EXP-002_camera_focal_sensitivity" / "src" /
        "focal_sweep.py", "camera translation equation",
        "tz(f) = cam_t[2] * f / f_pipeline; tx, ty unchanged")
    add(C0091 / "src" / "corrected_bone_fitter.py",
        "solver conventions reference", "UNDISTORT_ONCE_INTERNAL, N_ITER=32")
    add(RUNS / "CAM-EXP-009_2_robust_bilateral_focal" / "src" /
        "generate_stress_synthetic.py", "synthetic generator for the gate",
        "imported unchanged; its outputs are not overwritten")
    add(C0093 / "results" / "raw" / "geometry_templates.csv.gz",
        "CAM-EXP-009.3 reference geometry (context only)",
        "not an input to this method")

    write_csv(TAB / "source_audit.csv", rows)

    sc = read_scene_predictions()
    write_json(SUM / "source_audit.json", {
        "n_artifacts": len(rows),
        "missing": [r["artifact"] for r in rows if not r["exists"]],
        "scene_rows_after_focal_strip": len(sc),
        "scene_columns_after_strip": sorted(sc[0]) if sc else [],
        "gt_columns_present_after_strip":
            [k for k in (sc[0] if sc else {}) if k.startswith("gt_")],
    })
    for r in rows:
        print("%-38s %s %s" % (r["artifact"],
                               "OK " if r["exists"] else "MISSING",
                               r.get("sha256", "")[:12]))
    print("scene rows after strip:", len(sc),
          "| gt_ columns remaining:",
          [k for k in (sc[0] if sc else {}) if k.startswith("gt_")])


if __name__ == "__main__":
    main()
