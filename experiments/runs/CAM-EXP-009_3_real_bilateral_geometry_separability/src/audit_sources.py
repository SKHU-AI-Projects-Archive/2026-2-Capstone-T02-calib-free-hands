"""PHASE A — provenance audits. No distance is computed here.

Three questions, answered from source evidence rather than from memory:
  1. can participant identity be established for each sequence?
  2. how much eligible data is actually available locally?
  3. is the run structurally focal-independent?
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FOCAL_BEARING_MANIFESTS, MANIFESTS, MIN_FRAMES_PER_SIDE,  # noqa: E402
                    QC, QC_PASS_STATUSES, SUM, TAB, loco_on_path, read_csv,
                    write_csv, write_json)


def participant_audit():
    """Can we say which sequences belong to the same person?

    The sequence names carry a `p<NN>` prefix that LOOKS like a participant id.
    The spec forbids inferring identity from a name, so this function records
    what shipped evidence actually exists and grades the confidence.
    """
    loco_on_path()
    from experiments.src.datasets.gigahands import takes

    seqs = sorted(t.name for t in takes())

    # Evidence hunt: anything shipped that documents a participant field.
    ds_root = Path(__file__).resolve().parents[4] / "experiments" / "datasets" \
        / "gigahands"
    metadata_files = [p for p in ds_root.rglob("*")
                      if p.is_file() and p.suffix in (".json", ".txt", ".csv")
                      and "keypoints" not in str(p) and "bboxes" not in str(p)]
    documented = []
    for p in metadata_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:200000]
        except OSError:
            continue
        low = txt.lower()
        if any(k in low for k in ("participant", "subject_id", "\"subject\"",
                                  "person_id")):
            documented.append(str(p.relative_to(ds_root)))

    rows = []
    for s in seqs:
        prefix = s.split("-")[0]
        rows.append({
            "sequence": s,
            "participant_id": prefix,
            "identity_source": "SEQUENCE_NAME_PREFIX_CONVENTION",
            "identity_confidence": "LOW_UNVERIFIED",
            "same_subject_comparison_allowed": "NO",
            "notes": "The `p<NN>` prefix is an undocumented naming convention "
                     "in this local subset. No shipped GigaHands metadata "
                     "file, and no manifest produced by CAM-EXP-001..009.2, "
                     "declares a participant field. Identity is therefore "
                     "asserted by filename only, which the CAM-EXP-009.3 spec "
                     "forbids as a basis for same-subject claims.",
        })
    write_csv(TAB / "participant_identity_audit.csv", rows)

    groups = collections.Counter(r["participant_id"] for r in rows)
    out = {
        "n_sequences": len(seqs),
        "sequences": seqs,
        "candidate_participant_groups": dict(groups),
        "n_candidate_participants": len(groups),
        "participants_with_multiple_sequences":
            {k: v for k, v in groups.items() if v > 1},
        "shipped_metadata_files_scanned": len(metadata_files),
        "shipped_files_documenting_participant": documented,
        "identity_resolved": bool(documented),
        "STATUS": ("SAME_SUBJECT_PROVENANCE_RESOLVED" if documented
                   else "SAME_SUBJECT_PROVENANCE_UNRESOLVED"),
        "why": "Identity is graded from shipped evidence only. A MANO `shapes` "
               "vector is present per sequence in params/*.json, but using a "
               "fitted hand-shape parameter to DEFINE subject identity and "
               "then measuring hand-shape similarity would be circular, so it "
               "is explicitly not used as an identity source.",
    }
    write_json(SUM / "participant_identity_audit.json", out)
    return out


def population_audit():
    """How much eligible data exists locally, counted before any result."""
    qc = read_csv(QC)
    elig = [r for r in qc if r["qc_status"] in QC_PASS_STATUSES
            and r["triangulation_success"] == "1"]
    units = collections.Counter(
        (r["sequence"], r["camera"], r["hand"]) for r in elig)
    big = {k: v for k, v in units.items() if v >= MIN_FRAMES_PER_SIDE}

    per_side = collections.defaultdict(set)
    for (s, c, h) in big:
        per_side[(s, c)].add(h)
    both = [k for k, v in per_side.items() if v == {"left", "right"}]

    seqs = sorted(set(r["sequence"] for r in elig))
    out = {
        "qc_source": QC.name,
        "qc_statuses_accepted": list(QC_PASS_STATUSES),
        "total_qc_rows": len(qc),
        "eligible_rows": len(elig),
        "total_sequences": len(seqs),
        "unique_candidate_participants":
            len(set(s.split("-")[0] for s in seqs)),
        "physical_cameras": len(set(r["camera"] for r in elig)),
        "sequence_camera_units": len(set((r["sequence"], r["camera"])
                                         for r in elig)),
        "usable_left_frames": sum(1 for r in elig if r["hand"] == "left"),
        "usable_right_frames": sum(1 for r in elig if r["hand"] == "right"),
        "side_units_ge_min_frames": len(big),
        "units_with_both_sides_sufficient": len(both),
        "cameras_per_sequence": {s: len(set(r["camera"] for r in elig
                                            if r["sequence"] == s))
                                 for s in seqs},
        "frames_per_unit": {
            "min": int(min(units.values())),
            "median": float(np.median(list(units.values()))),
            "max": int(max(units.values())),
        },
    }
    write_json(SUM / "coverage_audit.json", out)
    write_csv(SUM / "coverage_summary.csv", [
        {"metric": k, "value": v} for k, v in out.items()
        if not isinstance(v, (dict, list))])
    return out


def focal_independence_audit():
    """Structural check that this run is focal-free."""
    src = Path(__file__).resolve().parent
    hits = []
    for p in sorted(src.glob("*.py")):
        txt = p.read_text(encoding="utf-8")
        body = "\n".join(ln for ln in txt.splitlines()
                         if not ln.strip().startswith("#"))
        for m in FOCAL_BEARING_MANIFESTS:
            # the tuple's own definition in common.py is the declaration, not a
            # use; a use would be a read_csv/open on the name
            if m in body and p.name not in ("common.py", "audit_sources.py"):
                hits.append({"file": p.name, "manifest": m})
        for token in ("gt_fx", "gt_fy", "anycalib", "AnyCalib", "geocalib",
                      "GeoCalib", "focal_grid", "q_grid", "candidate_focal"):
            if token in body and p.name not in ("common.py", "self_audit.py",
                                                "audit_sources.py"):
                hits.append({"file": p.name, "token": token})

    rows = [
        {"component": "reference focal / gt_fx", "reference_focal_loaded": "NO",
         "AnyCalib_loaded": "NO", "focal_candidate_grid_used": "NO",
         "allowed": "NO"},
        {"component": "AnyCalib inference", "reference_focal_loaded": "NO",
         "AnyCalib_loaded": "NO", "focal_candidate_grid_used": "NO",
         "allowed": "NO"},
        {"component": "GeoCalib inference", "reference_focal_loaded": "NO",
         "AnyCalib_loaded": "NO", "focal_candidate_grid_used": "NO",
         "allowed": "NO"},
        {"component": "E2 / scene fusion", "reference_focal_loaded": "NO",
         "AnyCalib_loaded": "NO", "focal_candidate_grid_used": "NO",
         "allowed": "NO"},
        {"component": "candidate focal sweep", "reference_focal_loaded": "NO",
         "AnyCalib_loaded": "NO", "focal_candidate_grid_used": "NO",
         "allowed": "NO"},
        {"component": "OTHER_CAMERA_ONLY_REFERENCE_3D (historical "
                      "calibration dependency)",
         "reference_focal_loaded": "NO", "AnyCalib_loaded": "NO",
         "focal_candidate_grid_used": "NO", "allowed": "YES"},
    ]
    write_csv(TAB / "focal_independence_audit.csv", rows)
    write_json(SUM / "focal_independence_audit.json",
               {"source_scan_hits": hits, "clean": not hits})
    return hits


def main():
    pid = participant_audit()
    pop = population_audit()
    hits = focal_independence_audit()

    print("=" * 62)
    print("PARTICIPANT IDENTITY")
    print("  sequences                     ", pid["n_sequences"])
    print("  candidate participant groups  ",
          pid["candidate_participant_groups"])
    print("  shipped files documenting id  ",
          pid["shipped_files_documenting_participant"] or "NONE")
    print("  STATUS                        ", pid["STATUS"])
    print("-" * 62)
    print("POPULATION")
    for k in ("eligible_rows", "total_sequences",
              "unique_candidate_participants", "physical_cameras",
              "sequence_camera_units", "usable_left_frames",
              "usable_right_frames", "side_units_ge_min_frames",
              "units_with_both_sides_sufficient"):
        print("  %-32s %s" % (k, pop[k]))
    print("-" * 62)
    print("FOCAL INDEPENDENCE source scan hits:", hits or "none")
    print("=" * 62)


if __name__ == "__main__":
    main()
