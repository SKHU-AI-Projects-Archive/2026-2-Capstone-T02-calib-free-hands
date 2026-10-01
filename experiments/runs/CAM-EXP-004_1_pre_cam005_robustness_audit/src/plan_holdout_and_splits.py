"""Reserve the final confirmatory holdout and freeze the GigaHands protocol.

Nothing is tuned here. The point is to decide, in writing and before CAM-EXP-005
starts, which data we are allowed to look at while developing a method and which
data is sealed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FRAMES8_CSV, MANIFESTS, RUN_DIR, read_csv, write_csv, write_json  # noqa: E402

CANDIDATES = [
    {
        "dataset": "HanCo (tester + calib_meta)",
        "locally_available": "YES",
        "rgb": "YES - 536 jpg crops, 1 sequence (0110), 8 cameras",
        "resolution": "224x224",
        "intrinsics": "per-frame K (3x3) and M (4x4 world->camera) for all 8 cameras",
        "distortion": "NONE - HanCo images ship already undistorted",
        "camera_identity": "8 fixed rig cameras",
        "static_camera": "NO, in image space: the released rgb/ images are "
                         "per-frame crops zoomed on the hand, so the effective "
                         "intrinsics change every frame (measured: fx varies by "
                         "up to 96 % relative within one camera across 67 frames, "
                         "cx by up to 119 px)",
        "usable_for": "single-frame calibration claims (CAM-EXP-003 / 003.1)",
        "not_usable_for": "static-camera multi-frame aggregation (CAM-EXP-004), "
                          "because the image-space camera is not static; and for "
                          "the distortion-aware claim, because the images carry "
                          "no distortion to handle",
        "licensing": "research use, see HanCo terms",
        "decision": "RESERVED_FINAL_CONFIRMATORY_HOLDOUT_SINGLE_FRAME",
    },
    {
        "dataset": "InterHand2.6M",
        "locally_available": "ANNOTATIONS ONLY (1.6 GB); no images downloaded",
        "rgb": "NO - the image release is ~80 GB and was explicitly excluded",
        "resolution": "512x334 (published)",
        "intrinsics": "campos/camrot + focal/princpt per camera, mm, no distortion",
        "distortion": "not modelled",
        "camera_identity": "80-140 fixed studio cameras",
        "static_camera": "YES - genuinely static multi-camera studio capture",
        "usable_for": "would be the strongest confirmatory set for BOTH the "
                      "single-frame and the static-camera multi-frame claims",
        "not_usable_for": "-",
        "licensing": "research use, see InterHand2.6M terms",
        "decision": "DESIGNATED_PRIMARY_HOLDOUT_PENDING_IMAGE_DOWNLOAD_APPROVAL",
    },
    {
        "dataset": "AssemblyHands",
        "locally_available": "ANNOTATIONS ONLY (391 MB)",
        "rgb": "NO - annotation-only download, no ego/exo images on disk",
        "resolution": "n/a", "intrinsics": "per-camera, available",
        "distortion": "n/a", "camera_identity": "ego + static exo rigs",
        "static_camera": "exocentric cameras are static",
        "usable_for": "nothing yet - no images",
        "not_usable_for": "any image-based calibration test until images exist",
        "licensing": "research use",
        "decision": "NOT_A_CANDIDATE_WITHOUT_IMAGES",
    },
    {
        "dataset": "Re:InterHand",
        "locally_available": "NO - intentionally not downloaded (hundreds of GB)",
        "rgb": "NO", "resolution": "n/a", "intrinsics": "n/a", "distortion": "n/a",
        "camera_identity": "n/a", "static_camera": "n/a",
        "usable_for": "later camera-diversity stage",
        "not_usable_for": "now",
        "licensing": "research use",
        "decision": "OUT_OF_SCOPE_FOR_NOW",
    },
    {
        "dataset": "GigaHands demo",
        "locally_available": "YES",
        "rgb": "YES - 175 usable (sequence, camera) views over 5 sequences",
        "resolution": "1280x720",
        "intrinsics": "COLMAP per camera, OpenCV [k1,k2,p1,p2]",
        "distortion": "YES, |k1| 0.34-0.43",
        "camera_identity": "40 unique physical cameras",
        "static_camera": "YES - verified constant GT within every view",
        "usable_for": "development and internal validation",
        "not_usable_for": "any confirmatory claim - it is the development set",
        "licensing": "research use",
        "decision": "DEVELOPMENT_SET",
    },
]


def main() -> None:
    write_csv(RUN_DIR / "tables" / "holdout_candidate_datasets.csv", CANDIDATES)

    rows = read_csv(FRAMES8_CSV)
    views = sorted({(r["sequence"], r["camera"]) for r in rows})
    seqs = sorted({v[0] for v in views})
    per_seq = {s: sorted(v[1] for v in views if v[0] == s) for s in seqs}

    split = {
        "id": "GIGAHANDS_DEVELOPMENT_VALIDATION_PROTOCOL_V1",
        "frozen_on": "2026-09-23",
        "applies_from": "CAM-EXP-005",
        "dataset": "GigaHands demo, 175 (sequence, camera) views",
        "structure": {s: len(v) for s, v in per_seq.items()},
        "protocol": "LEAVE_ONE_SEQUENCE_OUT",
        "why_not_a_fixed_two_way_split": (
            "there are only 5 sequences, and they are unbalanced (40/40/40/40/15 "
            "views). Any fixed 2-way split either puts a single sequence in "
            "validation - 15 to 40 views, and in the worst case the one small, "
            "atypical sequence - or leaves too little for development. "
            "Leave-one-sequence-out uses every sequence as validation exactly "
            "once, keeps the sequence as the unit of independence, and reports 5 "
            "held-out numbers instead of one lucky one. The retrospective E2 "
            "audit in this run already shows how much that spread matters: the "
            "held-out median ranged 5.73-8.80 % across folds."),
        "rules": [
            "method selection, thresholds and any tuning may only use the 4 "
            "training-fold sequences of the fold being evaluated",
            "the held-out sequence's results may be reported but never used to "
            "choose anything",
            "report all 5 held-out numbers, not only their mean",
            "the view (sequence, camera) is the statistical unit inside a fold; "
            "frames are never independent samples",
            "a method that needs a single headline number reports the mean of "
            "the 5 held-out folds, with the range",
        ],
        "folds": [{"held_out_sequence": s, "n_heldout_views": len(per_seq[s]),
                   "training_sequences": [x for x in seqs if x != s]}
                  for s in seqs],
        "status_of_numbers_produced_this_way":
            "RETROSPECTIVE_INTERNAL_VALIDATION - same rig, same 5 sequences. It "
            "is not, and must not be called, an independent test.",
    }
    write_json(MANIFESTS / "development_validation_split_v1.json", split)

    holdout = {
        "id": "FINAL_CONFIRMATORY_HOLDOUT_V1",
        "frozen_on": "2026-09-23",
        "purpose": "the last, single, unrepeated evaluation of whatever method "
                   "CAM-EXP-005/006 produces",
        "primary": {
            "dataset": "InterHand2.6M",
            "status": "DESIGNATED_PRIMARY_HOLDOUT_PENDING_IMAGE_DOWNLOAD_APPROVAL",
            "what_is_held_locally": "annotations and camera parameters only "
                                    "(1.6 GB); the ~80 GB image release has NOT "
                                    "been downloaded and downloading it needs an "
                                    "explicit decision",
            "why": "genuinely static multi-camera studio rig, different from "
                   "GigaHands in optics, lighting, scene and resolution, and the "
                   "only candidate that can confirm BOTH the single-frame and the "
                   "static-camera multi-frame claims",
            "contamination_risk": "currently zero - no image has ever been read",
        },
        "secondary": {
            "dataset": "HanCo (tester)",
            "status": "RESERVED_FINAL_CONFIRMATORY_HOLDOUT_SINGLE_FRAME",
            "scope": "single-frame calibration claims only",
            "why_limited": "the released rgb/ images are per-frame hand-centred "
                           "crops, so the image-space intrinsics change every "
                           "frame (fx varies up to 96 % relative within one "
                           "camera) and the static-camera premise does not hold; "
                           "the images are also already undistorted, so the "
                           "distortion-aware result cannot be confirmed on them",
            "size": "536 images, 8 cameras, 1 sequence, 224x224",
        },
        "contamination_rules": [
            "no result from a holdout dataset may influence method choice, "
            "ensemble membership, weights, thresholds or any cue rule",
            "no calibration model may be run on holdout images before the final "
            "experiment",
            "structural sanity checks are allowed and must be logged",
        ],
        "sanity_checks_performed_so_far": [
            {"dataset": "HanCo tester", "date": "2026-09-23",
             "what": "counted rgb directories and images, read image size and "
                     "mode of one file, read K and M shapes from the calib json, "
                     "and measured whether K is constant across frames",
             "what_was_NOT_done": "no calibration model was run, no image content "
                                  "was inspected visually, no metric was computed",
             "finding": "8 cameras, 536 images, 224x224, per-frame K varies "
                        "because the images are crops"},
            {"dataset": "InterHand2.6M", "date": "2026-09-23",
             "what": "read the existing preparation manifest only",
             "what_was_NOT_done": "no image was downloaded or read",
             "finding": "annotations only on disk"},
        ],
        "open_decision_for_the_user": (
            "Confirming the CAM-EXP-004 static-camera result needs a second rig "
            "with genuinely static cameras and full images. Nothing local "
            "provides that. The options are (a) download the InterHand2.6M image "
            "release, (b) download the full HanCo RGB release, or (c) accept that "
            "the multi-frame claim stays single-dataset. All three were "
            "previously excluded on size grounds, so this is a decision to make "
            "rather than something to assume."),
    }
    write_json(MANIFESTS / "final_confirmatory_holdout_v1.json", holdout)
    write_json(RUN_DIR / "results" / "summary" / "confirmatory_holdout_plan.json",
               {"holdout": holdout, "development_protocol": split})
    print("wrote holdout + split manifests")
    print(f"  sequences: {[(s, len(v)) for s, v in per_seq.items()]}")


if __name__ == "__main__":
    main()
