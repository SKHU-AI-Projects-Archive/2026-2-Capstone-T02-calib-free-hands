# Leakage audit

## The phase barrier

| phase | what happens | reference focal |
| --- | --- | --- |
| **A** | source audit, units, frames, folds, MANO prior, WiLoR cache, spec freeze | **CLOSED** |
| **B** | synthetic implementation gate | **CLOSED** |
| **C** | real label-free candidate score curves | **CLOSED** |
| **D** | nested fusion-lambda tuning | **TRAIN cameras only** |
| **E** | freeze test predictions | still no TEST focal |
| **F** | open the TEST reference focal | logged, see below |
| **G–I** | focal metrics, absolute 3D, controls, report | open |

The machine-readable version is `tables/reference_focal_access_audit.csv`.

## The structural problem, and how it is handled

The scene prediction cache reused from CAM-EXP-004.1 **physically contains**
`gt_fx`, `gt_fy`, `gt_cx`, `gt_cy` and derived error columns. A Phase A/B/C
script could therefore read the reference focal by accident.

Two defences:

1. **`common.read_scene_predictions()` drops every forbidden column at load.**
   Phase A/B/C code uses only that reader, so the reference focal is not present
   in the objects it handles. `build_units.py` asserts this at runtime.
2. **`self_audit.py` greps every Phase A/B/C script** for `gt_fx`, `gt_fy`,
   `gt_cx`, `gt_cy`, `gt_k1`, `gt_k2`, `reference_focal_map` and
   `reference_focal_all`, and fails on a hit.

## The opening log

`evaluate_focal.py` is the first script permitted to read the TEST reference
focal. It writes `results/raw/REFERENCE_FOCAL_OPENED.txt` containing the
timestamp, the git HEAD, and the SHA-256 of the frozen test predictions, the
method spec and the lambda selection.

`results/summary/frozen_predictions_hash.json` records the prediction hash at
freeze time, and the self-audit re-checks it afterwards. **Predictions are never
regenerated after the focal is opened**; a new question needs a new experiment.

## Never used in inference

| quantity | status |
| --- | --- |
| reference / GT focal | never an inference input |
| GT principal point, distortion, `fy/fx` ratio | never — the scene estimator's own principal point is used |
| absolute average hand size | never — only sum-normalised proportions |
| known object or table dimensions | never |
| EXIF | never |
| participant anatomy from another recording | never — `p_seq` is refit per unit |
| target-camera reference 3D | never in inference; evaluation only |
| `FINAL_CONFIRMATORY_HOLDOUT` (InterHand2.6M / HanCo) | **not opened** |

## Two subtler leakage paths, both closed

- **WiLoR's shape prior.** The network's predicted bone *lengths* would carry its
  own training hand-shape prior into a model meant to estimate anatomy. Only
  **unit bone directions** are taken; lengths are discarded
  (`common.bone_unit_directions`).
- **Prior built from the evaluation set.** The generic prior is the neutral MANO
  hand, not GigaHands statistics. No participant template, no test-unit bone
  mean, no CAM-EXP-009.3.1 per-bone value enters it.

## Selection hygiene

- `lambda_generic` and `lambda_side` were fixed on the **synthetic gate only**.
- `lambda_fusion` is selected per outer fold on **TRAIN physical cameras only**.
- The outer split unit is the **physical camera**, so no camera appears on both
  sides, and every sequence of a camera stays together.
- Eligibility was frozen before any focal result. No unit is dropped for a large
  focal error, an odd hand score, a boundary solution or a method
  underperforming — those are results.
- Two compute decisions (N=64 hand frames skipped; the wrong-bone control run
  for M3 only) were made from **measured runtime before any real focal result**
  and are recorded in the method spec with their reasoning.
