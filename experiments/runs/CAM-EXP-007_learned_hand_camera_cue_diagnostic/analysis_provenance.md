# CAM-EXP-007 — analysis provenance

Records what was frozen before the target was read, what happened after, and
the evidence level of each. No numerical result is changed by this document.

## 1. Phase separation

| phase | code | reads the target? |
| --- | --- | :-: |
| A | `audit_hand_model.py`, `freeze_spec.py`, `run_hand_inference.py`, `extract_hand_features.py`, `feature_quality_audit.py`, `verify_hook_identity.py` | **no** |
| B | `run_probes.py`, `evaluate.py`, `feature_associations.py`, `figures.py` | yes |

Verified at AST level by `src/verify_phase_separation.py`:
`PHASE_SEPARATION_HELD`. Phase A imports and uses no target symbol.

Frozen before any target was read: the frame grid (reused from CAM-EXP-006),
the five feature groups, the single latent layer and its pooling, the PCA
dimension, the outer splits, the probe list, the baselines, the controls, and
the four success criteria.

## 2. Implementation correction before performance inspection

This is the one event in CAM-007 whose provenance needs stating precisely.

**Label: `IMPLEMENTATION_CORRECTION_BEFORE_PERFORMANCE_INSPECTION`.**

| question | answer |
| --- | --- |
| had the target been loaded when the mismatch was found? | **yes** — `run_probes.py` is a Phase-B script and loads `view_targets.csv.gz` at start-up |
| had any performance metric been inspected? | **no** — `evaluate.py` had not been run; no error, correlation or ranking existed yet |
| what revealed the problem? | the run's own feature-count metadata: it printed `scene feats 62` |
| why was that wrong? | the frozen CAM-EXP-005 scene baseline (its P6) is an exact set of **31** features |
| what was the bug? | the selector matched CAM-005 feature names on their *base* name, so `x` and `x__iqr` both passed, silently redefining the frozen baseline as 62 features |
| what changed? | the selector was changed to exact-name matching, restoring 31 |
| what became the scientific result? | the corrected run only |

Canonical wording for the report:

> The target had already been loaded by the Phase-B probe script when the
> scene-baseline feature-count mismatch was detected. However, no model
> performance metric had been inspected. The mismatch was identified from the
> feature-count metadata itself (62 features instead of the frozen CAM-EXP-005
> P6 set of 31), and the selector was corrected to exact-name matching before
> performance evaluation.

**What this must not be called:** `PRE_TARGET_CORRECTION`,
`PRE_REGISTERED_FIX`, or "result-blind before target load". The target *was*
loaded. What is true, and weaker, is that the correction was driven by a
frozen-spec mismatch rather than by a performance number.

### 2.1 The superseded run

The initial 62-feature run produced fold predictions that were overwritten by
the corrected run. Those numbers appear in no table, figure or conclusion, and
are not reproduced here. Only the fact that the run existed is recorded.

## 3. Feature-level associations are descriptive only

`src/feature_associations.py` computes Spearman correlations with
physical-camera cluster-bootstrap confidence intervals. It applies **no formal
multiple-comparison correction**.

An earlier docstring in that file claimed a BH-FDR supplement was reported
alongside. It never was. The docstring was corrected; the computation is
unchanged, and no BH-FDR analysis was added in this correction pass, because
adding a new analysis is outside the scope of a documentation correction.

Consequence for interpretation: three of 78 pre-registered explicit scalar
features had intervals excluding zero. Because many features were inspected and
no correction was applied, these are **descriptive associations only**, not
statistically confirmed findings. None translated into unseen-camera predictive
performance.

## 4. No post-hoc analyses

No analysis was added after results were seen. The probe list, baselines,
controls, protocols and success criteria were all executed exactly as frozen.

## 5. Frozen manifests are not retro-edited

`cam_exp_007_feature_spec_v1.json`, `cam_exp_007_latent_spec_v1.json`,
`cam_exp_007_probe_spec_v1.json`, `cam_exp_007_control_spec_v1.json`,
`cam_exp_007_outer_splits_v1.json` and `cam_exp_007_frame_manifest_v1.csv.gz`
are left byte-identical. Their value is as a dated record of what was committed
to before results existed; corrections live in this document and in the report.
