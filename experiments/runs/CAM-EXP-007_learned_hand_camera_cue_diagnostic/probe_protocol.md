# CAM-EXP-007 — probe protocol

Frozen in `experiments/manifests/cam_exp_007_probe_spec_v1.json`.

## 1. Target

Reused **exactly** from CAM-EXP-005, not redefined here:

```
e_v = median_t log( f_AnyCalib(v,t) / f_ref(v) )
```

source `CAM-EXP-005/results/raw/view_targets.csv.gz`, column
`anycalib_signed_log_bias`, AnyCalib `anycalib_gen` + `radial:2`.

**Why not predict the focal directly.** The rig's reference focal has CV
1.90 %, so a model that emits a constant looks excellent. The residual bias is
the question that cannot be answered by memorising the rig.

## 2. Correction diagnostic

```
f_corrected = f_anycalib / exp(bias_hat)
```

Label: `LINEAR_PROBE_CORRECTION_DIAGNOSTIC`. Not a calibration method.

## 3. Model

Ridge regression, closed form, alpha in {0.01, 0.1, 1, 10, 100}, selected by
grouped CV **inside the training fold**. No network is trained.

## 4. Preprocessing, per outer fold

1. training-fold median imputation
2. training-fold standardisation
3. training-fold PCA (latent groups only)
4. inner grouped-CV alpha selection
5. full training fit
6. predict the held-out physical camera

The outer test set never influences any fitted quantity.

## 5. Protocols

| | folds | what it tests |
| --- | ---: | --- |
| `LOCO_PHYSICAL_CAMERA` | 40 | **PRIMARY.** A physical camera and all its views are held out. |
| `LOSO_SEQUENCE` | 5 | SECONDARY. Content shift; the same camera is usually in training, so this is *not* a new-camera test. |

## 6. Baselines and controls

| id | what | role |
| --- | --- | --- |
| B0 | raw AnyCalib | reference point |
| B1 | global-bias corrected AnyCalib | **PRIMARY PRACTICAL BASELINE** |
| B2 | CAM-005 scene-only, its frozen P6, 31 features | scene comparison |
| B3 | training-fold median reference focal | `ORACLE_SANITY_CONTROL`, not deployable |
| C1 | `log(f_anycalib)` alone | `SINGLE_FOCAL_PRIOR_CONFOUND_CONTROL`, never hand signal |
| C2 | hand features shuffled among views | **PRIMARY NEGATIVE CONTROL**, 10 seeds |
| C3 | random Gaussian features, matched dimension | secondary |
| C4 | training targets shuffled | secondary |

B2 reuses CAM-005's scene groups `F_SCENE_GEOM + F_IMAGE_GLOBAL +
F_BORDER_SCENE` exactly, matched on exact column names. CAM-005's P4 and P7 are
**not** used as the scene baseline: they contain the model's own prediction,
which is the confound C1 isolates.

## 7. Success criteria, frozen before results

`HAND_DERIVED_SIGNAL_PROMISING` requires **all four**:

1. versus B1: median relative focal error down ≥ 10 % relative, **or**
   within-±5 % up ≥ 5 pp
2. bias-prediction Spearman positive with the cluster-bootstrap CI lower bound
   above zero
3. clearly better than the `C2` shuffle control
4. the corrected focal is not a constant collapse

## 8. Statistics

Statistical unit: the `(sequence, camera)` view. Generalisation group and
bootstrap cluster: the physical camera, 10 000 iterations. Frames and hands are
repeated observations and are never counted as independent samples.

All paired comparisons run on the **common eligible set** of 160 views, so no
probe is scored on an easier subset than its baseline.

## 9. Feature-level associations

`src/feature_associations.py` is a **descriptive supplement**, not part of the
decision procedure. It reports Spearman correlation with physical-camera
cluster-bootstrap confidence intervals and applies **no formal
multiple-comparison correction**. Intervals excluding zero there are
descriptive associations, not statistically confirmed findings, and they are
never a success criterion.

## 10. Implementation correction during Phase B

The scene-baseline selector initially matched CAM-EXP-005 feature names on
their base name, admitting 62 features instead of the frozen 31. The target had
already been loaded when this was found, but no performance metric had been
inspected; the mismatch was visible in the feature-count metadata alone. See
[`analysis_provenance.md`](analysis_provenance.md) §2. Label:
`IMPLEMENTATION_CORRECTION_BEFORE_PERFORMANCE_INSPECTION`.

## 11. A note on reading the correlation column

B0, B1 and P0 emit a (near-)constant prediction. Their Spearman against the
actual bias is defined only through small fold-to-fold changes in the training
median, so the large negative values in their rows are artefacts of ranking a
near-constant and carry no meaning. Only the feature-based probes' correlations
are interpretable.
