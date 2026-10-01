# CAM-EXP-007 — Learned Hand-Derived Camera Cue Diagnostic

```
DECISION TAGS
  H_NO_GENERALIZABLE_HAND_DERIVED_SIGNAL
  G_SHUFFLE_CONTROL_NOT_DEGRADED
  F_SINGLE_FOCAL_PRIOR_CONFOUND   (controls only, as designed)
```

**One sentence:** Under the current GigaHands rig, selected frozen
AnyHand-WiLoR outputs and one mean-pooled backbone representation did not
provide a generalisable linearly decodable signal for AnyCalib's view-level
focal residual on unseen physical cameras — every hand feature group performed
at the level of its own shuffled control.

This is a negative result for the tested representations and linear-probe
protocol, **not** evidence that all learned hand models contain no camera
information. The narrow focal distribution of this rig is a major limitation,
and a varied-focal dataset is the most important next validation requirement.

## 1. Plain-language summary

CAM-EXP-006 tried to compute the focal length by fitting a known 3D hand onto
the image with geometry. That did not work, because a hand at this distance
does not constrain the focal.

CAM-EXP-007 asks something different. A hand-estimation network has seen an
enormous number of images. Perhaps its outputs, or its internal features,
quietly encode a hint like *"on this camera the focal estimate will come out
higher than usual"*. If such a hint exists, a simple linear probe should be
able to read it.

We were careful about one trap. On this rig every camera has almost the same
focal (it varies by only 1.9 %), so a model that simply memorises "about
920 px" looks excellent without understanding anything. To count as a real
cue, a hand feature had to (a) beat a plain constant correction, (b) work on
physical cameras never seen in training, and (c) get *worse* when we scramble
which hand belongs to which view.

**None of the hand features passed.** Scrambling the features changed almost
nothing, which is the clearest sign that no real hand-to-camera correspondence
was being used.

## 2. What was tested

| item | value |
| --- | --- |
| hand model | AnyHand-WiLoR, `models/anyhand_wilor.ckpt`, sha256 `9709eca6e77fb77d…` |
| detector | WiLoR YOLO, `models/detector.pt`, sha256 `5ef3df44e42d2db5…` |
| latent tap | `model.backbone` output index 3 (`vit_out`), (B,1280,16,12) → global average pool → **1280-d** |
| frames | the frozen CAM-EXP-006 16-frame nested grid, unchanged |
| target | CAM-EXP-005 `anycalib_signed_log_bias`, reused exactly |
| probe | ridge regression, α ∈ {0.01, 0.1, 1, 10, 100}, chosen by grouped CV **inside** the training fold |
| primary protocol | leave-one-physical-camera-out, 40 folds |

The latent hooks were verified **bit-identical**: the model's own outputs are
unchanged with the hooks attached (max abs difference `0.0`).

### 2.1 Coverage

| | views | physical cameras | sequences |
| --- | ---: | ---: | ---: |
| full benchmark | 175 | 40 | 5 |
| hand-feature eligible | **160** | 40 | 5 |
| with latent | 162 | — | — |

15 views were excluded by the pre-registered rule (fewer than 4 of 16 frames
with a usable hand); 13 of those had **zero** detections. They fall in
plant (8), dog (4) and boxing (3). All paired comparisons run on the common
eligible set of 160 views.

## 3. Results — leave-one-physical-camera-out

| probe | features | median focal err | within ±5 % | bias MAE | Spearman [CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| `B3` constant-reference **oracle** | 0 | **0.93 %** | 97.5 % | 0.0127 | +0.970 [+0.947, …] |
| `C1` AnyCalib-focal-only **control** | 1 | **1.03 %** | 96.9 % | 0.0132 | +0.969 [+0.946, …] |
| P1 F0 box / visibility | 20 | 4.77 % | 51.2 % | 0.0624 | −0.085 [−0.274, …] |
| **B1 global-bias corrected** | 0 | **4.80 %** | 53.1 % | 0.0617 | — |
| P0 training mean | 0 | 4.80 % | 50.6 % | 0.0621 | — |
| P2 F1 `pred_cam` head | 6 | 4.81 % | 50.6 % | 0.0642 | −0.420 |
| B2 CAM-005 scene | 31 | 4.87 % | 51.2 % | 0.0600 | **+0.279 [+0.098, …]** |
| P4 F3 temporal | 10 | 4.99 % | 50.6 % | 0.0630 | −0.224 |
| P8 scene + explicit | 51 | 5.07 % | 49.4 % | 0.0617 | +0.235 |
| P9 scene + latent | 31+ | 5.19 % | 49.4 % | 0.0660 | +0.032 |
| P7 explicit + latent | 78+ | 5.20 % | 46.2 % | 0.0658 | +0.015 |
| P5 all explicit | 78 | 5.23 % | 48.1 % | 0.0638 | +0.035 |
| P3 F2 hand geometry | 42 | 5.26 % | 46.2 % | 0.0633 | +0.087 |
| P6 **latent** | 1280→32 PCA | 5.44 % | 46.9 % | 0.0659 | +0.005 |
| P10 scene + all hand | — | 5.51 % | 46.2 % | 0.0690 | −0.054 |
| B0 raw AnyCalib | — | 9.70 % | 20.0 % | 0.1101 | — |

**Reading it:** correcting AnyCalib with a single constant (B1) halves the raw
error, 9.70 % → 4.80 %. *Every* hand feature group then sits within ±0.7 pp of
that constant, and most are worse. The best hand group, P1, improves on B1 by
0.7 % relative — the pre-registered threshold was 10 %.

A caution on the correlation column: B0, B1 and P0 emit a (nearly) constant
prediction, so their Spearman against the actual bias is numerically defined
only through small fold-to-fold differences in the training median. Those large
negative values are artefacts of ranking a near-constant and carry no meaning.
The correlations for the *feature-based* probes are the interpretable ones.

## 4. The controls decide this

### 4.1 Shuffling the hand features changes almost nothing

| probe | real features | view-shuffled | real Spearman | shuffled Spearman |
| --- | ---: | ---: | ---: | ---: |
| P5 all explicit | 5.23 % | 5.26 % | +0.035 | −0.063 |
| P6 latent | 5.44 % | **4.91 %** | +0.005 | −0.006 |
| P7 explicit+latent | 5.20 % | 5.19 % | +0.015 | −0.028 |

If a genuine hand-to-camera correspondence were being exploited, destroying it
should clearly hurt. It does not. For the latent the shuffled version is
*better* than the real one. Tag `G_SHUFFLE_CONTROL_NOT_DEGRADED`.

Two further controls land in the same band, confirming this is chance-level
behaviour: random Gaussian features of matched dimension give 5.04–5.77 %, and
shuffling the training targets gives 5.04–5.37 %.

### 4.2 The single-focal prior confound is enormous — and is a control

`C1`, a probe given **one** feature, `log(f_anycalib)`, reaches **1.03 %**
median error with Spearman **+0.969**. The `B3` constant-reference oracle
reaches 0.93 %.

This is exactly the `SINGLE_FOCAL_PRIOR_CONFOUND`. Because the reference focal
varies by only 1.90 % across the rig, `log(f_any / f_ref)` is almost a
deterministic function of `log(f_any)`. Both are **confound controls, not
methods**: `C1` consumes AnyCalib's own prediction and `B3` consumes training
target labels. Neither is ever counted as a hand signal, and neither was
allowed into any primary feature set.

This is also why the target was defined as the residual bias rather than the
focal itself — and the size of the C1 result vindicates that pre-registered
choice.

### 4.3 LOSO is no better, so this is not even rig repetition

| probe | LOCO camera | LOSO sequence |
| --- | ---: | ---: |
| P5 | 5.23 % | 5.72 % |
| P6 | 5.44 % | 5.72 % |
| P7 | 5.20 % | 6.09 % |

Tag `E_RIG_REPETITION_SIGNAL_ONLY` does **not** fire: the hand probes fail
under both protocols.

## 5. Incremental value over scene cues — none

| condition | median err | within ±5 % | bias MAE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| B2 scene only | 4.87 % | 51.2 % | 0.0600 | +0.279 |
| + all explicit hand | 5.07 % | 49.4 % | 0.0617 | +0.235 |
| + latent | 5.19 % | 49.4 % | 0.0660 | +0.032 |
| + all hand | 5.51 % | 46.2 % | 0.0690 | −0.054 |

Adding hand features makes the scene baseline worse on every metric. Tag
`D_HAND_SIGNAL_INCREMENTAL_TO_SCENE` does not fire.

Worth recording: CAM-005's scene features retain the only genuinely positive
bias correlation here, Spearman **+0.279, CI [+0.098, +0.470]** — a CI
excluding zero. That signal is weak and does not convert into a better focal,
but it is real and it is *not* matched by any hand feature.

## 6. Constant collapse and focal tracking

No probe collapses to a constant — corrected-focal CV ranges from 4 % to 12 %
against the reference CV of 1.90 %. The estimates are **scattered**, not
collapsed. So the failure is noise, not degenerate memorisation.

## 7. Non-default-focal stress subsets

| subset | views | cameras | B1 | B2 scene | P5 hand | P6 latent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S2 (≥2 % from rig median) | 33 | 21 | 5.60 % | 5.28 % | 5.85 % | 4.83 % |
| S5 (≥5 %) | 4 | 4 | 14.80 % | 11.53 % | 12.56 % | 11.62 % |

S5 is flagged `UNDERPOWERED_STRESS_SUBSET` (4 views, 4 cameras) and no
conclusion is drawn from it.

On the GT-defined S2 stress subset the latent probe had a lower median error
than B1 (**4.83 % vs 5.60 %**). This is a secondary stress diagnostic, **not**
evidence of a detected hand-derived camera cue: the subset is
reference-focal-defined and therefore analysis-only, it is not the headline
benchmark, it was not a pre-registered route to success, and the corresponding
shuffle-control superiority was not established for it. It is reported because
it exists, not because it supports a signal.

## 8. Most predictive individual features

Feature-level associations are reported descriptively using Spearman
correlation with physical-camera cluster-bootstrap confidence intervals. **No
formal multiple-comparison correction is applied**, and these univariate
associations are not a success criterion.

Three of 78 pre-registered explicit scalar features had intervals excluding
zero:

| feature | group | Spearman vs signed bias | CI |
| --- | --- | ---: | --- |
| `mano_beta_7__median` | F2 | −0.289 | [−0.440, −0.113] |
| `availability_rate` | F0 | −0.200 | [−0.360, −0.031] |
| `finger_spread__iqr` | F2 | −0.141 | [−0.264, −0.016] |
| `bbox_cy_norm__median` | F0 | +0.158 | [−0.064, +0.350] |
| `mano_beta_5__median` | F2 | −0.138 | [−0.282, +0.021] |

Because many features were inspected and no formal multiple-comparison
correction was applied, these are treated as **descriptive associations only**,
not statistically confirmed findings. Three out of 78 is close to what
inspecting that many features produces by itself, and crucially **none of them
translated into unseen-camera predictive performance** — the probes containing
them fail.

## 9. Hypotheses

| | outcome |
| --- | --- |
| H1 some explicit output is associated with the residual bias | **weakly**: 3 of 78 features have CIs excluding zero |
| H2 a real cue must beat B1 under LOCO-camera | **FAIL** for every group |
| H3 simple cues and the latent differ | both at chance; the latent is slightly worse |
| H4 latent beats explicit → hidden camera prior | **not observed** |
| H5 real features must beat shuffled | **FAIL** — shuffling barely changes anything |
| H6 LOSO-only gain → rig repetition | not applicable; LOSO also fails |
| H7 scene+hand beats scene-only | **FAIL** — it is worse |
| H8 nothing beats B1 and scene → no usable signal detected | **this is the outcome** |

## 10. What this does and does not establish

**Establishes.** Under this dataset and validation protocol, no linearly
decodable camera-residual signal was detected in the selected frozen
representation of AnyHand-WiLoR, nor in its explicit outputs, that generalises
to unseen physical cameras or improves on a single constant correction.

**Does not establish.**

* That the network contains no camera information. A *linear* probe on *one*
  pooled layer is a narrow instrument.
* That other layers, other poolings or a non-linear read-out would also fail.
  Only one layer was frozen, deliberately, so that this is a clean test rather
  than a search.
* That hand-derived camera cues are impossible in other regimes. GigaHands has
  a 1.9 % focal spread, which leaves very little for any per-camera method to
  demonstrate. That narrow spread is a major limitation, but it does not by
  itself establish that the representation side is adequate — this run tested
  one checkpoint, one layer, one pooling and one linear read-out, so a
  representation-side limitation cannot be excluded either.

The correct wording is: *no linearly decodable camera-residual signal was
detected in the selected frozen representation under this dataset and
validation protocol.*

## 11. CAM-EXP-007.1

**Recommendation: method development is not justified by the present
evidence.**

No pre-registered hand feature group beats the global-bias baseline by the
required margin, and the shuffle controls do not degrade. Building a correction
head on this basis would be fitting noise.

In addition, the narrow focal distribution of the current GigaHands rig is a
major limiting factor and prevents a strong conclusion about whether richer
hand-model representations could carry useful camera information. A
varied-focal dataset should precede any new correction head, and is the most
important next validation requirement.

Both constraints are real and neither is singled out: this experiment tested
one frozen hand model, one selected representation layer, global-average
pooling and a linear probe, so a representation-side limitation cannot be
excluded either.

If hand-derived cues are revisited later, the honest next step is a *different
instrument* — several layers, spatial structure retained, and a non-linear
read-out, pre-registered as a search with multiplicity accounted for — not a
correction head built on this null result.

## 12. Limitations

1. One linear probe, one pooled layer, one model. Deliberately narrow.
2. `SINGLE_FOCAL_RIG_CONFOUND`: reference focal CV 1.90 %; a constant scores
   0.93 %. This caps every per-camera conclusion on GigaHands.
3. 160/175 views eligible; the 15 excluded are concentrated in three
   sequences, so coverage is not uniform across content.
4. Mean-pooling the latent discards spatial structure, which could matter for
   a perspective cue.
5. The target is AnyCalib's residual. A different calibration model would give
   a different target; GeoCalib was excluded because it is non-deterministic.
6. S5 stress subset underpowered.
7. The external confirmatory holdout remains unopened.

## 13. Provenance

Phase A (source audit, frame manifest, feature groups, latent layer, splits,
feature extraction, quality audit) completed and froze before the target was
read. `src/extract_hand_features.py` and `src/run_hand_inference.py` contain no
reference to the target, the reference focal or any AnyCalib output.

Phase B (`src/run_probes.py` onward) is the first code that reads
`view_targets.csv.gz`.

No post-hoc analysis was added after seeing results.

One implementation correction is recorded as
`IMPLEMENTATION_CORRECTION_BEFORE_PERFORMANCE_INSPECTION`. The target had
already been loaded by the Phase-B probe script when the scene-baseline
feature-count mismatch was detected. However, **no model performance metric had
been inspected** — `evaluate.py` had not been run. The mismatch was identified
from the feature-count metadata itself (`scene feats 62` instead of the frozen
CAM-EXP-005 P6 set of 31 features), caused by matching CAM-005 feature names on
their base name so that `x` and `x__iqr` both passed. The selector was
corrected to exact-name matching before performance evaluation, and only the
corrected run is used as the scientific result.

This is deliberately **not** described as a correction made before the target
was read: it was not. Full account in
[`analysis_provenance.md`](analysis_provenance.md) §2.

## 14. Reproduce

```
python src/audit_hand_model.py
python src/freeze_spec.py                      # freeze BEFORE anything else
PYOPENGL_PLATFORM=win32 <venv-anyhand> src/run_hand_inference.py
python src/extract_hand_features.py            # target-blind
python src/feature_quality_audit.py            # target-blind
PYOPENGL_PLATFORM=win32 <venv-anyhand> src/verify_hook_identity.py
python src/run_probes.py                       # FIRST read of the target
python src/evaluate.py
python src/feature_associations.py
python src/figures.py
```
