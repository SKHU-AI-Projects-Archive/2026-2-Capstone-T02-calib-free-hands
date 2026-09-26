# CAM-EXP-005 — Static-view focal-bias attribution & scene/viewpoint cue analysis

**Question.** CAM-EXP-004/004.1 showed that a static camera's residual focal
error is a *stable per-view offset*, not frame noise, and that 1→64 frames do not
remove it. Why does a particular static view get a consistently high or low
focal prediction — and can any deployable RGB cue recover that offset on a camera
it has never seen?

**Two answers, and they point in opposite directions.**

1. **The bias is associated with physical camera identity, and repeats across sequences.** Physical camera identity describes the
   per-view bias (adjusted R² **0.383** in-sample); sequence identity describes
   essentially none of it (adjusted R² **−0.009**). Different sequences agree on
   which camera is biased (median Pearson r **+0.422** over the 10 sequence
   pairs), and 28 of 40 cameras keep the same sign of bias in every sequence they
   appear in. The structure sits with the view / camera, not with the activity
   or scene content.

2. **No deployable RGB cue recovers it on an unseen camera.** Under
   leave-one-physical-camera-out, no scene-geometry, image-global or border
   feature group beat the plain mean-bias baseline by enough to matter: the best
   was a **3.15 %** relative reduction in median focal error and **+2.29 pp**
   within ±5 %, against pre-registered thresholds of 10 % and +5 pp. And no
   *provided* camera property — k1, k2, principal-point offset, fx/fy, or camera
   pose — had a camera-cluster confidence interval excluding zero either.

**Decision tags: `PHYSICAL_CAMERA_PATTERN_STRONG`,
`NO_USEFUL_DEPLOYABLE_RGB_SIGNAL`, and a new one this run forced —
`SINGLE_FOCAL_RIG_CONFOUND`.** Details in §7.

Nothing here is a causal claim. Everything below is association, attribution or
predictive signal, in that language.

---

## 1. Setup

| | |
|---|---|
| views | **175** static `(sequence, camera)` views |
| sequences | **5** |
| unique physical cameras | **40**, every one appearing in more than one sequence |
| frames per view | 64, the frozen CAM-EXP-004.1 nested set |
| primary model | AnyCalib `anycalib_gen` + `radial:2`, predictions read from the frozen CAM-EXP-004.1 outputs — **no new inference** |
| target | `e_v = median_t log(f_pred / f_ref)`, see `target_definition.md` |
| observational unit | the view. Frames are repeated observations; 64 × 175 is never treated as 11,200 samples |
| association CI | physical-camera cluster bootstrap, 40 clusters, 10,000 iterations |

AnyCalib is primary because CAM-EXP-004.1 measured it as 100 % bit-identical
across repeated runs, so no run-to-run variation enters the target. GeoCalib is
0 % bit-identical there, and the frozen E2 ensemble — which contains it — is
secondary only and is not modified.

Median signed bias: **−0.0974 log = −9.28 %**, camera-cluster CI
[−10.94 %, −7.48 %].

## 2. The bias shows repeatable physical-camera structure (H1, H2 — both supported)

**Repeatability across sequences** (`results/summary/camera_across_sequence_repeatability.csv`):

* spread of the per-camera mean bias **across cameras**: sd 5.82 %
* spread of one camera's bias **across sequences**: median sd 4.62 %
* sign consistent in every sequence: **28 / 40** cameras
* between-camera share of the total sum of squares: **52.2 %**
* one-way ICC-like repeatability index: **0.388** (unbalanced design, 40 cameras
  over 5 sequences — a descriptive index, not an inferential ICC)

**Sequence-pair agreement** (`sequence_pair_camera_correlations.csv`): using only
the cameras two sequences share, Pearson r ranges **+0.171 to +0.804**, median
**+0.422**, over all 10 pairs.

**Descriptive models** (`camera_sequence_descriptive_models.csv`, in-sample fit,
not a causal variance share):

| model | params | R² | adjusted R² | MAE (log) |
|---|---|---|---|---|
| M0 intercept | 1 | 0.000 | 0.000 | 0.0616 |
| M_SEQ sequence id | 5 | 0.014 | **−0.009** | 0.0607 |
| M_CAM physical camera id | 40 | 0.522 | **0.383** | 0.0407 |
| M_BOTH | 44 | 0.537 | 0.385 | 0.0399 |

Adding sequence on top of camera buys almost nothing (0.383 → 0.385). Camera
identity is `DIAGNOSTIC_ONLY`: a new deployment camera has no row in any lookup
table, which is exactly why the probe below exists.

**Plain-language version.** If the same camera keeps predicting about 10 % low
no matter whether the person is making tea, boxing or handling a plant, the
problem is not what is happening in front of the lens on any one frame — it is
something about that camera's view.

## 3. Viewpoint audit

`tables/extrinsic_consistency_audit.csv`, 300 same-camera sequence pairs: the
provided extrinsics of one physical camera agree closely across sequences
(rotation difference median **0.86°**, camera-centre difference median **9.4 mm**).
Verdict **`COMPARABLE_ACROSS_SEQUENCES`**, so camera-orientation descriptors were
admissible as an oracle diagnostic. No top/front/diagonal label was invented;
only continuous descriptors (optical-axis elevation, camera height, distance)
were used.

## 4. What the bias is associated with (H3 — weakly supported at best)

`results/summary/feature_bias_associations.csv`, 42 pre-registered features,
Spearman rho with a camera-cluster CI. **7 of 42 have an interval excluding
zero.** Top-ranked among the pre-registered set:

| feature | group | rho | camera-cluster CI | CI excludes 0 |
|---|---|---|---|---|
| `self_pred_focal_px` | F_MODEL_SELF | **+0.969** | [+0.946, +0.982] | yes |
| `self_pred_k2` | F_MODEL_SELF | +0.702 | [+0.540, +0.834] | yes |
| `self_pred_k1` | F_MODEL_SELF | −0.632 | [−0.771, −0.462] | yes |
| `vp_consensus_strength` | F_SCENE_GEOM | **+0.229** | [+0.059, +0.378] | yes |
| `disagree_abs_log_any_vs_geo` | MODEL_DISAGREEMENT | −0.222 | [−0.403, −0.029] | yes |
| `disagree_signed_log_any_vs_geo` | MODEL_DISAGREEMENT | +0.206 | [+0.025, +0.373] | yes |
| `long_line_count` | F_SCENE_GEOM | +0.195 | [−0.000, +0.379] | no |
| `horizontal_line_fraction` | F_SCENE_GEOM | −0.188 | [−0.364, +0.002] | no |

**The top three are not findings.** `self_pred_focal_px` is the numerator of the
target, and with `f_ref` nearly constant on this rig the correlation is close to
definitional. The `k1`/`k2` associations are the same relationship seen through
AnyCalib's own coupled parameters.

The strongest genuinely-scene association is **vanishing-point consensus
strength, rho +0.229** — real but small, and small enough that it predicts
nothing useful (§6).

## 5. Oracle camera properties (H6 — not supported)

`results/summary/oracle_camera_property_associations.csv`, all marked
`ORACLE_DIAGNOSTIC_ONLY`. **No property's camera-cluster CI excludes zero.**
Strongest: fx/fy anisotropy rho +0.218 [−0.007, +0.427]; reference focal −0.145;
k2 −0.125; optical-axis elevation +0.121; principal-point offset +0.095; k1
+0.093.

On the distortion question CAM-EXP-003.1 left open: GT `k1` again shows
essentially no association with the residual bias (rho +0.093, CI spanning zero).
**This does not mean distortion is irrelevant.** GigaHands' |k1| spans only
0.34–0.43; the range is too narrow to resolve a dose-response relationship, the
same caveat as before.

So we could not name *which* camera property produces the offset, even with the
ground truth in hand. The bias shows clear per-camera structure (§2) but is not
explained by any single provided parameter we measured.

## 6. Can any cue predict the bias on a NEW camera? (H4 — no; H5 — not even that)

`LINEAR PROBE DIAGNOSTIC`, ridge, all preprocessing and alpha selection inside
the training fold. Not a calibration method.

**Leave-one-physical-camera-out (the primary criterion), bias prediction:**

| feature set | MAE of predicted bias (% equiv.) | R² | Spearman |
|---|---|---|---|
| P0 baseline (training mean) | 6.25 | −0.028 | — |
| P1 scene geometry | 6.18 | −0.000 | +0.133 |
| P2 image global | 6.35 | −0.054 | −0.170 |
| P3 border scene | 6.30 | −0.043 | −0.003 |
| P5 model disagreement | 6.13 | −0.018 | +0.032 |
| P6 all scene groups | 6.13 | +0.044 | +0.221 |
| P4 model self | **1.46** | +0.936 | +0.963 |
| P7 all deployable | 1.65 | +0.927 | +0.956 |
| D_CAMERA_ID *(diagnostic)* | 6.25 | −0.028 | — |

**Correction diagnostic** (`f_corr = f_pred / exp(b̂)`), leave-one-camera-out:

| | median rel. focal error | camera-cluster CI | within ±5 % | within ±10 % |
|---|---|---|---|---|
| original, uncorrected | 9.97 % | [7.67, 11.35] | 20.6 % | 50.3 % |
| P0 mean-bias baseline | 5.00 % | [3.92, 5.75] | 50.3 % | 80.0 % |
| P1 scene geometry | 4.84 % | [3.64, 5.84] | 52.6 % | 80.6 % |
| P6 all scene | 5.00 % | [3.71, 6.04] | 49.7 % | 81.1 % |
| P4 model self | 1.12 % | [0.83, 1.22] | 97.1 % | 99.4 % |
| **`CONSTANT_RIG_FOCAL_ORACLE`** *(POST_HOC_ORACLE_SANITY_CONTROL)* | **0.88 %** | [0.55, 1.18] | **97.7 %** | 99.4 % |

Against the pre-registered thresholds (`tables/decision_summary.csv`), on
leave-one-camera-out: P1 gives +3.15 % relative reduction and +2.29 pp within
±5 %; P2, P3, P5 and P6 give between −3.4 % and +2.8 %. **Every scene and
disagreement group fails both criteria.**

P4 and P7 pass both by a wide margin — **and this is the trap.** They do not beat
the control that ignores the image entirely. On this rig the reference focal
spans 824.9–986.6 px, CV **1.90 %**; always emitting the rig's median focal
(921.6 px) yields 0.88 % median error and 97.7 % within ±5 %, *better* than P4's
1.12 % / 97.1 %. P4's corrected focals have CV 0.82 % — it has collapsed to
approximately one number. What P4 learned is this rig's focal length, not how to
calibrate.

**Leave-one-sequence-out** tells the same story: P1 4.50 % vs P0 5.02 % median
error (MAE of predicted bias 6.06 vs 6.20). The LOSO/LOCO gap is small for every
deployable group, so this is not even the `RIG_OR_CAMERA_REPETITION_SIGNAL`
pattern of H5 — the scene features are weak in both protocols.

**Plain-language version.** We hoped the straight lines and perspective in the
scene would tell us which way a new camera's estimate is off. On cameras the
probe had never seen, they did not. And the one feature set that looked
spectacular was really just memorising that every camera in this lab has the
same lens.

## 7. Decision tags

* **`PHYSICAL_CAMERA_PATTERN_STRONG`** — the bias shows repeatable structure per physical camera
  across sequences (adjusted R² 0.383 vs −0.009 for sequence; pairwise r +0.422).
* **`NO_USEFUL_DEPLOYABLE_RGB_SIGNAL`** — no scene, image-global, border or
  disagreement group met either pre-registered threshold on a held-out camera.
* **`SINGLE_FOCAL_RIG_CONFOUND`** *(new; not in the original tag list, added
  because the data forced it)* — the rig's reference focal varies by only 1.90 %,
  so an image-free constant beats every probe and any feature set containing the
  predicted focal wins for the wrong reason. This limits what CAM-EXP-005 could
  have shown in the first place.
* **not** `SEQUENCE_CONTENT_EFFECT_STRONG` — sequence identity explains nothing.
* **not** `DEPLOYABLE_RGB_SIGNAL_PROMISING`, **not** `MODEL_SELF_DIAGNOSTIC_PROMISING`
  (P4 fails the control), **not** `ORACLE_VIEWPOINT_SIGNAL_ONLY` (no oracle
  property reached significance either).

## 8. What we may and may not say

| may say | may not say |
|---|---|
| the per-view bias is **associated with** physical camera identity and repeats across sequences | the camera viewpoint **causes** the bias |
| vanishing-point consensus is associated with the signed bias (rho +0.229, CI excludes zero) | scene geometry predicts the bias on a new camera |
| no deployable RGB group met the pre-registered thresholds on a held-out camera | RGB scene cues are useless in general — this is one rig with one focal length |
| no provided camera property reached significance | the camera's optics are irrelevant |
| GT k1 again shows no association | distortion does not matter (|k1| spans only 0.34–0.43 here) |
| P4 reaches 1.12 % median error | P4 is a good calibration method — it loses to an image-free constant |

## 9. Limitations

1. **The single-focal rig.** The dominant limitation. With reference focal CV
   1.90 %, this dataset cannot separate "predicting the focal" from "knowing the
   rig's focal", and it compresses the dynamic range any cue would need.
2. **Classical features only.** The pre-registered set is OpenCV-level geometry
   and image statistics. A learned representation might carry viewpoint
   information these do not. That was deliberately out of scope, and its absence
   is not evidence that no cue exists.
3. **40 cameras, 5 sequences, one lab.** The camera-cluster bootstrap respects
   the structure, but 40 is not many, and all 40 sit on one rig.
4. **In-sample descriptive fit.** The camera-identity R² is an upper bound with
   40 free parameters for 175 views, not a variance decomposition.
5. **Association only.** Nothing here isolates a mechanism.
6. **Hand cues untouched by design** — see below.

## 10. What was deliberately not done

No hand information of any kind entered this experiment: no provided 2D or 3D
annotation, no hand geometry, no MANO, no predicted joints or depth, no detector
box. The border-region features are generic image statistics, not a hand mask.
The final confirmatory holdout was not opened. The frozen E2 definition was not
modified. No new calibration model was built. Registered as
`OPEN_NEXT_EXPERIMENT`.

## 11. Next experiment

**Go to CAM-EXP-006 (hand cues), not CAM-EXP-005.1.**

`CAM-EXP-005.1` (scene-aware bias correction under strict grouped validation)
was conditional on tag C, `DEPLOYABLE_RGB_SIGNAL_PROMISING`. That tag was not
earned: the best scene group reached 3.15 % relative reduction against a 10 %
threshold, and 2.29 pp against a 5 pp threshold, on the new-camera test. Building
a scene-aware corrector on that would be building on noise.

Two things should accompany CAM-EXP-006:

* **A rig with genuinely different focal lengths.** Until the reference focal
  varies, no experiment on this data can distinguish calibration from constant
  memorisation. This strengthens the case already registered in
  `OPEN_ISSUES.md` for the external image download decision.
* **Hand geometry as the remaining untried cue.** The hand is the one object in
  the frame whose approximate metric size is known a priori, which is precisely
  the information a focal estimate needs and which no scene feature here
  supplied. CAM-EXP-005 has now closed the RGB-scene route under strict
  new-camera validation, which is the evidence CAM-EXP-006 was waiting for.

## 12. Artifacts

`config.json`, `environment.json`, `target_definition.md`,
`scene_feature_audit.md`, `probe_protocol.md`;
`results/raw/` (view targets, 11,200 frame features, probe fold predictions);
`results/summary/` (camera repeatability, sequence-pair correlations, descriptive
models, variance split, view features, feature associations, oracle associations,
both probe summaries, correction summary, extrinsic verdict);
`tables/` (feature definitions, quality audit, extrinsic audit, leakage audit,
statistical units, decision summary);
`figures/` 01–08 plus `CAM_EXP_005_MAIN_EXPLANATION.png`;
new frozen manifests `cam_exp_005_scene_feature_spec_v1.json`,
`cam_exp_005_probe_spec_v1.json`, `cam_exp_005_outer_splits_v1.json`.

CAM-EXP-001 through CAM-EXP-004.1 and `experiments/report_prep/progress_report_v1`
were read only and are unmodified.
