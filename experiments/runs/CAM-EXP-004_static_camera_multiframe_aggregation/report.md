# CAM-EXP-004 — Static-camera multi-frame aggregation & bias decomposition

**Setting.** Our deployment is monocular RGB video from a camera that is static
for the whole clip, with fixed intrinsics, available offline. That means we may
use every frame of the video to estimate one focal length. This experiment asks
what that actually buys.

**One-line answer: more frames remove frame-to-frame noise, and our best model
has almost no frame-to-frame noise to remove.** What is left is a stable
per-camera offset — a bias — and no amount of averaging touches a bias. The
thing that *did* help was combining two different calibration models, which cut
the median focal error from 9.89 % to 6.14 % and more than doubled the share of
cameras calibrated to within 5 % (20.0 % → 38.3 %).

### The intuition, before the tables

Point a fixed camera at a scene and take 8 photos. The true focal length is one
number — say 920 px — for all 8. If the model answers 1000, 1002, 998, 1001 …
then averaging helps a little: the scatter cancels. But our primary model
answers something like 835 every single time. Averaging a hundred of those still
gives 835. The 9 % error is not randomness we can average away; it is the model
being consistently wrong about *this* camera. Measured: **98.1 % of AnyCalib's
squared error is this fixed per-camera offset**, and only 1.9 % is scatter.

---

## Setup

- **Aggregation unit: `(sequence, camera)`** — one static camera inside one take,
  175 of them, 8 frames each. Frames were never pooled across sequences even
  where the same physical camera id reappears, because a deployment calibrates
  one video at a time.
- **Views**: the frozen CAM-EXP-003 manifest, `usable_for_camera_benchmark == 1`.
  Hand-annotation quality was **not** used as a filter; a camera-valid view is
  never discarded because its hand labels are REVIEW/EXCLUDE.
- **Static-camera sanity test**: `tables/static_camera_gt_check.csv` — all 175
  views have `fx, fy, cx, cy, k1, k2, p1, p2` **exactly constant** across their
  frames (max spread 0). `STATIC_CONFIRMED`. Without this the whole premise
  would be void.
- **PHASE A used zero model inference.** It reuses the per-frame predictions
  already produced by CAM-EXP-003 / 003.1 on the identical frozen frames. Those
  runs were read only and are unmodified.
- **Subsets**: every one of C(8,N) subsets is evaluated (8, 28, 70, 1), so no
  result depends on which frames one random draw happened to pick. Each view is
  then collapsed to its *expected* error over subsets, and **the view — not the
  subset — is the statistical unit** (175, not ~19 000).
- **Deployable models only**: AnyCalib `anycalib_gen`+`radial:2` (PRIMARY),
  GeoCalib `distorted`+`radial` (SECONDARY), PerspectiveFields uncentered on raw
  input (DIVERSITY BASELINE). CAM-EXP-003.1's GT-undistorted condition is an
  oracle and is excluded from every method here.

---

## Q1 — What happens from 1 → 2 → 4 → 8 frames?

Median relative focal error across the 175 views (median aggregation):

| N | AnyCalib-gen | GeoCalib-distorted | PF-uncentered | 2-model ensemble (E2) |
|---|---|---|---|---|
| 1 | 9.62 % | 13.37 % | 23.45 % | 7.73 % |
| 2 | 9.62 % | 13.37 % | 23.45 % | 7.36 % |
| 4 | 9.77 % | 11.65 % | 23.70 % | 6.61 % |
| 8 | **9.89 %** | **11.18 %** | **23.88 %** | **6.14 %** |

Paired view-level gain from N=1 to N=8, bootstrap 95 % CI:

| Model | gain (pp) | 95 % CI | views improved |
|---|---|---|---|
| AnyCalib-gen | **−0.02** | [−0.07, +0.05] | 49.1 % |
| GeoCalib-distorted | **+0.50** | [+0.31, +0.81] | 68.6 % |
| PF-uncentered | +0.19 | [−0.18, +0.36] | 54.9 % |

For the primary model the gain is indistinguishable from zero. GeoCalib gains a
real but small half a percentage point. The `within-5 %` rate tells the same
story more usefully: AnyCalib 18.9 % → 20.0 %, GeoCalib **14.3 % → 21.1 %**,
PF 6.9 % → 11.4 %.

**A provable reason, not just an observation.** With *mean* aggregation the
median error is bit-for-bit constant in N for AnyCalib (9.618 % at N=1,2,4,8) and
for PF (23.452 %). That is not a coincidence: if every frame of a view errs in
the same direction, then the mean of the absolute errors equals the absolute
error of the mean, so the expected error of a random N-frame subset is
*identical* to the single-frame error for every N. Measured share of views where
all 8 frames err in the same direction: **AnyCalib 91.4 %**, PF 82.9 %, GeoCalib
63.4 %. Averaging can only help where the sign flips — and for our best model it
almost never does.

## Q2 — Was Phase B (16/32/64 frames) needed?

**No. `SATURATED_BY_8_FRAMES`.** The stopping rule was written into
`config.json` and `tables/stopping_rule.csv` before the N-curve was inspected:
run Phase B only if the primary model's median error drops ≥ 10 % relative, or
its within-5 % rate rises ≥ 3 pp, from N=4 to N=8. Measured: median
9.767 % → 9.890 % (a **−1.26 %** relative *increase*), within-5 % 20.0 % → 20.0 %
(**+0.00 pp**). Neither criterion fires.

As a supplementary check outside the rule, the same thresholds were applied to
the secondary model and to the two best ensembles — GeoCalib 4.0 % relative drop
/ +0.57 pp, E4 3.7 % / −0.57 pp, E2 7.1 % / +0.00 pp. None would have triggered
Phase B either, so the decision does not rest on one model saturating while
others were still climbing. `results/summary/phase_b_decision.json`.

No new frames were decoded and no inference was run. The 16/32/64-frame manifest
was therefore never created.

## Q3 — Where did saturation happen?

Between N=1 and N=4 for the only model with real frame noise (GeoCalib: 13.37 →
11.65, then only 11.18 at N=8). For AnyCalib there was nothing to saturate.

## Q4 / Q5 — Best aggregation method

At N=8 the four aggregation rules are within a few hundredths of a percentage
point of one another for every model, and every paired CI spans zero:

| Model | mean | median | geomean | log-Huber |
|---|---|---|---|---|
| AnyCalib-gen | 9.62 | 9.89 | 9.63 | 9.73 |
| GeoCalib-distorted | 12.70 | **11.18** | 11.56 | 11.21 |
| PF-uncentered | 23.45 | 23.88 | 23.40 | 23.46 |

The choice of aggregator is close to irrelevant, with one exception that makes
sense: for the noisy model, the robust rules (median 11.18, log-Huber 11.21)
beat the mean (12.70) by ~1.5 pp, because GeoCalib has heavy-tailed outlier
frames (its p90 within-view sd is 40 %). The Huber tuning constant is the
standard `c = 1.345` with `scale = MAD/0.6745`; it was **not** tuned against GT.

Other static parameters, aggregated the same way (`results/summary/other_parameter_aggregation.csv`):
AnyCalib `k1` median abs error 0.078 and principal point 32.6 px; GeoCalib `k1`
0.150 (principal point `NOT_PREDICTED_BY_MODEL`, fixed at the image centre); PF
principal point 161 px, no distortion term. Nothing was invented where a model
does not predict it.

## Q6 — Cross-model ensemble

**This is the finding of the experiment.** In CAM-EXP-003.1 AnyCalib-gen's signed
bias was −9.3 % and GeoCalib-distorted's was +6.6 %: they are wrong in opposite
directions. A parameter-free, GT-blind combination exploits that. At N=8:

| Estimator | median | p90 | signed median | ≤5 % | ≤10 % | ≤20 % | gain vs primary (pp) |
|---|---|---|---|---|---|---|---|
| **E2** per-frame geomean, then median | **6.14 %** | 19.18 % | −2.36 % | **38.3 %** | 68.6 % | 90.3 % | **+2.23 [+1.32, +3.00]** |
| E4 per-view geomean | 6.37 % | 19.86 % | −2.25 % | 38.3 % | 67.4 % | 89.7 % | +2.29 [+1.45, +3.09] |
| E1 per-frame mean | 6.51 % | 20.56 % | −1.81 % | 38.3 % | 66.3 % | 89.1 % | +2.27 [+1.05, +2.99] |
| E3 per-view mean | 6.65 % | 21.05 % | −1.62 % | 38.9 % | 66.3 % | 89.1 % | +2.30 [+0.97, +2.96] |
| E5 three-model median | 8.54 % | 21.90 % | +1.72 % | 28.0 % | 57.1 % | 86.3 % | +0.56 [+0.00, +1.62] |
| single AnyCalib-gen | 9.89 % | 18.68 % | −9.25 % | 20.0 % | 51.4 % | 94.9 % | — |
| single GeoCalib-distorted | 11.18 % | 41.31 % | +6.20 % | 21.1 % | 41.7 % | 72.0 % | −2.10 |
| single PF-uncentered | 23.88 % | 66.46 % | +23.54 % | 11.4 % | 21.1 % | 44.0 % | −12.87 |

Yes: the two-model ensemble improves on the best single model by ~2.2 pp of
median error with the CI clear of zero, on 62–64 % of views, and the signed bias
collapses from −9.25 % to −2.36 %. Adding PF as a third member **hurts** (E5,
8.54 %) — PF is too weak and its +23.5 % bias drags the median. Whether to
combine per frame or per view barely matters; the geometric mean is marginally
better than the arithmetic one, consistent with focal error being multiplicative.

No weights were fitted. Doing so against this dataset's GT was forbidden and was
not done.

The gain does come at a cost worth stating: the ensemble's `≤20 %` rate (90.3 %)
is *below* AnyCalib alone (94.9 %), because GeoCalib's heavy tail leaks into it.
The ensemble is better in the middle of the distribution and slightly worse in
the far tail.

## Q7 — View bias vs frame noise

Decomposition of the squared log focal error, where `e_vt = log(f_pred_vt/f_gt_v)`,
view bias `b_v = mean_t e_vt`, residual `r_vt = e_vt − b_v`, and exactly
`mean(e²) = mean(b²) + mean(r²)`:

| Model | bias share | noise share | median \|view bias\| | median within-view sd |
|---|---|---|---|---|
| AnyCalib-gen | **98.1 %** | 1.9 % | 9.63 % | 1.43 % |
| GeoCalib-distorted | 53.4 % | 46.6 % | 11.56 % | 4.66 % |
| PF-uncentered | 96.0 % | 4.0 % | 23.40 % | 4.61 % |

This is the whole story in one table, and it is consistent with CAM-EXP-003.1's
finding that the residual after distortion handling is stable per-view bias.
GeoCalib is the only model with enough noise for averaging to matter — which is
exactly the model that gained from more frames (H2 confirmed; H1 confirmed).

Temporal-position diagnostic (`results/summary/temporal_position_diagnostic.csv`):
median centred log error by frame slot stays within ±1.5 % and is non-monotonic
for all three models, so there is no systematic early/middle/late drift to
exploit. Frames were sampled uniformly in time; no scene, hand or GT cue was
used to choose them.

## Q8 / Q9 — Oracle best-of-N, and is frame *selection* the problem?

`ORACLE_BEST_OF_N` picks the single frame in the subset with the smallest GT
focal error. **It uses GT, so it is not a method** and is never mixed with the
deployable numbers.

| Model | N=1 | N=2 | N=4 | N=8 | deployable median at N=8 | gap |
|---|---|---|---|---|---|---|
| AnyCalib-gen | 9.62 % | 8.62 % | 7.63 % | **6.85 %** | 9.89 % | 3.04 pp |
| GeoCalib-distorted | 13.37 % | 9.82 % | 6.58 % | **4.80 %** | 11.18 % | 6.39 pp |
| PF-uncentered | 23.45 % | 20.08 % | 17.19 % | 13.64 % | 23.88 % | 10.24 pp |

Read this carefully, because it is the hypothesis that dies here. For the
primary model, even *cheating* — knowing the answer and picking the single best
of 8 frames — only reaches 6.85 %, and the **deployable two-model ensemble
already beats it at 6.14 %**. So for AnyCalib the answer to Q9 is: there is no
hidden good frame to find; almost every frame carries the same bias. A perfect
frame-selection oracle would be worth at most ~3 pp, and we can get more than
that for free from a second model.

For GeoCalib the oracle gap is larger (6.4 pp, reaching 4.80 %) — but that is
precisely because GeoCalib is noisy, and "pick the luckiest sample of a noisy
estimator" is not a capability that survives contact with a deployment where the
GT is unknown. Frame selection is therefore rated *possible but low-priority*,
and clearly secondary to the ensemble.

## Q10 — AnyCam

**`NOT_APPLICABLE_REQUIRES_CAMERA_MOTION`** — recorded as
`METHOD_ASSUMPTION_MISMATCH`, not a failure. AnyCam was **not run**, and the
audit was done from the official paper and official code before any install
attempt (`anycam_applicability_audit.md`, `tables/anycam_applicability.csv`).

The decisive detail: AnyCam does not regress a focal, it scores 32 focal
*candidates* with `induce_flow_dist(depth, candidate_intrinsics, relative_pose)`
and takes the argmax. With a static camera the relative pose is the identity, so
the same intrinsics that unproject each pixel immediately reproject it — the
induced flow is identically zero for every candidate, and the objective that
selects the focal carries no information at all. The repository's own comment
states it: *"The last pose should be identity -> Induced flow should be zero"*.
Moving hands in front of a fixed camera do not rescue this; AnyCam's uncertainty
formulation is designed to down-weight independently moving content, not to
calibrate from it. Running it anyway would have measured its learned focal prior
and mislabelled the result as AnyCam's performance.

## Q11 — Does multi-frame alone reach ±5 %?

**No.** Best deployable result: median 6.14 %, with 38.3 % of static cameras
inside ±5 % and 68.6 % inside ±10 %. Nearly two cameras in three are still
outside the target.

For interpretation only, CAM-EXP-002's sensitivity (5 % ≈ 32.8 mm, 10 % ≈ 65.6 mm
of isolated depth displacement in that working-distance regime) puts a 6.14 %
median at roughly 40 mm. That is a reference conversion from a different
experiment, not a depth measurement made here.

---

## Hypotheses, as judged by the data

| | Verdict |
|---|---|
| H1 AnyCalib's low within-view noise means little aggregation gain | **CONFIRMED** — gain −0.02 pp, CI spanning zero, 98.1 % of its error is bias |
| H2 GeoCalib gains more from aggregation | **CONFIRMED** — +0.50 pp [+0.31, +0.81], within-5 % 14.3 → 21.1 % |
| H3 Opposite signed biases let a parameter-free ensemble cancel | **CONFIRMED** — 9.89 → 6.14 %, signed bias −9.25 → −2.36 % |
| H4 A large oracle best-of-N gap would justify frame-selection research | **PARTIALLY REJECTED** — the gap exists but is small for the primary model (3.04 pp) and the deployable ensemble already beats that oracle |
| H5 Otherwise, view-dependent model bias is the main limit | **CONFIRMED** — it is, and it survives both aggregation and oracle selection |

## Model agreement as a confidence signal

GT-blind disagreement between models does correlate with error, but weakly and
inconsistently across which pair and which target is used (Spearman ρ from
+0.083 to +0.351; the strongest, 2-model disagreement vs E5 error, gives
ρ = +0.351, p = 1.9e-6, with quartile medians 5.7 / 6.0 / 10.4 / 15.4 %). It is
suggestive — the most-agreeing quartile really is ~3× more accurate than the
least — but it is not consistent enough to build on, and it is reported as a
measurement, not promoted into a method.

## What this experiment does not establish

- Only 8 frames per view were available; the pre-registered rule said saturation,
  but N ≥ 16 was never measured, so "saturated" means "no gain detectable between
  4 and 8 frames", not a proof for all N.
- One dataset, one distortion regime, one resolution, all cameras static and all
  from the same rig. Bias that is common to this rig would be invisible here.
- No causal account of *why* each model has a per-camera bias; that is what a
  scene/viewpoint ablation would be for.
- The ensemble was evaluated, not tuned; a better combination rule may exist, but
  finding it on this GT is exactly what was forbidden here and would need a
  held-out split.
- Nothing here measures depth or hand accuracy.

---

## DECISION_AFTER_CAM_EXP_004

Two conclusions hold simultaneously.

**B. `MULTIFRAME_HELPS_BUT_BIAS_REMAINS`** — the primary conclusion. Using the
whole video instead of one frame is nearly free and mildly helpful for noisy
estimators, but 98 % of the primary model's error is a per-camera bias that no
amount of frames removes. The ±5 % goal is not reachable this way.

**D. `MODEL_ENSEMBLE_PROMISING`** — the actionable one. A parameter-free
two-model geometric-mean ensemble is the strongest deployable estimator found so
far (6.14 % median, 38.3 % within 5 %, signed bias −2.4 %) and should be carried
forward as the **strong baseline** that any hand-aware or scene-aware method must
beat.

**C. `FRAME_SELECTION_PROMISING` — not supported as a priority.** The oracle gap
is only 3.04 pp for the primary model and the deployable ensemble already
outperforms that oracle. Keep it on the list, below B and D.

### Recommended next step

**CAM-EXP-005: scene / background / viewpoint cue analysis** — ask *what makes a
given camera's view produce a consistent focal offset*, since that bias, not
frame count and not frame choice, is now the binding constraint. Carry the
two-model ensemble in as the baseline. Hand-aware calibration (CAM-EXP-006+)
stays out until the RGB-only ceiling is understood.

---

## Artifacts

`config.json` (including the pre-registered stopping rule), `environment.json`,
`anycam_applicability_audit.md`;
`tables/{static_camera_gt_check, anycam_applicability, aggregation_methods, stopping_rule}.csv`;
`results/raw/{phase_a_subset_results, per_view_aggregation, cross_model_ensemble, oracle_best_frame, bias_noise_decomposition}.csv.gz`;
`results/summary/` (error and thresholds vs frame count, aggregation methods,
ensembles, oracle, agreement, bias/noise, per camera, per sequence, temporal
position, `phase_b_decision.json`); 8 figures led by
`figures/CAM_EXP_004_MAIN_EXPLANATION.png`.

No Phase B files exist because Phase B was not run. CAM-EXP-001 through
CAM-EXP-003.1 were read only and are unmodified.
