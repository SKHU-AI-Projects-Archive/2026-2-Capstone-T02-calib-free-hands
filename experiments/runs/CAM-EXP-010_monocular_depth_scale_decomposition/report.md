# CAM-EXP-010 — Monocular Depth/Scale Error Decomposition and Identifiability Audit

```
TAGS  SEQUENCE_TRANSLATION_BIAS_DOMINATES
      SEQUENCE_DEPTH_SCALE_STABLE
      DEPTH_OFFSET_ADDS_VALUE
      XY_TRANSLATION_IS_A_MAJOR_COMPONENT
      METRIC_SCALE_NOT_GEOMETRICALLY_IDENTIFIABLE
      LARGE_IRREDUCIBLE_POSE_FLOOR

CAM-EXP-009.4 oracle row reproduced: YES, to 0.000 mm
ALL corrections below are ORACLE DIAGNOSTICS, not deployable methods.
```

**One sentence.** The ~78 mm that survived a perfect focal is **mostly a
per-video constant**: a single affine depth correction plus a constant x/y
offset per video — both fitted on held-out frames — cuts wrist/root error from
74.7 mm to **26.4 mm**, but none of those numbers is obtainable at deployment,
because the global metric scale is provably **not identifiable** from the cues
the task allows.

## 1. 쉬운 설명

CAM-009.4에서 focal을 완벽하게 맞춰도 손목 위치 오차가 약 78 mm 남았습니다.
이번 실험은 그 78 mm가 어디서 오는지 분해했습니다.

답은 이렇습니다. **영상마다 거의 일정한 치우침(bias)이 대부분입니다.**
영상 하나당 깊이 보정값 두 개(곱셈 + 덧셈)와 x/y 치우침 하나를 주면,
손목 오차가 74.7 mm → **26.4 mm**로 줄었습니다. 이건 그 영상의 다른 frame으로
검증한 값입니다.

그런데 **중요한 함정**이 있습니다. 그 보정값들은 정답 3D를 보고 구한 것이라
실제 현장에서는 구할 수 없습니다. 그리고 이번에 수학적으로 확인한 바로는,
지금 허용된 정보(단안 카메라, 모르는 작업자 손 크기, 기준 물체 없음)만으로는
**절대 크기(metric scale)를 원리적으로 결정할 수 없습니다**. 3D 전체와 카메라
거리를 같은 배율로 키우면 영상이 완전히 똑같아지기 때문입니다.

또 하나. x/y 오차가 42 mm로 생각보다 큽니다. 깊이만 고쳐서는 부족합니다.
그리고 손목을 완벽하게 맞춰도 관절 오차 **35.6 mm**는 그대로 남습니다.

## 2. Q2–Q4 — reproduction and population

| | |
| --- | --- |
| **Q2 CAM-EXP-009.4 oracle reproduced?** | **YES — 0.000 mm** on both wrist/root (74.95) and root-depth (53.46) |
| **Q3 PRIMARY common set** | **123 sequence-camera units**, 7,168 frames |
| **Q4 cameras / sequences / participants** | **36 cameras, 5 sequences, 4 participants** |
| FIT / EVAL frames | 3,516 / 3,652 |
| circularity (`target_camera_in_reference_count`) | **0** |
| new WiLoR inference | **none** — CAM-EXP-009.4's cache reused |

**A bug the reproduction gate caught.** My first pass took the predicted root to
be `cam_t`, assuming WiLoR's `keypoints_3d` were root-centred. They are not:
`kp3[0]` sits ~96 mm from the local origin, almost entirely in x (the left/right
mirror offset). Depth still reproduced to 1.04 mm, so a depth-only check would
have passed, but wrist/root came out at 120.14 mm against CAM-EXP-009.4's 74.95.
The correct root is `kp3[0] + cam_t`. Fixed, re-extracted, and the gate now
matches to 0.000 mm. Had the gate not existed, the decomposition would have
reported a ~96 mm phantom x-error as a real finding.

## 3. Q5–Q8 — the O0 error budget

Per frame, with the reference focal supplied as an oracle:

| component | median |
| --- | ---: |
| **Q5 wrist/root (3D)** | **78.3 mm** |
| **Q6 root depth `|dz|`** | **58.0 mm** (74 % of root) |
| **Q7 x/y** | **42.2 mm** (54 % of root) |
| `|dx|` | 32.5 mm |
| `|dy|` | 18.6 mm |
| **Q8 root-aligned pose** | **35.6 mm** |

(Medians of components do not add linearly to the median of the norm; the
percentages describe each component's typical size relative to the root error,
not a partition.)

**x/y is not a rounding term.** It is 42 mm — more than half the root error
budget — and it is dominated by `dx`. Fixing depth alone cannot get below it.

## 4. Q9–Q10 — is the depth error one number per video?

Frame-wise `log(z_ref / z_pred)`:

| | value |
| --- | ---: |
| median `alpha` = `exp(median log ratio)` | **1.0341** |
| alpha p25 / p75 across units | 0.966 / 1.093 |
| **within-unit MAD(log)** | **0.0524** |
| **across-unit MAD(log alpha)** | **0.0971** |
| **ratio within / across** | **0.54** |

WiLoR systematically places hands ~3.4 % too close. More importantly, the
frame-to-frame scatter inside a video is **about half** the spread between
videos. So the depth error is *substantially* but not *purely* a per-video
constant: a single number per video captures the larger share, and a real
frame-varying component remains.

## 5. Q11–Q20 — the oracle hierarchy (held-out EVAL frames)

| oracle | root | x/y | depth | abs MPJPE |
| --- | ---: | ---: | ---: | ---: |
| **O0** oracle focal only | 74.7 | 40.6 | 53.5 | 61.6 |
| **O1** sequence multiplicative z | 58.7 | 40.6 | 28.8 | 51.2 |
| **O2** sequence affine z | 48.0 | 40.6 | 13.3 | 42.6 |
| **O3** sequence constant 3D bias | 38.3 | 16.8 | 28.4 | 46.3 |
| **O4** sequence xy bias + affine z | **26.4** | 16.8 | 13.3 | 40.3 |
| **O5** per-frame depth oracle | 40.6 | 40.6 | 0.0 | 39.8 |
| **O6** per-frame root oracle | 0.0 | 0.0 | 0.0 | **35.6** |

Paired, camera-clustered bootstrap (10,000 iterations), versus O0:

| oracle | depth reduction | root reduction | root gain CI (mm) |
| --- | ---: | ---: | --- |
| **Q12/Q13/Q14 O1** | **+46.2 %** | **+21.4 %** | [5.63, 18.34] |
| **Q15/Q16 O2** | +75.1 % | +35.8 % | [14.94, 30.74] |
| **Q17 O3** | +46.9 % | +48.7 % | [27.92, 37.59] |
| **Q18 O4** | +75.1 % | **+64.7 %** | [39.67, 52.70] |
| **Q19 O5** | +100 % | +45.6 % | [23.35, 36.80] |
| **Q20 O6** | +100 % | +100 % | [67.97, 83.57] |

All CIs exclude zero.

**Q16 — the depth offset matters a lot.** O1 → O2 takes depth error from 28.8 mm
to 13.3 mm, a further 54 % reduction. `DEPTH_OFFSET_ADDS_VALUE = True`; this is
not a pure multiplicative scale.

**The result that reframes the problem: O4 (26.4 mm) beats O5 (40.6 mm).** Three
constants per video beat a *perfect per-frame depth oracle*, because O5 leaves
x/y untouched at 40.6 mm. Frame-varying depth noise is real but smaller than the
sequence-constant translation bias.

## 6. Q21–Q22 — the waterfall

**wrist/root:** 109.8 (scene focal, CAM-EXP-009.4) → **74.7** (perfect focal) →
58.7 (one z scale) → 26.4 (three constants) → 0 (perfect translation).

**absolute MPJPE:** 105.8 → **61.6** → 51.2 → 40.3 → **35.6 floor**.

Of the 61.6 mm absolute MPJPE left after a perfect focal, **35.6 mm is an
irreducible root-relative pose floor** and only ~26 mm is translation. Per-video
constants recover 21.3 mm of that 26 mm.

**Q21 consistency check:** O6's absolute MPJPE equals the root-aligned MPJPE to
**0.000 mm**, exactly as the algebra requires. The implementation is validated.

## 7. Q23–Q25 — does the correction transfer?

| | median \|log ratio\| | as a percentage |
| --- | ---: | ---: |
| **Q23 LEFT vs RIGHT alpha, same video** | 0.0776 | **8.1 %** |
| **Q24 same camera, different sequence** (177 pairs) | 0.0878 | 9.2 % |
| **Q25 p41 across its two sessions** | included above | — |

Left and right hands of the *same video* disagree by ~8 % on the required scale
(Spearman 0.63 — correlated, not identical). Crucially, the **same physical
camera in a different sequence disagrees by 9.2 %, no better than the two hands
of one video**. The required correction is a property of the *recording*, not of
the camera and not of the person — consistent with CAM-EXP-009.3.1, where p41's
geometry did not transfer across sessions either.

So a camera-specific calibration factor measured once would not carry over.

## 8. Q26–Q28 — what predicts the scale error?

Spearman rho against `log(z_ref / z_pred)`, all frames:

| predictor | rho |
| --- | ---: |
| is_right | +0.117 |
| **Q26 bbox fraction** | +0.113 |
| predicted depth | −0.086 |
| **Q27 cam_t z** | −0.083 |
| detection score | +0.079 |
| **Q28 hand extent** (pose-diversity proxy) | +0.051 |

**Every correlate is weak.** No single observable explains the scale error, so
there is no obvious deployable regressor hiding in these features. Reported as
effect sizes; no p-value hunting.

## 9. Q29–Q34 — component shares and tags

- **Q29 depth share:** 58.0 mm of a 78.3 mm root error.
- **Q30 x/y share:** 42.2 mm — large, and dominated by `dx`.
- **Q31 after perfect translation:** 35.6 mm of root-relative pose error remains.
- **Q32 `SEQUENCE_DEPTH_SCALE_STABLE`: True** (O1 gives +46.2 % depth and
  +21.4 % root, CI lower bound > 0).
- **Q33 `MULTIPLICATIVE_SCALE_DOMINATES`: False.**
- **Q34 `DEPTH_OFFSET_ADDS_VALUE`: True** (O2 improves on O1 by 54 % on depth).

## 10. Q35–Q39 — the identifiability audit

**Q35 `GLOBAL_SCALE_INVARIANCE_TEST`: PASS**, with its three companions:

| test | max difference | result |
| --- | ---: | --- |
| global scale invariance (s = 0.5 / 1 / 2 / 5) | 5.68e-14 px | PASS |
| **Q37** multi-frame (64 frames, one consistent hand) | 1.14e-13 px | PASS |
| **Q38** bilateral (both hands, both trajectories) | 1.14e-13 px | PASS |
| anatomy scale-free (sum-normalised proportions) | 2.78e-17 | PASS |

Scaling every 3D point **and** the camera translation by a common `s` leaves
`f·X/Z` unchanged — the `s` cancels exactly.

- **Q36: No.** With the allowed cues the metric scale is **not geometrically
  identifiable**. It is a one-parameter family of solutions, all of which explain
  the images equally well.
- **Q37: multi-frame does not break it.** A consistent hand of size `L` at
  distance `Z` and one of size `sL` at `sZ` generate identical video.
- **Q38: bilateral does not break it.** Left/right agreement constrains relative
  shape; without an absolute size for one hand it says nothing about `s`. And
  the sum-normalised anatomy used throughout CAM-EXP-009.x is scale-free *by
  construction*.

**Q39 — the distinction that matters most.** WiLoR *does* output metric numbers,
via a **learned statistical scale prior** absorbed from its training
distribution. That is a real estimator, but a prior, not a measurement: its
error is the gap between this worker's hands and the training distribution, and
no image evidence corrects it. `LEARNED STATISTICAL SCALE PRIOR` is not
`GEOMETRICALLY IDENTIFIABLE SCALE`.

The measured 3.4 % median `alpha` with 9.7 % spread between videos is a direct
estimate of how large that prior's gap is here.

And: **correct intrinsics are not metric world scale.** CAM-EXP-009.4's
oracle-focal residual is that statement in empirical form.

## 11. Q40–Q44 — implications

**Q40 is a metric anchor required?** For *metric* absolute 3D, yes — something
outside monocular geometry must supply the scale. This run does **not** adopt
any such assumption; it establishes that one is needed.

**Q41 what kind of information would suffice, in principle** (stated as logical
options, *not* adopted here): a known dimension visible in the scene, a
calibration object or second view used once, a depth sensor, a one-time
per-worker hand measurement, or an explicit decision to rely on the learned
prior with its error budgeted. Choosing among these is a project decision that
now has an error budget to reason with.

**Q42 next substantive experiment.** The evidence points at the
**sequence-constant translation bias**, which is the largest correctable block
(74.7 → 26.4 mm) — but the honest framing is not "estimate alpha", because §10
shows it is not estimable from geometry. The next question is which anchor the
task can afford, and what residual remains after it. A useful companion
measurement is the **35.6 mm pose floor**, which no camera work touches at all.

**Q43 should focal work continue now?** **No.** A perfect focal was worth
109.8 → 74.7 mm, and the remaining 74.7 mm is not a focal problem. CAM-EXP-009.4
already showed hand geometry adds little to the scene focal on this rig.

**Q44 varied-intrinsics dataset?** Still needed *if* focal work resumes — but it
is no longer the top priority, because focal is not the dominant error term.

## 12. What this does and does not establish

**Establishes**, on the evaluated 123-unit GigaHands common set with an oracle
focal: the residual absolute error is dominated by a **per-video translation
bias** in both depth and x/y; three constants per video remove 65 % of the
wrist/root error on held-out frames; a further frame-varying depth component
exists but is smaller; ~35.6 mm of root-relative pose error is untouched by any
translation correction; and the global metric scale is not identifiable from the
allowed cues.

**Does not establish** that any of these corrections is achievable at
deployment. **Every one was fitted on the reference 3D.** The reference-derived
alpha is not a proposed method, and §10 is the reason it cannot simply be
estimated instead.

**Does not say** hand-geometry research failed. CAM-EXP-009.1's clean synthetic
focal identification and CAM-EXP-009.3.1's generic bone-identity signal both
stand. Focal identifiability and metric-scale identifiability are **different
problems**, and this run is about the second.

## 13. Limitations

1. 4 participants, 5 sequences, one capture rig. The per-video bias could be
   specific to this rig's geometry.
2. The reference 3D carries reconstruction error of unknown size, which bounds
   how small a residual is meaningful — the 35.6 mm floor includes it.
3. `alpha` is fitted from as few as 6 FIT frames in some units.
4. O4's parameterisation is restricted (x/y bias + affine z) to stay
   identifiable; a full 3D scale-plus-bias was not fitted.
5. `ORACLE_ABSOLUTE_HAND_SCALE` was specified as optional and **not run**; no
   assumed hand size is used anywhere.
6. The camera-to-workspace transform is out of scope. Metric camera coordinates
   would still need their own transform and anchor.

## 14. Reproduce

```
python src/extract_translation_errors.py    # ~11 min, reuses CAM-009.4 WiLoR cache
python src/reproduce_cam0094_oracle.py      # gate: must match to 1 mm
python src/run_oracle_hierarchy.py
python src/analyze_scale_structure.py
python src/identifiability_audit.py
python src/figures.py && python src/self_audit.py
```
