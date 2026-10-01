# CAM-EXP-001.1 — GigaHands bad-view diagnosis

Run date 2026-09-22 · branch `kjh` · base commit `0c15483` · no GPU used.
Full preflight record in `environment.json`, parameters in `config.json`,
run log in `logs/run.log`.

## 1. Purpose

CAM-EXP-001 validated the GigaHands camera convention but left a heavy tail:
about a third of views disagreed with the 2D annotation, some by hundreds of
pixels. That report attributed the tail to "failed or mis-assigned per-view 2D
detections" without proving it. This experiment tests that claim and separates
the competing explanations:

1. plain 2D detection failure (H1)
2. frame synchronisation / temporal lag (H2)
3. camera mapping or camera-parameter mis-assignment (H3)
4. distortion model misuse (H4)
5. coordinate system / axis / flip misinterpretation (H5)
6. sequence- or hand-specific concentration (H6)
7. hand identity swap in the annotation (H7 — added after the figures made it
   visible; see §5)

CAM-EXP-001 is treated as read-only input. All new output lives in this folder.

## 2. Input data and paths

| Item | Path | Checked |
|---|---|---|
| Raw per-joint table (input) | `experiments/runs/CAM-EXP-001_gt_projection_validation/results/raw/reprojection_per_joint.csv.gz` | present |
| GigaHands root | `experiments/datasets/gigahands/demo_all/raw/hand_pose` | present, 5 sequences |
| Shared loaders / geometry / metrics / visualization | `experiments/src/` | present, reused unchanged |
| Branch / remote | `kjh` → `SKHU-AI-Projects-Archive/2026-2-Capstone-T02-calib-free-hands` | confirmed |

The per-view statistics are rebuilt entirely from the CAM-EXP-001 raw CSV. The
dataset itself is re-read only where a test needs something the CSV cannot
provide: alternative projections (lag, distortion, flips), the partner hand,
and video frames for the figures. Nothing was re-downloaded.

## 3. How good and bad views are defined

A **view** is one `(sequence, camera, frame, hand)` observation — 8000 in total.

The first finding reshapes the whole analysis: **23.5 % of views carry no
annotation at all.** Every joint of those views is exactly `(0, 0)` while the
per-joint confidence is reported as `1.0`. This is GigaHands' undetected-hand
sentinel, not a measurement. Because the confidence is 1.0, the confidence
filter used in CAM-EXP-001 did not remove them, and their "error" is simply the
distance from the hand to the image origin — which is what produced the ~690 px
mode in the original report.

These views are set aside as their own class. For the remainder, the threshold
is **derived from the data** rather than fixed in code: Otsu's method on
`log10(per-view median error)` puts the split at **30.11 px**, and the classes
are defined as

| Class | Rule | Views | Share | Median of view medians |
|---|---|---|---|---|
| `good` | median ≤ min(otsu/2, 10) = 10 px | 3516 | 44.0 % | 5.58 px |
| `medium` | between | 1685 | 21.1 % | 16.00 px |
| `bad` | median ≥ max(otsu×2, 50) = 50 px | 920 | 11.5 % | 204.26 px |
| `zero_sentinel` | every GT 2D joint is (0,0) | 1879 | 23.5 % | n/a (no annotation) |

So the honest headline is: of the views that actually carry an annotation,
**57 % are good, 27 % medium, 15 % bad**, and a further quarter of all views
were never annotated in the first place.

## 4. How the representative cases were chosen

`results/raw/representative_cases.csv` records the reason for every case:

- **good** — lowest per-view median error, spread across distinct sequences so
  the examples are not all from one take.
- **bad** — highest per-view median error among *non-sentinel* views, so the
  failures shown are real disagreements rather than missing annotations.
- **paired** — same `sequence`, `frame` and `hand`, one good camera and one bad
  camera. Everything except the viewpoint is held constant, which is what makes
  the comparison interpretable.
- **sentinel** — two views showing what a `(0,0)` annotation looks like.

6 good, 6 bad, 6 pairs (12 panels) and 2 sentinel cases were produced.

## 5. Hypothesis results

Verdicts use the scale *supported / partially supported / insufficient evidence
/ close to rejected*. Ranking is in `results/summary/root_cause_ranking.csv`,
full metrics in `results/raw/hypothesis_tests.csv`.

### H1 — plain 2D detection failure → **supported**

- 23.5 % of all views are `(0,0)` sentinels (no detection at all).
- Among non-sentinel bad views, 95 % of joints exceed 50 px and **83.5 % of
  bad views fail as a whole view** (>90 % of joints wrong together).
- 519 frames have exactly one hand as a sentinel while the other hand is
  annotated normally — the failure follows the hand, not the geometry.

A camera-model error would perturb all joints of a view smoothly and would
never produce a literal `(0,0)` annotation. This is annotation-side failure.

### H7 — hand identity swap → **supported for a substantial share of bad views**

Prompted by the paired figures, where green (annotation) and red (projection)
repeatedly sat on *different real hands*. Measured by centroid, since a left
and a right skeleton use mirrored joint ordering and would retain a per-joint
residual even under a perfect swap:

| Measure | Value (n = 184 bad views with both hands available) |
|---|---|
| Annotation centroid closer to the **other** hand's projection | 77.7 % |
| Annotation falls within one hand-width of the other hand | 77.7 % |
| Median centroid distance to its **own** projected hand | 224 px |
| Median centroid distance to the **other** projected hand | 15 px |

A 15 px centroid distance is essentially a hit. The annotation labelled "left"
is frequently describing the right hand, and vice versa.

### H6 — sequence / hand concentration → **partially supported**

| Sequence | sentinel rate | bad rate (excl. sentinel) |
|---|---|---|
| p36-tea-0010 | 0.241 | 0.150 |
| p41-boxing-0021 | 0.246 | 0.120 |
| p41-plant-0004 | 0.309 | 0.053 |
| p44-dog-0004 | 0.239 | **0.265** |
| p52-instrument-0034 | **0.139** | 0.153 |

Failure varies by a factor of ~5 across sequences, so scene content, occlusion
and two-hand proximity matter. Between hands the difference is small (bad rate
left 0.141 vs right 0.160; sentinel rate 0.238 vs 0.232), so this is **not** a
systematic left/right handedness bug in our loader.

### H3 — camera mapping / parameter mis-assignment → **close to rejected as a calibration fault**

Per-camera bad rates span 0.00–0.98, so failure genuinely is camera-dependent;
`brics-odroid-027_cam1` is bad in 98 % of its non-sentinel views. But that is
not evidence of wrong parameters:

- the same camera also fails to detect the hand at all in **68 %** of its views;
- across the rig, per-camera bad rate and sentinel rate correlate at **r = 0.69**;
- 24 of 40 cameras behave inconsistently from frame to frame.

Wrong intrinsics or a swapped camera identity would be a fixed property of that
camera, would fail on *every* frame, and would leave 2D detection untouched.
What we see instead is that the cameras which project badly are the ones that
rarely see the hand. The camera dependence is a **visibility** effect.

### H5 — coordinate/axis/flip misinterpretation → **close to rejected**

| Repair attempted on bad views | Fraction fixed (< 20 px) |
|---|---|
| mirror x, mirror y, swap u/v, negate x (best of) | **0.0 %** |
| pure translation removed | 4 % |
| full similarity transform removed | 38 % |

No flip helps at all. That 38 % become consistent under a similarity transform
while only 4 % are explained by translation means the annotated skeleton keeps
the correct *shape* but is placed at the wrong position and scale — the
signature of the wrong hand being annotated (H7), not of a broken camera model.

### H2 — frame synchronisation / lag → **close to rejected**

Sweeping the 3D frame index over t−5…t+5 for 200 bad views:

- **0 %** of bad views are brought under 20 px by any lag;
- only 2 % even halve their error at a non-zero lag;
- the median best lag is **0**.

`figures/lag_sweep_plot.png` shows good views with a sharp minimum at lag 0
(as expected for correctly synchronised data) and bad views essentially flat —
no lag rescues them. A real desynchronisation would show a consistent non-zero
best lag shared across views.

### H4 — distortion model misuse → **close to rejected**

| Views | distortion applied | distortion disabled |
|---|---|---|
| good (n=200) | 6.07 px | 11.45 px |
| bad (n=200) | 199.35 px | 212.43 px |

Disabling distortion repairs **0** of 200 bad views and makes them slightly
worse, while on good views distortion clearly helps. The distortion model is
being applied correctly and is not what separates good from bad.

### Observed failure modes

From a 400-view census of bad + sentinel views
(`results/summary/failure_mode_census.csv`):

| Mode | Share | Meaning |
|---|---|---|
| A. no detection sentinel | 66.3 % | GT 2D is `(0,0)`, conf 1.0 |
| F. structural mismatch | 19.0 % | annotation on a different hand/shape |
| B. shape kept, placement wrong | 7.8 % | similarity transform fits |
| D. partial joint corruption | 5.3 % | only some joints off |
| C. whole-view translation | 1.8 % | pure shift fits |

## 6. Images to look at

Start with the first two:

| File | What it shows |
|---|---|
| `figures/paired_good_bad_comparison_grid.png` | **Most informative.** Same sequence/frame/hand, good camera vs bad camera. In the bad panels green and red sit on two *different real hands*. |
| `figures/failure_mode_gallery.png` | Each failure mode with 1–2 examples and a caption. |
| `figures/good_view_examples_grid.png` | Successes: green and red coincide on the hand. |
| `figures/bad_view_examples_grid.png` | The largest non-sentinel failures. |
| `figures/error_rank_bar.png` | All 6121 annotated views ranked by median error, thresholds marked. |
| `figures/lag_sweep_plot.png` | H2: no lag repairs bad views. |
| `figures/distortion_effect_plot.png` | H4: distortion helps good views, does not explain bad ones. |
| `figures/cases/good_case_01..06.png` | Six-panel detail: raw, GT, projection, both, zoom, metadata. |
| `figures/cases/bad_case_01..06.png` | Same layout for failures. `bad_case_01` is a clear hand swap. |
| `figures/cases/pair_good_01..06.png` / `pair_bad_01..06.png` | Both halves of each pair at full detail. |
| `figures/cases/sentinel_case_01..02.png` | What a `(0,0)` annotation looks like: red skeleton on a real hand, no green anywhere. |

## 7. Most likely cause, on current evidence

The bad views are **annotation-side failures in GigaHands `keypoints_2d`**, in
two distinct forms:

1. **Missing detections written as `(0,0)` with confidence 1.0** — 23.5 % of all
   views, and 66 % of everything that looks like a failure. These are not
   errors at all; they are absent annotations that any naive confidence filter
   will silently accept.
2. **Hand identity swaps** — in 77.7 % of the remaining bad views the
   annotation sits on the other hand (median centroid distance 15 px to the
   other hand vs 224 px to its own).

The camera convention validated in CAM-EXP-001 is **not** implicated: no flip,
axis swap, lag or distortion setting repairs these views, and in every failure
figure the red projection lands on a genuine hand in the image, with the correct
shape and scale — it is simply the hand the annotation did not name.

The per-camera and per-sequence concentration is real but secondary, and is
consistent with visibility: cameras and scenes where hands are occluded or far
away both fail to detect and, when they do detect, are likelier to confuse two
nearby hands.

## 8. What remains uncertain

- The census is based on a 400-view sample of bad/sentinel views and a 200-per-
  class sample for the dataset re-read, not on all 8000 views. Shares are
  therefore estimates, though the effects are large relative to the sample.
- 19 % of failures fall into "structural mismatch" and are not individually
  explained. They are consistent with wrong-instance annotation, but we did not
  prove that case by case.
- H7 is measured only where **both** hands are present and non-sentinel
  (n = 184). Bad views whose partner hand is also missing cannot be tested this
  way.
- We have not confirmed against GigaHands documentation or code that `(0,0)`
  is the *intended* sentinel; the interpretation rests on the observed pattern
  (exactly `(0,0)`, all joints, confidence 1.0, all-or-nothing per view).
- Whether the 2D annotation or the triangulated 3D is at fault in the swap
  cases cannot be settled from a single view. The 3D is multi-view consistent
  and lands on a real hand, which favours the 2D being wrong, but a dedicated
  multi-view consistency test would be needed to state this firmly.
- The `medium` band (21 % of views, ~16 px median) was not diagnosed here and
  may be ordinary annotation noise or mild occlusion.

## 9. Next steps

1. **Fix the filter, not the geometry.** Drop `(0,0)` sentinel rows in the
   GigaHands loader regardless of confidence, and re-derive the CAM-EXP-001
   headline numbers on the annotated subset. This is a small change to
   `experiments/src/datasets/gigahands.py` and would change that report's
   distribution materially.
2. **Re-associate rather than trust.** Before using GigaHands `keypoints_2d` as
   a reference, assign each per-view 2D detection to the nearer projected hand,
   or reject views whose per-view median exceeds a derived threshold.
3. **Settle 2D-vs-3D fault** with a multi-view consistency test: triangulate
   from the annotated 2D only and compare against the provided 3D.
4. Carry this into CAM-EXP-002 as a data-quality gate, so camera-error studies
   are not measuring annotation noise.

## WHAT WE KNOW

- 23.5 % of GigaHands views have **no 2D annotation**: all joints exactly
  `(0,0)` with confidence 1.0. This alone accounts for about two thirds of what
  CAM-EXP-001 counted as reprojection failure.
- Among annotated views, 57 % are good (≤10 px, median 5.6 px), 27 % medium,
  15 % bad.
- In 77.7 % of testable bad views the annotation describes the **other hand**
  (median centroid distance 15 px to the other hand, 224 px to its own).
- Bad views fail as whole views (83.5 %), not as scattered joints.
- Lag does not explain them (0 % fixed, median best lag 0).
- Distortion does not explain them (0 of 200 fixed; distortion helps good views).
- No mirror, axis swap or sign flip fixes any bad view (0 %).
- Per-camera bad rate correlates with per-camera sentinel rate at r = 0.69, so
  camera dependence reflects visibility, not calibration bookkeeping.
- The camera convention established in CAM-EXP-001 survives every test here.

## WHAT WE DO NOT KNOW

- Whether `(0,0)` is GigaHands' documented sentinel or an artefact of their
  export; we have only the observed pattern.
- The exact cause of the 19 % "structural mismatch" failures.
- Whether the 2D or the 3D is wrong in swap cases — evidence favours the 2D,
  but it has not been proven by an independent multi-view test.
- How much the `medium` band (21 % of views) reflects annotation noise versus
  residual model error.
- Whether these rates hold beyond the 5 demo sequences and the frames sampled
  by CAM-EXP-001.

## NEXT ACTIONS

1. Add a `(0,0)`-sentinel filter to the GigaHands loader and re-run CAM-EXP-001
   to get a clean baseline.
2. Re-state the CAM-EXP-001 GigaHands headline on annotated views only.
3. Add a per-view quality gate (2D-to-projection association + median-error
   rejection) for all downstream experiments.
4. Run a triangulation-based 2D-vs-3D consistency check to settle which side is
   at fault in swap cases.
5. Proceed to CAM-EXP-002 with that gate in place.

## Artifacts

```
config.json / environment.json / logs/run.log
results/raw/per_view_error_summary.csv        8000 view rows
results/raw/per_frame_error_summary.csv       4400 frame rows
results/raw/hypothesis_tests.csv              H1..H7 verdicts + metrics
results/raw/lag_sweep_results.csv             H2 sweep, t-5..t+5
results/raw/distortion_compare.csv            H4 on/off per view
results/raw/alignment_pattern_summary.csv     H5 flips, similarity, translation
results/raw/hand_identity_check.csv           H7 centroid test
results/raw/sequence_hand_concentration.csv   H6
results/raw/representative_cases.csv          case list + selection reason
results/summary/good_vs_bad_summary.csv       class sizes and statistics
results/summary/root_cause_ranking.csv        ranked causes
results/summary/failure_mode_census.csv       failure-mode shares
results/summary/human_readable_summary.csv    plain-language answers
tables/{top_good_views,top_bad_views,per_camera_stats,per_sequence_stats,
        per_camera_family_stats,per_hand_stats}.csv
figures/*.png, figures/cases/*.png
src/{bad_view_common,step1_view_stats,step2_hypotheses,step3_figures,run_all}.py
```

Reproduce with `python src/run_all.py` (needs `PYTHONPATH` at the repository
root so `experiments.src` resolves).

## 10. Git commit

Committed on branch `kjh`; hash recorded in the commit log and in
`environment.json` (`git_commit` there is the parent commit at run time,
`0c15483`).
