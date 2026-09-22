# CAM-EXP-001.3 — GigaHands independent multi-view triangulation & QC adjudication

Run date 2026-09-22 · branch `kjh` · base commit `7066b70` · no GPU.
Config in `config.json`, environment in `environment.json`, log in `logs/run.log`.

## 1. Purpose

Decide, for the GigaHands hand-identity mismatches found in CAM-EXP-001.1/001.2:

* **A** — the per-view 2D annotation of a particular camera is wrong, or
* **B** — the released 3D hand pose is wrong, or
* **C** — the available information cannot decide.

The method is to reconstruct the hands **from the released 2D observations and
the camera calibration only**, holding out the camera under test, and to read
the released 3D **only afterwards**, purely as something to compare against.

### A note on wording

Neither side of this comparison is external ground truth. GigaHands 2D comes
from an automatic pipeline (HaMeR, with ViTPose for hand detection) and its 3D
is triangulated from that 2D. This run is an **independent implementation of
multi-view reconstruction from the released 2D observations** — not independent
ground truth. Throughout, "provided 2D/3D annotation" or "reference 3D" is used
rather than "ground truth".

## 2. Non-circularity

The claim of this experiment rests entirely on the reconstruction not having
seen what it is testing, so this is enforced mechanically, not by inspection
(`src/tests/test_no_circularity.py`, all passing):

| Rule | How it is enforced |
|---|---|
| Reconstruction uses only 2D + calibration | static AST check that `gather_observations`, `reconstruct`, `reconstruct_all`, `refit_excluding`, `project_hypothesis` never reference `joints3d` |
| Provided 3D is never read during reconstruction | dynamic tripwire: `joints3d` is replaced by a function that raises, and a full reconstruction still succeeds with **0 calls** |
| The camera under test contributes nothing | held-out camera absent from the observation set; holding out removes exactly one camera (25 → 24) |
| Placeholder 3D rows are unreachable | `joints3d` returns `None` outside `chosen_frames` (M3) |
| All-(0,0) 2D never used as an observation | checked over sampled frames (M5) |
| Earlier verdicts are not truth | the CAM-EXP-001.2 status selects *which* cases to inspect and is never an inlier criterion or a label |

The reconstruction module itself imports nothing from any dataset loader
(`__future__`, `cv2`, `dataclasses`, `numpy` only).

## 3. Method

**Triangulation** (`experiments/src/geometry/triangulation.py`, reusable):

1. observations → normalised undistorted coordinates (`cv2.undistortPoints`
   with the camera's own `[k1,k2,p1,p2]`);
2. per joint, RANSAC over camera pairs: DLT hypothesis → reprojection residuals
   in **pixels with distortion re-applied** → consensus set;
3. refit on the consensus, re-deciding membership;
4. report inlier count, median/p90 residual and the **maximum ray angle**
   (geometric conditioning).

A plain least-squares fit over all views is unusable here: one camera that
annotated the wrong hand drags the solution by hundreds of millimetres.

**Leave-one-camera-out**: for each held-out camera C, reconstruct LEFT and
RIGHT separately from every other camera, project both into C, and compare with
C's own two annotations:

```
E_LL = LEFT annotation  vs LEFT reconstruction     E_LR = LEFT annotation  vs RIGHT reconstruction
E_RR = RIGHT annotation vs RIGHT reconstruction    E_RL = RIGHT annotation vs LEFT reconstruction
```

Identity questions are judged on **centroids**, not per-joint distance: left and
right skeletons use mirrored joint ordering, so a true swap still leaves a large
per-joint residual (this is why CAM-EXP-001.1 saw only 9.1 % of swaps under
20 px per joint while centroids showed 15 px).

**Scaling**: the full pass runs RANSAC once per (frame, hand) and judges each
held-out camera against a hypothesis **refitted from the consensus inliers
excluding that camera**. For a camera that disagrees — exactly the cases of
interest — it is not in the consensus at all, so the hypothesis is entirely
independent of it. Measured against the strict per-camera variant on 15 frames:
median absolute difference **0.72 px**, max 2.60 px, **case agreement 96.7 %**
(`results/summary/fast_vs_strict_validation.csv`).

## 4. Implementation validation (Q1)

Ten synthetic tests with a known answer (`src/tests/test_triangulation_synthetic.py`):

| Test | Result |
|---|---|
| undistort inverts project | max error 2e-4 (3 cameras) |
| noiseless recovery | 2.4e-16 m |
| 1.5 px pixel noise | median 1.58 mm |
| with distortion | median 1.09 mm |
| one camera reports the other hand | reconstruction still 1.06 mm; outlier rejected in **100 %** of joints |
| 20 % gross outliers | median 0.74 mm |
| only 3 cameras | median 1.47 mm |
| single view | reports `insufficient_observations` |
| low vs wide baseline | 2.9° vs 141.7° ray angle, correctly distinguished |
| no dataset-3D reference in executable code | pass |

**Clean-control validation on real data** (20 controls, run before any suspected
case, `tables/good_control_metrics.csv`):

| Metric | Median | p90 |
|---|---|---|
| triangulation residual | 3.07 px | 3.28 px |
| held-out 2D vs **same**-hand reconstruction | 8.33 px | 12.5 px |
| held-out 2D vs **other**-hand reconstruction | 192.7 px | 347 px |
| reconstruction vs provided **same** 3D | **3.30 mm** | 6.24 mm |
| reconstruction vs provided **other** 3D | 180.4 mm | 322.7 mm |
| inlier cameras | 14 | 15 |

Same-hand beats cross-hand by ~23× in 2D and ~55× in 3D. **Q1: yes** — the
triangulation behaves correctly on clean data, and the run only proceeded past
this gate because the check passed (`run_all.py` aborts otherwise).

## 5. Thresholds (§22 of the brief)

Derived in two stages, because the bootstrap pass uses a looser inlier gate that
inflates inlier counts and ray angles; the operating criteria come from controls
measured **at the operating configuration** (`results/summary/_thresholds.json`).

| Threshold | Value | Origin |
|---|---|---|
| `triangulation_inlier_px` | 6.0 | p99 of control triangulation residual, floored |
| `min_inliers_for_reconstruction` | 4 | geometric requirement |
| `same_hand_reprojection_px` | 16.0 | p99 of control same-hand error, floored |
| `provided3d_mpjpe_mm` | 10.0 | p99 of control MPJPE (4.4 mm), floored |
| `qc_min_inlier_cameras` | 8 | control p5, clipped to [4, 8] |
| `min_ray_angle_deg` | 30.0 | conditioning floor (controls sit at ~133°) |
| `identity_margin_factor` | 3.0 | opposite hand must be 3× closer |
| `identity_min_gap_px` | 30.0 | and 30 px closer in absolute terms |

The geometric requirements are deliberately conditioning floors rather than
control percentiles: demanding control-level redundancy would reject
well-conditioned but less redundant frames. An earlier iteration that used the
control p5 directly failed 13 of 20 clean controls, which is how the flaw was
caught.

## 6. Results

Full pass: **every chosen frame** of all five demo sequences × all 40 annotated
cameras × both hands = **108,240 observations** (54,120 leave-one-out rows). No
RGB decoded; numerical only. Checkpointed per 50 frames and resumable.

### Q2 — reconstruction vs provided 3D

`results/summary/triangulation_summary.csv`, n = 107,640:

| Metric | Median | p90 | p95 | p99 | Max |
|---|---|---|---|---|---|
| triangulation residual | 3.08 px | 3.35 | 3.43 | 3.58 | 3.84 |
| **vs provided SAME hand** | **4.00 mm** | 6.48 | 7.34 | 9.72 | 24.26 |
| vs provided OTHER hand | 192.3 mm | 309.5 | 328.4 | 380.3 | 408.1 |

An independent re-triangulation of the released 2D reproduces the released 3D to
**4 mm median, under 10 mm at p99, never worse than 24 mm**. Given a hand is
~180 mm across and the two hands are ~190 mm apart, the same-hand and other-hand
distributions do not overlap at all.

### Q3/Q4 — adjudicated cases

`results/summary/identity_verdict_summary.csv`:

| Case | n | Share |
|---|---|---|
| `MULTIVIEW_CONFIRMED_GOOD` | 48,856 | 45.1 % |
| `NO_USABLE_2D_ANNOTATION` | 25,852 | 23.9 % |
| `BAD_2D_GEOMETRY` | 15,615 | 14.4 % |
| `UNRESOLVED_INSUFFICIENT_GEOMETRY` | 10,207 | 9.4 % |
| **`CONFIRMED_2D_HAND_IDENTITY_SWAP`** | **7,101** | **6.6 %** |
| `NOT_CHOSEN_FRAME` | 600 | 0.6 % |
| `CONFIRMED_..._UNVERIFIED_3D` | 9 | 0.01 % |
| **`SUSPECT_PROVIDED_3D_IDENTITY_ERROR`** | **0** | **0 %** |

**The answer is A.** 7,101 observations meet the full confirmation bar of §28:
triangulation succeeded without the held-out camera, with enough inlier cameras
and ray angle, the reconstruction agrees with the provided 3D of *its own* hand
(≤10 mm), and the held-out annotation is at least 3× and 30 px closer to the
**other** hand's reconstruction.

**Q4: no case of provided-3D identity error was found** — zero out of 108,240.
Wherever the comparison was possible, the released 3D matched our independent
reconstruction of the same hand. The released 3D is not the wrong side.

On the smoke set this direction is visible case by case, e.g.
`p41-boxing-0021 / brics-odroid-026_cam0 / f0 / left`: E_LL = 170.7 px,
E_LR = 34.6 px, while our LEFT reconstruction matches the provided LEFT 3D at
**4.2 mm** and the provided RIGHT 3D at 181.8 mm.

### Q3 in the terms of CAM-EXP-001.2

001.2 flagged 518 `likely_hand_identity_error` records on a stride-20 sample
using a heuristic. 001.3 covers every chosen frame (20× more observations) and
adjudicates independently, so the two sets are not row-comparable. In the
targeted smoke test drawn from that group, **12 of 20 (60 %)** were confirmed as
2D identity swaps, 7 became `BAD_2D_GEOMETRY` and 1 was undecidable — none was
overturned in favour of the 3D being wrong.

### Q5 — what the `bad_unexplained` group became

Of the 20 sampled from 001.2's 204 `bad_unexplained` records: **16
`BAD_2D_GEOMETRY`** (the annotation matches neither reconstruction) and
**4 `UNRESOLVED_INSUFFICIENT_GEOMETRY`**. None became a confirmed swap, and none
implicated the 3D. Across the full pass `BAD_2D_GEOMETRY` is the second-largest
real defect at 15,615 observations (14.4 %).

### Q6 — the medium group

Of 20 sampled `medium` records: 7 `MULTIVIEW_CONFIRMED_GOOD`, 8
`BAD_2D_GEOMETRY`, 5 `UNRESOLVED`. So the medium band **does** split, but not
cleanly — roughly a third are recoverable as strict passes and the rest are
genuinely defective or undecidable. They are not admitted to `PASS_STRICT`
unless they meet every criterion on their own.

### Q7 — where the errors concentrate

`results/summary/per_camera_summary.csv` and
`figures/per_camera_identity_error_heatmap.png`:

| Highest swap rate | | Lowest |
|---|---|---|
| `brics-odroid-029_cam0` | 0.288 | `brics-odroid-024_cam1` 0.000 |
| `brics-odroid-026_cam0` | 0.284 | `brics-odroid-024_cam0` 0.000 |
| `brics-odroid-026_cam1` | 0.260 | `brics-odroid-023_cam0` 0.000 |
| `brics-odroid-030_cam1` | 0.255 | |
| `brics-odroid-027_cam1` | 0.196 | |

This **independently reproduces** CAM-EXP-001.2's finding from a completely
different method: the swaps cluster in cameras 026–030 and are absent in
023–024. Per sequence, the strict-pass rate ranges 0.368 (`p44-dog-0004`) to
0.655 (`p52-instrument-0034`), and the swap rate 0.013 (`p41-plant-0004`) to
0.126 (`p52-instrument-0034`).

### Q8/Q9 — final QC

`experiments/manifests/gigahands_demo_qc_v1.json`, 108,240 observations:

| Status | n | Share |
|---|---|---|
| `PASS_STRICT` | 32,826 | 30.3 % |
| `PASS_SINGLE_HAND` | 16,005 | 14.8 % |
| `REVIEW` | 10,236 | 9.5 % |
| `EXCLUDE` | 49,173 | 45.4 % |

48,831 observations (45.1 %) are clean at the observation level; of those,
32,826 also have a clean partner hand and so keep `PASS_STRICT`, while 16,005
are downgraded to `PASS_SINGLE_HAND`.

**Bimanual clean subset: 16,413 frames** where LEFT and RIGHT are both
`PASS_STRICT` (`gigahands_demo_bimanual_clean_v1.csv.gz`).

### Q10 — reusability

Yes. The triangulation lives in `experiments/src/geometry/triangulation.py` and
the take accessor in `experiments/src/datasets/gigahands.py`, both dataset-path
driven. Running the same pipeline on the full GigaHands release needs only the
data present; the pass is chunk-checkpointed and resumable. Runtime is ~0.3 s
per (frame × 40 cameras × 2 hands), i.e. ~7 minutes for the 1,353 demo frames.

## 7. Figures

| File | Shows |
|---|---|
| `figures/CAM_EXP_001_3_MAIN_EXPLANATION.png` | the whole experiment on one sheet, including the QC distribution |
| `figures/confirmed_swaps/confirmed_2d_swap_grid.png` | green pairs with orange and blue with red — the annotation on the other hand |
| `figures/diagrams/leave_one_camera_out_explainer.png` | the method and the four distances |
| `figures/3d_consistency_plot.png` | same-hand vs other-hand MPJPE distributions |
| `figures/per_camera_identity_error_heatmap.png` | where swaps concentrate |
| `figures/triangulation_examples/good_control_grid.png` | controls: green with red, blue with orange |
| `figures/unresolved/unresolved_grid.png` | cases still undecided |
| `figures/qc_examples/qc_status_grid.png` | PASS_STRICT / REVIEW / EXCLUDE examples |

In every figure the **red/orange skeletons are our reconstruction from the other
cameras only**, not the dataset's 3D reprojected. Each figure says so.

## 8. Manifests

The dataset is **immutable**: nothing was deleted, moved or rewritten. The
manifests are the source of truth for usability.

| File | Contents |
|---|---|
| `experiments/manifests/gigahands_demo_qc_v1.csv.gz` | 108,240 per-observation records with every QC metric |
| `experiments/manifests/gigahands_demo_qc_v1.json` | summary, thresholds, status definitions |
| `experiments/manifests/gigahands_demo_bimanual_clean_v1.csv.gz` | 16,413 both-hands-clean frames |
| `experiments/manifests/gigahands_demo_camera_benchmark_v1.csv.gz` | 200 rows of RGB + calibration usability, independent of hand QC |

A loader filters `qc_status == PASS_STRICT`. `REVIEW` must not be used in core
experiments. Hand-annotation quality deliberately does **not** gate camera
usability: 175 of 200 sequence×camera rows remain usable for camera work even
where the hand QC fails (the other 25 are the `p52-instrument-0034` cameras
whose annotated video segment was never downloaded, M6).

## 9. Updates to earlier conclusions

**PREVIOUS CONCLUSION (CAM-EXP-001.1/001.2)** — bad views are annotation-side,
most plausibly per-camera hand-identity swaps, but whether the 2D or the 3D was
the wrong side could not be settled, and `LIKELY_DATASET_ANNOTATION_ERROR` was
deliberately withheld.

**NEW EVIDENCE** — an independent reconstruction that never sees the released 3D
reproduces it to 4 mm median across 107,640 observations, and in 7,101 cases the
held-out camera's annotation sits on the *other* hand's independently
reconstructed position while the reconstruction agrees with the provided 3D of
its own hand. Zero observations show the released 3D on the wrong side.

**UPDATED CONCLUSION** — the defect is in the **per-view 2D annotation of
specific cameras**, not in the released 3D. The released 3D for the demo
sequences is reliable to a few millimetres wherever it is defined. Nothing in
CAM-EXP-001/001.1/001.2 was overturned; the open question they left is now
answered in direction A. Their raw outputs are unmodified.

## 10. Limitations

- Both sides derive from the same released 2D, so a systematic error in the 2D
  pipeline shared across most cameras would be invisible to this test. What is
  demonstrated is *self-consistency of the 2D with the 3D*, and that the
  minority cameras are the deviant ones.
- 10,207 observations (9.4 %) remain `UNRESOLVED_INSUFFICIENT_GEOMETRY`.
- 15,615 `BAD_2D_GEOMETRY` observations are identified as defective but their
  mechanism is not individually diagnosed.
- The full pass uses the consensus-refit leave-one-out variant; it agrees with
  the strict variant at 96.7 % on a 15-frame check, not 100 %.
- Thresholds come from 20 clean controls from one sampling; a larger control set
  would tighten them.
- `p52-instrument-0034` has the highest strict-pass rate (0.655) *and* the
  highest swap rate (0.126); 25 of its cameras have no downloadable video, so
  those observations are numerically judged but never visually checked.
- The confirmation bar is conservative by design, so the 7,101 confirmed swaps
  are a lower bound; some of the 15,615 `BAD_2D_GEOMETRY` are likely swaps that
  missed one criterion.

## 11. Next steps

1. Use `PASS_STRICT` (and `PASS_SINGLE_HAND` for single-hand work) in
   CAM-EXP-002 onwards; never `REVIEW`.
2. Start bimanual geometry work from the 16,413-frame clean subset.
3. Treat camera usability separately via the camera-benchmark manifest.
4. If the full GigaHands release is obtained, re-run this pipeline unchanged to
   produce a v2 manifest.
5. Optionally, diagnose the `BAD_2D_GEOMETRY` group, which is now the largest
   unexplained defect class.

## Reproduce

```bash
PYTHONPATH=<repo root> python src/run_all.py                    # full pipeline
PYTHONPATH=<repo root> python src/tests/test_triangulation_synthetic.py
PYTHONPATH=<repo root> python src/tests/test_no_circularity.py
```
