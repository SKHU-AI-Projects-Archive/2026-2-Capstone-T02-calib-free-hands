# CAM-EXP-006 — Reference-Hand Geometry Focal Information Diagnostic

**Status: complete. The result is strongly negative, and is reported unchanged.**

```
DECISION TAGS
  B_REFERENCE_HAND_FOCAL_INFORMATION_ABSENT
  D_SIGNAL_EXPLAINED_BY_SINGLE_FOCAL_RIG_CONFOUND
  E_WEAKLY_IDENTIFIABLE
  F_SATURATED_BY_FEW_FRAMES
```

## 1. The question

If we already knew the exact 3D shape of the hand in the image — an oracle we
will never have at deployment time — could we read the camera's focal length
off a single view of it?

This sets an **information ceiling**. If the answer is no even with an oracle
hand, then no hand-based focal estimator, however good its hand model, can work
under these imaging conditions. That is why the experiment is worth running
despite being undeployable by construction.

Scope tag: `INTERNAL_REFERENCE_HAND_DIAGNOSTIC`. Every number below is an upper
bound under an oracle reference hand. None of them is achievable performance,
and none describes a method.

## 2. Prerequisite: the manual hand-QC gate

The spec made a human review of the hand annotations a hard prerequisite. All
200 stratified cases were visually reviewed (17 contact sheets, 7 cases
re-opened at full resolution).

```
MANUAL_QC_READINESS_GATE = PASS
```

0 / 200 catastrophic identity or association errors, 0 failures across all 5
sequences and all 39 physical cameras, 200 / 200 reconstructions visually
plausible. Details, caveats and the three open issues it raised are in
[`manual_qc_report.md`](manual_qc_report.md). **0 / 200 is a readiness gate on
an audit sample; it is not a dataset accuracy figure.**

## 3. Method

`PROFILED_PNP_FOCAL_DIAGNOSTIC`, frozen before any result was seen. One scalar
focal, principal point fixed at the image centre, per-hand pose profiled out by
PnP at every candidate focal, over a 121-point log grid in
`gamma = f / max(W, H)` spanning 0.25 – 5.0. Full protocol in
[`solver_protocol.md`](solver_protocol.md).

The reference 3D is `OTHER_CAMERA_ONLY_REFERENCE_3D`: the hand triangulated
from every *other* camera, with the camera under test removed **before**
reconstruction. Provenance and the non-circularity audit are in
[`reference_3d_provenance.md`](reference_3d_provenance.md).

| non-circularity check | result |
| --- | --- |
| camera under test ever an inlier of its own reference hand | **0 of 5600** |
| GT focal, GT extrinsics or GT distortion as a solver input | none |
| provided principal point as a solver input | no (fixed at image centre) |
| bilateral apparent-size ratio used as a focal estimator | no (explicitly forbidden) |

Coverage: 175 views × 16 frames × 2 hands = 5600 cases, of which 4271 (76.3 %)
had enough paired joints to solve.

## 4. Results

### 4.1 The headline

| condition | N=1 | N=2 | N=4 | N=8 | N=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **reference hand (main)** | **110.0 %** | 131.1 % | 212.2 % | 189.7 % | **202.2 %** |
| control: joints permuted | 65.7 % | 195.2 % | 289.3 % | 336.0 % | 462.2 % |
| control: wrong pose | 70.5 % | 113.0 % | 137.4 % | 211.1 % | 201.4 % |
| control: planarized | 57.9 % | 59.0 % | 60.7 % | 57.6 % | 60.0 % |

Median relative focal error, 175 views. Cluster-bootstrap 95 % interval for
main at N=16: **[150.7 %, 305.4 %]**.

Against the frozen comparators:

| comparator | median relative focal error |
| --- | ---: |
| `CONSTANT_RIG_FOCAL_ORACLE` | **0.85 %** |
| `E2_N8` (frozen exploratory ensemble) | 6.37 % |
| `ANYCALIB_N8` | 9.97 % |
| `GEOCALIB_N8` | 11.28 % |
| **reference hand, N=16** | **202.16 %** |

### 4.2 Pre-registered hypotheses

| hypothesis | threshold | outcome |
| --- | --- | --- |
| **H1** the reference hand carries focal information | beat the constant-rig oracle, paired CI excluding zero | **FAIL** — worse by 200.1 pp, CI [148.9, 304.9], entirely on the wrong side of zero |
| **H2** the information is geometric | every control ≥ 2× worse than main | **FAIL** — the planarized control is **3.4× better** than main, not worse |
| **H3** more frames help | N=16 beats N=1 by ≥ 2 pp | **FAIL** — N=16 is 92 pp *worse* than N=1 |

All three fail. Not one is marginal.

### 4.3 Identifiability

| N | share `FLAT_PROFILE` | share `BOUNDARY_SOLUTION` | share cleanly identifiable |
| ---: | ---: | ---: | ---: |
| 1 | 65.1 % | 16.6 % | 15.4 % |
| 16 | 86.9 % | 25.7 % | **4.0 %** |

At N=16 only **4 % of views** produce a focal estimate that is identifiable at
all. Adding frames makes this worse, not better, because averaging flat curves
produces a flatter curve.

## 5. Why: the solver is correct, the information is absent

This is the part that matters, and it is the reason the negative result is
trustworthy rather than a suspected bug.

A separate diagnostic ([`results/summary/solver_sanity_check.json`](results/summary/solver_sanity_check.json),
Fig 06 and Fig 07) evaluates the objective *at the true focal* and compares it
to the profile minimum:

| where the objective is evaluated | median reprojection error |
| --- | ---: |
| at the profile minimum | 6.03 px |
| **at the true focal** | **6.83 px** — only 11 % worse |
| at `gamma = 5.0`, a focal ≈ 7× too large | **6.53 px** — *better than the true focal* |
| at `gamma = 0.25`, a focal far too small | 10.71 px |

The solver finds the true focal to be an almost-as-good explanation of the
data — so it is not broken. But a focal seven times too large explains the same
hand *slightly better*. The objective simply does not distinguish the focal.

This is the **weak-perspective degeneracy**. A hand is a small object (~18 cm)
at moderate depth. Over that shallow depth range, scaling the focal and pushing
the object further away produce nearly identical images, so focal and distance
trade off almost exactly. Fig 06 shows the curve falling and then flattening
into a plateau that begins at roughly the true focal and never rises again:
there is no minimum to find.

This also explains the two most surprising rows in the table:

* **Why the planarized control "wins".** Flattening the hand removes the last
  residual depth cue, so the objective becomes perfectly scale-degenerate and
  the argmin is pinned near the low end of the grid. Being pinned near the low
  end happens to land nearer the true focal than the plateau does. It is an
  artefact of where a flat curve's argmin falls, not evidence that a flat hand
  is informative. A control beating the main condition is itself a signature of
  absent signal.
* **Why more frames hurt.** Averaging N flat curves in log space suppresses the
  per-frame noise that was the only thing creating a local minimum, so the
  argmin migrates further onto the plateau and toward the grid boundary
  (boundary solutions rise from 16.6 % to 25.7 %).

## 6. What this does and does not establish

**Establishes.** Under GigaHands imaging conditions, hand geometry carries
essentially no usable focal information, even given an oracle 3D hand. A
single-view reprojection objective cannot separate focal from depth for an
object of this size at this distance. This is a geometric limit, not a modelling
limit, so it will not be fixed by a better hand model.

**Does not establish.** That hand-based focal estimation is impossible in
general. The result is conditioned on this rig, this object scale, this depth
range and this single focal setting. A hand filling much more of the frame, or
imaged at a much shorter distance, would sit in a more perspective-dominated
regime and could behave differently.

**Consequence for CAM-EXP-007.** The planned predicted-hand experiment
(WiLoR / AnyHand cues) inherits this ceiling. A predicted hand cannot carry more
focal information than the oracle hand tested here, so CAM-007 should be
re-scoped: as a *focal estimator* it is already ruled out by this result. It
remains worth running only for a different question, such as whether predicted
hands are consistent enough to serve some other diagnostic purpose.

## 7. Important caveats

1. **`SINGLE_FOCAL_RIG_CONFOUND`.** The GT reference focal varies by only
   1.90 % (CV) across the 175 views, which is why a single constant achieves
   0.85 %. On this rig a per-view focal method has almost no dynamic range to
   demonstrate skill in. The constant-rig oracle is a
   `POST_HOC_ORACLE_SANITY_CONTROL`, not a method — it uses the evaluation
   targets and could not be formed without them.
2. **The reference 3D is not independent of the rig's calibration.** It is
   triangulated with the rig's own provided camera parameters, so any error
   shared across the whole rig is inherited and undetectable from inside this
   dataset.
3. **76.3 % coverage.** The usable subset excludes hands that are out of frame
   or poorly reconstructed, so the ceiling is measured on easier-than-average
   hands. A generous bias, which makes the negative result stronger.
4. **The planarized control uses a different PnP back-end** (IPPE/ITERATIVE
   rather than SQPNP, which rejects coplanar input), so it is not a perfectly
   matched comparison. Registered as `PLANAR_PNP_BACKEND_DIFFERS`.
5. **One dataset, one rig, one focal setting.**

All open issues: [`OPEN_ISSUES.md`](OPEN_ISSUES.md).

## 8. Artefacts

| kind | path |
| --- | --- |
| frozen specs | `experiments/manifests/cam_exp_006_{reference_hand,controls,evaluation}_spec_v1.json` |
| frozen frame grid | `experiments/manifests/cam_exp_006_reference_hand_frames_v1.csv.gz` |
| manual-QC manifest | `experiments/manifests/cam_exp_006_manual_hand_qc_v1.csv` |
| non-circularity audit | `results/raw/circularity_audit.csv` |
| solver output | `results/raw/focal_profile_solutions.csv.gz` |
| per-view estimates | `results/raw/view_focal_estimates.csv.gz` |
| verdict | `results/summary/cam006_verdict.json` |
| solver sanity check | `results/summary/solver_sanity_check.json` |
| tables | `tables/solver_summary.csv`, `tables/comparators.csv`, `tables/manual_qc_summary.csv`, `tables/manual_qc_breakdown.csv` |
| figures | `figures/fig01…fig07`, `figures/CAM_EXP_006_MAIN_EXPLANATION.png` |

## 9. Reproduce

```
python src/build_manual_qc.py        # audit panels (decodes video)
python src/assemble_manual_qc.py     # manifest + contact sheets
python src/freeze_spec.py            # freeze all specs BEFORE solving
python src/build_frame_manifest.py
python src/build_reference_3d.py
python src/run_solver.py
python src/evaluate.py               # first read of GT focal
python src/solver_sanity_check.py
python src/figures.py
python src/figures_profile.py
```
