# CAM-EXP-008 — Offline Sequence-Level Scene + Hand Geometry Focal Fusion

```
DECISION TAGS
  NO_INCREMENTAL_HAND_GEOMETRY_SIGNAL
  CONTROLS_NOT_DEGRADED

MODE  FULL_DURATION_64_APPROXIMATION
      (the whole-sequence run hit the pre-registered compute blocker)
```

**One sentence:** On exactly the same video and exactly the same frames, adding
sequence-level reference hand geometry moved the scene-anchored focal estimate
by a median of **+0.03 percentage points** — a 2.2 % relative reduction against
a pre-registered bar of 10 % — and a **view-shuffled hand profile achieved a
larger gain (+0.048 pp)**, so the change is not attributable to hand geometry.

## 1. Plain-language summary

This was not an experiment about reading the focal length off a hand.

First, an existing scene-based calibration looks at the whole recorded video
and proposes a focal length. Then we ask whether the many hand poses observed
in that same video can nudge that proposal toward a better answer.

To keep the comparison fair, "scene-only" and "scene+hand" used **the same
video, the same frames, the same scene predictions and the same candidate
range**. The only difference was whether the hand term was switched on. That
identity was machine-verified.

The answer is no. The hand term barely moved anything, and when we deliberately
scrambled which hand belonged to which video — destroying all real hand
information — we got a *slightly bigger* improvement. That is the clearest sign
that the tiny gain was not hand information.

## 2. How this differs from CAM-EXP-006

| | CAM-EXP-006 | CAM-EXP-008 |
| --- | --- | --- |
| who decides the focal | the hand alone, by PnP | the scene proposes; the hand may only express a preference nearby |
| search range | γ = f/max(W,H) ∈ [0.25, 5.0] | q = f/f_scene ∈ [0.5, 1.5] |
| distortion | **not modelled** (the error CAM-006.1 found) | scene-estimated k1, k2, in both conditions |
| principal point | image centre | scene-estimated, in both conditions |
| hand scale | absolute reconstruction | **scale removed** — no hand-size prior |
| unit | one frame set | one whole recorded video |

CAM-006's failure does **not** imply CAM-008's. That was the explicit
hypothesis: a hand that cannot determine a focal on its own might still break a
tie inside a region the scene has already narrowed. It was worth testing, and
it was tested.

## 3. Did we really use the whole sequence? No — and here is exactly why

```
FULL_SEQUENCE_COMPUTE_BLOCKER
```

The ALL_COMMON (whole-video) set was built and attempted: **30,031 frames
across 157 views, 40 physical cameras, 5 sequences**, median 162 frames per
view. It was abandoned on measured throughput, not guessed:

| stage | measured rate | projected |
| --- | --- | ---: |
| AnyCalib scene predictions | ~0.75 views/min | ~3.5 h |
| other-camera-only reference 3D | ~0.5 views/min | ~5 h |
| hand focal profiles (real + control) | from the CAM-006 PnP rate | ~3 h |
| **total** | | **~8 h** |

The pre-registered rule (spec §13) fires a blocker above **6 GPU-hours**. The
frame-count criterion did *not* fire (30,031 ≪ 100,000); the runtime one did.
**No arbitrary subsampling was performed** — the partial caches (9 scene views,
6 reference-3D views) were discarded and contribute to nothing.

The run therefore proceeds on `FULL_DURATION_64_APPROXIMATION`: the frozen
CAM-EXP-004.1 64-frame grid, which is spread across each recording's whole
duration. Its AnyCalib predictions already existed at 100 % success, so **no
new scene inference was required**. This is never called "the full sequence".

**What is unaffected:** the primary paired comparison. **What is affected:** we
cannot say how much a whole video's worth of frames would add.

### 3.1 What was actually used

| item | value |
| --- | --- |
| candidate frames | **6,677** |
| sequence-camera units | **154** |
| physical cameras | **40** |
| sequences | **5** |
| frames per unit | min 1, median **51**, max 64 |
| hand observations (reference 3D) | **12,860** |
| scene predictions reused, none recomputed | 6,677 |

**Naming.** Historical result files use the key `ALL_COMMON` for all usable
frames within the frozen 64-frame grid. This is **not** the attempted
30,031-frame whole-video ALL_COMMON candidate set of §3. Where the narrative
needs to be unambiguous this report calls it the
**`64GRID_COMMON_SET`** — every QC-passing frame within the 64-frame grid,
median 51 per view. The frozen manifests and result-file keys are not
retroactively renamed.

## 4. The paired comparison held

```
PAIRED_INPUT_IDENTITY_HELD          (980 view-subset rows)
```

| check | result |
| --- | --- |
| identical frame IDs | **True** for every row |
| identical scene predictions | **True** |
| identical candidate grid | **True** |
| identical scene nuisance (f_scene, σ) | **True** |
| `C3` flat-hand reproduces `S0` exactly | **True** |
| camera under test ever an inlier of its own reference 3D | **0** |

S0 and S1 are computed from one shared per-(view, subset) scene cost and one
shared grid; the hand term is the only thing switched on.

## 5. Primary result — leave-one-physical-camera-out, ALL_COMMON, 151 units

| | scene-only S0 | scene+hand S1 |
| --- | ---: | ---: |
| median relative focal error | **8.866 %** | **8.672 %** |
| within ±5 % | 22.5 % | 21.9 % |
| tracking Spearman vs reference focal | +0.095 | +0.097 |
| estimate CV | — | 7.98 % (reference CV 1.73 %) |

| paired S1 − S0 | value |
| --- | ---: |
| median paired gain | **+0.0317 pp** |
| cluster-bootstrap 95 % CI | [+0.0116, +0.0654] |
| relative reduction | **+2.19 %** |
| within ±5 % gain | **−0.67 pp** |
| units improved | 62.3 % |
| median selected λ | 3.0 |

### 5.1 The frozen success criteria

| criterion | threshold | result |
| --- | --- | --- |
| 1. beats S0 | ≥ 10 % relative **or** +5 pp within ±5 % | **FAIL** (2.19 %, −0.67 pp) |
| 2. gain CI excludes zero | — | PASS |
| 3. beats the controls | — | **FAIL** (see §6) |
| 4. not a constant collapse | — | PASS |

All four were required. Two failed, including the decisive one.

## 6. The controls decide it

| condition | median paired gain | CI | relative reduction |
| --- | ---: | --- | ---: |
| **S1 real hand** | **+0.0317 pp** | [+0.0116, +0.0654] | +2.19 % |
| C1 wrong-frame hand | +0.0006 pp | [+0.0001, +0.1341] | −4.76 % |
| **C2 view-shuffled hand** | **+0.0478 pp** | [−0.0170, +0.1336] | −1.57 % |

A hand profile taken from an entirely different video — carrying no
information about this camera at all — produced a **larger** median gain than
the real hand. Whatever the +0.03 pp is, it is not hand geometry. It is the
generic effect of adding a small bounded perturbation to a robust scene prior.

Tag: `CONTROLS_NOT_DEGRADED`.

## 7. Why the hand term is powerless here

`results/summary/hand_term_diagnostic.json`. The hand profile is both **shallow
and biased**:

| property | value |
| --- | --- |
| total hand reprojection excess across the whole ±50 % focal range | median **1.88 px** (p10 0.62, p90 9.65) |
| focal the hand prefers, relative to the scene estimate | median **1.184×** |
| views whose hand profile prefers the grid boundary | **22.7 %** |

Two consequences:

1. **Shallow.** Over a ±50 % focal sweep the entire hand evidence is under two
   pixels. Against a robust scene prior it has almost nothing to say.
2. **Biased.** Where it does speak, it systematically prefers a focal ~18 %
   *larger* than the scene estimate, and nearly a quarter of views run to the
   grid boundary. So the little force it exerts does not point at the reference
   focal.

This is why nested λ selection is bimodal (λ = 0.01, i.e. effectively off, or
λ = 10) and why the net effect is ~0.

Hand-only, for context, is far worse than scene-only at every frame count:

| | N8 | N16 | N32 | N64 | ALL_COMMON |
| --- | ---: | ---: | ---: | ---: | ---: |
| scene-only | 9.02 % | 8.61 % | 9.78 % | 10.24 % | 8.87 % |
| hand-only (H0) | 22.99 % | 25.00 % | 23.40 % | 27.65 % | 24.19 % |

So H0 fails, consistent with CAM-006 — and S1 fails too. The hypothesis that
scene-anchoring would rescue a weak hand signal is **not** supported.

## 8. More frames did not help

| subset | units | scene-only | scene+hand | paired gain |
| --- | ---: | ---: | ---: | ---: |
| N8 | 143 | 9.020 % | 8.741 % | +0.110 pp |
| N16 | 133 | 8.611 % | 8.976 % | +0.056 pp |
| N32 | 106 | 9.775 % | 9.168 % | +0.151 pp |
| N64 | **23** | 10.243 % | 10.243 % | +0.000 pp |
| ALL_COMMON | 151 | 8.866 % | 8.672 % | +0.032 pp |

There is no coherent trend, and every value is far below the materiality bar.

**A note on N64.** Only 23 of 154 units have 64 QC-passing frames on the grid
(the median is 51), so N64 is thin and its units are not representative. The
frozen spec (§36) pre-registers ALL_COMMON as the preferred primary with N64
only as a fallback, so ALL_COMMON is reported as primary. **The verdict is
identical under either choice** (N64: +0.0002 pp, 0.00 % relative).

## 9. Pose diversity did not help either

Compared as **paired gains**, because the subsets contain different frames and
their scene baselines differ:

| subset | units | scene-only | scene+hand | paired gain |
| --- | ---: | ---: | ---: | ---: |
| LOW16 | 106 | 10.083 % | 10.059 % | **+0.084 pp** |
| DIVERSE16 | 106 | 9.931 % | 10.184 % | +0.000 pp |
| LOW32 | 106 | 9.801 % | 9.718 % | **+0.039 pp** |
| DIVERSE32 | 106 | 9.740 % | 9.772 % | +0.029 pp |

`G_diverse > G_low` is the criterion for `POSE_DIVERSITY_HELPS_HAND_FUSION`. It
is **not met** at either N — if anything the low-diversity subsets gained
slightly more. Both are negligible.

## 10. Secondary results

**LOSO sequence (N64):** gain +0.021 pp — no better than LOCO, so this is not a
case of a local signal that fails to generalise. Tag
`HAND_SIGNAL_LOCAL_BUT_NOT_GENERALIZABLE` does not fire.

**Stress subsets:** S2 has only 6 views and 5 physical cameras and is flagged
`UNDERPOWERED`; S5 is empty. No conclusion is drawn. (For the record, S2 gave
+0.095 pp for the real hand and +0.063 pp for the wrong-frame control.)

**Constant-rig oracle:** 0.906 % against scene-only 8.87 %. On a rig whose
reference focal varies by only 1.73 % (CV), a single constant still beats every
image-based estimate by roughly 10×. This is an `ORACLE_SANITY_CONTROL`, not a
method, and it caps what any per-camera conclusion on GigaHands can show.

**Constant collapse:** not present. S1's estimates vary with CV 7.98 % against
a reference CV of 1.73 % — scattered, not collapsed.

## 11. Answers to the pre-registered questions

| | question | answer |
| --- | --- | --- |
| Q1 | did scene+hand beat scene-only on identical input? | Not materially: +0.0317 pp, 2.19 % relative, against a 10 % bar |
| Q2 | by how much? | 8.866 % → 8.672 % median error; within ±5 % got *worse* by 0.67 pp |
| Q3 | on unseen physical cameras? | That *is* the primary protocol; the answer is still no |
| Q4 | do wrong-frame / shuffled hands give the same gain? | **Yes — the shuffled hand gained more** |
| Q5 | does more temporal evidence help? | No coherent trend; N64 gave exactly zero |
| Q6 | does pose diversity help? | No; low-diversity subsets gained marginally more |
| Q7 | does hand-only fail while scene+hand succeeds? | Hand-only fails (24 % vs 9 %) and scene+hand also fails |
| Q8 | do scene distortion/PP errors limit the gain? | **No.** Answered by the closure run: with the provided PP, distortion and fy/fx the profile deepens only 1.82 -> 2.86 px, the q bias is unchanged, and fusion gets marginally worse |
| Q9 | camera-specific improvement beyond a constant prior? | No. A constant still beats everything by ~10× |

## 12. What this does and does not establish

**Establishes.** Under the evaluated GigaHands regime and this offline
scene-anchored sequence-level fusion formulation, the reference hand geometry
did not provide a measurable improvement over the paired scene-only baseline,
and the small change it did produce was matched or exceeded by controls that
carry no hand information.

**Does not establish.** That hand information is useless. This tested one
fusion formulation (a robust scene prior plus a PnP reprojection profile), one
rig with a 1.73 % focal spread, and a 64-frame approximation rather than whole
videos. A different objective, a wider focal range across cameras, or a hand
much closer to the camera could behave differently.

Note also that this was the **information ceiling**: the hand geometry used
here is other-camera-only reference 3D paired with dataset-provided 2D
observations, which a single deployed camera does not have. A deployable
predicted hand would be strictly noisier.

## 13. Consequence for CAM-EXP-008.1

**Do not proceed with the predicted-hand practical fusion for now.** The
pre-registered logic was explicit: if even the oracle hand does not help, there
is little motivation to repeat the same explicit scene-plus-reprojection
formulation with noisier predicted geometry. The oracle hand did not help, and
the controls show the residual movement was not hand information.

This does **not** establish that every predicted-hand formulation or learned
hand-derived constraint must fail — only that this one has no support.

The binding constraint remains what CAM-EXP-007 also ran into: on a rig where
the reference focal varies by 1.73 % and a constant scores 0.9 %, there is very
little per-camera variation for any cue to explain. A varied-focal dataset is
the most important next validation requirement. That said, this run also tested
only one fusion objective, so an objective-side limitation cannot be excluded
either.

## 14. Limitations

1. `FULL_DURATION_64_APPROXIMATION`, not whole videos (§3). Median 51 frames
   per unit.
2. `SINGLE_FOCAL_RIG_CONFOUND`: reference focal CV 1.73 %; a constant scores
   0.906 %.
3. The oracle-nuisance diagnostic was not run *in this run*. It has since
   been completed in `CAM-EXP-008_closure_camera_nuisance`, which found the
   negative result robust to camera-nuisance error
   (`CAM008_NEGATIVE_ROBUST_TO_CAMERA_NUISANCE`). Q8 is answered there.
4. The E2 secondary anchor was not run, for the same reason.
5. S2 underpowered (6 views, 5 cameras); S5 empty.
6. One dataset, one rig, one focal setting, one worker per sequence.
7. The hand term is an internal information ceiling, not deployable.

## 15. Reproduce

```
python src/build_common_frame_manifest.py    # ALL_COMMON candidate set
python src/build_64grid_inputs.py            # the approximation actually used
python src/freeze_spec.py                    # freeze BEFORE the target
python src/build_reference_hand_geometry.py
python src/build_frame_subsets.py
python src/compute_hand_profiles.py
python src/compute_hand_profiles.py --wrong-frame
python src/run_fusion.py
python src/verify_paired_inputs.py           # must print PAIRED_INPUT_IDENTITY_HELD
python src/evaluate.py                       # FIRST read of the reference focal
python src/diagnose_hand_term.py
python src/figures.py
```
