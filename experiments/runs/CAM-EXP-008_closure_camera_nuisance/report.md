# CAM-EXP-008 CLOSURE — oracle camera-nuisance diagnostic

```
VERDICT  CAM008_NEGATIVE_ROBUST_TO_CAMERA_NUISANCE
```

**One sentence:** Giving the hand objective the *provided* principal point,
distortion and aspect ratio made the hand profile only 1.6× deeper, left its
focal bias essentially unchanged (q 1.263 → 1.272), and made the fused estimate
slightly *worse* than scene-only — so CAM-EXP-008's negative result was not
caused by scene-estimated camera-parameter error.

## 1. What this closes

CAM-EXP-008 registered `ORACLE_NUISANCE_CONDITION_NOT_RUN` as an open issue:
the diagnostic that gives the fusion the true camera nuisance parameters was
skipped once the hand profile proved shallow. This run performs it.

Everything is held identical to CAM-008 — the same sequence-camera units, the
same frame IDs, the same dataset-provided 2D, the same reference 3D, the same
candidate focal grid and the same `f_scene` anchor. The **only** thing that
changes is which camera nuisance parameters the hand reprojection uses.

| condition | principal point | distortion | fy/fx |
| --- | --- | --- | --- |
| `N0` | scene | scene | scene | *(= CAM-008 primary)* |
| `N1` | **provided** | scene | scene |
| `N2` | scene | **provided** | scene |
| `N3` | **provided** | **provided** | scene |
| `N4` | **provided** | **provided** | **provided** |

`N1`–`N4` are `ORACLE_DIAGNOSTIC_ONLY`. The provided focal **magnitude** is
never used anywhere.

## 2. Does the true camera model sharpen the hand profile?

| condition | hand profile depth (px) | preferred q | boundary rate | correct direction |
| --- | ---: | ---: | ---: | ---: |
| N0 scene nuisance | 1.820 | 1.263 | 0.24 | 0.66 |
| N1 + true PP | 1.797 | 1.221 | 0.23 | 0.67 |
| N2 + true distortion | 2.524 | 1.317 | 0.24 | 0.70 |
| N3 + true PP & distortion | 2.797 | 1.299 | 0.24 | 0.71 |
| **N4 full oracle except focal** | **2.861** | **1.272** | **0.25** | **0.72** |

**Depth:** 1.82 px → 2.86 px. Deeper, but still under three pixels across a
±50 % focal sweep. That is not a profile that can argue with a robust scene
prior.

**Bias:** the preferred focal stays ~27 % above the scene estimate. Supplying
the true camera model did **not** remove the bias — it barely moved it, and N2
and N3 actually made it slightly worse.

**Boundary rate:** unchanged at ~24 %. A quarter of views still prefer the edge
of the search range.

**Direction:** the share of views whose hand profile points from the scene
estimate *toward* the reference focal rises 0.66 → 0.72. Better than chance but
far from decisive, and it does not convert into accuracy.

## 3. Does the fused estimate improve?

Leave-one-physical-camera-out, λ selected on training cameras only, 151 units:

| condition | scene-only | scene+hand | paired gain | relative reduction |
| --- | ---: | ---: | ---: | ---: |
| N0 | 8.866 % | 8.672 % | +0.0317 pp | +2.19 % |
| N1 | 8.866 % | 8.705 % | +0.0449 pp | +1.82 % |
| N2 | 8.866 % | 8.695 % | +0.0139 pp | +1.92 % |
| N3 | 8.866 % | 8.712 % | +0.2600 pp | +1.73 % |
| **N4** | 8.866 % | **8.877 %** | +0.0279 pp | **−0.13 %** |

Under the full oracle the fused estimate is marginally **worse** than
scene-only. No condition approaches the 10 % materiality bar CAM-008 froze.

## 4. Interpretation

`CAM008_NEGATIVE_ROBUST_TO_CAMERA_NUISANCE`.

The three pre-registered signs of `CAM008_NUISANCE_LIMITED` — a markedly
sharpened profile, a reduced q bias, and materially better fusion — are all
absent. The shallow, upward-biased hand profile CAM-008 reported is a property
of the hand-geometry evidence in this imaging regime, not an artefact of
imperfect scene-estimated camera parameters.

This **strengthens** CAM-EXP-008's original negative conclusion. Its numbers
are unchanged.

### 4.1 A note on the two q figures

CAM-008's `hand_term_diagnostic.json` reports a preferred q of 1.184; the N0
row here reports 1.263. These are consistent, not contradictory: CAM-008's
diagnostic took a plain median over every stored profile in a view, while this
closure restricts to the `ALL_COMMON` frame set and uses the frame-balanced
aggregation. Both show the same upward bias of roughly 18–26 %.

## 5. What is unchanged

CAM-EXP-008's `results/raw`, `results/summary`, `figures` and frozen manifests
are untouched. This run writes only into its own directory. Its headline
numbers remain:

> S0 8.8658 %, S1 8.6719 %, relative reduction 2.1871 %, within ±5 % 22.52 % →
> 21.85 %, real-hand paired gain +0.0317 pp, view-shuffled control +0.0478 pp,
> tags `NO_INCREMENTAL_HAND_GEOMETRY_SIGNAL` and `CONTROLS_NOT_DEGRADED`.

## 6. Limitations

1. The oracle conditions consume provided camera metadata and are not
   deployable; they exist only to test an explanation.
2. This closure inherits CAM-008's `FULL_DURATION_64_APPROXIMATION` scope —
   median 51 frames per unit, not whole videos.
3. `SINGLE_FOCAL_RIG_CONFOUND` still applies: the reference focal varies by
   1.73 % and a constant beats every image-based estimate here.
4. The fy/fx ratio contributes almost nothing on this rig because it is within
   0.13 % of unity, so N4 and N3 are nearly the same condition.

## 7. Reproduce

```
python src/run_nuisance_profiles.py --condition N1   # and N2, N3, N4
python src/evaluate_closure.py
```
