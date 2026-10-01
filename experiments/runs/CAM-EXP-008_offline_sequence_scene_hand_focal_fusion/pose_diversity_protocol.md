# CAM-EXP-008 — pose diversity protocol

## 1. Question

At the same frame count, do more varied hand poses make the hand term more
useful?

## 2. Descriptor (target-blind)

For each frame: the root-relative reference hand, scale-normalised by its own
norm, with global rotation removed by Procrustes alignment to the view's medoid
pose. What remains is articulation. No reference focal, no focal error and no
fusion outcome takes any part.

Distance: Euclidean between aligned, scale-normalised joint configurations.

## 3. Selection

* `DIVERSE_N` — greedy farthest-point, seeded at the medoid
* `LOW_N` — the N poses nearest the medoid

N in {16, 32}. Available for 106 of 154 units.

## 4. The interpretation rule that matters

The two subsets contain **different frames**, so their scene-only baselines
differ. Comparing `scene+hand` errors across subsets would confound pose
diversity with which frames the scene happened to like.

The comparison is therefore between **paired incremental gains**:

```
G_diverse = err(scene-only, DIVERSE) - err(scene+hand, DIVERSE)
G_low     = err(scene-only, LOW)     - err(scene+hand, LOW)
```

`POSE_DIVERSITY_HELPS_HAND_FUSION` requires `G_diverse > G_low`.

## 5. Result

| subset | scene-only | scene+hand | paired gain |
| --- | ---: | ---: | ---: |
| LOW16 | 10.083 % | 10.059 % | **+0.084 pp** |
| DIVERSE16 | 9.931 % | 10.184 % | +0.000 pp |
| LOW32 | 9.801 % | 9.718 % | **+0.039 pp** |
| DIVERSE32 | 9.740 % | 9.772 % | +0.029 pp |

The criterion is **not met** at either N. If anything the low-diversity subsets
gained marginally more. Both are negligible, and neither approaches the
materiality bar.
