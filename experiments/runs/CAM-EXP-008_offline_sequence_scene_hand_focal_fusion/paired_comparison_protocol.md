# CAM-EXP-008 — paired comparison protocol

Everything in this run exists to make one comparison trustworthy:

```
S0   L_total(f) = L_scene(f)
S1   L_total(f) = L_scene(f) + lambda * L_hand(f)
```

If S0 and S1 differ in anything other than the hand term, the primary result is
invalid and must not be reported as a paired comparison.

## 1. What is held identical

| quantity | how it is guaranteed |
| --- | --- |
| sequence and physical camera | the same `(sequence, camera)` unit |
| frame IDs | one frame set per (view, subset), used by every condition |
| scene predictions | one per-frame AnyCalib record set, hashed |
| `f_scene`, `sigma_scene` | computed once from those frames |
| principal point, distortion, fy/fx | one scene-derived set, used by both |
| candidate focal grid | one absolute grid `f = q * f_anchor`, hashed |
| aggregation rules | identical |
| outer split | identical |
| evaluation target | identical |

`S0` and `S1` are not two pipelines. They are two readings of **one** cost
surface computed once per (view, subset): `L_scene` alone, and `L_scene` plus
the hand term.

## 2. Machine verification

`src/verify_paired_inputs.py` re-derives, per (view, subset), the frame hash,
the scene-prediction hash, the grid hash and the scene nuisance values for
every condition, and asserts they match.

```
same_frames_all              True
same_scene_all               True
same_grid_all                True
same_scene_nuisance_all      True
C3_flat_hand_reproduces_S0   True

PAIRED_INPUT_IDENTITY_HELD   (980 view-subset rows)
```

`C3` is the sharpest of these: a hand term forced to zero must reproduce `S0`
*exactly*. It does, to within 1e-9, which shows the fusion machinery adds
nothing of its own when the hand says nothing.

Table: `tables/paired_input_identity_audit.csv`.

## 3. Why the scene cost is built this way

```
L_scene(f) = Huber( log(f / f_scene) / sigma_scene ),  delta = 1.345
sigma_scene = max( 1.4826 * MAD(log per-frame pred_fx), 0.02 )
```

The property that matters: at `lambda = 0` the minimum of `L_total` is exactly
`f_scene`. So `S0` is not an approximation of the scene baseline — it *is* the
scene baseline, reproduced by the same code path the fusion uses. Any
difference in `S1` is then attributable to the hand term alone.

`sigma_scene` is the sequence's own scene dispersion, so a view whose scene
predictions agree strongly resists being moved, and a view whose predictions
scatter is more open to the hand. That is the intended behaviour.

## 4. Why the hand cost is an absolute pixel excess

```
L_hand(f) = C_hand(f) - min_f C_hand(f)      [pixels]
```

with `C_hand` aggregated hands → frame (median) → sequence (median).

It is **not** per-view min-max normalised. Min-max normalisation would rescale
every view's curve to the same range, turning a flat, uninformative profile
into an apparently confident one and letting noise push the scene estimate
around. Keeping pixels means a flat profile contributes a flat, near-zero term
and correctly fails to move anything.

The frame-level median also means a two-hand frame does not get double weight —
the aggregation mismatch found in CAM-EXP-006 is not repeated here.

## 5. lambda is never chosen on the held-out camera

λ ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10}, selected by grouped CV over the
**training physical cameras only**, inside each outer fold. The held-out
camera's reference focal never influences λ. Figure 09 shows the selected
values.

## 6. The controls

| id | what it destroys | what it keeps |
| --- | --- | --- |
| `C1` wrong-frame hand | pose correspondence | hand size, worker, sequence, side |
| `C2` view-shuffled hand | all correspondence to this camera | the statistical shape of a hand profile |
| `C3` flat hand | the hand term entirely | must reproduce `S0` |

The decisive rule, fixed in advance: if `S1` beats `S0` but `C1` or `C2` gain
similarly, the gain is not attributed to hand geometry. That is what happened —
`C2` gained more than `S1`.
