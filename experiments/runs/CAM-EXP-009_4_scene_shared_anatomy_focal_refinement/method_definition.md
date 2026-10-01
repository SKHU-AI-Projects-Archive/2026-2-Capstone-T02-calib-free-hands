# Method definition

## The four methods

| method | scene | hand anatomy |
| --- | --- | --- |
| **M0_SCENE_ONLY** | yes | none — the baseline |
| M1_SCENE_GENERIC_INDEPENDENT | yes | generic prior, LEFT/RIGHT fitted independently, no shared `p_seq` |
| M2_SCENE_SHARED_NO_GENERIC | yes | shared `p_seq`, uniform base instead of the MANO prior |
| **M3_SCENE_SHARED_GENERIC_FULL** | yes | shared `p_seq` + generic prior + side deviations |

**PRIMARY comparison: M3 vs M0.** M1 and M2 exist so that a gain can be
attributed - M2 isolates the generic prior's contribution, M1 isolates the
shared sequence anatomy's.

Diagnostics that are **not** deployment methods: `H_ONLY` (hand score with no
scene term), `RIG_MEDIAN_TRAIN_ONLY` (a dataset-confound diagnostic) and
`REF_FOCAL_ORACLE` (an upper diagnostic for absolute 3D only).

## Parameterisation

Softmax over log-space offsets guarantees positivity and sum-to-one without any
absolute-size parameter:

```
p_seq = softmax(log(p0 + 1e-8) + a)
p_L   = softmax(log(p_seq + 1e-8) + delta_L)
p_R   = softmax(log(p_seq + 1e-8) + delta_R)
```

Softmax is shift-invariant, so `a` and each `delta` are mean-centred after every
update to remove the redundant gauge direction.

## Objective

```
L_fit = L_reproj + lambda_generic * ||a||^2
                 + lambda_side * (||delta_L||^2 + ||delta_R||^2)
```

`lambda_generic` and `lambda_side` were both fixed at **1.0 on the synthetic
gate only**. They were never tuned against the real reference focal.

## Solver conventions, inherited unchanged from CAM-EXP-009.1

- `UNDISTORT_ONCE_INTERNAL` — undistort once with the candidate K, then solve
  with `distCoeffs=None`. The distortion is never applied twice and a pinhole
  prediction is never scored against raw distorted pixels.
- 32 alternating iterations; deterministic initialisation.
- Per frame the pose `(R, T)` is a free nuisance solved by PnP; the anatomy is
  shared across all frames of the unit and across both hands through `p_seq`.

## Held-out scoring

Hand frames are split FIT/EVAL by a deterministic SHA-256 bucket of
`(sequence, camera, side, frame)` — never by fit quality. On EVAL frames the
anatomy is **frozen** and only the pose is refitted, and the score is the median
reprojection error divided by the image diagonal, so it is dimensionless.

LEFT and RIGHT need not appear in the same frame: `p_seq` is a sequence-level
parameter, so each side contributes whatever eligible frames it has.
