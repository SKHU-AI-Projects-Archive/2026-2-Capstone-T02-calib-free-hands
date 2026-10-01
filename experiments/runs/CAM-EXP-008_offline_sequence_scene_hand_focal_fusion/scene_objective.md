# CAM-EXP-008 — the scene objective

## 1. Model

AnyCalib `anycalib_gen` / `radial:2`, the canonical primary of this programme.
Chosen because CAM-EXP-004.1 measured it as 100 % bit-identical across repeated
runs while GeoCalib was 0 %, so the primary conclusion does not depend on
run-to-run variation.

## 2. Sequence-level aggregation

From the per-frame predictions on the common frames of a unit:

| parameter | definition |
| --- | --- |
| `f_scene` | median per-frame `pred_fx` |
| `r_fy` | median(`pred_fy` / `pred_fx`) |
| `cx`, `cy` | median `pred_cx`, `pred_cy` |
| `k1`, `k2` | median predicted radial coefficients |
| `sigma_scene` | max(1.4826 x MAD(log `pred_fx`), 0.02) |

Every one of these is used **identically** in S0 and S1.

## 3. Cost

```
L_scene(f) = Huber( log(f / f_scene) / sigma_scene ),  delta = 1.345
```

Its minimum is exactly `f_scene`, so S0 reproduces the scene baseline through
the same code path the fusion uses.

## 4. Why distortion is modelled

CAM-EXP-006.1 found that feeding raw distorted 2D to a solver assuming no
distortion, on a rig whose median k1 is about -0.39, substantially inflated the
focal error. The hand objective here therefore uses the **scene-estimated**
distortion at every candidate focal, inside both `solvePnP` and
`projectPoints`. GT distortion is never used in the primary condition.

## 5. Context

The scene-only numbers here (8.9 % median at ALL_COMMON) are in the range
CAM-EXP-004 reported for static-camera aggregation. They are contextual, not a
like-for-like reproduction: the eligibility set differs.
