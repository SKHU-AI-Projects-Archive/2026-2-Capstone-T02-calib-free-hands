# Geometry representation

## Topology - 20 connected bones

Verified against the dataset's 21-joint hand and frozen in the spec manifest:

```
thumb   0-1   1-2    2-3    3-4
index   0-5   5-6    6-7    7-8
middle  0-9   9-10  10-11  11-12
ring    0-13 13-14  14-15  15-16
pinky   0-17 17-18  18-19  19-20
```

Each bone connects a joint to its parent. Arbitrary joint pairs such as `4-8`,
`4-12` or `8-20` span no bone and are **not used**; `self_audit.py` fails if
any appears.

## Per frame, per hand

1. 20 connected bone lengths `l_b` from the reference 3D.
2. **Absolute size removed:** `p_b = l_b / sum_b l_b`, so `sum_b p_b = 1`.
   A person with larger hands is never separated for that reason alone.
3. **PRIMARY representation:** `z_b = log(p_b + 1e-8)`.
   The epsilon is frozen at `1e-8` for continuity with CAM-EXP-009.2.

## Template

A template is the **bone-wise median of `z`** over a unit's eligible frames.
The median is used rather than the mean so that a few badly reconstructed
frames cannot drag the template.

Three families, all from the same frame vectors:

- `FULL` — one per (sequence, camera, hand)
- `HALF` — one per (sequence, camera, hand, A|B), for the repeatability split

LEFT and RIGHT are **not** required to appear in the same frame. Hand anatomy
is treated as a sequence-level property, so each side's template is built from
whatever eligible frames that side has.

## PRIMARY distance

```
D(A,B) = median_b | z_A,b - z_B,b |
```

the median absolute log-proportion difference. **This one metric is the
primary.** Four secondaries (mean absolute log, raw proportion L1, cosine,
finger-internal ratio) were frozen at the same time and are reported as
sensitivity analysis only. The primary conclusion is not re-chosen from
whichever secondary looks best.
