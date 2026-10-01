# CAM-EXP-006.1 — implementation audit of CAM-EXP-006

Full table: `tables/cam006_aggregation_implementation_audit.csv`.
Machine verdict: `results/summary/cam006_implementation_audit.json`.

## 1. The mismatch

```
CAM006_ORIGINAL_HAND_WEIGHTED
```

`focal_profile_solver.combine()` carries the docstring

> "Mean over frames of the per-frame log objective, as frozen in the spec."

but the argument it receives, built in `run_solver.py`, is

```python
ps = [p for f in order[:n] for p in prof.get((cond, f), [])]
```

a **flat list of per-hand profiles**. `combine()` then takes
`np.nanmean(lg, axis=0)` over that list. A frame in which both hands are usable
therefore contributes two entries and receives twice the weight of a one-hand
frame. The effective aggregation unit was the hand, not the frame.

## 2. The other eight stages

| stage | verdict |
| --- | --- |
| joint to hand | MATCH |
| hands within one frame | **MISMATCH** |
| frames to view | MATCH_IN_FORM (correct once stage 2 is fixed) |
| PnP back-end and planar fallback | MATCH_WITH_DOCUMENTED_EXTENSION |
| principal point | MATCH (centre, as specified) |
| distortion | MATCH (none, as specified) |
| fx / fy | MATCH |
| focal grid and refinement | MATCH |
| flat / boundary flags | MATCH |

The implementation matched its own frozen spec everywhere except stage 2. Note
that "MATCH" at stages 5 and 6 means the code did what the spec said — it does
not mean the spec was a good choice. Section 4.3 of the report shows the
distortion decision was the consequential one.

## 3. Materiality

`FRAME_WEIGHTING_NON_MATERIAL`. At N=16 the correction moves the median error
by 0.96 pp, the flat share by 1.14 pp and the identifiable share by 1.14 pp,
all under the 10 pp thresholds frozen beforehand.

The mismatch was real, is now fixed, and did not drive CAM-EXP-006's
conclusion.

## 4. Equivalence check

`camera_model_solver.profile_hand` with `cx=W/2, cy=H/2, dist=None,
fy_over_fx=1.0` was verified to reproduce
`CAM-EXP-006/src/focal_profile_solver.profile_one` to a maximum absolute
difference of **0.0e+00**, and `pick()` returns an identical gamma and status.
That exactness is what makes R0's reproduction of 202.16 % meaningful.
