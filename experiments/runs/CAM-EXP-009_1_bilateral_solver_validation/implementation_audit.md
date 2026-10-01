# CAM-EXP-009.1 — implementation audit of CAM-EXP-009

Machine-readable: `tables/cam009_solver_implementation_audit.csv`.

## 1. DOUBLE_DISTORTION_CONVENTION_BUG (high)

`fit_bones` built its working set as

```python
prepped.append((dirs, _undistort(uv, K, dist), use))
```

and then, inside the loop, called

```python
R, T = _pose_from(l, dirs, uv, use, K, dist)
```

where `uv` is the **already undistorted** coordinate from `prepped` and `dist`
is still the original non-zero coefficient vector. `solvePnP` therefore removed
the distortion a second time from points that no longer had any.

On this rig k1 is about -0.39, so the second removal is not a rounding-level
effect.

## 2. RAW_VS_PINHOLE_REPROJECTION_MISMATCH (high)

`run_synthetic.eval_reproj` received RAW pixels, ran a distortion-aware PnP -
consistent so far - and then scored with

```python
proj = fx * Xc/Zc + cx        # pinhole, no distortion
errs.append(median(norm(proj[use] - uv[use])))   # uv is RAW DISTORTED
```

A no-distortion prediction compared against with-distortion measurements. This
was the held-out metric, i.e. gate g3 of CAM-009.

## 3. INSUFFICIENT_ALTERNATION_ITERATIONS (high)

CAM-009 used `n_iter = 6`. Measured convergence of the same fitter from a
uniform start, at the TRUE focal:

| iterations | reprojection (px) |
| ---: | ---: |
| 8 | 0.425 |
| 16 | 0.021 |
| 32 | 0.00067 |
| 64 | 0.00061 |
| 128 | 0.00054 |

CAM-009's observation that "even at the true focal the reprojection is ~3 px,
not near zero" was an unconverged fit, not a property of the geometry.

## 4. NONDETERMINISTIC_SEEDING (medium)

`abs(hash(condition_name))` was used to seed conditions. Python's `hash` is
salted per process unless `PYTHONHASHSEED` is fixed, so the synthetic data was
not reproducible across runs. CAM-009.1 uses a SHA-256 based `stable_seed` and
verifies identity across two independent processes.

## 5. MISLEADING_METRIC_NAME (medium)

CAM-009 recorded `bilateral_held` as

```python
out["bilateral_held"].append(bilateral_distance(lL, lR, ...))
```

which is the identical expression used for `bilateral`. Nothing was held out.
CAM-009.1 separates `BILATERAL_FIT_SHAPE_DISTANCE` from
`BILATERAL_EVAL_SHAPE_DISTANCE`, the latter being an independent refit on the
held-out frames.

## 6. Why this mattered so much

These faults do not merely add noise. Faults 1 and 2 add a **systematic**,
focal-dependent error that grows toward the image edges, and fault 3 leaves the
fit short of its optimum by an amount that also varies with the candidate
focal. Together they produced a score that decreased monotonically toward
larger focals - which is exactly the "runs to the grid boundary" behaviour
CAM-009 interpreted as a geometric degeneracy.
