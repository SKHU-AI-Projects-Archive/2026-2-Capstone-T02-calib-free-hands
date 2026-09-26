# CAM-EXP-009.1 — bone fitter validation (G2)

At the true focal, with realistic distortion and no noise.

| initialisation | reprojection (ideal px) | bone-vector L1 vs truth |
| --- | ---: | ---: |
| I0 uniform (the deployable one) | **6.7e-04** | 4.8e-07 |
| I1 multi-start, 8 deterministic positive perturbations | 6.6e-04 | 4.8e-07 |
| I2 true-bone initialisation (ORACLE, diagnostic only) | 1.0e-05 | 4.4e-09 |

Gate: 0.05 px. All three pass.

Cosine similarity between the uniform-start estimate and the truth: **0.9999**.

## Shape identifiability

Multi-start shape spread (median pairwise L1 between the vectors recovered from
eight different starts): **2.7e-04**.

Every initialisation converges to the same bone proportions, so there is **no
shape non-identifiability** at the true focal. `SHAPE_NONIDENTIFIABILITY` does
not fire.

## What changed from CAM-EXP-009

Nothing in the equations. The fitter was correct; it was being stopped after 6
alternations, which is roughly a factor of five short of convergence from a
uniform start. Raising it to 32 was done during validation, judged only against
this positive control, before any focal sweep.
