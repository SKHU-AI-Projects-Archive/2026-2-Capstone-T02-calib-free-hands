# CAM-EXP-009.1 — camera model validation (G0, G1)

## G0 round trip

| direction | median | max | gate |
| --- | ---: | ---: | --- |
| raw: projectPoints(X_true, K, dist) vs the generated pixels | **0.0 px** | 0.0 px | median < 1e-6, max < 1e-4 |
| ideal: undistortPoints(raw, K, dist, P=K) vs pinhole(X_true) | **2.2e-09 px** | 8.4e-04 px | median < 1e-5, max < 1e-3 |

Both pass. The generator and the two scoring conventions agree with each other
to numerical precision.

## G1 true geometry plus PnP

Using the TRUE normalised bone vector and recovering only the pose:

| space | median reprojection | gate |
| --- | ---: | --- |
| ideal | **3.9e-07 px** | 0.05 px |
| raw | **3.8e-07 px** | 0.05 px |

So at the true focal, with the true shape, the projection and pose machinery is
exact. CAM-009's ~3 px at the true focal had nothing to do with geometry.

## G4 scoring consistency

Ideal and raw scores rank candidate focals almost identically: Spearman
**0.999**, and both place their minimum at q = 1.021 with the true shape fixed.
So the two conventions are not in conflict, and the choice between them does
not drive any conclusion.
