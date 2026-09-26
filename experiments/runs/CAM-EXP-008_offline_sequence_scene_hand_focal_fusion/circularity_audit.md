# CAM-EXP-008 — circularity audit

Machine-readable: `tables/circularity_audit.csv`.

## 1. Per condition

| condition | target 2D in hand objective | target 2D in reference 3D | target extrinsics | target reference focal in optimisation | GT distortion | GT principal point | scene prediction | allowed primary |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| S0 | NO | NO | NO | NO | NO | NO | YES | YES |
| S1 | YES | **NO** | NO | NO | NO | NO | YES | YES |
| H0 | YES | NO | NO | NO | NO | NO | YES | diagnostic |
| C1 | YES | NO | NO | NO | NO | NO | YES | control |
| C2 | YES | NO | NO | NO | NO | NO | YES | control |
| C3 | n/a | NO | NO | NO | NO | NO | YES | sanity |

## 2. The distinction that matters

The target camera's 2D observation appears **once**, as the thing the reference
hand is reprojected against. It never appears in building that reference hand.
If it did, the hand would already know the answer it is being tested on.

Verified per observation: the camera under test was an inlier of its own
reference reconstruction in **0 of 12,860** cases.

## 3. What the reference focal is used for

Only evaluation, and only in `src/evaluate.py`, which runs after every estimate
is frozen on disk. It also drives nested lambda selection — but strictly on
**training** physical cameras inside each outer fold, never on the held-out
camera.

## 4. Frame selection

Frame eligibility uses pre-existing QC and data availability only. No frame was
selected or rejected by focal error, reference focal, hand PnP score, proximity
to a candidate minimum or whether fusion succeeded. No hand observation was
filtered by whether it improved the focal.
