# CAM-EXP-008 — the hand geometry objective

## 1. Source

`OTHER_CAMERA_ONLY_REFERENCE_3D`: the hand triangulated from every *other*
calibrated camera in the sequence, with the camera under test removed **before**
reconstruction. CAM-EXP-001.3's frozen reconstruction is reused unchanged.

Audit: the camera under test appeared among its own reference hand's inlier
cameras in **0** of 12,860 hand observations.

This is an **internal information ceiling**, not a deployable input: a single
deployed camera has no other cameras to triangulate from.

## 2. SEQUENCE_CONSISTENT_REFERENCE_HAND

Per `(sequence, camera, anatomical side)`:

1. **bone lengths** = the median over that view's own reference
   reconstructions. One worker's bones do not change between frames, so this is
   a legitimate sequence-level constraint.
2. **scale is removed** — the template is normalised by its own bone-length
   sum. No absolute hand-size prior is used anywhere, because hand size differs
   between factories and workers. What survives is perspective information, not
   a metric ruler.
3. **articulation is per-frame** — each frame keeps its own bone directions,
   re-assembled along the kinematic tree with the shared normalised lengths.

The template for a view is built only from that view's own reconstructions.
Borrowing another view's would re-admit the target camera's 2D and break
non-circularity.

## 3. Kinematic tree

The 21-joint topology is the dataset convention — the same connection list used
for the CAM-EXP-006 manual-QC overlays that were visually verified to land on
real hands. Parent indices are not invented here.

## 4. Per-candidate solve

At every candidate focal `f = q * f_anchor`:

```
K(f) = [[f, 0, cx_scene], [0, r_fy*f, cy_scene], [0, 0, 1]]
dist  = [k1_scene, k2_scene, 0, 0]
```

`cv2.solvePnP` (SQPNP, with the documented IPPE/ITERATIVE fallback reused from
CAM-EXP-006.1) recovers a free rotation and translation per frame and hand. GT
extrinsics are never used.

## 5. Aggregation

```
joints -> hand      median reprojection error
hands  -> frame     median      (a two-hand frame does not count twice)
frames -> sequence  median
L_hand(f) = C_hand(f) - min_f C_hand(f)    [pixels, not normalised]
```

## 6. What it turned out to be

Measured across 154 units:

* total excess over a +-50 % focal sweep: median **1.88 px**
* preferred focal relative to the scene estimate: median **1.184x**
* views preferring the grid boundary: **22.7 %**

Shallow and biased. That is the mechanical reason the fusion gain is about zero.
