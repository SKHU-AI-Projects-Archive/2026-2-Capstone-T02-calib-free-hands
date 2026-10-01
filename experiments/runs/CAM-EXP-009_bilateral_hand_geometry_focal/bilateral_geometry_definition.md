# CAM-EXP-009 — bilateral geometry definition

## 1. The hypothesis

For one person, anatomically corresponding bones of the LEFT and RIGHT hand
have related proportions. If the candidate focal is correct, the bone
proportions required to explain the left-hand images and those required to
explain the right-hand images should agree better than at a wrong focal.

## 2. Bones and correspondence

Only true parent-child bones of the 21-joint tree are used — 20 of them.
Arbitrary joint pairs such as 4-8 or 8-20 are not bones and are never used.

```
thumb   0-1  1-2   2-3   3-4
index   0-5  5-6   6-7   7-8
middle  0-9  9-10 10-11 11-12
ring    0-13 13-14 14-15 15-16
little  0-17 17-18 18-19 19-20
```

The tree is identical for both hands, so anatomically corresponding bones share
the same index: left bone b corresponds to right bone b. Recorded explicitly in
`tables/bone_correspondence.csv`.

## 3. What is NOT done

**Fixed reference lengths are not compared.** The reference 3D reconstruction
already fixes each hand's bone lengths, so any distance between them is a
constant with respect to the candidate focal. Its derivative is zero. Measured
range over a +-50 % sweep: exactly 0.

That is the single most important design decision in this run, and it is why
the obvious implementation of the user's idea was rejected before it was built.

## 4. What IS done

`CANDIDATE_CONDITIONED_SHARED_BONE_FIT`. For every candidate focal f and each
side separately:

* the bone DIRECTIONS come from the other-camera-only reference reconstruction
  (per frame, articulation only)
* the bone LENGTHS are unknown, and one shared vector is estimated across the
  whole sequence
* the per-frame pose R_t, T_t is free
* the objective is reprojection error against that side's own 2D

Then

```
L_bilateral(f) = mean_b | l_L(f)_b - l_R(f)_b |
```

## 5. Scale and symmetry

Both vectors are normalised to sum to 1, so **absolute hand size is never used
as a cue**. Factories and workers differ in hand size, and a metric hand-size
prior would be both wrong and an unearned constraint.

Left and right are **not** forced to be equal. Real people are not perfectly
symmetric; the synthetic sweep deliberately includes 0 %, 1 %, 2 % and 5 %
asymmetry. A side-scale nuisance r_LR would be redundant once both vectors are
sum-normalised, so it is not introduced.

## 6. Why the fit is tractable

With directions fixed, the hand is linear in the bone-length vector:

```
X_j = sum_b M[j,b] * l_b * dir_b
```

so with the pose held, the collinearity equations are linear in l. The fit
alternates a PnP step for the pose with a linear least-squares step for l. The
candidate focal enters those equations explicitly, which is exactly why the
fitted l depends on it.

Initialisation is uniform. The reference bone lengths are never used as a
starting point or a prior centre — doing so would smuggle the answer in.
