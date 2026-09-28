# Oracle diagnostics

**Every correction on this page is fitted using the reference 3D. None of them
is a deployable method.** A deployment has no reference 3D; if it did, it would
not need any of this. They exist to attribute the residual error to components.

| | name | what it assumes it knows |
| --- | --- | --- |
| **O0** | `ORACLE_FOCAL_ONLY` | the true focal; `cam_t` untouched |
| **O1** | `SEQUENCE_MULTIPLICATIVE_Z_SCALE` | one `alpha` per video, fitted on FIT frames |
| **O2** | `SEQUENCE_AFFINE_Z` | `alpha` and `beta` per video |
| **O3** | `SEQUENCE_CONSTANT_3D_BIAS` | a constant `(bx, by, bz)` per video |
| **O4** | `SEQUENCE_SCALE_PLUS_BIAS` | x/y bias + affine z per video |
| **O5** | `PER_FRAME_DEPTH_ORACLE` | the true depth of **every frame** |
| **O6** | `PER_FRAME_ROOT_TRANSLATION_ORACLE` | the true root translation of every frame |
| - | root-aligned pose floor | the translation is irrelevant; what remains is articulation error |

## How to read the ladder

The ladder is ordered by how much knowledge each rung assumes, and the gaps
between rungs are the attribution:

- **O0 -> O1** - how much is a single constant per video? This is the rung that
  matters for the project, because a per-video constant is the kind of thing a
  one-time calibration step could in principle supply.
- **O1 -> O2** - is a multiplicative scale enough, or is there also a constant
  depth offset?
- **O0 -> O3** - how much is a rigid 3D offset rather than a depth scaling?
- **O2/O4 -> O5** - how much depth error is frame-to-frame rather than
  sequence-wide? A large gap here means no single per-video number can fix it.
- **O5 -> O6** - how much x/y translation error remains once depth is perfect?
- **O6 -> floor** - O6 sets the translation exactly right, so its absolute
  MPJPE should equal the root-aligned MPJPE. A gap above 2 mm means the
  implementation is wrong, not that the model is interesting.

## O4 identifiability caution

`T_corrected = alpha * T_pred + b` is over-parameterised: a scale and a bias on
the same axis trade off against each other. O4 is therefore restricted to a
constant x/y bias plus an affine z, which is identifiable. This restriction was
fixed before results.

## Optional and deliberately not run as a method

`ORACLE_ABSOLUTE_HAND_SCALE` - supplying the true absolute hand size to see how
much of the depth error one number would fix - is an **error-budget diagnostic**
only. **No assumed typical human hand size is used anywhere**, as a deployment
input or otherwise.

## The rule this page exists to enforce

If a sequence-wide scale correction turns out to remove most of the error, that
is **not** a result saying "use a per-video alpha". The reference-derived alpha
is unobtainable at deployment, and
`metric_scale_identifiability.md` shows it is not recoverable from the allowed
cues by geometry. The correct consequence is a question about what metric anchor
the task can afford, not a proposed estimator.
