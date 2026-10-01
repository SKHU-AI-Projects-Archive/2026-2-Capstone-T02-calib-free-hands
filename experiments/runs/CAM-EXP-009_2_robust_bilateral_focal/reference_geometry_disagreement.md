# Reference left/right geometry disagreement in GigaHands

**NOT MEASURED — `REAL_FOCAL_PHASE_NOT_RUN`.**

This document was to hold the measured disagreement between the left and right
reference bone proportions in GigaHands, per subject, reconstructed with
`OTHER_CAMERA_ONLY_REFERENCE_3D`. It is the prerequisite that would tell us
whether the synthetic asymmetry levels tested in this run (1 %, 2 %, 5 %, in
four families) bracket what real data actually exhibits.

It was not measured, because the synthetic gate failed on two required criteria
and the pre-registered rule is that the GigaHands phase then does not run. See
`report.md` §6 and §8, and `real_data_protocol.md` step 1.

Two constraints that would apply if it were ever measured, recorded here so
they are not lost:

1. Any observed left/right disagreement in GigaHands would contain
   reconstruction error of unknown size, and must **not** be described as true
   human anatomical asymmetry.
2. `SINGLE_FOCAL_RIG_CONFOUND` applies to any focal number derived from this
   dataset: the reference focal has a CV of only 1.73-1.90 %, and a constant
   predictor scores 0.85-0.91 %.
