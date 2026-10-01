# Research question

## The question

CAM-EXP-009.4 supplied the **reference focal as an oracle** and still measured
about **78 mm** of wrist/root error, ~58 mm of root-depth error and ~68 mm of
absolute MPJPE. A perfect camera focal did not fix absolute 3D.

This run decomposes that residual:

> After removing focal error, how much of the remaining absolute hand-position
> error is explained by (1) a sequence-shared multiplicative depth scale,
> (2) a sequence-shared additive depth bias, (3) a sequence-shared 3D
> translation bias, (4) frame-varying translation/depth error, and (5)
> root-relative pose error?

and asks the prior question underneath it:

> Can any dominant sequence-level metric scale be identified from the
> deployment-allowed information at all, or is it fundamentally ambiguous
> without a metric anchor?

## What this run is, and is not

It is an **error decomposition**, an **oracle correctability diagnostic** and a
**metric-scale identifiability audit**.

It is **not** a new depth estimator. Every correction measured here is fitted
using the reference 3D, which a deployment does not have. They are labelled
`ORACLE DIAGNOSTIC` throughout, and none of them is proposed as a method - not
even if it turns out to remove most of the error.

The reference focal is likewise an oracle diagnostic *input* here. Using it
removes focal error so the remaining translation error can be studied on its
own. It is not a deployable input, and CAM-EXP-009.4 already showed the real
scene-estimated focal is far from it.

## Deployment conditions, unchanged

Static monocular RGB, one worker per video, that worker's two hands, one
continuous sequence, intrinsics fixed within the sequence, whole video available
offline, no real-time requirement, and the goal is a metric absolute 3D hand
trajectory.

Explicitly still not available: EXIF, known table or object dimensions, an
assumed `typical human hand size`, a participant template from another video,
multi-camera input at deployment, or an external metric marker.

**Using the reference 3D for evaluation does not add it to the deployment
assumptions.**
