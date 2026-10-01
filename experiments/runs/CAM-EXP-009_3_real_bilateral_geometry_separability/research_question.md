# Research question

## What this run asks

> In the GigaHands reference geometry we can actually obtain, is a sequence's
> LEFT hand bone structure closer to its OWN RIGHT hand than to a RIGHT hand
> from a different sequence?

And, one step further:

> Is that difference large enough to matter, compared with how much the
> reference reconstruction of a *single* hand wobbles when you measure it
> twice?

## What this run is not

This is **not a focal estimation experiment**. No focal is estimated, no
candidate focal grid is built, no calibrator is run, and the dataset reference
focal is never loaded. See `focal_independence.md`.

## Why now

CAM-EXP-009.1 showed that in clean, symmetric synthetic geometry the bilateral
cue can identify a focal almost exactly, so focal-sensitive information does
exist in corresponding left/right bone structure.

CAM-EXP-009.2 showed that once bone-level mismatch, 2D noise and articulation
error are added, the formulation becomes unstable, and at `COMBINED_MODERATE`
a right hand from a different synthetic subject worked as well as the matched
one.

But those synthetic subjects all came from one generator with a shared
bone-length distribution, so a "different subject" was only a different draw
from the same family. That leaves the question open on real data, and it is
the question this run measures directly - before any further modelling effort
is spent on a subject-specific bilateral prior.

## The decision this feeds

If a sequence's own left/right pairing is clearly and reliably closer than a
cross-sequence pairing, there is a basis for CAM-EXP-009.4: a joint
shared-anatomy model that fits `p_shared + delta_L + delta_R` instead of
fitting each side independently and comparing afterwards.

If it is not - or if the separation is no bigger than the reconstruction's own
repeatability - then a strong subject-specific bilateral prior is not
justified by this evidence, whatever the synthetic results suggested.
