# Research question

## What this is

A **post-result provenance correction and participant-aware reanalysis** of
CAM-EXP-009.3. It is explicitly **not** a preregistered confirmatory
experiment: CAM-EXP-009.3's results were already known when this analysis was
designed, and every manifest records `created_after_cam0093_results = true`.

No new success threshold was invented after seeing those results. No PASS/FAIL
gate is declared. The output is a tag set and a claim-survival table.

## What it asks

CAM-EXP-009.3 could not establish participant identity from the local files, so
it compared WITHIN_SEQUENCE against CROSS_SEQUENCE. With the official naming
convention verified, three quantities become separable that CAM-EXP-009.3 could
not separate:

1. **`D_WITHIN_SESSION`** — same participant AND same session
2. **`D_SAME_PARTICIPANT_CROSS_SESSION`** — same participant, different session
3. **`D_CROSS_PARTICIPANT`** — different participants

The question is which of CAM-EXP-009.3's claims survive that correction, and
specifically:

> Is a participant's left/right hand structure closer to itself than to another
> participant's — and does that relationship survive into a different recording
> session?

## Why the second half matters more than the first

A within-session pair shares far more than a person: the same session, the same
calibration solution, the same capture conditions and the same reconstruction
context. So `D_WITHIN_SESSION < D_CROSS_PARTICIPANT` on its own cannot
establish a person-specific anatomical signature.

Only the cross-session comparison can separate those, and in this local subset
exactly one participant (`p41`) has two sessions.

## What is not done

No focal is estimated or read. No reference 3D is recomputed — CAM-EXP-009.3's
14,015 reconstructions are reused byte-identically. No new distance metric is
introduced. No per-bone feature selection is performed.
