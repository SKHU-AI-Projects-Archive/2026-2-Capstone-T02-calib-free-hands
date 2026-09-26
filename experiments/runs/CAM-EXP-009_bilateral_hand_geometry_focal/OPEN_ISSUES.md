# CAM-EXP-009 — open issues

## Raised here

| id | issue |
| --- | --- |
| `BILATERAL_FOCAL_SIGNAL_NOT_IDENTIFIABLE_SYNTHETICALLY` | the candidate-conditioned bilateral objective is focal-dependent but minimises at the search boundary in 90-100 % of synthetic trials, at every distance, asymmetry, noise level and pose diversity tested. |
| `FREE_BONE_LENGTHS_ABSORB_FOCAL_ERROR` | the mechanism. 20 free shape parameters per side let the fit reshape the hand to explain a wrong focal, and a larger focal always fits slightly better. The signal the formulation was meant to expose is consumed by its own free parameters. |
| `HELD_OUT_SPLIT_DOES_NOT_CATCH_SHARED_BIAS` | the FIT/EVAL guard was included to catch over-fitting and did not fire: a hand fitted at the wrong focal is consistently wrong and generalises fine to held-out frames. Held-out data exposes variance, not shared bias. |
| `G4_UNINFORMATIVE_WHEN_G2_FAILS` | the wrong-correspondence control could not degrade a result that was already at the floor. Its failure is a consequence of g2's, not independent evidence. |
| `LOW_DIMENSIONAL_VARIANT_NOT_RUN` | `FINGER_CHAIN_PROPORTIONS` was registered as a secondary formulation and was not run, because running a variant after seeing the primary fail would be a search rather than a test. |
| `SYNTHETIC_BONE_DISTRIBUTION_INVENTED` | the generator uses a plausible but invented bone-length distribution. The conclusion rests on a structural degeneracy rather than those particular numbers, but this is stated rather than assumed. |
| `MANIFESTS_TRANSCRIBED_AFTER_RUN` | the gate criteria were fixed in `src/run_synthetic.py` before execution; the manifest JSONs record them and were written afterwards, so they are a faithful transcription rather than an independent pre-registration. |

## Carried forward

| id | issue |
| --- | --- |
| `SINGLE_FOCAL_RIG_CONFOUND` | would have applied to the real phase: reference focal CV ~1.7-1.9 %, where a constant beats every image-based estimate. |
| `FINAL_CONFIRMATORY_HOLDOUT_UNOPENED` | InterHand2.6M / HanCo remain reserved and unopened. |

## Next

| item | status |
| --- | --- |
| this bilateral formulation | **closed.** Not identifiable; do not carry it to real data. |
| a tightly-constrained bilateral variant | open and untested. The failure is caused by free shape parameters, so a formulation that restricts bone proportions to a low-dimensional anatomical subspace, or ties them across subjects, would not share this degeneracy. |
| varied-focal dataset | still the binding requirement for the whole line, as in CAM-EXP-007 and CAM-EXP-008. |
