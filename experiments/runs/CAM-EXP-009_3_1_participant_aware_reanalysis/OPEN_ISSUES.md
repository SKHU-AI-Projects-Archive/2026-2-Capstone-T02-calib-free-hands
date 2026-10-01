# Open issues — CAM-EXP-009.3.1

## 1. The decisive result rests on one participant

`CROSS_SESSION_STABILITY_NOT_SUPPORTED_FOR_P41` is the finding that drives the
recommendation against a subject-specific prior, and it comes from **a single
participant's two sessions**. `p41-boxing-0021` vs `p41-plant-0004` is the only
same-participant cross-session comparison the local subset permits.

It is a strong signal within itself — all four camera-matched comparisons
(L-L, R-R, L-R, R-L) exceed the cross-participant level, and left-vs-left is the
worst of them — but n = 1. A second multi-session participant could change the
picture entirely.

**This is the single highest-value data acquisition for the programme**: more
participants with multiple sessions each.

## 2. Session and subject are still confounded for the within-session result

`D_within_session` (0.0289) vs `D_cross_participant` (0.0494) is a real
separation, but a within-session pair shares the session, the calibration
solution, the capture conditions and the reconstruction context in addition to
the person. Nothing in this reanalysis isolates the person's contribution.

The design needed is the same participant in multiple sessions **and** multiple
participants in one session. The present data has the first only, once.

## 3. Verifying identity did not fix power

`OFFICIAL_PARTICIPANT_MAPPING_VERIFIED` resolves the provenance question that
blocked CAM-EXP-009.3. It does **not** change that there are 4 participants
against a pre-set bar of 8. `PARTICIPANT_REANALYSIS_UNDERPOWERED` stands, and no
population-level claim is made.

## 4. The participant permutation weakened relative to v1

| test | p |
| --- | ---: |
| CAM-EXP-009.3, 5! = 120 sequence relabelings | 0.0167 |
| CAM-EXP-009.3.1, 4! = 24 participant blocks | **0.0833** |

This is not a discrepancy to reconcile — it is the correction working. The v1
test let `p41`'s two sessions permute independently, as if they were two
different people. Binding them into one block removes that, and the apparent
effect weakens. Neither test is load-bearing: the participant version's floor is
1/24 = 0.0417 with 4 units.

## 5. `p36` is reversed and was not investigated

`p36-tea-0010` has `D_within_session` = 0.0521, above the cross-participant
level of 0.0494 — its own two hands are further apart than a typical pair of
different people's. CAM-EXP-009.3 saw the same reversal.

No cause was investigated here. Doing so after seeing the result, and then
excluding the sequence, would be post-hoc filtering. It is flagged for a future
pre-registered look.

## 6. The participant template for p41 is an average of disagreeing sessions

Per the protocol, `p41`'s participant template is the equal-weight median of its
two sequence templates. §6 of the report shows those two sessions disagree more
than different participants do, so that template represents neither session
well. The participant-level identification diagnostic (50 %, 2/4) is therefore
weakest for exactly the participant that carries the most information.

## 7. Generic vs subject-specific signal must not be conflated later

The bone-permutation control degrades the distance by +857 %. That is a
**generic bone-identity** result: the metric knows which bone is which. It is
not evidence of a subject-specific signal, and the two are reported separately.

A future design claiming a subject-specific effect needs a control that breaks
subject pairing while preserving bone identity — which, with identity now
verified, is finally constructible, but needs more participants to be
informative.

## 8. This is a post-result reanalysis, and is labelled as such

CAM-EXP-009.3's results were known when this was designed. Every manifest
carries `created_after_cam0093_results = true` and
`is_preregistered_confirmatory_experiment = false`. No new PASS/FAIL threshold
was invented; the output is a tag set and a claim-survival table rather than a
gate.

## 9. Not addressed

- Any focal estimate. None computed, none read.
- Whether OPTION B (sequence-specific shared anatomy) or OPTION C (generic
  anatomical prior) actually improves focal recovery. That is the next
  substantive experiment and needs its own frozen spec.
- Whether a better reference reconstruction would change the cross-session
  result. The mismatch is equally consistent with session-dependent
  reconstruction error as with anything about hands.
