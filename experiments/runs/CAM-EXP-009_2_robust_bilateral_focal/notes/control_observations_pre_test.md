# Control observations — recorded BEFORE the TEST run finished

Written while `run_synth_test.py` was still executing, so that the reading of
these controls cannot have been shaped by the TEST numbers.

Condition: `COMBINED_MODERATE`, 8 synthetic subjects, selected method
`M4_HUBER_005`, EVAL (odd-frame) profiles.

| control | median focal error | boundary rate | score-curve range |
| --- | ---: | ---: | ---: |
| C0_CORRECT | 7.41 % | 0.12 | 3.53e-01 |
| C1_WRONG_BONE_MAPPING | 44.44 % | 0.88 | 2.45e-01 |
| C2_SUBJECT_SWAP | 6.84 % | 0.00 | 2.43e-01 |
| C3_TEMPORAL_SHUFFLE | 6.84 % | 0.00 | 2.43e-01 |
| C4_FIXED_LENGTH_INVARIANT | 44.44 % | 1.00 | 0.00e+00 |

## 1. C3 is vacuous by construction, and this is a design fault, not a result

C2 and C3 are **bit-identical**, per subject, not merely close (subject 0:
781.3 px for both; subject 1: 960.0; subject 2: 836.8).

The reason is structural. The objective fits `p_L` from all left frames and
`p_R` from all right frames independently, and only then compares the two
proportion vectors. **It never pairs a left frame with a right frame.** There
is therefore no temporal pairing for a temporal shuffle to destroy, and
replacing the right frames with another sequence's right frames is exactly what
C2 already does.

C3 was pre-registered before this was noticed. It is reported as
`CONTROL_VACUOUS_BY_CONSTRUCTION` and carries no evidential weight in either
direction. It is **not** replaced with a different control after the fact.

## 2. C2 does not degrade — it is marginally better than the correct pairing

This is the consequential observation. `C2_SUBJECT_SWAP` gives the right hand
of a **different synthetic subject**, so the bilateral correspondence the method
is built on is entirely absent. Under the pre-registered control logic — "a cue
that identifies the focal must lose that ability when the bilateral
correspondence it relies on is destroyed" — C2 should degrade. It does not
(6.84 % vs C0's 7.41 %).

The straightforward reading is that under `COMBINED_MODERATE` the focal
preference is **not** coming from left–right correspondence. What varies with
the candidate focal is each side's own fitted profile; comparing the left
profile against *some* plausible right-hand profile appears to be enough to
produce a similar minimum, whether or not that profile belongs to the same
person.

Note also that C0 itself is at 7.41 % here, already outside the 5 % moderate
gate, so this comparison is between two conditions that both estimate the focal
poorly. It should not be read as "the swap works".

## 3. C1 and C4 behave as required

`C1_WRONG_BONE_MAPPING` degrades hard (44 %, 88 % at a grid boundary), so the
score is not indifferent to which bone is compared with which.
`C4_FIXED_LENGTH_INVARIANT` has a score-curve range of **exactly 0.00e+00**,
reconfirming CAM-EXP-009's exact result that the naive fixed-length formulation
cannot depend on the candidate focal. (C4 is an oracle diagnostic — it is handed
the true bone lengths and could never run on real data.)

## 4. Consequence for the verdict

The synthetic gate requires the controls to degrade. On this evidence C2 does
not. Whatever the TEST numbers show, the control outcome must be carried into
the verdict rather than set aside, and the real-data phase decision follows the
pre-registered rule.
