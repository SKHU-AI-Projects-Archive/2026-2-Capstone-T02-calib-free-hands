# CAM-EXP-009.3 — Real Reference-Geometry Bilateral Separability Diagnostic

```
VERDICT  REAL_REFERENCE_BILATERAL_SIGNAL_WEAK
       + SUBJECT_SEPARABILITY_UNDERPOWERED
       + SAME_SUBJECT_PROVENANCE_UNRESOLVED

FOCAL ESTIMATED: NONE.  This run reads no focal and runs no calibrator.
```

**One sentence.** In the GigaHands reference geometry a sequence's own LEFT/RIGHT
bone-proportion pair is, on average, closer than a pair drawn across sequences
(0.0283 vs 0.0458, ratio 0.62, separation 1.6× the reconstruction's own
repeatability, exact permutation p = 0.017) — but the effect is not reliable per
unit (only **53.7 %** of units are closer to their own partner than to the
nearest outsider, against a 75 % bar), one of the five sequences is outright
reversed, and the whole advantage is confounded with sequence/session context
rather than shown to be a property of a person.

## 1. Plain-language summary

CAM-EXP-009.1 showed that on clean synthetic hands, the matching bone structure
of the left and right hand can pin down the camera's focal length almost
exactly. CAM-EXP-009.2 showed that once realistic-looking error is added the
method becomes unreliable, and a different synthetic person's right hand worked
just as well as the correct one.

This experiment estimates **no focal at all**. It asks a simpler question first:
in the real GigaHands data we actually have, is one recording's left hand more
similar to its own right hand than to a right hand from a different recording?

The answer is: **somewhat, but not dependably.**

On average, yes — a recording's own pair is meaningfully closer, and that gap is
bigger than the amount the measurement wobbles when you simply measure the same
hand twice. But if you ask the sharper question — "for this particular left
hand, is its own partner the closest one?" — you only get the right answer about
54 % of the time (chance is 25 %). And in one of the five recordings the own
pair is actually *farther* than the average outsider.

There is a more awkward finding underneath. The two recordings whose names share
a `p41` prefix — the only pair that might be the same person — turned out to be
the **furthest apart of anything we measured**, further than ordinary
cross-recording pairs. So this geometry does not carry a stable signature across
sessions. Whatever makes a recording's two hands look alike seems to be
substantially about the recording, not about the person.

That matters for what comes next: it weakens the case for building a strong
person-specific left/right prior.

## 2. What was frozen, and when

All of Phase A was written before any distance was computed
(`created_before_distance_results = true`):

| artefact | contents |
| --- | --- |
| `cam_exp_0093_geometry_spec_v1.json` | topology, representation, primary distance, eligibility, thresholds |
| `cam_exp_0093_eligible_units_v1.csv.gz` | 315 side-units |
| `cam_exp_0093_repeatability_split_v1.csv.gz` | the deterministic A/B frame split |
| `cam_exp_0093_identity_mapping_v1.csv` | identity, with its confidence grade |
| `cam_exp_0093_controls_spec_v1.json` | the bone permutation, fixed in advance |
| `cam_exp_0093_permutation_spec_v1.json` | the exact 120-relabeling null |

SHA-256 for each in `results/summary/manifest_hashes.json`. The frame cap (48
per unit, evenly spaced by index) was fixed from a measured 0.122 s per
reconstruction, before any result.

## 3. Q1–Q2 — how much data, and do we know who is who?

| question | answer |
| --- | --- |
| **Q1 unique participants** | **4 candidate groups** (`p36`, `p41`, `p44`, `p52`) — below the pre-set bar of 8 |
| **Q2 identity provenance** | **UNRESOLVED** |

Sequences 5 · physical cameras 40 · sequence-camera units 177 · QC-eligible
rows 48,831 (22,252 left / 26,579 right) · side-units with ≥8 frames 315 ·
units with both sides sufficient 145.

**No shipped GigaHands metadata declares a participant field.** Every `.json`,
`.txt` and `.csv` in the dataset tree outside the keypoint and bbox payloads was
scanned, as were all manifests from CAM-EXP-001 onward. The `p<NN>` prefix is an
undocumented naming convention, and the spec forbids inferring identity from a
filename.

A MANO `shapes` vector *is* shipped per sequence and would be suggestive
evidence — it is **deliberately not used**, because defining subject identity
from fitted hand shape and then measuring whether same-subject hands have
similar shape would be circular.

**Consequence, decided before any distance existed:** the contrast is reported
as **WITHIN_SEQUENCE vs CROSS_SEQUENCE**, not same-subject vs cross-subject, and
the one same-prefix pair is held out of the primary cross set. Details in
`participant_identity_provenance.md`.

## 4. Q3–Q7 — the three distances

Primary metric: `median_b |log(p_L,b + 1e-8) − log(p_R,b + 1e-8)|` over the 20
connected bones, absolute hand size removed. Cluster bootstrap on sequence,
10,000 iterations.

| quantity | median | 95 % CI |
| --- | ---: | --- |
| **Q3 `D_repeat`** same hand, half A vs half B | **0.0109** | [0.0076, 0.0151] |
| `D_view_repeat` same hand, two cameras | 0.0113 | — |
| **Q4 `D_within`** same sequence, LEFT vs RIGHT | **0.0283** | [0.0210, 0.0459] |
| **Q5 `D_cross`** different sequences, LEFT vs RIGHT | **0.0458** | [0.0341, 0.0638] |

- **Q6 ratio `D_within / D_cross` = 0.618** — passes the ≤ 0.80 bar.
- **Q7 `S_sep` = (0.0458 − 0.0283) / 0.0109 = 1.61** — passes the ≥ 1.0 bar. The
  separation is about 1.6× the reconstruction's own repeatability, so it is not
  purely measurement wobble.

The within and cross bootstrap intervals **overlap** ([0.021, 0.046] vs
[0.034, 0.064]). With five sequence clusters these intervals are wide and the
label is unstable; they are supplemental, as pre-registered.

`D_view_repeat` (0.0113) is essentially identical to `D_repeat` (0.0109): moving
to an entirely different camera's reconstruction costs no more than splitting
one camera's frames in half. Camera/view choice is therefore not a major source
of variability here.

## 5. Q8–Q10 — per-unit discrimination, where it fails

Averages hide the important part. For each LEFT template we asked whether its
own sequence's RIGHT is the *nearest* among all camera-matched candidates.

| | LEFT→RIGHT | RIGHT→LEFT | combined |
| --- | ---: | ---: | ---: |
| **Q8 fraction with `M_nearest` > 0** | 56.0 % | 51.4 % | **53.7 %** |
| median `M_nearest` | 0.0021 | 0.0007 | — |
| median `M_median` | 0.0111 | 0.0188 | — |
| **Q9 top-1 accuracy** | 56.0 % | 51.4 % | **53.7 %** |
| **Q10 chance** | 25.0 % | 25.0 % | **25.0 %** |
| n | 141 | 144 | 285 |

**This is the criterion that fails.** The pre-registered bar was 75 % of units
with `M_nearest` > 0; the result is 53.7 %. Against the *typical* outsider the
margin is comfortable (`M_median` positive, and top-1 at 53.7 % is well above
25 % chance), but against the *nearest* outsider it is close to a coin flip. The
median nearest-margin is 0.0021 — about a fifth of `D_repeat`, i.e. well inside
the measurement's own noise.

Top-1 accuracy and the `M_nearest` fraction are identical by construction: with
camera-matched candidates, "own partner is nearest" and "top-1 correct" are the
same event. Both directions were run, as specified; neither is materially better.

## 6. Q13 — the cross-session result, and why it is the most important number

The only sequence pair sharing a name prefix — `p41-boxing-0021` and
`p41-plant-0004` — is the one pair that might be the same person.

| comparison | median distance |
| --- | ---: |
| within-sequence (own pair) | 0.0283 |
| cross-sequence, primary set | 0.0458 |
| **same-prefix pair, same side (L↔L, R↔R)** | **0.0852** |
| **same-prefix pair, opposite side (L↔R)** | **0.0770** |

**The same-prefix pair is the furthest apart of anything measured** — further
than ordinary cross-sequence pairs, and even the same *side* of the two
recordings (left vs left) does not match.

Two readings, and this experiment cannot separate them:

1. the two recordings are not the same person, and the prefix means something
   else; or
2. they are the same person, and the reference geometry simply does not carry a
   stable signature across sessions.

Either way it **undermines the assumption that this geometry encodes a
person-level template**, and it is direct evidence that the name prefix should
not have been trusted as identity.

## 7. Q11–Q12 — controls

**Q11 bone-mapping permutation.** Right-hand bones permuted within each finger,
frozen before results:

| | median `D_within` | top-1 |
| --- | ---: | ---: |
| C0 correct mapping | 0.0283 | 56.0 % |
| C1 permuted mapping | **0.2706** | 41.8 % |

Degradation **+857 %**; top-1 drops 14.2 pp. The criterion (≥ 20 % degradation
**or** ≥ 20 pp top-1 drop) passes decisively on the first arm. The distance
plainly uses real anatomical bone correspondence — it is not indifferent to
which bone is compared with which.

Note what this does *not* show: a generic bone-identity signal is not the same
thing as a subject-specific one. C1 confirms the former; §5 and §6 are what bear
on the latter.

**Q12 sequence-label permutation.** Exact over all 5! = 120 relabelings, identity
permutation included in the null.

| | value |
| --- | ---: |
| observed `median(CROSS) − median(WITHIN)` | +0.01741 |
| null median | −0.00642 |
| null maximum | +0.01753 |
| **p (one-sided)** | **0.0167** (2/120) |

Direction is consistent, and p = 0.017. But **one relabeling scored higher than
the truth**, the permutation unit is the sequence, and there are only five, so
the smallest attainable p-value is 0.0083. This test is weak by construction —
that was written down before it was run — and with 4 candidate participants no
conclusion here rests on it.

## 8. Sequence-level results — one reversal

Camera views are never counted as independent participants; distances are
aggregated per sequence.

| sequence | `D_repeat` | `D_within` | `D_cross` | within < cross? |
| --- | ---: | ---: | ---: | :--- |
| p36-tea-0010 | 0.0126 | 0.0521 | 0.0494 | **NO — reversed** |
| p41-boxing-0021 | 0.0091 | 0.0182 | 0.0760 | yes, strongly |
| p41-plant-0004 | 0.0155 | 0.0231 | 0.0324 | yes |
| p44-dog-0004 | 0.0162 | 0.0268 | 0.0295 | marginal |
| p52-instrument-0034 | 0.0066 | 0.0309 | 0.0466 | yes |

**4 of 5 sequences** show the expected ordering; `p36-tea-0010` is reversed, and
`p44-dog-0004` is nearly a tie. The pooled effect is carried substantially by
`p41-boxing-0021`.

## 9. Q14 — per-bone structure (descriptive only)

| pattern | bones |
| --- | --- |
| most reliable (`R_b` = V_cross / V_repeat high, low repeat noise) | `pinky_dip` (8.1), `middle_mcp` (5.8), `pinky_tip` (6.5), `middle_pip` (5.6) |
| least reliable | `pinky_pip` (1.7), `pinky_mcp` (2.6), `thumb_pip` (3.2), `thumb_mcp` (3.2) |
| within ≥ cross (no separation at all) | `ring_pip` (0.0447 vs 0.0436), `pinky_pip` (0.0462 vs 0.0281) |

Fingertip bones carry the largest cross-sequence spread but also the largest
repeat noise; the MCP bones of the middle and ring fingers are the most stable.

**This is descriptive and is not acted on.** The primary distance keeps all 20
frozen bones. Selecting the best-looking subset and recomputing would be feature
selection after the fact; it is offered only as a design input to a future,
separately pre-registered experiment.

## 10. The decision criteria

| # | criterion | threshold | result | pass |
| --- | --- | --- | ---: | :--- |
| 1 | `D_within` ≤ 0.80 × `D_cross` | 0.80 | 0.618 | **YES** |
| 2 | ≥ 75 % of units with `M_nearest` > 0 | 75 % | 53.7 % | **NO** |
| 3 | `S_sep` ≥ 1.0 | 1.0 | 1.61 | **YES** |
| 4 | bone permutation degrades ≥ 20 % or top-1 −20 pp | — | +857 % | **YES** |
| 5 | permutation direction consistent | — | p = 0.017 | **YES** |

Four of five pass. Criterion 2 fails, so the result is not
`..._SUPPORTED`; the ordering `D_repeat < D_within < D_cross` does hold and the
separation does exceed repeatability, so it is not
`REFERENCE_VARIABILITY_DOMINATES` either.

**Verdict: `REAL_REFERENCE_BILATERAL_SIGNAL_WEAK`**, tagged
`SUBJECT_SEPARABILITY_UNDERPOWERED` (4 candidate participants < 8) and
`SAME_SUBJECT_PROVENANCE_UNRESOLVED`.

## 11. What this means — and what it does not

**It means.** In the evaluated GigaHands reference geometry, within-sequence
left/right normalised bone structure is on average more similar than
cross-sequence structure, by a margin larger than the reconstruction's own
repeatability. The distance genuinely uses anatomical bone correspondence.

**It does not mean:**

- that physical human anatomy has been characterised — the reference 3D mixes
  annotation, triangulation, calibration and association error with anatomy;
- that any focal estimate improved — no focal was estimated or read;
- that a same-**subject** effect was demonstrated — identity is unresolved, so
  the measured contrast is within- vs cross-**sequence**;
- that human left and right hands are unrelated, if one reads the weak result
  negatively;
- anything about factory deployment or about surviving image noise.

**The confound that matters most.** A within-sequence pair shares far more than
a person: the same session, the same calibration solution, the same capture
conditions and the same reconstruction context. A cross-sequence pair is
camera-matched but shares none of those. So the observed advantage is **not
separated from sequence/session-level reconstruction context**, and §6 is
positive evidence that session context is doing real work — the one pair that
might be the same person across two sessions was the furthest apart of all.

This experiment cannot distinguish a person-specific bilateral signal from a
session-specific one, and with 5 sequences and unresolved identity it was never
going to.

## 12. Q15–Q16 — is there a basis for CAM-EXP-009.4?

**Q15 participants sufficient?** No. 4 candidate groups against a pre-set bar of
8, and identity unverified.

**Q16 basis for a joint shared-anatomy model?** **Not on this evidence.**

The case for `p_shared + delta_L + delta_R` rests on a person's two hands
sharing a stable template. What was found is an average-level separation that is
unreliable per unit (53.7 %), reversed in one of five sequences, carried
substantially by a single sequence, confounded with session context, and
contradicted across sessions by the only pair that might be the same person.

Per the pre-registered branch for a weak signal, **do not jump to a strong
subject-specific bilateral prior.** The better-supported options are:

1. **Resolve identity first.** Obtain the official GigaHands participant
   metadata. Without it the central question cannot be asked properly, and this
   is the cheapest blocking fix.
2. **More participants.** 4 candidate groups cannot support a participant-level
   claim at any effect size.
3. **Separate session from subject.** The design needed is the same participant
   across multiple sessions *and* multiple participants within one session. The
   present data has neither.
4. **A generic anatomical prior rather than a subject-specific bilateral one.**
   The C1 control shows a strong generic bone-identity signal (+857 %) even
   where the subject-specific signal is weak. A generic hand-proportion prior is
   the better-supported modelling direction from these results.
5. **Improve the reference geometry** if the bilateral route is pursued anyway —
   `D_repeat` = 0.011 against a nearest-margin of 0.002 means the measurement
   noise is several times the quantity of interest.

## 13. Limitations

1. 5 sequences, 4 candidate participant groups, identity unverified. Everything
   at participant level is underpowered by design, and was flagged as such
   before the run.
2. The within/cross contrast is confounded with session context (§11).
3. The permutation test has 120 relabelings and 5 sequence units; its floor is
   p = 0.0083 and one relabeling beat the observed statistic.
4. Bootstrap intervals use 5 clusters and overlap; they are supplemental.
5. `D_within` uses left and right templates that need not come from the same
   frames — anatomy is treated as a sequence-level property. A frame-paired
   variant was not tested.
6. One reconstruction pipeline, one QC definition, one distance metric as
   primary. Four secondaries were frozen but the primary conclusion is not
   re-chosen from them.
7. The cross-session diagnostic rests on a single sequence pair.

## 14. Reproduce

```
python src/audit_sources.py       # PHASE A provenance + population + focal audit
python src/freeze_spec.py         # PHASE A freeze (before any distance)
python src/extract_bone_vectors.py  # ~20 min, 14,015 reconstructions
python src/run_all.py             # templates -> distances -> controls -> verdict
```
