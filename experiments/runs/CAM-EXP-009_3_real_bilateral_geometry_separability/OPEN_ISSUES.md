# Open issues — CAM-EXP-009.3

## 1. Participant identity is unresolved, and it blocks the stated question

The experiment was specified as same-subject vs cross-subject. No shipped
GigaHands metadata declares a participant field, so the contrast actually
measured is **within-sequence vs cross-sequence**.

This is the single cheapest blocking fix in the programme: obtain the official
GigaHands participant metadata. Until then, no participant-level claim about
bilateral structure can be made from this dataset, regardless of effect size.

The MANO `shapes` vector shipped per sequence is **not** an acceptable
substitute — defining identity from fitted hand shape and then measuring hand
shape similarity is circular.

## 2. Within vs cross is confounded with session/reconstruction context

A within-sequence pair shares the session, the calibration solution, the capture
conditions and the reconstruction context. A cross-sequence pair is
camera-matched but shares none of them.

So the measured advantage (ratio 0.62, `S_sep` 1.61) is **not attributable to a
person**. §6 of the report is positive evidence that session context matters:
the only sequence pair that might be the same person across two sessions
(`p41-boxing-0021` / `p41-plant-0004`) came out at 0.0852 / 0.0770 — the
**furthest apart of any comparison in the run**, worse than ordinary
cross-sequence pairs.

Resolving this needs a design the present data cannot provide: the same
participant in multiple sessions **and** multiple participants in one session.

## 3. The per-unit criterion failed and the pooled effect is carried by one sequence

Only 53.7 % of units are closer to their own partner than to the nearest
outsider (bar: 75 %). The median nearest-margin is 0.0021 — roughly a fifth of
`D_repeat`, i.e. inside the measurement's own noise.

At sequence level, 4 of 5 show the expected ordering, `p36-tea-0010` is
**reversed** (within 0.0521 > cross 0.0494), and `p44-dog-0004` is nearly a tie.
The pooled separation leans heavily on `p41-boxing-0021` (0.0182 vs 0.0760).

## 4. The permutation test is weak by construction

5 sequences → 120 relabelings → the smallest attainable p-value is 0.0083. The
observed p = 0.0167 with **one relabeling scoring higher than the truth**
(observed 0.01741, null max 0.01753). Direction is consistent, but nothing in
the verdict rests on this test, as pre-registered.

Bootstrap intervals use 5 clusters and overlap substantially
(within [0.021, 0.046], cross [0.034, 0.064]).

## 5. C1 shows a generic signal, which is not the signal under test

The bone-mapping permutation degrades the distance by +857 %. That establishes a
strong **generic bone-identity** signal: the metric knows which bone is which.

It does **not** establish a subject-specific one. These are different claims and
the report keeps them apart. A future design that wants to claim a
subject-specific effect needs a control that breaks subject pairing while
preserving bone identity — which is exactly what an identity-resolved
subject-swap control would be, and which could not be run here.

## 6. Per-bone results are descriptive and were deliberately not acted on

`pinky_dip`, `middle_mcp` and `pinky_tip` look most reliable; `pinky_pip` and
`ring_pip` show no separation at all (within ≥ cross). Selecting the
best-looking subset and recomputing the primary metric would be post-hoc feature
selection. The primary distance keeps all 20 frozen bones.

These ratios may be used as **design input** to a separately pre-registered
CAM-EXP-009.4, never as a re-analysis of this run.

## 7. Measurement noise is large relative to the quantity of interest

`D_repeat` = 0.0109 against a median nearest-margin of 0.0021. If the bilateral
route is pursued, improving the reference reconstruction is likely worth more
than improving the comparison metric.

`D_view_repeat` (0.0113) ≈ `D_repeat` (0.0109), so camera/view choice is not the
dominant noise source — the noise is in the reconstruction itself.

## 8. Not addressed by this run

- Any focal estimate. None was computed and none was read.
- Whether a frame-paired within-sequence variant (left and right from the *same*
  frames) behaves differently; templates here are built per side independently.
- Whether the weak result would strengthen with a better reference geometry, more
  participants, or resolved identity — all three are confounded in the present
  answer.
