# Open issues — CAM-EXP-009.4

## 1. The shared-anatomy term was inactive — my parameterisation fault

`p_L = softmax(log p0 + a + delta_L)` depends only on the **sum** `a + delta_L`,
so with `lambda_generic = lambda_side = 1.0` the ridge penalty merely splits one
offset between two names instead of tying the hands together through `p_seq`.

Evidence: M1 and M3 curves coincide to a median max difference of **5.25e-06**
(typical score scale 5.3e-04); the ablation gain is **exactly 0.000 pp**;
‖a‖ = 0.055 vs ‖delta_L‖ = 0.061.

So the headline M3-vs-M0 number is really *"scene + generic-prior,
independent-side anatomy"* vs *"scene"*. The experiment did not test sequence
sharing at all.

**Not fixed here.** TRAIN focal errors had already been seen when this surfaced,
so re-tuning the weights would be tuning on results. A future run must set
`lambda_side >> lambda_generic` (or impose a hard tie) **and** include an
M1-vs-M3 comparison *inside the synthetic gate*, which is the check this run
lacked — the gate ran M3 only.

## 2. The real-data wrong-bone control failed

Permuting right-hand bones within each finger **improved** the hand-only estimate
(37.15 % → 31.66 %) and halved the boundary rate (0.97 → 0.37). Synthetic showed
the opposite (+33.1 pp degradation).

The hand term on real data is therefore not demonstrably using correct
anatomical bone correspondence. This directly undercuts attributing the +0.65 pp
focal gain to hand geometry, and it is the strongest reason to treat that gain
as shrinkage rather than signal.

Why the gate missed it: the synthetic generator produces bones from the same
topology the solver assumes, with clean directions. Real WiLoR articulation is
noisy enough that bone identity apparently stops mattering. A future gate should
include a *real-data* control before the full run, not only a synthetic one.

## 3. The dataset cannot answer the question

Reference focal CV **0.53 %**, relative span **1.87 %**. `RIG_MEDIAN_TRAIN_ONLY`
— a constant — scores **0.869 %**, ~10x better than any real method, within 5 %
on 99.2 % of units. And **90.2 %** of M3's corrections move toward that constant.

No focal-refinement method can be distinguished from a constant on this data. A
varied-intrinsics dataset is now the **blocking requirement** for further work in
this direction, not an optional follow-up.

## 4. The hand term has no interior minimum on real data

`H_ONLY` boundary rate is **0.97**: in 97 % of units the hand score slides to the
edge of the candidate range. It contributes only as a small perturbation to an
estimate the scene term has already anchored. Reported, not hidden — and it is
consistent with issues 2 and 3.

## 5. Focal is not the dominant absolute-3D error source

`REF_FOCAL_ORACLE` leaves **78.3 mm** of wrist/root error against M0's 109.8 mm.
So ~32 mm of ~110 mm is attributable to the focal and ~78 mm is not — it comes
from the monocular depth/scale in WiLoR's `cam_t`.

This reframes the programme: further focal work has a hard ceiling of about
32 mm on this task, while the untouched depth-scale term is more than twice as
large.

## 6. Not addressed

- Whether a correctly-tied shared anatomy (issue 1) would behave differently.
- Whether the hand term helps on a dataset with real focal diversity.
- The monocular depth/scale error itself, which issue 5 identifies as the larger
  target.
- N = 64 hand frames (skipped on measured runtime, pre-result).

## 7. What survived and should not be re-litigated

- The synthetic gate passed cleanly (clean 0.272 %, moderate 3.42 %, wrong-bone
  +33.1 pp) — the solver implementation is sound.
- The generic MANO prior helped in both settings (+5.6 pp synthetic, +6.5 %
  relative real) and is the one component worth keeping.
- The downstream implementation is validated by the root-aligned control moving
  **exactly 0.000 mm**.
- The leakage barrier held: focal closed through Phases A-C, tuned on TRAIN
  cameras only, predictions frozen and verified unchanged after the focal was
  opened.
