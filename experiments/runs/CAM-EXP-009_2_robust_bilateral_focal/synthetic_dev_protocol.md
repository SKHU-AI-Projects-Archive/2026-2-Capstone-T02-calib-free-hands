# DEV protocol — method selection

## Purpose, and what DEV is not

The DEV set exists for exactly one purpose: to choose one method out of the
nine frozen candidates. **Its numbers are never reported as evidence that the
cue works.** That is what the disjoint TEST set is for. Figure F1 carries this
warning in its own title.

## Design

Seven cells, 8 synthetic subjects each (56 trials), spanning the stress axes
CAM-EXP-009.1 showed the cue is sensitive to:

- `DEV_CLEAN`
- `DEV_ASYM_DENSE_02`
- `DEV_ASYM_SPARSE_05`
- `DEV_ASYM_FINGER_03`
- `DEV_NOISE_1PX`
- `DEV_ARTIC_1DEG`
- `DEV_COMBINED_MODERATE`

## The selection rule, fixed in advance

Selection score = the median over the seven cells of each cell's median **EVAL**
focal error. Ties break by the frozen `METHODS` order, so the rule is fully
deterministic and no second look at the numbers is involved.

EVAL (odd-numbered frames) is used rather than FIT so the choice is not made on
the very frames the shape was fitted on.

## Seeds

DEV seeds come from the namespace `CAM0092_DEV` and TEST seeds from
`CAM0092_TEST`, both via SHA-256 (`stable_seed`). Python's built-in `hash()` is
salted per process and is never used for seeding anywhere in this programme.
Disjointness is asserted at freeze time and re-checked by `self_audit.py`.

## Trial counts were fixed from runtime, before any result

One trial - 12 frames, both sides, FIT and EVAL profiles, a 161-point candidate
grid, warm-started - was measured at **10.9 s**. DEV was set at 56 trials
(~10 min) and TEST at 400 trials (~1.21 h) on that measurement alone. Counts
were not revisited after seeing performance.
