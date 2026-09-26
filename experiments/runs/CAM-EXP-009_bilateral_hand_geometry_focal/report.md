# CAM-EXP-009 — Bilateral Sequence Hand-Geometry Focal Identifiability

```
VERDICT  BILATERAL_FOCAL_SIGNAL_NOT_IDENTIFIABLE_SYNTHETICALLY

REAL GIGAHANDS PHASE: NOT RUN
  (pre-registered consequence of a failed synthetic gate)
```

**One sentence:** The naive version of the idea — comparing the two hands'
fixed reference bone lengths — is provably useless because it does not depend
on the focal at all (measured range exactly **0**); the repaired version, which
refits each side's bone proportions at every candidate focal, *does* depend on
the focal but is minimised at the edge of the search range in 90–100 % of
synthetic trials, at every distance, asymmetry and noise level tested, so it
never prefers the true focal.

## 1. Plain-language summary

The proposal was: a person's left and right hands have matching bone structure,
so the camera setting that makes the two hands look most consistent with each
other is probably the right one.

There is a trap in the obvious way of doing this. If you take the already-known
3D bone lengths of each hand and compare them, that number is fixed — changing
the camera's assumed focal length does not change it at all. We measured this,
and the value moved by exactly zero across a ±50 % sweep. So the obvious version
cannot estimate anything.

The repaired version asks, for each candidate focal: *what bone proportions
would I need to assume to explain the left-hand images, and what would I need
for the right-hand images?* Those answers **do** change with the candidate
focal, so a comparison between them could in principle be informative.

It turned out not to be. Once you let the bone lengths be unknown, the fit can
reshape the hand to absorb a wrong focal — and a larger focal always explains
the pictures slightly better. So the score just slides to the edge of the search
range instead of settling on the truth.

Because this failed on clean synthetic data where we knew the answer, the rule
fixed in advance says: do not run it on the real data. We did not.

## 2. What was tested and why it is not CAM-EXP-008 repeated

CAM-EXP-008 stabilised each hand's bone lengths **separately** per side and
used them for reprojection. It never asked whether the left and right bones
agree. CAM-EXP-009 adds exactly that link.

| | CAM-EXP-008 | CAM-EXP-009 |
| --- | --- | --- |
| left / right templates | independent, fixed from the reference | **re-estimated per candidate focal, then compared** |
| bone lengths | taken from the reference reconstruction | **unknown, fitted** |
| what is compared | hand vs its own 2D | **left's required proportions vs right's** |

## 3. The focal-dependence audit — the first thing that had to be true

| claim | measured | verdict |
| --- | ---: | --- |
| fixed reference lengths, compared directly | range over ±50 % sweep = **0.000e+00** | **focal-INVARIANT** |
| candidate-conditioned refit | range 4.97e-03, relative dynamic range **1.97** | **focal-DEPENDENT** |

This is a genuinely useful negative: it shows precisely *why* the intuitive
formulation cannot work, and it is the reason CAM-EXP-009 did not simply
implement it. `results/summary/focal_dependence_audit.json`.

## 4. The synthetic gate — and why it failed

Every condition, 20 trials each, 161-point grid, true focal at q = 1.0.

| gate | criterion | result |
| --- | --- | --- |
| g1 | bilateral objective is not focal-invariant | **PASS** |
| g2 | true focal near the minimum, noiseless (≤ 5 % error) | **FAIL** — 50.0 % |
| g3 | held-out reprojection identifies the focal | **FAIL** — 50.0 % |
| g4 | wrong bone correspondence degrades | **FAIL** — cannot, the real mapping is already at the floor |
| g5 | fixed-length control is flat | **PASS** |

g1, g2 and g4 were all required. Two failed.

### 4.1 Every sweep sits at the same ceiling

| sweep | condition | bilateral focal error | boundary rate |
| --- | --- | ---: | ---: |
| distance / diameter | 2× | 50.0 % | 0.90 |
| | 4× | 50.0 % | 0.95 |
| | 8× | 50.0 % | 0.95 |
| | 16× | 50.0 % | 0.95 |
| | 32× | 50.0 % | 1.00 |
| bilateral asymmetry | 0 % | 50.0 % | 0.95 |
| | 1 % | 50.0 % | 1.00 |
| | 2 % | 50.0 % | 0.90 |
| | 5 % | 50.0 % | 1.00 |
| 2D noise | 0 px | 50.0 % | 1.00 |
| | 0.5 px | 50.0 % | 0.95 |
| | 1 px | 50.0 % | 0.95 |
| | 2 px | 50.0 % | 0.95 |
| | 4 px | 50.0 % | 0.95 |
| pose diversity | LOW | 50.0 % | 0.80 |
| | MEDIUM | 50.0 % | 0.95 |
| | HIGH | 50.0 % | 0.90 |

A "50.0 % error" here means the minimum landed on the grid edge at q = 1.5 —
the largest focal the search allows. Not a noisy answer: a systematically
wrong one. Even at 2× hand diameters, the most perspective-rich regime tested,
and even with perfect bilateral symmetry and zero noise.

### 4.2 Why — the mechanism

The bilateral score simply inherits the shape of the underlying fit quality,
and that fit quality has no minimum at the truth:

| candidate q | reprojection of the fit (px) |
| ---: | ---: |
| 0.50 | 13.63 |
| 0.78 | 4.88 |
| **1.00 (truth)** | **~3.0** |
| 1.20 | 2.41 |
| 1.50 | 1.82 |

With 20 free bone lengths per side plus a free pose per frame, a larger focal
is always a slightly better explanation: the fit lengthens and reshapes the
hand to compensate. The focal error is absorbed by the shape parameters rather
than being exposed by them.

**The held-out frame split did not rescue this** (g3). That guard was included
precisely to catch over-fitting, but it does not fire here: a hand fitted at
the wrong focal is *consistently* wrong, so it generalises perfectly well to
other frames of the same hand. Held-out data exposes variance, not this kind of
shared bias.

And this is also why g4 could not pass. A control can only degrade performance
that exists; the real correspondence was already pinned at the boundary, so
mismapping the bones had nothing to spoil.

## 5. What this does and does not establish

**Establishes.** Comparing fixed reference bone lengths between the hands is
focal-invariant and cannot estimate a focal — this is exact, not empirical.
And in this candidate-conditioned formulation, where the shared bone
proportions are free parameters, bilateral consistency does not identify the
focal on clean synthetic data at any tested distance, asymmetry, noise level or
pose diversity.

**Does not establish** that left–right hand structure is useless. The failure
is specific and mechanical: the formulation grants 20 free shape parameters per
side, and those absorb exactly the signal they were meant to reveal. A
formulation that constrains the bone proportions far more tightly — for example
to a low-dimensional anatomical subspace, or with a strong prior shared across
subjects — would not have this degeneracy and has not been tested here.

`FINGER_CHAIN_PROPORTIONS` was registered as a secondary low-dimensional
formulation. It was not run, because the primary failed at the identifiability
gate rather than at the stability stage; running a variant after seeing the
primary fail would be a search, not a test.

## 6. Why the real-data phase was not run

The gate rule was fixed before results: if the bilateral objective cannot
identify a known focal on clean synthetic data, the GigaHands phase does not
run. It failed, so it did not run. The GigaHands reference focal was never
read by this experiment — see `tables/leakage_audit.csv`.

This is the intended function of a synthetic gate: it cost a few minutes of
compute and saved a multi-hour real-data run that could only have produced an
uninterpretable result.

## 7. Limitations

1. One articulation source (other-camera-only reference directions), one
   camera model, one hand topology.
2. 20 trials per condition — enough to establish a 90–100 % boundary rate,
   not enough for fine effect sizes.
3. The synthetic hand generator uses a plausible but invented bone-length
   distribution; the conclusion rests on the degeneracy, which is structural,
   not on those particular numbers.
4. Only the L1 distance between sum-normalised proportion vectors was tried as
   the bilateral distance.
5. The secondary low-dimensional formulation was not run (§5).

## 8. Reproduce

```
python src/audit_focal_dependence.py    # the two claims
python src/run_synthetic.py             # the gate; exits non-zero on failure
python src/figures.py
```
