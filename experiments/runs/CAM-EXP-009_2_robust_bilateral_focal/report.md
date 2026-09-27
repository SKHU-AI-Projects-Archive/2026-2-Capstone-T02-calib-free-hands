# CAM-EXP-009.2 — Asymmetry- and Noise-Robust Bilateral Hand-Geometry Focal Fusion

```
VERDICT  ROBUST_BILATERAL_SYNTHETIC_GATE_FAILED
         The selected robust bilateral formulation did not satisfy the
         pre-specified synthetic robustness gate.

SYNTHETIC GATE: FAILED  (g2 and g7, both required)
REAL GIGAHANDS PHASE: NOT RUN
  (pre-registered consequence of a failed synthetic gate)

SELECTED METHOD  M4_HUBER_005   (chosen on DEV seeds only)
```

**One sentence.** A robust bilateral comparison does extend the reach of the
CAM-EXP-009.1 cue — it recovers the focal exactly under clean conditions, under
a pure left/right size difference, and under missing joints and low per-side
visibility — but under the pre-specified `COMBINED_MODERATE` synthetic stress it
lands at **10.8 %** focal error against a 5 % bar, and at that same condition
replacing the right hand with one from a different synthetic subject did not
degrade performance, so this experiment found **no measurable same-subject
bilateral advantage under that condition**.

That control was evaluated only at `COMBINED_MODERATE`. It does not establish
whether same-subject pairing provides information in the clean condition or in
any other stress regime.

## 1. Plain-language summary

The idea under test: a person's left and right hands have related bone
structure, so the camera focal that makes the two hands most consistent with
each other is probably the correct one. CAM-EXP-009.1 showed this works when
the two hands are *identical*, and collapses when they are not. This run asked
whether comparing them in a way that tolerates a few badly mismatched bones
rescues it.

Partly, and not enough. Where the two hands differ only by an overall size
factor, the method is perfect — but that is because such a difference is
mathematically invisible to it, not because it handled it. Where the hands
differ bone-by-bone, or where the 2D joint positions are noisy, the estimate
degrades roughly in step with the amount of stress, and at the pre-specified
moderate combination of stresses it is off by about a tenth of the focal
length.

The finding that decides the gate is a control. We gave the method the right
hand of a **different synthetic person**, breaking the intended same-subject
pairing. It did just as well — slightly better. So at that stress level the
correct same-subject pairing bought nothing measurable. We tested this at one
stress level only, so it does not tell us whether same-subject pairing helps in
the clean condition, where the cue does recover the focal exactly.

Because the pre-registered gate failed, the GigaHands phase was not run.

## 2. What was frozen, and when

| step | artefact | before |
| --- | --- | --- |
| A candidate family M0–M6 | `cam_exp_0092_robust_candidate_spec_v1.json` | any DEV result |
| B stress grid / generator | same manifest | any DEV result |
| C DEV seeds | `cam_exp_0092_synthetic_dev_trials_v1.csv.gz` | any DEV result |
| D TEST seeds | `cam_exp_0092_synthetic_test_trials_v1.csv.gz` | any DEV result |
| E DEV run | `results/raw/dev_trials.csv.gz` | — |
| F winner selection | deterministic pre-registered rule | — |
| G winner manifest | `cam_exp_0092_selected_method_v1.json` | any TEST result |
| H TEST run | `results/raw/test_trials.csv.gz` | — |

Trial counts were fixed from a **measured 10.9 s per trial**, not from observed
performance. The CAM-EXP-009.1 solver was imported unchanged
(`UNDISTORT_ONCE_INTERNAL`, `N_ITER = 32`, D20 only); `self_audit.py` fails if
anything in that run directory is modified. Its D10/D5 groupings were **not**
reused — they force equal bone lengths within a group, which no real hand
satisfies.

Bilateral asymmetry is drawn **once per synthetic subject** and held fixed for
the whole sequence. The two hands are never assumed identical.

## 3. Method selection (DEV — a selection instrument, not evidence)

| method | DEV selection score (% focal error) |
| --- | ---: |
| M0_BASELINE_RAW_L1 | 8.215 |
| M1_LOG_L1 | 6.362 |
| M2_LOG_MEDIAN | 11.216 |
| M3_LOG_TRIMMED | 7.082 |
| M4_HUBER_001 | 6.390 |
| M4_HUBER_002 | 7.020 |
| **M4_HUBER_005** | **5.085** |
| M5_FINGER_INTERNAL_RATIO | 8.282 |
| M6_FINGER_HUBER | 7.211 |

These numbers chose the method and are reported for transparency only. They are
not evidence that the cue works; the disjoint TEST set is.

## 4. TEST results

Grid anchored on an arbitrary `F_NOMINAL_SYNTH = 1000` while the true focal is
900, so the answer sits at q = 0.9, **away from the grid centre**. The grid step
is 0.689 %, and the nearest grid point to the truth is 902.449 px — so
**0.272 % is the quantisation floor: an exact hit at the resolution of the
frozen candidate grid**, not a residual error.

| cell | M4_HUBER_005 | M0 | rel. reduction | within 5 % | boundary |
| --- | ---: | ---: | ---: | ---: | ---: |
| TEST_CLEAN | **0.27** | 0.27 | 0.0 % | 100 % | 0.00 |
| ASYM_GLOBAL 1 / 2 / 5 % | **0.27 / 0.27 / 0.27** | 0.27 | 0.0 % | 100 % | 0.00 |
| ASYM_DENSE 1 / 2 / 5 % | 1.72 / 5.84 / 17.26 | 2.72 / 6.06 / 20.59 | +37 / +4 / +16 % | 88 / 38 / 19 % | 0.00 / 0.00 / 0.12 |
| ASYM_SPARSE 1 / 2 / 5 % | 1.10 / 3.06 / 5.44 | 0.96 / 0.96 / 3.77 | −14 / **−218** / −44 % | 100 / 63 / 50 % | 0.00 |
| ASYM_FINGER 1 / 2 / 5 % | 1.77 / 4.13 / 17.02 | 4.10 / 4.85 / 16.12 | +57 / +15 / −6 % | 75 / 56 / 25 % | 0.00 |
| NOISE 0.5 / 1 / 2 px | 5.80 / 11.04 / 19.23 | 4.43 / 10.16 / 12.59 | −31 / −9 / **−53 %** | 38 / 31 / 19 % | 0.00 / 0.00 / 0.06 |
| ARTIC 0.5 / 1 / 2° | 8.21 / 12.73 / 26.18 | 7.21 / 11.08 / 31.85 | −14 / −15 / +18 % | 13 / 25 / 13 % | 0.00 / 0.00 / 0.19 |
| MISSING 10 / 20 % | **0.27 / 0.27** | 0.27 | 0.0 % | 100 % | 0.00 |
| VISIBILITY 75 / 50 / 25 % | **0.27 / 0.27 / 0.27** | 0.27 | 0.0 % | 100 % | 0.00 |
| DIST 2× / 8× / 16× | **0.27 / 0.27 / 0.27** | 0.27 | 0.0 % | 100 % | 0.00 |
| **COMBINED_MODERATE** | **10.78** | 15.58 | +30.8 % | 25 % | 0.00 |
| COMBINED_STRONG | 44.44 | 44.44 | 0.0 % | 19 % | 0.50 |

Subject-clustered bootstrap, 10,000 iterations, on the headline cell:
M4_HUBER_005 **10.78 % [7.02, 21.69]**, M0 **15.58 % [10.16, 22.76]**. The
intervals overlap across most of their range, so the 30.8 % point reduction
should not be read as a well-separated improvement.

### 4.1 The robust score is not uniformly better than the baseline

It wins on dense and finger asymmetry at low levels, and on the moderate
combination. It **loses** on sparse asymmetry (−218 % at the 2 % level) and on
2D noise (−53 % at 2 px). A single robust loss chosen on a 7-cell DEV set does
not dominate the plain L1 distance across the stress space; it trades one
regime for another. This is reported as measured, not summarised as a gain.

### 4.2 The cells at exactly 0.272 % are not successes of robustness

`ASYM_GLOBAL`, `MISSING`, `VISIBILITY` and `DIST` all return the grid floor at
every level, for both methods. For `ASYM_GLOBAL` the reason is structural and
was predicted in advance: both proportion vectors are normalised to sum to 1,
so a pure left/right size ratio is absorbed by the normalisation and cannot
change any score. The `audit_r_LR_redundancy()` check measured the largest
possible score change under such a rescaling as **5.7e-14** across all nine
methods, and these cells confirm the same thing end to end through the solver.

A separate `r_LR` nuisance parameter is therefore **redundant, not omitted** —
adding one would be fitting a quantity the objective cannot see.

For `MISSING`, `VISIBILITY` and `DIST` the reason is simpler: those cells hold
asymmetry, 2D noise and articulation noise at zero, so the two hands are
identical and the problem reduces to the clean case. They show the cue tolerates
missing joints and sparse per-side observation — which is a real and useful
negative-space result — but they are not evidence about asymmetry.

## 5. The controls — the decisive evidence

`COMBINED_MODERATE`, 8 subjects, selected method, EVAL profiles.

| control | median focal error | boundary rate | score-curve range |
| --- | ---: | ---: | ---: |
| C0_CORRECT | 7.41 % | 0.12 | 3.53e-01 |
| C1_WRONG_BONE_MAPPING | 44.44 % | 0.88 | 2.45e-01 |
| **C2_SUBJECT_SWAP** | **6.84 %** | 0.00 | 2.43e-01 |
| C3_TEMPORAL_SHUFFLE | 6.84 % | 0.00 | 2.43e-01 |
| C4_FIXED_LENGTH_INVARIANT | 44.44 % | 1.00 | **0.00e+00** |

Recorded in `notes/control_observations_pre_test.md` **before** the TEST run
finished, so this reading cannot have been shaped by the TEST numbers.

**C2 does not degrade.** Replacing the matched right hand with a right hand
from a different synthetic subject — breaking the intended same-subject pairing
— gives 6.84 % against the correct pairing's 7.41 %. Under the pre-registered
control logic, at `COMBINED_MODERATE` the correct same-subject pairing had
**no advantage** over the subject-swap control: the residual focal preference
there was not specifically attributable to same-subject left–right
correspondence. What varies with the candidate focal is each side's own fitted
profile, and comparing the left profile against *some* plausible right-hand
profile produced a similar minimum whether or not it belonged to the same
subject.

Two limits on how far this reaches. **First**, this control was evaluated only
at `COMBINED_MODERATE`; it does not establish whether same-subject pairing
provides information in the clean condition or in other stress regimes.
**Second**, the synthetic subjects are drawn from one generator with a shared
bone-length distribution, so a swapped right hand is not anatomically unrelated
to the left — it is a different draw from the same family. The control breaks
the *intended pairing*, not all shared anatomical structure.

Both conditions are poor in absolute terms (7 % against a 5 % bar). This is not
a finding that the subject swap works — it is a finding that neither pairing
identifies the focal well at this stress level, and that the correct pairing
had no measurable advantage there.

**C1 and C4 behave as required.** A wrong bone mapping degrades hard (44 %,
88 % at a boundary), so the score is not indifferent to which bone is compared
with which. C4's score-curve range is **exactly zero**, reconfirming
CAM-EXP-009's exact result that the naive fixed-length formulation cannot
depend on the candidate focal. (C4 is an oracle diagnostic — handed the true
bone lengths, and not runnable on real data.)

**C3 is vacuous by construction, and that is a design fault of mine.** It is
bit-identical to C2 per subject (subject 0: 781.3 px for both; subject 1:
960.0; subject 2: 836.8). The objective fits each side independently across all
of that side's frames and never pairs a left frame with a right frame, so there
is no temporal pairing for a shuffle to destroy. It is recorded as
`CONTROL_VACUOUS_BY_CONSTRUCTION` with no evidential weight in either
direction, and was **not** replaced with a different control after the fact.

## 6. The frozen gate

| gate | criterion | result | required |
| --- | --- | --- | --- |
| g1 | clean median error ≤ 1 % | **PASS** — 0.27 % | no |
| g2 | COMBINED_MODERATE ≤ 5 % | **FAIL** — 10.78 % | **yes** |
| g3 | ≥ 30 % relative reduction vs M0 | **PASS** — 30.8 % | **yes** |
| g4 | boundary rate ≤ 0.10 | **PASS** — 0.00 | no |
| g5 | FIT vs EVAL differ ≤ 5 pp | **PASS** | no |
| g6 | wrong mapping degrades ≥ 10 pp | **PASS** — +37 pp | **yes** |
| g7 | subject swap degrades ≥ 10 pp | **FAIL** — −0.6 pp | **yes** |
| g8 | fixed-length curve exactly flat | **PASS** — range 0 | no |

Two of the four required gates failed. g3 passed on the point estimate, but its
bootstrap intervals overlap (§4), and g3 is in any case a *relative* criterion:
being 30 % better than a baseline that is also wrong does not make an estimate
usable.

## 7. What this does and does not establish

**Establishes**, on synthetic data generated by the same model the solver
assumes:

1. A robust (Huber, τ = 0.05, log-proportion) bilateral comparison recovers a
   known focal exactly under clean conditions, and tolerates joint missingness
   up to 20 %, per-side visibility down to 25 %, and distances from 2× to 16×
   hand diameters.
2. A pure left/right size difference is invisible to this objective by
   construction — measured, not assumed — so a global scale nuisance parameter
   would be redundant.
3. Under bone-by-bone asymmetry, 2D noise and articulation-direction error the
   estimate degrades roughly in proportion to the stress, reaching 10.8 % at the
   pre-registered moderate combination and 44 % at the strong one.
4. The robust loss does not dominate the plain L1 baseline across the stress
   space; it wins in some regimes and loses in others.
5. At `COMBINED_MODERATE`, a right hand from a different synthetic subject
   serves as well as the matched one, so at that condition the residual focal
   preference was not specifically attributable to same-subject correspondence.
   This was measured at that one stress level only.
6. The naive fixed-length formulation is exactly focal-invariant
   (CAM-EXP-009's result, reconfirmed).

**Does not establish** that left–right hand structure carries no focal
information, that the cue is not bilateral, or that same-subject geometry is
irrelevant in general. Point 5 is a statement about *this* formulation at *one*
stress level: 20 free shape parameters per side, fitted independently, compared
after sum-normalisation, at `COMBINED_MODERATE`. A formulation that constrained
the two sides jointly — a shared low-dimensional anatomical subspace, or a prior
tying corresponding bones across subjects — would not have the same degeneracy
and was not tested.

**Does not establish anything about real data.** No GigaHands focal was
estimated in this run. Equally, had the synthetic gate passed, that would not
have licensed a claim about GigaHands or about a factory deployment either —
the synthetic generator uses the solver's own hand and camera model, which real
data does not.

### 7.1 Canonical conclusion

CAM-EXP-009.2 shows that the selected robust bilateral formulation retains
focal information in clean synthetic conditions, but does not satisfy the
pre-specified robustness criteria under the evaluated combined synthetic
perturbations. At `COMBINED_MODERATE`, same-subject left–right pairing did not
outperform a subject-swapped right-hand control. Therefore the formulation is
not ready for the planned GigaHands evaluation.

**No conclusion about real GigaHands performance is drawn from CAM-EXP-009.2,
because the real-data phase was not run.** `COMBINED_MODERATE` and
`COMBINED_STRONG` are pre-specified, controlled synthetic stress conditions.
They were not built by measuring an error distribution from GigaHands or from a
deployment, so a failure at those conditions is a failure of the
pre-specified synthetic gate, not a measured failure on real data.

## 8. Why the real-data phase was not run

The rule was fixed before any result: if the required synthetic gates fail, the
GigaHands phase does not run. They failed, so it did not. Status
`REAL_FOCAL_PHASE_NOT_RUN`; the full protocol that *would* have run is in
`real_data_protocol.md`, written before the gate was evaluated.

A cue that cannot recover a known focal from data produced by the very model
the solver assumes will not become interpretable on real data, where the hand
model, the articulation source and the camera are all additionally wrong.

## 9. Limitations

1. One articulation source, one camera model, one hand topology, one invented
   bone-length distribution.
2. 16 subjects per primary cell, 8 per secondary cell — enough for the
   order-of-magnitude statements above, not for fine effect sizes. The
   headline bootstrap intervals are wide.
3. The controls were run at `COMBINED_MODERATE` only. C2's failure to degrade
   is established at that stress level, not across the grid.
4. `C3_TEMPORAL_SHUFFLE` was vacuous (§5), so the pre-registered control set is
   effectively four conditions, not five.
5. The DEV set is 7 cells and 8 subjects; a different DEV set could well have
   selected a different member of the family. §4.1 shows the family members are
   not ordered consistently across the stress space.
6. `MISSING`, `VISIBILITY` and `DIST` were specified with zero asymmetry and
   zero noise, so they do not test those stresses in combination. Only the two
   `COMBINED` cells do.

## 10. Reproduce

```
python src/freeze_spec.py            # A-D: family, grid, DEV and TEST seeds
python src/run_synth_dev.py          # E
python src/select_robust_method.py   # F-G: winner frozen to a manifest
python src/run_synth_test.py         # H  (~62 min measured)
python src/run_synth_controls.py
python src/evaluate.py               # the frozen gate
python src/figures.py
python src/self_audit.py
```
