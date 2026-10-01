# CAM-EXP-009 — focal dependence audit

Run before anything else, on noiseless, perfectly symmetric, moderate-
perspective synthetic data where the focal is known by construction. This is
the most favourable case: if the objective is flat here it is flat everywhere.

## Claim 1 — the naive loss is focal-invariant

Comparing the FIXED reference bone lengths of the two hands:

```
value range over a +-50 % candidate focal sweep :  0.000e+00
```

Exactly zero. The quantity does not depend on f, so dL/df = 0 and it cannot
estimate a focal. **This is why CAM-EXP-009 does not use it**, and it is worth
recording as a positive piece of knowledge rather than a footnote.

## Claim 2 — the candidate-conditioned loss does vary

Refitting l_L(f) and l_R(f) at every candidate:

```
value range                :  4.97e-03
relative dynamic range     :  1.97      (197 % of its own median)
```

So focal DEPENDENCE is achieved.

## But dependence is not identifiability

The same audit recorded where that curve is minimised:

```
argmin q = 1.500   (the grid boundary; truth is q = 1.0)
```

The curve is monotone decreasing toward larger focals, not peaked at the truth.
That distinction — dependent but not identifiable — is the whole finding of
this run, and it is what the synthetic gate then confirmed across every
distance, asymmetry, noise level and pose-diversity condition.

`results/summary/focal_dependence_audit.json`,
`results/raw/focal_dependence_curve.csv`.
