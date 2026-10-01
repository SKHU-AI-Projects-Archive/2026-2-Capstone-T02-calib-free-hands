# Statistical protocol

## The unit problem

40-49 cameras observe the **same pair of hands** in a sequence. They are
emphatically not independent samples of a person. The aggregation is therefore:

| level | independent? | role |
| --- | --- | --- |
| frame | no | pooled into a template |
| sequence-camera | no | the raw comparison unit |
| sequence | partial | **primary aggregation level** |
| candidate participant | unverified | only 4 groups, identity unresolved |

Camera views are never counted as independent participants, and cross pairings
are **camera-matched** so the two contrasts are not separated artificially by
camera/view reconstruction differences.

## Cross-pairing explosion

One sequence's LEFT can be compared against many other RIGHTs. The raw pair
table is kept, but those pairs are **not** counted as independent samples: the
primary margin is summarised per sequence-camera and then per sequence.

## The three distances

- `D_repeat` — same sequence, camera and hand; half-A template vs half-B.
  The split is the SHA-256 parity of `(sequence|camera|hand|frame)`:
  deterministic and independent of fit quality.
- `D_view_repeat` — same sequence and hand, two different cameras.
- `D_within` / `D_cross` — the contrast of interest.

## Signal-to-reconstruction ratio

```
S_sep = ( median(D_cross) - median(D_within) ) / median(D_repeat)
```

`S_sep > 1` means the within-vs-cross separation is larger than the
reconstruction's own repeatability wobble. `S_sep < 1` means the wanted signal
is comparable to or smaller than the measurement noise.

## Permutation test

The statistic is `median(CROSS) - median(WITHIN)`, camera-matched, and the null
permutes **sequence labels**. With 5 sequences all `5! = 120` relabelings are
enumerable, so the null is computed **exactly** rather than sampled, and the
identity permutation is included in the null as a valid permutation test
requires.

**The smallest attainable p-value is therefore 1/120 = 0.0083, and the
permutation unit is the sequence, of which there are five.** The test is weak
by construction, whatever it returns.

## Bootstrap

Cluster bootstrap on **sequence**, 10,000 iterations. With 5 clusters the
intervals are wide and the label is unstable; they are supplemental.

## Power

Fewer than 8 unique participants triggers `SUBJECT_SEPARABILITY_UNDERPOWERED`.
The experiment still runs and the descriptive mechanism result is reported, but
no strong population-generalisation claim is made and no conclusion rests on a
p-value alone.
