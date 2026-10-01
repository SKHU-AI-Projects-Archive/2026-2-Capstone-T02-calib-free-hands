# Generic anatomy prior

## Source — and what it is NOT

The prior `p0` is the **neutral MANO hand**: `mano_data/MANO_RIGHT.pkl`, shape
`betas = 0`, rest pose, loaded through `smplx.MANO(use_pca=False,
flat_hand_mean=True)`.

It is emphatically **not** built from GigaHands. Using the evaluation set's own
bone statistics as a prior would be leakage. Specifically excluded:

- GigaHands subject mean hand size
- GigaHands test-unit bone means
- any per-participant template (`p36`, `p41`, `p44`, `p52`)
- CAM-EXP-009.3.1's measured per-bone values
- the reference focal
- test 3D geometry of any kind

CAM-EXP-009.3.1's +857 % bone-permutation result is a reason to use **correct
bone identity**. It is not a reason to import the test set's bone ratios.

## Joint construction — audited, not guessed

`smplx.MANO` returns 16 joints in the MANO order (`0` wrist, `1-3` index,
`4-6` middle, `7-9` pinky, `10-12` ring, `13-15` thumb). The five fingertips are
the vertices named in `smplx.vertex_ids['mano']`:

```
{
  "thumb": 744,
  "index": 320,
  "middle": 443,
  "ring": 554,
  "pinky": 671
}
```

These are mapped into the OpenPose 21-joint order documented in
`demo/hand_topology.py` (`0` wrist, `1-4` thumb, `5-8` index, `9-12` middle,
`13-16` ring, `17-20` pinky), and the 20 connected bones are taken from that.

## The mapping was verified empirically

A wrong mapping would not produce a hand-shaped bone profile. Checked against
WiLoR's own 21-joint output over 406 cached hands:

| check | value |
| --- | ---: |
| Spearman rank correlation of bone proportions | **0.9985** |
| max absolute proportion difference | **0.0012** |

WiLoR is used **only as a check on the mapping**; no WiLoR geometry enters the
prior.

## Absolute size is discarded

Only the sum-normalised proportions are kept (total neutral hand length
0.7867 m is thrown away), so no absolute hand-size assumption enters the method.
The LEFT prior uses the same bone proportions: the bone set is
mirror-symmetric, and only proportions are used.

Full values in `experiments/manifests/cam_exp_0094_generic_mano_prior_v1.json`.
