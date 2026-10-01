# CAM-EXP-007 — latent protocol

Frozen in `experiments/manifests/cam_exp_007_latent_spec_v1.json`,
`created_before_target_analysis = true`.

## 1. The tap

```
WiLoR.forward_step:
  temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x[:, :, :, 32:-32])
  pred_mano_params, pred_cam = self.refine_net(vit_out, ...)
```

| item | value |
| --- | --- |
| layer | `model.backbone` |
| which output | index 3 of the returned tuple, `vit_out` |
| tensor | `(B, 1280, 16, 12)` |
| pooling | global average over the two spatial axes → `(B, 1280)` |
| extraction | forward hook; **no model source is modified** |

## 2. Why this layer

It is the image encoder's final feature map, immediately before the MANO and
camera heads consume it. If the network holds an image- or camera-level prior
that is not exposed in its explicit geometric outputs, this is where it would
be. The choice was made from architecture semantics with no target read.

## 3. Exactly one layer

Only `vit_out` was probed. `pred_mano_feats` is a second candidate and was
deliberately **not** used: probing several layers and reporting the best is
selection on the result. The cost of this discipline is that other layers
remain untested, recorded as `SINGLE_LAYER_FROZEN_BY_DESIGN` in
`OPEN_ISSUES.md`.

## 4. Pooling choice

Global average pooling is the simplest deterministic reduction of a spatial
feature map, and was fixed before results. It discards spatial arrangement,
which is registered as a limitation — a perspective cue could plausibly live in
the spatial structure.

## 5. The hook does not change the model

Verified empirically on real frames rather than assumed:

```
frames tested 4, hands compared 5
max absolute difference 0.0
verdict HOOKS_DO_NOT_ALTER_MODEL_OUTPUT
```

(`results/summary/hook_identity_check.json`.) PyTorch forward hooks returning
`None` cannot alter the output, but the check confirms it.

## 6. Dimension reduction

PCA by SVD, **fit on training-fold views only** inside each outer fold, primary
dimension `min(32, n_train_views - 2, input_dim)`. The dimension was never
changed after seeing test performance; 16 and 64 were registered as secondary
sensitivity and were not needed, because the primary result is a clean null.
