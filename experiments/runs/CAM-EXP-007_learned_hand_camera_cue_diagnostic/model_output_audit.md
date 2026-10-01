# CAM-EXP-007 — hand model output audit

Machine-readable: `results/summary/hand_model_audit.json`,
`tables/hand_model_output_inventory.csv`.

## 1. Model identity

| item | value |
| --- | --- |
| model | AnyHand-WiLoR (WiLoR architecture, AnyHand fine-tuned checkpoint) |
| entry point | `rgb_predictor.AnyHandPredictor, backend 'wilor'` |
| model class | `WiLoR/wilor/models/wilor.py :: WiLoR (pl.LightningModule)` |
| detector | WiLoR YOLO hand detector (models/detector.pt) |
| wilor checkpoint | `models/anyhand_wilor.ckpt` |
| wilor sha256 | `9709eca6e77fb77d2cfcf6ee641660f98dca6941e9fc1cf7a5be19fe62164b77` |
| wilor bytes | 2,565,031,198 |
| detector checkpoint | `models/detector.pt` |
| detector sha256 | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| detector bytes | 53,582,271 |
| preprocessing | ViTDetDataset crop around the detector box, rescale_factor 2.0, model input 256x192 after the [:, :, :, 32:-32] crop inside forward_step |
| model source modified | **none** |

## 2. Forward path

```
WiLoR.forward_step: backbone(x[:,:,:,32:-32]) -> (temp_mano_params, pred_cam, pred_mano_feats, vit_out); refine_net(vit_out, ...) -> (pred_mano_params, pred_cam)
```

## 3. Output inventory

| output | group | focal-dependent | allowed as primary | note |
| --- | --- | :-: | :-: | --- |
| `pred_cam.s` | F1_CAMERA_HEAD_OUTPUT | no | yes | crop-space weak-perspective scale; NOT converted to metric translation |
| `pred_cam.tx` | F1_CAMERA_HEAD_OUTPUT | no | yes | crop-space; left-hand sign already mirrored |
| `pred_cam.ty` | F1_CAMERA_HEAD_OUTPUT | no | yes | crop-space |
| `pred_mano_params.betas` | F2_EXPLICIT_HAND_GEOMETRY | no | yes | MANO shape coefficients |
| `pred_mano_params.global_orient` | F2_EXPLICIT_HAND_GEOMETRY | no | yes | converted to axis-angle |
| `pred_mano_params.hand_pose` | F2_EXPLICIT_HAND_GEOMETRY | no | yes | converted to axis-angle |
| `pred_keypoints_3d` | F2_EXPLICIT_HAND_GEOMETRY | no | yes | ROOT-RELATIVE: cam_t is never added, so this carries no focal |
| `pred_keypoints_2d` | F0_HAND_VISIBILITY_AND_BOX | no | yes | normalised crop coordinates; used only for crop-space spread |
| `box_center` | F0_HAND_VISIBILITY_AND_BOX | no | yes | detector crop geometry |
| `box_size` | F0_HAND_VISIBILITY_AND_BOX | no | yes | detector crop geometry |
| `bbox` | F0_HAND_VISIBILITY_AND_BOX | no | yes | screen geometry, not a hand representation |
| `score` | F0_HAND_VISIBILITY_AND_BOX | no | yes |  |
| `is_right` | F0_HAND_VISIBILITY_AND_BOX | no | yes | used only as a side-consistency statistic |
| `backbone vit_out` | F4_LEARNED_HAND_LATENT | no | yes | image-encoder feature map before the MANO/camera heads; the frozen latent tap for this run |
| `pred_mano_feats` | NOT_USED | no | no | a second candidate latent; NOT used, because only one latent layer is frozen and trying several would be selection on the result |
| `cam_t / pred_cam_t` | FORBIDDEN | yes | no | metric translation computed WITH the pipeline focal; using it would re-inject the camera focal into the predictor |
| `focal_length` | FORBIDDEN | yes | no | PIPELINE_BASELINE_FOCAL, a training convention constant per image size |
| `vertices (absolute)` | FORBIDDEN | yes | no | absolute placement depends on the focal |
| `keypoints_2d in original pixels` | NOT_USED_PRIMARY | no | no | derived from the crop transform, not focal-contaminated, but it duplicates box geometry; excluded to keep F0 minimal |

## 4. The forbidden outputs

Three outputs are focal-contaminated and are excluded from every
primary feature set:

* `cam_t / pred_cam_t`
* `focal_length`
* `vertices (absolute)`

The contamination enters at
`cam_t_full = _cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal)`.
Anything downstream of that multiplication carries the pipeline focal,
so using it as a predictor would make the question circular.

## 5. Latent tap

| item | value |
| --- | --- |
| layer | `model.backbone` |
| which output | `index 3 of the returned tuple (vit_out)` |
| tensor shape | `(B, 1280, 16, 12)` |
| pooling | `global average pool over the two spatial axes -> (B, 1280)` |
| extraction method | `forward hook; no model source is modified` |

the image encoder's final feature map, before the MANO and camera heads consume it. Chosen on architecture semantics before any target was read.

Full detail in `latent_protocol.md`.

## 6. Reuse of the CAM-EXP-002 inference cache

**Decision: NOT_REUSED.**

Cache: `experiments/runs/CAM-EXP-002_camera_focal_sensitivity/cache/hand_inference (1189 frames, 135 views)`.

it stores no pred_cam, no mano_pose, no box_size and no latent - three of the four primary feature groups - and its frames do not lie on the frozen CAM-006 16-frame grid. Partial reuse would mix frame sets across feature groups, so CAM-007 runs its own inference. The historical cache is not modified.
