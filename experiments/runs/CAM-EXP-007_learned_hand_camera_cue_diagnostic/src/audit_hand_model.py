"""Source audit of the frozen hand model, before any feature is extracted.

Records exactly which outputs exist, which are focal-contaminated and therefore
forbidden as primary features, and where a latent representation can be tapped
without modifying model code.

Reads no target.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, REPO, SUM, TAB, sha256, write_csv, write_json  # noqa: E402

INVENTORY = [
    # name, source, shape, focal_dependent, allowed_primary, group, note
    ("pred_cam.s", "WiLoR forward_step output['pred_cam'][0]", "scalar",
     0, 1, "F1_CAMERA_HEAD_OUTPUT",
     "crop-space weak-perspective scale; NOT converted to metric translation"),
    ("pred_cam.tx", "output['pred_cam'][1]", "scalar", 0, 1,
     "F1_CAMERA_HEAD_OUTPUT", "crop-space; left-hand sign already mirrored"),
    ("pred_cam.ty", "output['pred_cam'][2]", "scalar", 0, 1,
     "F1_CAMERA_HEAD_OUTPUT", "crop-space"),
    ("pred_mano_params.betas", "output['pred_mano_params']['betas']", "(10,)",
     0, 1, "F2_EXPLICIT_HAND_GEOMETRY", "MANO shape coefficients"),
    ("pred_mano_params.global_orient", "(1,3,3) rotation matrix", "(3,3)",
     0, 1, "F2_EXPLICIT_HAND_GEOMETRY", "converted to axis-angle"),
    ("pred_mano_params.hand_pose", "(15,3,3) rotation matrices", "(15,3,3)",
     0, 1, "F2_EXPLICIT_HAND_GEOMETRY", "converted to axis-angle"),
    ("pred_keypoints_3d", "output['pred_keypoints_3d']", "(21,3)", 0, 1,
     "F2_EXPLICIT_HAND_GEOMETRY",
     "ROOT-RELATIVE: cam_t is never added, so this carries no focal"),
    ("pred_keypoints_2d", "output['pred_keypoints_2d']", "(21,2)", 0, 1,
     "F0_HAND_VISIBILITY_AND_BOX",
     "normalised crop coordinates; used only for crop-space spread"),
    ("box_center", "ViTDetDataset batch['box_center']", "(2,)", 0, 1,
     "F0_HAND_VISIBILITY_AND_BOX", "detector crop geometry"),
    ("box_size", "batch['box_size']", "scalar", 0, 1,
     "F0_HAND_VISIBILITY_AND_BOX", "detector crop geometry"),
    ("bbox", "YOLO detector box in original pixels", "(4,)", 0, 1,
     "F0_HAND_VISIBILITY_AND_BOX", "screen geometry, not a hand representation"),
    ("score", "YOLO detector confidence", "scalar", 0, 1,
     "F0_HAND_VISIBILITY_AND_BOX", ""),
    ("is_right", "detector side flag", "bool", 0, 1,
     "F0_HAND_VISIBILITY_AND_BOX", "used only as a side-consistency statistic"),
    ("backbone vit_out", "WiLoR backbone 4th return value",
     "(B,1280,16,12)", 0, 1, "F4_LEARNED_HAND_LATENT",
     "image-encoder feature map before the MANO/camera heads; the frozen "
     "latent tap for this run"),
    ("pred_mano_feats", "WiLoR backbone 3rd return value", "varies", 0, 0,
     "NOT_USED", "a second candidate latent; NOT used, because only one "
                 "latent layer is frozen and trying several would be "
                 "selection on the result"),
    # ---- forbidden ----
    ("cam_t / pred_cam_t", "_cam_crop_to_full(pred_cam, ..., scaled_focal)",
     "(3,)", 1, 0, "FORBIDDEN",
     "metric translation computed WITH the pipeline focal; using it would "
     "re-inject the camera focal into the predictor"),
    ("focal_length", "scaled_focal = FOCAL_LENGTH/IMAGE_SIZE * max(W,H)",
     "scalar", 1, 0, "FORBIDDEN",
     "PIPELINE_BASELINE_FOCAL, a training convention constant per image size"),
    ("vertices (absolute)", "MANO mesh placed by cam_t", "(778,3)", 1, 0,
     "FORBIDDEN", "absolute placement depends on the focal"),
    ("keypoints_2d in original pixels", "_kp2d_crop_to_full", "(21,2)", 0, 0,
     "NOT_USED_PRIMARY",
     "derived from the crop transform, not focal-contaminated, but it "
     "duplicates box geometry; excluded to keep F0 minimal"),
]


def main() -> None:
    rows = []
    for (name, src, shape, focal_dep, allowed, group, note) in INVENTORY:
        rows.append({
            "output": name, "source": src, "shape": shape,
            "focal_dependent": focal_dep,
            "allowed_as_primary_feature": allowed,
            "feature_group": group, "note": note,
        })
    write_csv(TAB / "hand_model_output_inventory.csv", rows)

    ck = {}
    for k, p in MODELS.items():
        ck[k] = {"path": str(p.relative_to(REPO)).replace("\\", "/"),
                 "exists": p.exists(),
                 "bytes": p.stat().st_size if p.exists() else None,
                 "sha256": sha256(p) if p.exists() else None}

    meta = {
        "model": "AnyHand-WiLoR (WiLoR architecture, AnyHand fine-tuned "
                 "checkpoint)",
        "entry_point": "rgb_predictor.AnyHandPredictor, backend 'wilor'",
        "model_class": "WiLoR/wilor/models/wilor.py :: WiLoR "
                       "(pl.LightningModule)",
        "detector": "WiLoR YOLO hand detector (models/detector.pt)",
        "checkpoints": ck,
        "forward_path": "WiLoR.forward_step: backbone(x[:,:,:,32:-32]) -> "
                        "(temp_mano_params, pred_cam, pred_mano_feats, "
                        "vit_out); refine_net(vit_out, ...) -> "
                        "(pred_mano_params, pred_cam)",
        "latent_tap": {
            "layer": "model.backbone",
            "which_output": "index 3 of the returned tuple (vit_out)",
            "why": "the image encoder's final feature map, before the MANO "
                   "and camera heads consume it. Chosen on architecture "
                   "semantics before any target was read.",
            "tensor_shape": "(B, 1280, 16, 12)",
            "pooling": "global average pool over the two spatial axes -> "
                       "(B, 1280)",
            "extraction_method": "forward hook; no model source is modified",
        },
        "preprocessing": "ViTDetDataset crop around the detector box, "
                         "rescale_factor 2.0, model input 256x192 after the "
                         "[:, :, :, 32:-32] crop inside forward_step",
        "modifications_to_model_source": "none",
        "cam002_cache_reuse": {
            "decision": "NOT_REUSED",
            "cache": "experiments/runs/CAM-EXP-002_camera_focal_sensitivity/"
                     "cache/hand_inference (1189 frames, 135 views)",
            "reason": "it stores no pred_cam, no mano_pose, no box_size and "
                      "no latent - three of the four primary feature groups - "
                      "and its frames do not lie on the frozen CAM-006 "
                      "16-frame grid. Partial reuse would mix frame sets "
                      "across feature groups, so CAM-007 runs its own "
                      "inference. The historical cache is not modified.",
        },
        "forbidden_outputs": [r["output"] for r in rows
                              if r["feature_group"] == "FORBIDDEN"],
    }
    write_json(SUM / "hand_model_audit.json", meta)
    print(f"inventory rows: {len(rows)}")
    for k, v in ck.items():
        print(f"  {k}: exists={v['exists']} sha256="
              f"{(v['sha256'] or '')[:16]} bytes={v['bytes']}")
    print(f"  forbidden: {meta['forbidden_outputs']}")


if __name__ == "__main__":
    main()
