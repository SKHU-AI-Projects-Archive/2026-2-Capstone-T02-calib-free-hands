"""
samples/1_insert.mp4 를 파이프라인 기본값 focal 과 GeoCalib 추정 focal 두 조건으로 각각
렌더링해서 비교 영상 두 개를 만든다. GT focal 은 이 영상에 실측 카메라 캘리브레이션이
없어(EXIF 없음) 스킵.

demo_hand_mano.py 의 기존 함수(stage1_detect/stage2_fit_mano/stage3_to_camera_space/
stage4_visualize/make_video 등)를 그대로 재사용한다 — WiLoR forward pass 는 프레임당
1회만 실행하고(GigaHands 실험과 동일한 패턴), _cam_crop_to_full 를 몽키패치해서 잡은
raw 값(pred_cam/box_center/box_size/img_size)에 GeoCalib focal 만 새로 넣어 cam_t 를
다시 계산한다. 파이프라인 기본값 조건은 hand.cam_t 를 그대로 쓴다(오버라이드 없음).

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 tools/render_focal_comparison.py
"""
import copy
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import sys
import time
from pathlib import Path

import numpy as np
import torch

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "demo"))
sys.path.insert(0, str(_REPO_ROOT))

import demo_hand_mano as dm  # noqa: E402
from hand_topology import JOINTS_PER_HAND, LEFT, RIGHT  # noqa: E402

INPUT = _REPO_ROOT / "samples" / "1_insert.mp4"
OUT_PIPELINE = _REPO_ROOT / "out" / "1_insert_pipeline_default"
OUT_GEOCALIB = _REPO_ROOT / "out" / "1_insert_geocalib"


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t_start = time.time()

    OUT_PIPELINE.mkdir(parents=True, exist_ok=True)
    OUT_GEOCALIB.mkdir(parents=True, exist_ok=True)

    print("=== 모델 로드 ===")
    predictor = dm.load_predictor(checkpoint="anyhand", det_conf=0.3, det_iou=0.3, rescale_factor=2.0)
    from geocalib import GeoCalib
    device = predictor.device
    geo = GeoCalib(weights="pinhole").to(device)
    geo.eval()
    print(f"  로드 시간: {time.time()-t_start:.1f}s")

    import rgb_predictor
    ORIG_CAM_CROP = rgb_predictor._cam_crop_to_full
    captured = []

    def capturing(pred_cam, box_center, box_size, img_size, focal_length):
        captured.append((pred_cam.detach().cpu().clone(), box_center.detach().cpu().clone(),
                          box_size.detach().cpu().clone(), img_size.detach().cpu().clone()))
        return ORIG_CAM_CROP(pred_cam, box_center, box_size, img_size, focal_length)
    rgb_predictor._cam_crop_to_full = capturing

    def cam_t_from_raw(raw, focal_px):
        pc, bc, bs, isz = raw
        ct = ORIG_CAM_CROP(pc.unsqueeze(0), bc.unsqueeze(0), bs.reshape(1), isz, focal_px)
        return ct[0].numpy().astype(np.float32)

    records_pipeline, records_geocalib = [], []
    t_infer0 = time.time()
    n_frames = 0
    for frame_idx, frame_bgr in dm.iter_frames(INPUT, stride=1, max_frames=0):
        meta = dm.iter_frames.meta  # (fps, W, H, total)
        fps, W, H, _ = meta

        # ---- GeoCalib: 프레임당 focal_x 추정 ----
        rgb = frame_bgr[..., ::-1].copy()
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            out = geo.calibrate(img_t)
        focal_geocalib = float(out["camera"].f[0, 0].item())

        # ---- WiLoR forward pass: 1회만 ----
        captured.clear()
        boxes, is_right, scores = dm.stage1_detect(predictor, frame_bgr)
        hands = dm.stage2_fit_mano(predictor, frame_bgr, boxes, is_right, scores)
        raw_list = []
        for pc, bc, bs, isz in captured:
            for i in range(pc.shape[0]):
                raw_list.append((pc[i], bc[i], bs[i] if bs.dim() else bs, isz))
        assert len(raw_list) == len(hands), f"캡처 불일치 frame {frame_idx}: {len(raw_list)} vs {len(hands)}"

        rec_pipe = dm.empty_record(frame_idx)
        rec_geo = dm.empty_record(frame_idx)
        for hand, raw in zip(hands, raw_list):
            side = RIGHT if hand.is_right else LEFT
            b = dm.HAND_BASE[side]

            def fill(rec, hand_variant):
                if rec["detected"][side] and rec["scores"][side] >= hand_variant.score:
                    return
                kp3d_abs, verts_abs = dm.stage3_to_camera_space(hand_variant)
                rec["kp3d_abs"][b:b + JOINTS_PER_HAND] = kp3d_abs
                rec["kp2d"][b:b + JOINTS_PER_HAND] = hand_variant.keypoints_2d
                rec["cam_t"][side] = hand_variant.cam_t
                rec["mano_pose"][side] = hand_variant.mano_pose
                rec["mano_shape"][side] = hand_variant.mano_shape
                rec["bbox"][side] = hand_variant.bbox
                rec["scores"][side] = hand_variant.score
                rec["detected"][side] = True

            fill(rec_pipe, hand)   # 파이프라인 기본값: hand.cam_t 그대로

            hand_geo = copy.copy(hand)
            hand_geo.cam_t = cam_t_from_raw(raw, focal_geocalib)
            fill(rec_geo, hand_geo)

        records_pipeline.append(rec_pipe)
        records_geocalib.append(rec_geo)
        n_frames += 1
        if n_frames % 100 == 0:
            print(f"  frame {n_frames} ({time.time()-t_infer0:.0f}s elapsed)")

    print(f"추론 완료: {n_frames}프레임, {time.time()-t_infer0:.0f}s")

    for name, records, out_dir in (("파이프라인 기본값", records_pipeline, OUT_PIPELINE),
                                    ("GeoCalib", records_geocalib, OUT_GEOCALIB)):
        t0 = time.time()
        view_lims = dm.view_limits(records)
        png_paths = []
        cap = None
        import cv2
        cap = cv2.VideoCapture(str(INPUT))
        for rec in records:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            out_png = out_dir / f"frame_{rec['frame_idx']:06d}.png"
            dm.stage4_visualize(frame_bgr, rec, view_lims, out_png)
            png_paths.append(out_png)
        cap.release()
        dm.make_video(png_paths, fps, out_dir / "demo.mp4")
        print(f"[{name}] PNG {len(png_paths)}개 + demo.mp4 완료 ({time.time()-t0:.0f}s) -> {out_dir}")

    print(f"\n총 소요시간: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
