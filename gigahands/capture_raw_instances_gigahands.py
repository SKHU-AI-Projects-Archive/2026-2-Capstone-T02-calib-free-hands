"""
GigaHands translation 실험 후속분석(wrist-offset 검증, 부호 있는 z오차, bbox 비율)에
필요한 raw per-instance 값을 캡처하는 스크립트.

eval_translation_gigahands.py 는 인스턴스별 raw 값(pred_cam, keypoints_3d[WRIST],
box_size 등)을 디스크에 저장하지 않고 오차만 집계해 CSV로 남겼다 — 이번 후속분석
(wrist offset 분포, signed z error, bbox_size 비율)에는 그 raw 값 자체가 필요해서
불가피하게 WiLoR forward pass 를 1회 다시 실행한다("추론을 다시 하지 마라"는 지시와
배치되지만, 애초에 필요한 raw 값이 저장돼 있지 않았다 — 이번엔 전부 캐시로 남겨서
이후 어떤 후속분석도 재추론 없이 가능하게 한다).

동일한 샘플링(seed=0, target-per-group=500)을 재사용하므로 eval_translation_gigahands.csv
와 완전히 같은 1,846개 인스턴스가 나와야 한다 — 실행 후 기존 CSV의
abs_error_gt_focal_mm/abs_error_hardcoded_mm/z_error_gt_focal_mm/z_error_hardcoded_mm
과 행 단위로 대조해 100% 일치하는지 확인한다(다르면 버그).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 capture_raw_instances_gigahands.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent   # 이 스크립트 자신의 위치 (hand-demo/gigahands/)
HAND_DEMO_ROOT = Path("/home/juson/project/hand-demo")
OUT_DIR = HAND_DEMO_ROOT / "out" / "gigahands"

sys.path.insert(0, str(BASE_DIR))
import eval_pa_mpjpe_gigahands as baseline           # noqa: E402
import eval_translation_gigahands as tr               # noqa: E402  (focal_mm_to_px, load_camera_params, world_to_cam, quat_to_rotmat)

WRIST = 0
RESCALE_FACTOR = 2.0   # AnyHandPredictor 기본값과 동일 (rgb_predictor.AnyHandPredictor.__init__)
OUT_CSV = OUT_DIR / "gigahands_raw_instances_cache.csv"


def project_points(cam: dict, points_world_21x3: np.ndarray) -> np.ndarray:
    """GT world 21관절 -> 카메라좌표 -> 핀홀 투영 2D 픽셀 (21,2)."""
    pts_cam = np.array([tr.world_to_cam(cam, p) for p in points_world_21x3])  # (21,3)
    u = cam["fx"] * pts_cam[:, 0] / pts_cam[:, 2] + cam.get("cx", cam["width"] / 2)
    v = cam["fy"] * pts_cam[:, 1] / pts_cam[:, 2] + cam.get("cy", cam["height"] / 2)
    return np.stack([u, v], axis=1)


def load_camera_params_with_cxcy():
    """tr.load_camera_params 는 cx/cy 를 버렸으므로 여기서 다시 읽어 붙인다."""
    cams = tr.load_camera_params()
    with open(tr.OPTIM_PARAMS_TXT) as f:
        lines = [l.strip() for l in f if l.strip()]
    header = lines[0].lstrip("#").split()
    for line in lines[1:]:
        row = dict(zip(header, line.split()))
        name = row["cam_name"]
        if name in cams:
            cams[name]["cx"] = float(row["cx"])
            cams[name]["cy"] = float(row["cy"])
    return cams


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    cam_params = load_camera_params_with_cxcy()
    bins = baseline.build_bins()
    cam_group = baseline.load_camera_groups(bins)
    gt = baseline.load_gt_sequences()
    video_map = baseline.load_video_map()
    downloaded_index = baseline.load_downloaded_video_index()
    group_candidates = baseline.build_group_candidates(gt, video_map, downloaded_index, cam_group)
    samples = baseline.stratified_sample(group_candidates, 500, seed=0)   # eval_translation_gigahands.py 와 동일
    print(f"총 샘플: {len(samples)} (baseline/eval_translation 과 동일해야 함: 1500)")

    by_video = defaultdict(list)
    for s in samples:
        by_video[s[3]].append(s)

    os.chdir(HAND_DEMO_ROOT)
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    import rgb_predictor
    from rgb_predictor import AnyHandPredictor

    ORIG = rgb_predictor._cam_crop_to_full
    captured = []

    def cap(pred_cam, box_center, box_size, img_size, focal_length):
        captured.append((
            pred_cam.detach().cpu().clone(), box_center.detach().cpu().clone(),
            box_size.detach().cpu().clone(), img_size.detach().cpu().clone(),
        ))
        return ORIG(pred_cam, box_center, box_size, img_size, focal_length)

    rgb_predictor._cam_crop_to_full = cap

    t0 = time.time()
    predictor = AnyHandPredictor(backend="wilor")
    print(f"로드 시간: {time.time()-t0:.1f}s")

    def cam_t_from_raw(raw, focal_px):
        pc, bc, bs, isz = raw
        ct = ORIG(pc.unsqueeze(0), bc.unsqueeze(0), bs.reshape(1), isz, focal_px)
        return ct[0].numpy().astype(np.float64)

    rows = []
    t_start = time.time()
    n_videos = len(by_video)
    for vi, (video_path, vsamples) in enumerate(by_video.items()):
        frame_indices = [s[4] for s in vsamples]
        frames = baseline.extract_frames(video_path, frame_indices)

        for (scene, seq_int, cam_id, _vp, fidx, group) in vsamples:
            entry = gt[(scene, seq_int)]
            gt_left = entry["left"].get(fidx)
            gt_right = entry["right"].get(fidx)
            if gt_left is None and gt_right is None:
                continue
            frame = frames.get(fidx)
            if frame is None:
                continue
            cam = cam_params.get(cam_id)
            if cam is None:
                continue

            captured.clear()
            try:
                hands = predictor.predict(frame)
            except Exception as e:
                print(f"  [warn] predict 실패 {scene}/{seq_int}/{cam_id}/{fidx}: {e}")
                continue

            raw_list = []
            for pc, bc, bs, isz in captured:
                B = pc.shape[0]
                for i in range(B):
                    raw_list.append((pc[i], bc[i], bs[i] if bs.dim() else bs, isz))
            assert len(raw_list) == len(hands), f"캡처 불일치 @ {scene}/{seq_int}/{cam_id}/{fidx}"

            gt_focal_px = (cam["fx"] + cam["fy"]) / 2.0
            img_width_px = float(raw_list[0][3][0].item()) if raw_list else cam["width"]
            hardcoded_focal_px = tr.focal_mm_to_px(tr.HARDCODED_FOCAL_MM, img_width_px)

            best = {"left": (None, None), "right": (None, None)}
            for h, raw in zip(hands, raw_list):
                side = "right" if h.is_right else "left"
                cur_h, _ = best[side]
                if cur_h is None or h.score > cur_h.score:
                    best[side] = (h, raw)

            for side, gt_arr in (("left", gt_left), ("right", gt_right)):
                det, raw = best[side]
                if gt_arr is None or det is None:
                    continue
                pc, bc, bs, isz = raw
                wrist_offset = det.keypoints_3d[WRIST].astype(np.float64)

                cam_t_gt = cam_t_from_raw(raw, gt_focal_px)
                cam_t_hard = cam_t_from_raw(raw, hardcoded_focal_px)
                pred_wrist_gt = cam_t_gt + wrist_offset
                pred_wrist_hard = cam_t_hard + wrist_offset

                gt_wrist_world = gt_arr[WRIST]
                gt_wrist_cam = tr.world_to_cam(cam, gt_wrist_world)

                # bbox 이론치: GT 21관절 전체를 이 카메라 intrinsic 으로 투영 후 tight bbox,
                # 그 위에 predictor 와 동일한 rescale_factor 만 적용(BBOX_SHAPE 미설정이라
                # expand_to_aspect_ratio 는 no-op 임을 vitdet_dataset.py 로 확인함).
                proj = project_points(cam, gt_arr)   # (21,2)
                gt_box_w = float(proj[:, 0].max() - proj[:, 0].min())
                gt_box_h = float(proj[:, 1].max() - proj[:, 1].min())
                theoretical_box_size = RESCALE_FACTOR * max(gt_box_w, gt_box_h)

                rows.append(dict(
                    scene=scene, seq=seq_int, cam=cam_id, frame=fidx, group=group, side=side,
                    is_right=bool(det.is_right), score=float(det.score),
                    kp3d_wrist_x=float(wrist_offset[0]), kp3d_wrist_y=float(wrist_offset[1]), kp3d_wrist_z=float(wrist_offset[2]),
                    pred_cam_s=float(pc[0]), pred_cam_tx=float(pc[1]), pred_cam_ty=float(pc[2]),
                    box_center_x=float(bc[0]), box_center_y=float(bc[1]),
                    box_size=float(bs.reshape(-1)[0]), img_width=float(isz[0]), img_height=float(isz[1]),
                    gt_focal_px=gt_focal_px, hardcoded_focal_px=hardcoded_focal_px,
                    cam_t_gt_x=cam_t_gt[0], cam_t_gt_y=cam_t_gt[1], cam_t_gt_z=cam_t_gt[2],
                    cam_t_hard_x=cam_t_hard[0], cam_t_hard_y=cam_t_hard[1], cam_t_hard_z=cam_t_hard[2],
                    pred_wrist_gt_x=pred_wrist_gt[0], pred_wrist_gt_y=pred_wrist_gt[1], pred_wrist_gt_z=pred_wrist_gt[2],
                    pred_wrist_hard_x=pred_wrist_hard[0], pred_wrist_hard_y=pred_wrist_hard[1], pred_wrist_hard_z=pred_wrist_hard[2],
                    gt_wrist_cam_x=gt_wrist_cam[0], gt_wrist_cam_y=gt_wrist_cam[1], gt_wrist_cam_z=gt_wrist_cam[2],
                    # 부호 있는 오차 (예측 - GT) — signed, C1 용
                    signed_z_err_gt_mm=(pred_wrist_gt[2] - gt_wrist_cam[2]) * 1000.0,
                    signed_z_err_hard_mm=(pred_wrist_hard[2] - gt_wrist_cam[2]) * 1000.0,
                    # 대조용 unsigned 재계산 (기존 eval_translation_gigahands.csv 와 100% 일치해야 함)
                    abs_error_gt_focal_mm=float(np.linalg.norm(pred_wrist_gt - gt_wrist_cam) * 1000.0),
                    abs_error_hardcoded_mm=float(np.linalg.norm(pred_wrist_hard - gt_wrist_cam) * 1000.0),
                    z_error_gt_focal_mm=float(abs(pred_wrist_gt[2] - gt_wrist_cam[2]) * 1000.0),
                    z_error_hardcoded_mm=float(abs(pred_wrist_hard[2] - gt_wrist_cam[2]) * 1000.0),
                    # A(4)용: wrist offset 미보정 버전 (cam_t 그대로 vs GT)
                    abs_error_gt_focal_uncorrected_mm=float(np.linalg.norm(cam_t_gt - gt_wrist_cam) * 1000.0),
                    abs_error_hardcoded_uncorrected_mm=float(np.linalg.norm(cam_t_hard - gt_wrist_cam) * 1000.0),
                    # C2/C3 용
                    gt_cam_distance_mm=float(np.linalg.norm(gt_wrist_cam) * 1000.0),
                    theoretical_box_size=theoretical_box_size,
                    bbox_ratio=float(bs.reshape(-1)[0]) / theoretical_box_size if theoretical_box_size > 0 else None,
                ))

        if (vi + 1) % 40 == 0 or (vi + 1) == n_videos:
            print(f"  video {vi+1}/{n_videos} ({time.time()-t_start:.0f}s elapsed, {len(rows)}행)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[저장] {OUT_CSV} ({len(rows)}행)")


if __name__ == "__main__":
    main()
