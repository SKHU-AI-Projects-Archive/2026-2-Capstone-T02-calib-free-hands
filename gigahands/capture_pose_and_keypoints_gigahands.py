"""
coverage-aware 지표(MPJPE-p, EPE-p, GO-p)에 필요한 raw 출력 — 예측 21관절
keypoints_3d 전체와 mano_pose(48, 앞 3=global orient axis-angle) — 을 기존
1,846개 매칭 인스턴스에 대해 캐싱한다.

기존 스크립트들(eval_translation_gigahands.py 등)은 이 값들을 저장한 적이
없어서(PA-MPJPE/절대오차 스칼라만 남김) 불가피하게 WiLoR forward pass 를
1회 다시 실행한다 — 단, 이미 찾아낸(검출 성공한) 1,846개 인스턴스에 대해서만
돌리는 것이지, 새로운 검출/크롤링을 하는 게 아니다.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 capture_pose_and_keypoints_gigahands.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
HAND_DEMO_ROOT = Path("/home/juson/project/hand-demo")
OUT_DIR = HAND_DEMO_ROOT / "out" / "gigahands"

sys.path.insert(0, str(BASE_DIR))
import eval_pa_mpjpe_gigahands as baseline  # noqa: E402

OUT_CSV = OUT_DIR / "gigahands_pose_keypoints_cache.csv"


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    bins = baseline.build_bins()
    cam_group = baseline.load_camera_groups(bins)
    gt = baseline.load_gt_sequences()
    video_map = baseline.load_video_map()
    downloaded_index = baseline.load_downloaded_video_index()
    group_candidates = baseline.build_group_candidates(gt, video_map, downloaded_index, cam_group)
    samples = baseline.stratified_sample(group_candidates, 500, seed=0)   # 기존과 동일
    print(f"총 샘플: {len(samples)} (baseline 과 동일해야 함: 1500)")

    by_video = defaultdict(list)
    for s in samples:
        by_video[s[3]].append(s)

    os.chdir(HAND_DEMO_ROOT)
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    from rgb_predictor import AnyHandPredictor

    t0 = time.time()
    predictor = AnyHandPredictor(backend="wilor")
    print(f"로드 시간: {time.time()-t0:.1f}s")

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

            try:
                hands = predictor.predict(frame)
            except Exception as e:
                print(f"  [warn] predict 실패 {scene}/{seq_int}/{cam_id}/{fidx}: {e}")
                continue

            best = {"left": None, "right": None}
            for h in hands:
                side = "right" if h.is_right else "left"
                if best[side] is None or h.score > best[side].score:
                    best[side] = h

            for side, gt_arr in (("left", gt_left), ("right", gt_right)):
                det = best[side]
                if gt_arr is None or det is None:
                    continue
                row = dict(scene=scene, seq=seq_int, cam=cam_id, frame=fidx, group=group, side=side,
                           score=float(det.score))
                kp3d = det.keypoints_3d.astype(np.float64)   # (21,3) root-relative
                for j in range(21):
                    row[f"kp3d_{j}_x"] = kp3d[j, 0]
                    row[f"kp3d_{j}_y"] = kp3d[j, 1]
                    row[f"kp3d_{j}_z"] = kp3d[j, 2]
                mp = det.mano_pose.astype(np.float64)   # (48,) axis-angle, [0:3]=global_orient
                for k in range(48):
                    row[f"mano_pose_{k}"] = mp[k]
                row["cam_t_x"], row["cam_t_y"], row["cam_t_z"] = det.cam_t.astype(np.float64)
                rows.append(row)

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
