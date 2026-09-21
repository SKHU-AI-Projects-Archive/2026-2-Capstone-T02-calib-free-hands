"""
FAcc / Precision / F1 계산에 필요한 "프레임당 YOLO 검출 전체 목록"(매칭 안 된 여분·false
positive 포함)을 캐싱한다. 기존 baseline/translation 스크립트는 손별 최고점수 1개만 남기고
나머지 검출은 버렸기 때문에 이 정보가 어디에도 없다 — WiLoR 전체가 아니라 검출(stage 1,
YOLO) 단계만 다시 돌린다(크롭·MANO·pose 추정 없음, 훨씬 가벼움).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 detect_only_gigahands.py
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

OUT_CSV = OUT_DIR / "gigahands_detection_frames.csv"


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    bins = baseline.build_bins()
    cam_group = baseline.load_camera_groups(bins)
    gt = baseline.load_gt_sequences()
    video_map = baseline.load_video_map()
    downloaded_index = baseline.load_downloaded_video_index()
    group_candidates = baseline.build_group_candidates(gt, video_map, downloaded_index, cam_group)
    samples = baseline.stratified_sample(group_candidates, 500, seed=0)
    print(f"총 샘플(=프레임): {len(samples)} (baseline 과 동일해야 함: 1500)")

    by_video = defaultdict(list)
    for s in samples:
        by_video[s[3]].append(s)

    os.chdir(HAND_DEMO_ROOT)
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    from rgb_predictor import AnyHandPredictor

    t0 = time.time()
    predictor = AnyHandPredictor(backend="wilor")   # 검출기(YOLO)는 backend 무관하게 항상 로드됨
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
                boxes, is_right, scores = predictor.detect_hands(frame)   # stage 1 만
            except Exception as e:
                print(f"  [warn] detect_hands 실패 {scene}/{seq_int}/{cam_id}/{fidx}: {e}")
                continue

            n_det_left = int(np.sum(~is_right))
            n_det_right = int(np.sum(is_right))
            n_gt = int(gt_left is not None) + int(gt_right is not None)
            n_det = len(boxes)

            # 손별 최고점수 후보 = 기존 파이프라인이 채택한 TP 후보와 동일 로직
            def top_score(mask):
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return None
                return idx[np.argmax(scores[idx])]

            top_left = top_score(~is_right)
            top_right = top_score(is_right)

            # TP: GT 있고 해당 side 최고점수 후보 존재 -> 1개
            tp_left = int(gt_left is not None and top_left is not None)
            tp_right = int(gt_right is not None and top_right is not None)
            n_tp = tp_left + tp_right

            # FP: 그 side 에 GT 가 없는데 검출된 것 전부 + GT 는 있지만 최고점수 아닌 나머지 후보
            fp_left = n_det_left - (1 if (gt_left is not None and top_left is not None) else 0) \
                if gt_left is None else max(n_det_left - 1, 0)
            fp_right = n_det_right - (1 if (gt_right is not None and top_right is not None) else 0) \
                if gt_right is None else max(n_det_right - 1, 0)
            n_fp = fp_left + fp_right

            rows.append(dict(
                scene=scene, seq=seq_int, cam=cam_id, frame=fidx, group=group,
                n_gt=n_gt, n_det=n_det, n_det_left=n_det_left, n_det_right=n_det_right,
                tp_left=tp_left, tp_right=tp_right, n_tp=n_tp, n_fp=n_fp,
                frame_correct=int(n_det == n_gt),
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
