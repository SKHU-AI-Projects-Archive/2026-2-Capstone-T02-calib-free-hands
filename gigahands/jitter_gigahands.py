"""
Jitter(시간적 흔들림) — 화각 그룹당 대표 시퀀스에서 연속 프레임을 디코드해 WiLoR 를 돌리고,
예측 keypoint 위치의 2차 미분(가속도)을 계산한다. GT 불필요(예측끼리만 비교하는 지표라
"-p" 보정 대상도 아님) — 원본 영상에서 바로 연속 프레임을 뽑으므로 GT의 sparse
chosen_frames 제약과 무관하게 항상 연속.

대표 시퀀스(기존 표본에서 인스턴스가 가장 많이 몰린 것 기준):
  25mm: p005-noodle-fastfood seq9  cam brics-odroid-011_cam0
  26mm: p005-noodle-fastfood seq9  cam brics-odroid-001_cam0
  27mm: p005-tea             seq17 cam brics-odroid-009_cam1

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 jitter_gigahands.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
HAND_DEMO_ROOT = Path("/home/juson/project/hand-demo")
OUT_DIR = HAND_DEMO_ROOT / "out" / "gigahands"

sys.path.insert(0, str(BASE_DIR))
import eval_pa_mpjpe_gigahands as baseline  # noqa: E402

N_FRAMES = 30
TARGETS = [
    ("25mm", "p005-noodle-fastfood", 9, "brics-odroid-011_cam0"),
    ("26mm", "p005-noodle-fastfood", 9, "brics-odroid-001_cam0"),
    ("27mm", "p005-tea", 17, "brics-odroid-009_cam1"),
]

OUT_CSV = OUT_DIR / "gigahands_jitter.csv"


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    video_map = baseline.load_video_map()
    downloaded_index = baseline.load_downloaded_video_index()

    os.chdir(HAND_DEMO_ROOT)
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    from rgb_predictor import AnyHandPredictor

    t0 = time.time()
    predictor = AnyHandPredictor(backend="wilor")
    print(f"로드 시간: {time.time()-t0:.1f}s")

    rows = []
    for group, scene, seq_int, cam_id in TARGETS:
        cams = video_map.get((scene, seq_int))
        if cams is None or cam_id not in cams:
            print(f"  [skip] {group} {scene}/{seq_int}/{cam_id}: video_map 에 없음")
            continue
        basename = cams[cam_id].split("/")[-1]
        video_path = downloaded_index.get(basename)
        if video_path is None:
            print(f"  [skip] {group} {scene}/{seq_int}/{cam_id}: 영상 파일 없음")
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_grab = min(N_FRAMES, n_total)
        print(f"\n[{group}] {scene}/seq{seq_int}/{cam_id}  fps={fps:.2f}  전체프레임={n_total}  뽑을프레임={n_grab}")

        wrist_left, wrist_right = [], []
        for i in range(n_grab):
            ok, frame = cap.read()
            if not ok:
                print(f"    frame {i}: 디코드 실패 — 여기서 연속구간 끊김, 중단")
                break
            try:
                hands = predictor.predict(frame)
            except Exception as e:
                print(f"    frame {i}: predict 실패({e}) — 이 프레임 skip(연속성 깨짐 처리)")
                wrist_left.append(None); wrist_right.append(None)
                continue

            best_l = best_r = None
            for h in hands:
                if h.is_right:
                    if best_r is None or h.score > best_r.score:
                        best_r = h
                else:
                    if best_l is None or h.score > best_l.score:
                        best_l = h
            wrist_left.append((best_l.keypoints_3d[0] + best_l.cam_t) if best_l else None)
            wrist_right.append((best_r.keypoints_3d[0] + best_r.cam_t) if best_r else None)
        cap.release()

        for side, seq in (("left", wrist_left), ("right", wrist_right)):
            # 연속으로 검출된 run만 사용 (None 이 끼면 그 지점에서 run 이 끊김)
            run = []
            runs = []
            for v in seq:
                if v is not None:
                    run.append(v)
                else:
                    if len(run) >= 3:
                        runs.append(run)
                    run = []
            if len(run) >= 3:
                runs.append(run)

            for run in runs:
                arr = np.stack(run) * 1000.0  # m -> mm
                vel = np.diff(arr, axis=0) * fps            # mm/s
                acc = np.diff(vel, axis=0) * fps             # mm/s^2
                acc_mag = np.linalg.norm(acc, axis=1)
                rows.append(dict(
                    group=group, scene=scene, seq=seq_int, cam=cam_id, side=side,
                    run_len=len(run), fps=fps,
                    jitter_mean_mm_s2=float(acc_mag.mean()),
                    jitter_median_mm_s2=float(np.median(acc_mag)),
                    jitter_std_mm_s2=float(acc_mag.std()),
                ))
                print(f"    {side}: run_len={len(run)}  jitter(mean accel)={acc_mag.mean():.1f} mm/s^2")

    with open(OUT_CSV, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n[저장] {OUT_CSV} ({len(rows)}행)")


if __name__ == "__main__":
    main()
