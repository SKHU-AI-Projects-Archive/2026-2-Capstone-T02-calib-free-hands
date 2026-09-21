"""
GigaHands 고유 프레임(1,150개, eval_translation_gigahands.csv의 1,846 인스턴스에서
중복 프레임 제거)에 대해 GeoCalib으로 focal_x(px)를 추정한다.

WiLoR 추론은 다시 하지 않는다 — 영상에서 프레임만 디코드(baseline.extract_frames 재사용,
ffmpeg 재추출·재다운로드 없음)해서 GeoCalib(단일 이미지 카메라 캘리브레이션 모델)에
바로 넣는다. 프레임당 1회만 계산(같은 프레임의 좌/우 손은 같은 focal_x_geocalib 공유).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 run_geocalib_gigahands.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent   # 이 스크립트 자신의 위치 (hand-demo/gigahands/)
OUT_DIR = Path("/home/juson/project/hand-demo/out/gigahands")
sys.path.insert(0, str(BASE_DIR))
import eval_pa_mpjpe_gigahands as baseline  # noqa: E402  (load_video_map, load_downloaded_video_index, extract_frames 재사용)

from geocalib import GeoCalib  # noqa: E402

OUT_CSV = OUT_DIR / "geocalib_focal_estimates.csv"
SEQ_STATS_CSV = OUT_DIR / "geocalib_focal_seq_stats.csv"


def compute_seq_stats(rows):
    """(scene, seq, cam) = 영상(카메라 1대의 녹화 파일) 하나 단위로 GeoCalib 추정 focal_x 의
    평균/표준편차/min/max 를 계산한다. (scene, seq) 만으로 묶지 않는 이유: 같은 take 라도
    카메라마다 물리적으로 다른 렌즈(실제 초점거리)를 쓰므로, scene+seq 로만 묶으면 서로 다른
    실제 초점거리를 가진 카메라들의 추정치가 섞여 평균이 무의미해진다 — by_video 그룹핑(영상
    파일 1개 = (scene,seq,cam) 1개) 관례와 동일하게 맞춘다.

    이번 실험에서 실제로 GeoCalib 을 돌린 것은 (25/26/27mm 계층 샘플링으로 뽑힌) 일부 프레임
    뿐이므로, 이 평균은 "영상 전체 프레임 평균"이 아니라 "이번에 샘플링된 프레임들의 평균"이다.
    """
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[(r["scene"], r["seq"], r["cam"])].append(r["focal_x_geocalib"])

    seq_rows = []
    for (scene, seq_int, cam_id), vals in by_seq.items():
        arr = np.array(vals, dtype=np.float64)
        seq_rows.append(dict(
            scene=scene, seq=seq_int, cam=cam_id, n_frames=len(arr),
            focal_x_geocalib_mean=float(arr.mean()),
            focal_x_geocalib_std=float(arr.std()),
            focal_x_geocalib_min=float(arr.min()),
            focal_x_geocalib_max=float(arr.max()),
        ))

    with open(SEQ_STATS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        w.writeheader()
        for r in seq_rows:
            w.writerow(r)
    print(f"[저장] {SEQ_STATS_CSV} ({len(seq_rows)}개 영상)")

    stds = np.array([r["focal_x_geocalib_std"] for r in seq_rows])
    n_multi = int((np.array([r["n_frames"] for r in seq_rows]) >= 2).sum())
    print(f"  영상(카메라) {len(seq_rows)}개 중 프레임 2개 이상(=std 의미 있음): {n_multi}개")
    print(f"  영상별 focal_x_geocalib 표준편차: 평균 {stds.mean():.1f}px, 중앙값 {np.median(stds):.1f}px, "
          f"최대 {stds.max():.1f}px")
    return seq_rows


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if "--seq-stats-only" in sys.argv:
        # GeoCalib 을 다시 돌리지 않고, 이미 저장된 geocalib_focal_estimates.csv 로만
        # (scene,seq,cam) 별 평균/지터 통계를 재계산한다.
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        compute_seq_stats(rows)
        return

    trans = pd.read_csv(OUT_DIR / "eval_translation_gigahands.csv")
    uniq = trans[["scene", "seq", "cam", "frame"]].drop_duplicates().reset_index(drop=True)
    print(f"전체 인스턴스 {len(trans)}개 -> 고유 프레임 {len(uniq)}개")

    video_map = baseline.load_video_map()
    downloaded_index = baseline.load_downloaded_video_index()

    # (scene, seq, cam) -> video_path 매핑, baseline 이 쓴 것과 동일한 경로 해석
    by_video = defaultdict(list)   # video_path -> [(scene, seq, cam, frame), ...]
    unresolved = []
    for _, row in uniq.iterrows():
        scene, seq_int, cam_id, fidx = row["scene"], int(row["seq"]), row["cam"], int(row["frame"])
        cams = video_map.get((scene, seq_int))
        if cams is None or cam_id not in cams:
            unresolved.append((scene, seq_int, cam_id, fidx))
            continue
        rel_path = cams[cam_id]
        basename = rel_path.split("/")[-1]
        video_path = downloaded_index.get(basename)
        if video_path is None:
            unresolved.append((scene, seq_int, cam_id, fidx))
            continue
        by_video[video_path].append((scene, seq_int, cam_id, fidx))

    print(f"영상 경로 해석 실패: {len(unresolved)}개 (0이어야 정상 — baseline과 동일 소스)")
    print(f"디코드할 고유 영상 수: {len(by_video)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = GeoCalib(weights="pinhole").to(device)
    model.eval()

    rows = []
    n_bad = 0
    t_start = time.time()
    n_videos = len(by_video)
    for vi, (video_path, items) in enumerate(by_video.items()):
        frame_indices = [it[3] for it in items]
        frames = baseline.extract_frames(video_path, frame_indices)
        for (scene, seq_int, cam_id, fidx) in items:
            frame_bgr = frames.get(fidx)
            if frame_bgr is None:
                n_bad += 1
                continue
            rgb = frame_bgr[..., ::-1].copy()          # BGR -> RGB
            img = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0  # (3,H,W) in [0,1]

            t0 = time.time()
            with torch.no_grad():
                out = model.calibrate(img)
            dt = time.time() - t0

            f = out["camera"].f[0].detach().cpu().numpy()   # (fx, fy)
            fx, fy = float(f[0]), float(f[1])
            w, h = out["camera"].size[0].detach().cpu().numpy()

            rows.append(dict(
                scene=scene, seq=seq_int, cam=cam_id, frame=fidx,
                focal_x_geocalib=fx, focal_y_geocalib=fy,
                img_width=float(w), img_height=float(h),
                infer_time_s=dt,
            ))

        if (vi + 1) % 20 == 0 or (vi + 1) == n_videos:
            elapsed = time.time() - t_start
            print(f"  video {vi+1}/{n_videos} ({elapsed:.1f}s elapsed, {len(rows)}개 프레임 처리)")

    total_time = time.time() - t_start
    print(f"\n총 소요시간: {total_time:.1f}s ({len(rows)}개 프레임, "
          f"프레임당 평균 {np.mean([r['infer_time_s'] for r in rows])*1000:.1f}ms 추론)")
    print(f"디코드 실패 프레임: {n_bad}개")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[저장] {OUT_CSV} ({len(rows)}행)")

    # ---- (scene,seq,cam) 단위 평균/지터 통계 ----
    compute_seq_stats(rows)

    # ---- 이상치 점검 ----
    fx_vals = np.array([r["focal_x_geocalib"] for r in rows])
    widths = np.array([r["img_width"] for r in rows])
    bad_mask = (fx_vals <= 0) | (fx_vals > widths * 3) | ~np.isfinite(fx_vals)
    print(f"\n이상치(음수/0/이미지폭*3 초과/NaN·Inf): {bad_mask.sum()}개")
    if bad_mask.sum() > 0:
        print(pd.DataFrame(rows)[bad_mask.tolist() + [False]*(0)].to_string() if False else "")
        bad_rows = [r for r, b in zip(rows, bad_mask) if b]
        for r in bad_rows[:20]:
            print(" ", r)


if __name__ == "__main__":
    main()
