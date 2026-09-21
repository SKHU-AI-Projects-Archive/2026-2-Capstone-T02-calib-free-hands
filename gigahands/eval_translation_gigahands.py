"""
Absolute translation(cam_t) 오차 + 양손 간 상대거리 오차 평가 (hand-demo 파이프라인,
AnyHandPredictor/WiLoR backend) — GigaHands GT 대비, focal_length=GT카메라 실측값 vs
focal_length=24mm(35mm 환산) 하드코딩 두 가지를 나란히 비교.

eval_pa_mpjpe_gigahands.py 의 PA-MPJPE 는 Procrustes 정렬(회전+이동+스케일 제거) 후
오차라서 focal_length 오차의 영향(주로 cam_t, 특히 깊이 방향)을 구조적으로 반영하지
못한다 — 이 스크립트가 그 빈틈을 메운다.

재사용 (새로 짜지 않음):
  - eval_pa_mpjpe_gigahands.py 의 샘플링/후보구성/GT로딩/FOV그룹 함수 전부를 그대로
    import 해서 쓴다 — 같은 seed=0 이라 baseline 과 완전히 동일한 1,500프레임/2,804
    GT 인스턴스 표본이 재현된다 (실행 후 교차확인).
  - rgb_predictor._cam_crop_to_full 를 그대로 재사용하되, focal_length 만 바꿔 인스턴스당
    두 번 호출한다. WiLoR forward pass(추론) 자체는 프레임당 정확히 1회만 실행 —
    _cam_crop_to_full 를 몽키패치해서 그 순간의 raw 입력(pred_cam, box_center,
    box_size, img_size)을 가로채 저장해두고, 이후 GPU 없이 순수 파이썬으로 두 번째
    호출을 반복한다 (baseline 은 이 raw 값을 저장해두지 않았으므로, 이번에 1회만
    다시 추론해서 캡처한다 — MANO pose/shape 재추정이 목적이 아니라 이 raw 값을
    얻기 위한 부수효과일 뿐, "추론 중복 실행"이 아니다).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 eval_translation_gigahands.py --target-per-group 500
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent   # 이 스크립트 자신의 위치 (hand-demo/gigahands/)
HAND_DEMO_ROOT = Path("/home/juson/project/hand-demo")

# 원본 GigaHands 데이터셋은 hand-demo 프로젝트에 넣지 않고 /public/data 에 그대로 둔다
DATA_DIR = Path("/public/data/gigahands")
OUT_DIR = HAND_DEMO_ROOT / "out" / "gigahands"

sys.path.insert(0, str(BASE_DIR))
import eval_pa_mpjpe_gigahands as baseline  # noqa: E402  — 샘플링/GT/FOV 로직 재사용

OPTIM_PARAMS_TXT = DATA_DIR / "gigahands_optim_params_sample.txt"
WRIST = 0
SENSOR_WIDTH_MM = 36.0        # fov_lib.py 와 동일한 35mm-equivalent(풀프레임) 가정
HARDCODED_FOCAL_MM = 24.0


# ---------------------------------------------------------------------------
# mm -> px 초점거리 환산 (프레임의 실제 해상도 기준, 절대 하드코딩하지 않음)
# ---------------------------------------------------------------------------

def focal_mm_to_px(focal_mm: float, image_width_px: float, sensor_width_mm: float = SENSOR_WIDTH_MM) -> float:
    """35mm(풀프레임) 환산 초점거리[mm] -> 이 프레임의 실제 픽셀 폭 기준 초점거리[px]."""
    return focal_mm / sensor_width_mm * image_width_px


# ---------------------------------------------------------------------------
# GigaHands 카메라 GT (내부·외부 파라미터) — gigahands_optim_params_sample.txt
#   컬럼: cam_id width height fx fy cx cy k1 k2 p1 p2 cam_name qvecw qvecx qvecy qvecz tvecx tvecy tvecz
#   qvec/tvec 는 COLMAP 관례: X_cam = R(qvec) @ X_world + tvec
# ---------------------------------------------------------------------------

def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def load_camera_params():
    """cam_name -> dict(fx, fy, width, height, R (3,3), t (3,))."""
    with open(OPTIM_PARAMS_TXT) as f:
        lines = [l.strip() for l in f if l.strip()]
    header = lines[0].lstrip("#").split()
    cams = {}
    for line in lines[1:]:
        row = dict(zip(header, line.split()))
        R = quat_to_rotmat(float(row["qvecw"]), float(row["qvecx"]), float(row["qvecy"]), float(row["qvecz"]))
        t = np.array([float(row["tvecx"]), float(row["tvecy"]), float(row["tvecz"])], dtype=np.float64)
        cams[row["cam_name"]] = dict(
            fx=float(row["fx"]), fy=float(row["fy"]),
            width=float(row["width"]), height=float(row["height"]),
            R=R, t=t,
        )
    return cams


def world_to_cam(cam: dict, p_world: np.ndarray) -> np.ndarray:
    return cam["R"] @ np.asarray(p_world, dtype=np.float64) + cam["t"]


def verify_extrinsic_convention(cam: dict, p_world: np.ndarray, expected_z_range=(0.05, 5.0)) -> np.ndarray:
    """world_to_cam 변환의 방향(월드->카메라 vs 카메라->월드)이 맞는지 z(깊이) 부호/범위로
    경험적으로 검증. 맞지 않으면 역변환(카메라->월드로 가정하고 뒤집기)을 시도하고,
    그래도 안 맞으면 즉시 에러를 낸다 — 잘못된 컨벤션으로 조용히 진행하지 않는다."""
    p_cam = world_to_cam(cam, p_world)
    lo, hi = expected_z_range
    if lo <= p_cam[2] <= hi:
        return p_cam
    # 반대 컨벤션 시도: 저장된 (R,t) 가 camera->world 라면 world->cam 은 R^T @ (p - t)
    p_cam_alt = cam["R"].T @ (np.asarray(p_world, dtype=np.float64) - cam["t"])
    if lo <= p_cam_alt[2] <= hi:
        raise RuntimeError(
            f"[극성] optim_params.txt 의 qvec/tvec 는 world->cam 이 아니라 반대 방향(cam->world)"
            f"으로 보임 (z={p_cam[2]:.3f} 대신 {p_cam_alt[2]:.3f}). world_to_cam()/verify_extrinsic_convention()"
            f"의 부호를 R^T @ (p - t) 로 뒤집어야 함."
        )
    raise RuntimeError(
        f"[극성] 두 방향 모두 z 가 타당 범위 {expected_z_range} 밖 (z={p_cam[2]:.3f} / {p_cam_alt[2]:.3f})."
        f" 카메라 파라미터·GT 좌표 단위/컨벤션을 다시 확인해야 함."
    )


# ---------------------------------------------------------------------------
# GeoCalib_영상평균 — 프레임별 GeoCalib 추정치 대신, 같은 영상((scene,seq,cam) 하나 = 카메라
# 1대의 녹화 파일 하나) 안에서 샘플링된 프레임들의 평균 focal 하나를 그 영상 전체에 고정
# 적용했을 때의 절대 translation 오차. WiLoR 재추론 없음(GPU/torch 불필요) — 이미 캐시된
# raw 값(gigahands_raw_instances_cache.csv)과 시퀀스별 평균(geocalib_focal_seq_stats.csv,
# run_geocalib_gigahands.py --seq-stats-only 로 생성)만으로 _cam_crop_to_full 공식을
# 순수 파이썬으로 재계산한다.
# ---------------------------------------------------------------------------

def recompute_geocalib_seqavg():
    import csv as csv_mod

    raw_csv_path = OUT_DIR / "gigahands_raw_instances_cache.csv"
    seq_stats_path = OUT_DIR / "geocalib_focal_seq_stats.csv"
    trans_csv_path = OUT_DIR / "eval_translation_gigahands.csv"
    interhand_csv_path = OUT_DIR / "eval_interhand_distance_gigahands.csv"
    report_path = OUT_DIR / "geocalib_seqavg_comparison_report.md"

    with open(seq_stats_path, newline="") as f:
        seq_stats_rows = list(csv_mod.DictReader(f))
    seq_focal = {(r["scene"], r["seq"], r["cam"]): float(r["focal_x_geocalib_mean"]) for r in seq_stats_rows}
    print(f"시퀀스별 GeoCalib 평균 focal 로드: {len(seq_focal)}개 영상 ({seq_stats_path.name})")

    with open(raw_csv_path, newline="") as f:
        raw_rows = list(csv_mod.DictReader(f))
    print(f"raw 캐시 로드: {len(raw_rows)}행 ({raw_csv_path.name})")

    # ---- 인스턴스별 cam_t_geocalib_seqavg 재계산 (_cam_crop_to_full 공식과 동일, numpy) ----
    n_missing = 0
    for r in raw_rows:
        key = (r["scene"], r["seq"], r["cam"])
        focal = seq_focal.get(key)
        if focal is None:
            n_missing += 1
            for col in ("focal_geocalib_seqavg", "cam_t_geocalib_seqavg_x", "cam_t_geocalib_seqavg_y",
                        "cam_t_geocalib_seqavg_z", "pred_wrist_geocalib_seqavg_x", "pred_wrist_geocalib_seqavg_y",
                        "pred_wrist_geocalib_seqavg_z", "abs_error_geocalib_seqavg_mm",
                        "z_error_geocalib_seqavg_mm", "signed_z_err_geocalib_seqavg_mm"):
                r[col] = ""
            continue

        s = float(r["pred_cam_s"]); tx_crop = float(r["pred_cam_tx"]); ty_crop = float(r["pred_cam_ty"])
        cx_box = float(r["box_center_x"]); cy_box = float(r["box_center_y"])
        cx_img = float(r["img_width"]) * 0.5; cy_img = float(r["img_height"]) * 0.5
        box_size_val = float(r["box_size"])
        denom = s * box_size_val + 1e-9
        cam_t = np.array([
            tx_crop + 2.0 * (cx_box - cx_img) / denom,
            ty_crop + 2.0 * (cy_box - cy_img) / denom,
            2.0 * focal / denom,
        ], dtype=np.float64)

        wrist_offset = np.array([float(r["kp3d_wrist_x"]), float(r["kp3d_wrist_y"]), float(r["kp3d_wrist_z"])])
        pred_wrist = cam_t + wrist_offset
        gt_wrist_cam = np.array([float(r["gt_wrist_cam_x"]), float(r["gt_wrist_cam_y"]), float(r["gt_wrist_cam_z"])])

        r["focal_geocalib_seqavg"] = focal
        r["cam_t_geocalib_seqavg_x"], r["cam_t_geocalib_seqavg_y"], r["cam_t_geocalib_seqavg_z"] = cam_t.tolist()
        r["pred_wrist_geocalib_seqavg_x"], r["pred_wrist_geocalib_seqavg_y"], r["pred_wrist_geocalib_seqavg_z"] = pred_wrist.tolist()
        r["abs_error_geocalib_seqavg_mm"] = float(np.linalg.norm(pred_wrist - gt_wrist_cam) * 1000.0)
        r["z_error_geocalib_seqavg_mm"] = float(abs(pred_wrist[2] - gt_wrist_cam[2]) * 1000.0)
        r["signed_z_err_geocalib_seqavg_mm"] = float((pred_wrist[2] - gt_wrist_cam[2]) * 1000.0)

    print(f"시퀀스 평균 focal 매칭 실패: {n_missing}개 (0이어야 정상 — 같은 표본에서 나온 값이므로)")

    # ---- 수기(공식) 재현 검증 — rgb_predictor._cam_crop_to_full 로 5개 대조 ----
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    import rgb_predictor  # noqa: E402  (모델 로드 없이 순수 함수만 사용)
    checked = 0
    for r in raw_rows:
        if checked >= 5:
            break
        if r["focal_geocalib_seqavg"] == "":
            continue
        pc = torch.tensor([[float(r["pred_cam_s"]), float(r["pred_cam_tx"]), float(r["pred_cam_ty"])]])
        bc = torch.tensor([[float(r["box_center_x"]), float(r["box_center_y"])]])
        bs = torch.tensor([float(r["box_size"])])
        isz = torch.tensor([float(r["img_width"]), float(r["img_height"])])
        ct = rgb_predictor._cam_crop_to_full(pc, bc, bs, isz, float(r["focal_geocalib_seqavg"]))[0].numpy()
        expect = np.array([float(r["cam_t_geocalib_seqavg_x"]), float(r["cam_t_geocalib_seqavg_y"]),
                            float(r["cam_t_geocalib_seqavg_z"])])
        assert np.allclose(ct, expect, atol=1e-4), f"수기 재현 불일치: {ct} vs {expect}"
        checked += 1
    print(f"  수기(공식) 재현 검증 {checked}개 통과 (atol=1e-4)")

    # ---- raw 캐시 CSV 갱신(새 컬럼 추가) ----
    with open(raw_csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    print(f"[갱신] {raw_csv_path}")

    # ---- eval_translation_gigahands.csv 에 abs/z error 컬럼 병합 ----
    key_cols = ("scene", "seq", "cam", "frame", "side")
    seqavg_by_key = {
        tuple(r[k] for k in key_cols): (r["abs_error_geocalib_seqavg_mm"], r["z_error_geocalib_seqavg_mm"])
        for r in raw_rows
    }
    with open(trans_csv_path, newline="") as f:
        trans_rows = list(csv_mod.DictReader(f))
    n_matched = 0
    for r in trans_rows:
        vals = seqavg_by_key.get(tuple(r[k] for k in key_cols))
        if vals is None:
            r["abs_error_geocalib_seqavg_mm"] = r["z_error_geocalib_seqavg_mm"] = ""
            continue
        r["abs_error_geocalib_seqavg_mm"], r["z_error_geocalib_seqavg_mm"] = vals
        n_matched += 1
    print(f"eval_translation_gigahands.csv 병합: {n_matched}/{len(trans_rows)}행 매칭")
    with open(trans_csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        w.writeheader()
        for r in trans_rows:
            w.writerow(r)
    print(f"[갱신] {trans_csv_path}")

    # ---- 양손 상대거리(dx/dy/dz) 병합 — gt_vec = gt_wrist_cam_left - gt_wrist_cam_right
    # (world_to_cam 의 평행이동 t 가 좌우 차분에서 상쇄되므로 cam["R"]@(월드좌표 차) 와 동일) ----
    by_frame = defaultdict(dict)
    for r in raw_rows:
        if r["abs_error_geocalib_seqavg_mm"] == "":
            continue
        by_frame[(r["scene"], r["seq"], r["cam"], r["frame"])][r["side"]] = r

    interhand_seqavg = {}
    for key, sides in by_frame.items():
        if "left" not in sides or "right" not in sides:
            continue
        rl, rr = sides["left"], sides["right"]
        pred_l = np.array([float(rl[f"pred_wrist_geocalib_seqavg_{a}"]) for a in "xyz"])
        pred_r = np.array([float(rr[f"pred_wrist_geocalib_seqavg_{a}"]) for a in "xyz"])
        gt_l = np.array([float(rl[f"gt_wrist_cam_{a}"]) for a in "xyz"])
        gt_r = np.array([float(rr[f"gt_wrist_cam_{a}"]) for a in "xyz"])
        d = ((pred_l - pred_r) - (gt_l - gt_r)) * 1000.0
        interhand_seqavg[key] = (float(d[0]), float(d[1]), float(d[2]), float(np.linalg.norm(d)))

    with open(interhand_csv_path, newline="") as f:
        interhand_rows = list(csv_mod.DictReader(f))
    n_ih_matched = 0
    for r in interhand_rows:
        vals = interhand_seqavg.get((r["scene"], r["seq"], r["cam"], r["frame"]))
        if vals is None:
            r["dx_err_geocalib_seqavg_mm"] = r["dy_err_geocalib_seqavg_mm"] = ""
            r["dz_err_geocalib_seqavg_mm"] = r["euclid_err_geocalib_seqavg_mm"] = ""
            continue
        (r["dx_err_geocalib_seqavg_mm"], r["dy_err_geocalib_seqavg_mm"],
         r["dz_err_geocalib_seqavg_mm"], r["euclid_err_geocalib_seqavg_mm"]) = vals
        n_ih_matched += 1
    print(f"eval_interhand_distance_gigahands.csv 병합: {n_ih_matched}/{len(interhand_rows)}행 매칭")
    with open(interhand_csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(interhand_rows[0].keys()))
        w.writeheader()
        for r in interhand_rows:
            w.writerow(r)
    print(f"[갱신] {interhand_csv_path}")

    # ---- 보고서용 통계 집계 ----
    def arr(rows, col):
        return np.array([float(r[col]) for r in rows if r.get(col, "") != ""], dtype=np.float64)

    matched_raw = [r for r in raw_rows if r["abs_error_geocalib_seqavg_mm"] != ""]
    abs_err = arr(matched_raw, "abs_error_geocalib_seqavg_mm")
    z_err = arr(matched_raw, "z_error_geocalib_seqavg_mm")
    focal_vals = arr(matched_raw, "focal_geocalib_seqavg")
    gt_focal_vals = arr(matched_raw, "gt_focal_px")
    rel_err_pct = (focal_vals - gt_focal_vals) / gt_focal_vals * 100.0

    abs_err_frame = arr(raw_rows, "abs_error_geocalib_mm")   # 기존 프레임별 GeoCalib 조건 (비교용)
    z_err_frame = arr(raw_rows, "z_error_geocalib_mm")

    by_group = {}
    for g in ("25mm", "26mm", "27mm"):
        rows_g = [r for r in matched_raw if r["group"] == g]
        by_group[g] = dict(
            n=len(rows_g),
            abs_mean=float(np.mean([float(r["abs_error_geocalib_seqavg_mm"]) for r in rows_g])) if rows_g else float("nan"),
        )

    euclid_seqavg = arr(interhand_rows, "euclid_err_geocalib_seqavg_mm")
    dx_seqavg = arr(interhand_rows, "dx_err_geocalib_seqavg_mm")
    dy_seqavg = arr(interhand_rows, "dy_err_geocalib_seqavg_mm")
    dz_seqavg = arr(interhand_rows, "dz_err_geocalib_seqavg_mm")

    n_multi = sum(1 for r in seq_stats_rows if int(r["n_frames"]) >= 2)
    stds = np.array([float(r["focal_x_geocalib_std"]) for r in seq_stats_rows if int(r["n_frames"]) >= 2])

    report = f"""# GeoCalib_영상평균 — 프레임별 지터를 영상 단위 평균으로 줄였을 때의 절대오차

같은 영상((scene,seq,cam) 하나 = 카메라 1대의 녹화 파일 하나) 안에서 샘플링된 프레임들의
GeoCalib 추정 focal 평균 하나를 그 영상 전체 인스턴스에 고정 적용한 조건. WiLoR 재추론 없음
(`gigahands_raw_instances_cache.csv`의 raw 값 + `geocalib_focal_seq_stats.csv`의 시퀀스별
평균만으로 `_cam_crop_to_full` 후처리를 재계산). 기존 5-way 비교(`geocalib_comparison_report.md`)
와 완전히 동일한 표본(1,846 인스턴스, seed=0).

## 1. 시퀀스별 GeoCalib 추정 focal 의 원래 흩어짐 (평균 내기 전)

- 영상(=`(scene,seq,cam)`) 총 {len(seq_stats_rows)}개, 그중 프레임 2개 이상(표준편차가 의미
  있는 경우) {n_multi}개.
- 영상별 focal_x_geocalib 표준편차: 평균 {stds.mean():.1f}px, 중앙값 {np.median(stds):.1f}px,
  최대 {stds.max():.1f}px, 최소 {stds.min():.1f}px.
- 영상 단위로 평균 낸 값 하나를 쓰면, 정의상 이 프레임 간 표준편차는 0이 된다(같은 영상의
  모든 인스턴스가 동일한 focal 을 공유하게 되므로) — "지터 제거"는 focal 값 자체에는 완전히
  적용되고, 그 결과가 절대오차에 실질적으로 도움이 되는지는 아래 2절에서 본다.
- 상세: `geocalib_focal_seq_stats.csv` ({len(seq_stats_rows)}행: scene, seq, cam, n_frames,
  focal_x_geocalib_mean/std/min/max)

## 2. GeoCalib_영상평균 자체의 focal 추정 정확도

| | GeoCalib_영상평균 | GeoCalib(프레임별, 기존) |
|---|---:|---:|
| focal_px 평균 | {focal_vals.mean():.1f} | (기존 보고서: 1,174.6) |
| GT 대비 상대오차(signed, 평균) | {rel_err_pct.mean():+.2f}% | (기존 보고서: +28.35%) |
| GT 대비 상대오차(signed, 중앙값) | {np.median(rel_err_pct):+.2f}% | (기존 보고서: +18.70%) |
| 상대오차 표준편차 | {rel_err_pct.std():.2f}%p | (기존 보고서: 24.09%p) |

## 3. 절대 translation 오차 (mm) — 6-way 비교

`geocalib_comparison_report.md`의 종합 비교표에 이번 조건을 한 행 추가:

| 조건 | focal_px 평균 | GT 대비 상대오차(%) | 절대 translation 오차 평균(mm) | 절대오차 중앙값(mm) | z오차 평균(mm) |
|---|---:|---:|---:|---:|---:|
| GT focal | 913.8 | 0% | 130.9 | 120.4 | 94.5 |
| 24mm 하드코딩 | 853.3 | −6.9% | 149.9 | 141.2 | 112.5 |
| 파이프라인 기본값 | 5,000.0 | +445.5% | 2,840.5 | 3,097.5 | 2,838.6 |
| GeoCalib(프레임별) | 1,174.6 | +28.3% | 240.4 | (기존 보고서 미기재) | 210.6 |
| **GeoCalib_영상평균** | **{focal_vals.mean():.1f}** | **{rel_err_pct.mean():+.1f}%** | **{abs_err.mean():.1f}** | **{np.median(abs_err):.1f}** | **{z_err.mean():.1f}** |

(참고용 — 이번에 재계산한 프레임별 GeoCalib 조건의 절대오차: 평균 {abs_err_frame.mean():.1f}mm,
중앙값 {np.median(abs_err_frame):.1f}mm, 표준편차 {abs_err_frame.std():.1f}mm / 영상평균 조건:
표준편차 {abs_err.std():.1f}mm — 프레임별 대비 영상평균의 절대오차 표준편차가
{'줄었다' if abs_err.std() < abs_err_frame.std() else '오히려 늘거나 비슷하다'}.)

### 화각 그룹별 절대오차 평균(mm)

| 그룹 | n | GeoCalib_영상평균 |
|---|---:|---:|
| 25mm | {by_group['25mm']['n']} | {by_group['25mm']['abs_mean']:.1f} |
| 26mm | {by_group['26mm']['n']} | {by_group['26mm']['abs_mean']:.1f} |
| 27mm | {by_group['27mm']['n']} | {by_group['27mm']['abs_mean']:.1f} |

## 4. 양손 간 상대거리 오차 (mm)

| | GeoCalib_영상평균 |
|---|---:|
| dx 오차(평균) | {dx_seqavg.mean():.2f} |
| dy 오차(평균) | {dy_seqavg.mean():.2f} |
| dz 오차(평균) | {dz_seqavg.mean():.2f} |
| 전체 유클리드 오차(평균) | {euclid_seqavg.mean():.1f} |

(기존 조건들과 비교하려면 `translation_benchmark_report.md`/`geocalib_comparison_report.md`의
동일 표 참고 — GT −3.99mm, 24mm −8.11mm, 파이프라인 기본값 +283.4mm, 전체 유클리드는 각각
107.4/105.6/579.7mm.)

## 5. 결론

GeoCalib_영상평균의 절대오차는 평균 {abs_err.mean():.1f}mm — 기존 프레임별 GeoCalib(240.4mm)
{'보다 개선됐다' if abs_err.mean() < 240.4 else '와 비슷하거나 더 나쁘다'}. 다만 focal 추정
자체의 GT 대비 편향(+{rel_err_pct.mean():.1f}% 근방)은 프레임을 평균 내도 구조적으로 남는다 —
영상 단위 평균은 "프레임 간 무작위 흔들림(분산)"만 줄일 뿐, GeoCalib이 이 도메인(GigaHands
근접 손 클로즈업)에서 갖는 **체계적 편향(bias)**은 그대로다. 즉 이 실험은 "GeoCalib을 더 안정
적으로 쓰는 방법"에 대한 답이지, "GeoCalib의 정확도 자체를 개선하는 방법"에 대한 답은 아니다
— 근본 원인(도메인 불일치로 인한 편향)은 `geocalib_comparison_report.md` 4절의 논의가 여전히
유효하다.

## 산출물

- `geocalib_focal_seq_stats.csv` (신규, {len(seq_stats_rows)}행): (scene,seq,cam)별 GeoCalib
  focal 평균/표준편차/min/max (`run_geocalib_gigahands.py --seq-stats-only`로 재생성 가능,
  GeoCalib 재추론 없음)
- `gigahands_raw_instances_cache.csv`: `focal_geocalib_seqavg`, `cam_t_geocalib_seqavg_{{x,y,z}}`,
  `pred_wrist_geocalib_seqavg_{{x,y,z}}`, `abs_error_geocalib_seqavg_mm`, `z_error_geocalib_seqavg_mm`,
  `signed_z_err_geocalib_seqavg_mm` 컬럼 추가
- `eval_translation_gigahands.csv`: `abs_error_geocalib_seqavg_mm`, `z_error_geocalib_seqavg_mm` 컬럼 추가
- `eval_interhand_distance_gigahands.csv`: `dx/dy/dz_err_geocalib_seqavg_mm`,
  `euclid_err_geocalib_seqavg_mm` 컬럼 추가
- `eval_translation_gigahands.py --recompute-geocalib-seqavg`: 재실행 가능(GPU/재추론 불필요)
"""
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[저장] {report_path}")
    print(f"\n=== 요약 ===\nGeoCalib_영상평균 절대오차: 평균 {abs_err.mean():.1f}mm / 중앙값 {np.median(abs_err):.1f}mm "
          f"(기존 GeoCalib 프레임별: 240.4mm)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-geocalib-seqavg", action="store_true",
                     help="GPU/WiLoR 재추론 없이 캐시에서 GeoCalib_영상평균 조건만 재계산")
    ap.add_argument("--target-per-group", type=int, default=500)
    ap.add_argument("--out-csv", type=str, default=str(OUT_DIR / "eval_translation_gigahands.csv"))
    ap.add_argument("--out-interhand-csv", type=str, default=str(OUT_DIR / "eval_interhand_distance_gigahands.csv"))
    ap.add_argument("--out-summary", type=str, default=str(OUT_DIR / "eval_translation_summary.json"))
    ap.add_argument("--n-manual-check", type=int, default=8, help="수기 재현 검증할 인스턴스 수")
    args = ap.parse_args()

    if args.recompute_geocalib_seqavg:
        # GPU/카메라 로드/WiLoR 없이 캐시에서만 재계산 — 아래 본 파이프라인과 분리된 경로.
        recompute_geocalib_seqavg()
        return

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---- 카메라 GT 파라미터 ----
    cam_params = load_camera_params()
    print(f"카메라 파라미터 로드: {len(cam_params)}대")

    # ---- 컨벤션 검증 (임의 카메라 1대 + 임의 GT 포인트 1개) ----
    bins = baseline.build_bins()
    cam_group = baseline.load_camera_groups(bins)
    gt = baseline.load_gt_sequences()
    sample_key = next(iter(gt))
    sample_side_dict = gt[sample_key]["left"] or gt[sample_key]["right"]
    sample_frame = next(iter(sample_side_dict))
    sample_wrist_world = sample_side_dict[sample_frame][WRIST]
    sample_cam_name = next(iter(cam_params))
    verify_extrinsic_convention(cam_params[sample_cam_name], sample_wrist_world)
    print(f"[검증] extrinsic 컨벤션(world->cam) OK — 샘플 z={world_to_cam(cam_params[sample_cam_name], sample_wrist_world)[2]:.3f} m")

    # ---- baseline 과 동일한 표본 구성 (동일 함수 + seed=0) ----
    print("\n=== 데이터 로드 (baseline 함수 재사용) ===")
    downloaded_index = baseline.load_downloaded_video_index()
    video_map = baseline.load_video_map()
    group_candidates = baseline.build_group_candidates(gt, video_map, downloaded_index, cam_group)
    samples = baseline.stratified_sample(group_candidates, args.target_per_group)
    print(f"  총 샘플: {len(samples)}개")

    n_gt_instances = sum(
        int(gt[(s, q)]["left"].get(f) is not None) + int(gt[(s, q)]["right"].get(f) is not None)
        for (s, q, _c, _v, f, _g) in samples
    )
    print(f"  GT 인스턴스: {n_gt_instances}개")
    baseline_json = OUT_DIR / "eval_results_gigahands.json"
    if baseline_json.exists() and args.target_per_group == 500:
        with open(baseline_json) as f:
            bj = json.load(f)
        match_total = (bj["total_samples"] == len(samples)) and (bj["gt_hand_instances"] == n_gt_instances)
        print(f"  [교차확인] baseline(total_samples={bj['total_samples']}, gt_hand_instances={bj['gt_hand_instances']})"
              f" 대비 {'일치' if match_total else '불일치(!)'}")
        if not match_total:
            raise RuntimeError("표본이 baseline 과 다릅니다 — 샘플링 재사용에 버그가 있을 수 있습니다.")

    # ---- 화각 그룹별 카메라 다양성 ----
    group_cams = defaultdict(set)
    for (_s, _q, cam_id, _v, _f, group) in samples:
        group_cams[group].add(cam_id)
    print("  그룹별 사용 카메라 종류 수:", {g: len(c) for g, c in group_cams.items()})

    by_video = defaultdict(list)
    for s in samples:
        by_video[s[3]].append(s)

    # ---- 모델 로드 + _cam_crop_to_full 캡처용 몽키패치 ----
    print(f"\n=== 모델 로드 (backend=wilor) ===")
    os.chdir(HAND_DEMO_ROOT)
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    import rgb_predictor  # noqa: E402
    from rgb_predictor import AnyHandPredictor  # noqa: E402

    _ORIGINAL_CAM_CROP_TO_FULL = rgb_predictor._cam_crop_to_full
    _captured = []

    def _capturing_cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length):
        _captured.append((
            pred_cam.detach().cpu().clone(),
            box_center.detach().cpu().clone(),
            box_size.detach().cpu().clone(),
            img_size.detach().cpu().clone(),
        ))
        return _ORIGINAL_CAM_CROP_TO_FULL(pred_cam, box_center, box_size, img_size, focal_length)

    rgb_predictor._cam_crop_to_full = _capturing_cam_crop_to_full

    t0 = time.time()
    predictor = AnyHandPredictor(backend="wilor")
    print(f"  로드 시간: {time.time() - t0:.1f}s")

    def cam_t_from_raw(raw, focal_px):
        pc, bc, bs, isz = raw
        ct = _ORIGINAL_CAM_CROP_TO_FULL(
            pc.unsqueeze(0), bc.unsqueeze(0), bs.reshape(1), isz, focal_px,
        )
        return ct[0].numpy().astype(np.float64)

    trans_records = []      # per-instance (scene,seq,cam,frame,group,side, errors...)
    interhand_records = []  # per-dual-hand-frame
    manual_check_pool = []  # (raw, focal_px, computed_cam_t) 몇 개만 모아 뒤에서 수기 재현

    t_start = time.time()
    n_videos = len(by_video)
    n_processed_frames = 0
    n_predict_errors = 0

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

            _captured.clear()
            try:
                hands = predictor.predict(frame)
            except Exception as e:
                n_predict_errors += 1
                print(f"  [warn] predict 실패 {scene}/{seq_int}/{cam_id} frame{fidx}: {e}")
                continue
            n_processed_frames += 1

            raw_list = []
            for pc, bc, bs, isz in _captured:
                B = pc.shape[0]
                for i in range(B):
                    raw_list.append((pc[i], bc[i], bs[i] if bs.dim() else bs, isz))
            assert len(raw_list) == len(hands), (
                f"캡처된 raw 개수({len(raw_list)})와 검출된 손 개수({len(hands)}) 불일치 "
                f"@ {scene}/{seq_int}/{cam_id}/{fidx}"
            )

            gt_focal_px = (cam["fx"] + cam["fy"]) / 2.0
            img_width_px = float(raw_list[0][3][0].item()) if raw_list else cam["width"]
            hardcoded_focal_px = focal_mm_to_px(HARDCODED_FOCAL_MM, img_width_px)

            best_left = best_right = None
            best_left_raw = best_right_raw = None
            for h, raw in zip(hands, raw_list):
                if h.is_right:
                    if best_right is None or h.score > best_right.score:
                        best_right, best_right_raw = h, raw
                else:
                    if best_left is None or h.score > best_left.score:
                        best_left, best_left_raw = h, raw

            per_side_camt_gt = {}
            per_side_camt_hard = {}

            for side, gt_arr, det, raw in (
                ("left", gt_left, best_left, best_left_raw),
                ("right", gt_right, best_right, best_right_raw),
            ):
                if gt_arr is None or det is None:
                    continue
                gt_wrist_world = gt_arr[WRIST]
                gt_wrist_cam = world_to_cam(cam, gt_wrist_world)

                # [수정] cam_t 자체는 절대 손목 위치가 아니다 — keypoints_3d[WRIST](root-relative)가
                # 실측상 0이 아니라 손마다 고정 오프셋(~±9.5cm, focal_length 와 무관)을 갖고 있음을
                # 디버그로 확인했다. 파이프라인 규약(demo_hand_mano.stage3_to_camera_space 와 동일:
                # keypoints_3d + cam_t = 절대 위치)대로 더해야 focal_length 효과만 순수하게 비교된다.
                # (지시서는 "cam_t 를 GT 와 직접 비교"라 했으나, 그대로 하면 focal_length 와 무관한
                # 고정 오프셋이 오차에 섞여 실험 취지를 훼손하므로 이렇게 보정했다.)
                wrist_offset = det.keypoints_3d[WRIST].astype(np.float64)
                cam_t_gt_focal = cam_t_from_raw(raw, gt_focal_px)
                cam_t_hard = cam_t_from_raw(raw, hardcoded_focal_px)
                pred_wrist_gt_focal = cam_t_gt_focal + wrist_offset
                pred_wrist_hard = cam_t_hard + wrist_offset
                per_side_camt_gt[side] = pred_wrist_gt_focal
                per_side_camt_hard[side] = pred_wrist_hard

                if len(manual_check_pool) < args.n_manual_check:
                    manual_check_pool.append((raw, gt_focal_px, cam_t_gt_focal))
                    manual_check_pool.append((raw, hardcoded_focal_px, cam_t_hard))

                abs_err_gt = float(np.linalg.norm(pred_wrist_gt_focal - gt_wrist_cam) * 1000.0)
                abs_err_hard = float(np.linalg.norm(pred_wrist_hard - gt_wrist_cam) * 1000.0)
                z_err_gt = float(abs(pred_wrist_gt_focal[2] - gt_wrist_cam[2]) * 1000.0)
                z_err_hard = float(abs(pred_wrist_hard[2] - gt_wrist_cam[2]) * 1000.0)

                trans_records.append(dict(
                    scene=scene, seq=seq_int, cam=cam_id, frame=fidx, group=group, side=side,
                    abs_error_gt_focal_mm=abs_err_gt, abs_error_hardcoded_mm=abs_err_hard,
                    z_error_gt_focal_mm=z_err_gt, z_error_hardcoded_mm=z_err_hard,
                ))

            if gt_left is not None and gt_right is not None and "left" in per_side_camt_gt and "right" in per_side_camt_gt:
                gt_vec = cam["R"] @ (np.asarray(gt_left[WRIST], dtype=np.float64) - np.asarray(gt_right[WRIST], dtype=np.float64))

                pred_vec_gt = per_side_camt_gt["left"] - per_side_camt_gt["right"]
                pred_vec_hard = per_side_camt_hard["left"] - per_side_camt_hard["right"]

                d_gt = (pred_vec_gt - gt_vec) * 1000.0
                d_hard = (pred_vec_hard - gt_vec) * 1000.0

                interhand_records.append(dict(
                    scene=scene, seq=seq_int, cam=cam_id, frame=fidx, group=group,
                    dx_err_gt_focal_mm=float(d_gt[0]), dy_err_gt_focal_mm=float(d_gt[1]), dz_err_gt_focal_mm=float(d_gt[2]),
                    euclid_err_gt_focal_mm=float(np.linalg.norm(d_gt)),
                    dx_err_hardcoded_mm=float(d_hard[0]), dy_err_hardcoded_mm=float(d_hard[1]), dz_err_hardcoded_mm=float(d_hard[2]),
                    euclid_err_hardcoded_mm=float(np.linalg.norm(d_hard)),
                ))

        if (vi + 1) % 20 == 0 or (vi + 1) == n_videos:
            elapsed = time.time() - t_start
            print(f"  video {vi+1}/{n_videos} ({elapsed:.0f}s elapsed, {n_processed_frames} frames 처리, "
                  f"{n_predict_errors} predict 실패)")

    # ---- 수기 재현 검증 (docstring 공식 그대로, _cam_crop_to_full 을 아예 안 부르고 재계산) ----
    print(f"\n=== 수기 재현 검증 ({len(manual_check_pool)}개) ===")
    for (pc, bc, bs, isz), focal_px, computed in manual_check_pool:
        s = float(pc[0]); tx_crop = float(pc[1]); ty_crop = float(pc[2])
        cx_box = float(bc[0]); cy_box = float(bc[1])
        cx_img = float(isz[0]) * 0.5; cy_img = float(isz[1]) * 0.5
        box_size_val = float(bs.reshape(-1)[0])
        denom = s * box_size_val + 1e-9
        tz = 2.0 * focal_px / denom
        tx = tx_crop + 2.0 * (cx_box - cx_img) / denom
        ty = ty_crop + 2.0 * (cy_box - cy_img) / denom
        manual = np.array([tx, ty, tz], dtype=np.float64)
        assert np.allclose(manual, computed, atol=1e-4), (
            f"수기 재현 불일치: manual={manual}, script={computed}"
        )
    print(f"  모두 일치 (atol=1e-4) — {len(manual_check_pool)}개")

    # ---- 저장 ----
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = ["scene", "seq", "cam", "frame", "group", "side",
                      "abs_error_gt_focal_mm", "abs_error_hardcoded_mm",
                      "z_error_gt_focal_mm", "z_error_hardcoded_mm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trans_records:
            w.writerow(r)
    print(f"\n[저장] {args.out_csv} ({len(trans_records)}행)")

    with open(args.out_interhand_csv, "w", newline="") as f:
        fieldnames = ["scene", "seq", "cam", "frame", "group",
                      "dx_err_gt_focal_mm", "dy_err_gt_focal_mm", "dz_err_gt_focal_mm", "euclid_err_gt_focal_mm",
                      "dx_err_hardcoded_mm", "dy_err_hardcoded_mm", "dz_err_hardcoded_mm", "euclid_err_hardcoded_mm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in interhand_records:
            w.writerow(r)
    print(f"[저장] {args.out_interhand_csv} ({len(interhand_records)}행)")

    # ---- 요약 통계 ----
    def stats(vals):
        a = np.array(vals, dtype=np.float64)
        if len(a) == 0:
            return dict(n=0, mean=None, median=None, std=None)
        return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()))

    groups = ("25mm", "26mm", "27mm")
    summary = dict(
        target_per_group=args.target_per_group,
        total_samples=len(samples),
        gt_hand_instances=n_gt_instances,
        n_translation_instances=len(trans_records),
        n_interhand_frames=len(interhand_records),
        group_camera_diversity={g: len(group_cams.get(g, set())) for g in groups},
        translation=dict(
            abs_error_gt_focal_mm=stats([r["abs_error_gt_focal_mm"] for r in trans_records]),
            abs_error_hardcoded_mm=stats([r["abs_error_hardcoded_mm"] for r in trans_records]),
            z_error_gt_focal_mm=stats([r["z_error_gt_focal_mm"] for r in trans_records]),
            z_error_hardcoded_mm=stats([r["z_error_hardcoded_mm"] for r in trans_records]),
            by_group={
                g: dict(
                    abs_error_gt_focal_mm=stats([r["abs_error_gt_focal_mm"] for r in trans_records if r["group"] == g]),
                    abs_error_hardcoded_mm=stats([r["abs_error_hardcoded_mm"] for r in trans_records if r["group"] == g]),
                    z_error_gt_focal_mm=stats([r["z_error_gt_focal_mm"] for r in trans_records if r["group"] == g]),
                    z_error_hardcoded_mm=stats([r["z_error_hardcoded_mm"] for r in trans_records if r["group"] == g]),
                ) for g in groups
            },
        ),
        interhand=dict(
            euclid_err_gt_focal_mm=stats([r["euclid_err_gt_focal_mm"] for r in interhand_records]),
            euclid_err_hardcoded_mm=stats([r["euclid_err_hardcoded_mm"] for r in interhand_records]),
            dx_err_gt_focal_mm=stats([r["dx_err_gt_focal_mm"] for r in interhand_records]),
            dy_err_gt_focal_mm=stats([r["dy_err_gt_focal_mm"] for r in interhand_records]),
            dz_err_gt_focal_mm=stats([r["dz_err_gt_focal_mm"] for r in interhand_records]),
            dx_err_hardcoded_mm=stats([r["dx_err_hardcoded_mm"] for r in interhand_records]),
            dy_err_hardcoded_mm=stats([r["dy_err_hardcoded_mm"] for r in interhand_records]),
            dz_err_hardcoded_mm=stats([r["dz_err_hardcoded_mm"] for r in interhand_records]),
            by_group={
                g: dict(
                    euclid_err_gt_focal_mm=stats([r["euclid_err_gt_focal_mm"] for r in interhand_records if r["group"] == g]),
                    euclid_err_hardcoded_mm=stats([r["euclid_err_hardcoded_mm"] for r in interhand_records if r["group"] == g]),
                    dz_err_gt_focal_mm=stats([r["dz_err_gt_focal_mm"] for r in interhand_records if r["group"] == g]),
                    dz_err_hardcoded_mm=stats([r["dz_err_hardcoded_mm"] for r in interhand_records if r["group"] == g]),
                ) for g in groups
            },
        ),
    )

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[저장] {args.out_summary}")
    print("\n=== 요약 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
