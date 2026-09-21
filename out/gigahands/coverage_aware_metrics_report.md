# Coverage-aware 지표 (arXiv 2606.30308 방식) — GigaHands 결과

기존 baseline(recall 65.8%, PA-MPJPE 8.87mm)과 translation 실험(CT 130.9mm)에 FN(검출 실패)
페널티를 반영한 "-p"(presence) 버전을 계산했다. n_TP+n_FN = 1846+958 = **2804** (= 전체 GT
인스턴스 수, 검증 통과).

## 전체 요약: TP만(기존) vs -p(coverage-aware)

| 지표 | TP만(기존) | **-p (TP+FN)** | 단위 |
|---|---:|---:|---|
| Recall | — | 65.8 | % |
| FAcc | — | 52.7 | % |
| F1 | — | 77.6 | % (Precision 94.4%) |
| MPJPE | 60.7 (중앙값 54.5) | **89.8 (중앙값 74.5)** | mm |
| PA-MPJPE | 9.85 (중앙값 8.87) | **16.5 (중앙값 11.3)** | mm |
| EPE | 3773 (중앙값 301)† | **4192 (중앙값 605)†** | px |
| GO | 31.2 (중앙값 26.8) | **71.7 (중앙값 40.6)** | deg |
| CT | 130.9 (중앙값 120.4) | **343.2 (중앙값 152.7)** | mm |
| Jitter | 21,652 (GT 불필요, -p 대상 아님) | — | mm/s² |

†EPE는 평균이 중앙값의 12배 — 아래 "EPE 유의사항" 참고.

**FN을 반영하면 전부 나빠진다** — CT가 2.6배, GO가 2.3배, MPJPE가 1.5배, PA-MPJPE가 1.7배.
검출 실패를 빼고 보던 기존 지표들이 얼마나 낙관적이었는지 정량적으로 드러난다.

## 화각 그룹별 (-p 버전)

| 그룹 | Recall | FAcc | F1 | MPJPE-p | PA-MPJPE-p | GO-p | CT-p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25mm | 90.9% | 84.0% | 93.0% | 56.1mm | 10.3mm | 28.4° | 208.8mm |
| 26mm | 68.4% | 50.8% | 80.8% | 95.1mm | 16.3mm | 83.6° | 213.1mm |
| **27mm** | **35.9%** | **23.4%** | **50.4%** | **120.5mm** | **23.4mm** | **105.4°** | **629.4mm** |

27mm 그룹은 기존 TP-only PA-MPJPE(9.85mm 근처)로는 오히려 좋아 보였는데, coverage-aware로
보면 모든 지표에서 최악이다 — recall 35.9%(3분의 2를 놓침)가 반영되니 실제 실사용 성능은
가장 나쁜 그룹이라는 게 드러난다.

## 계산 중 발견/수정한 버그 (결과 확정 전에 잡음)

계산 과정에서 두 가지 좌표계 버그를 발견해 수정했다 — **초안 수치를 그대로 보고하지 않고
바로잡은 뒤의 최종값**이다:

1. **MPJPE/GO-p 초안이 각각 178mm/147°로 비정상적으로 컸다.** 원인: GigaHands GT는 world
   좌표계, WiLoR 예측은 카메라 좌표계(X우/Y하/Z전방)인데, 상대좌표(MPJPE)·회전(GO)을 비교할
   때 카메라 extrinsic 의 회전(R)을 안 곱하고 그냥 뺐다(절대위치 비교인 CT/EPE는 애초에
   `world_to_cam`을 정상적으로 썼어서 문제없었음). GT 상대벡터·회전에 카메라 R을 합성해
   카메라 프레임으로 옮긴 뒤 재계산 → MPJPE 178→**60.7mm**, GO 147→**31.2deg**로 정상화.
2. **왼손 GO만 138°로 비정상**(오른손은 32°로 정상)이었던 2차 버그. 원인: `rgb_predictor.py`가
   왼손 crop을 좌우반전해서 모델에 넣고 keypoints/vertices는 나중에 x를 되돌리지만,
   `mano_pose`(global_orient 포함)는 그 보정을 받지 않는다 — 왼손 global_orient가 "거울에
   비친 오른손" 좌표계로 남아있었다. 거울 켤레변환(R_true = M·R_raw·M, M=diag(-1,1,1))을
   왼손에만 적용해 수정 → 왼손 GO 138.7→**30.9deg**(오른손 31.7deg와 거의 대칭, 정상화 확인).

## EPE 유의사항 (평균 vs 중앙값 12배 차이)

EPE는 평균(3773px, TP만)이 중앙값(301px)보다 12배 크다 — 87/1846(4.7%) 인스턴스가 5,000px를
넘는 극단치(최대 579,681px)를 만들어낸다. 원인 확인: 손목 자체의 예측 깊이는 정상(0.58~1.22m)
인데, 일부 프레임에서 **손목이 아닌 다른 관절(손가락 등)의 예측이 심하게 어긋나** 그 관절의
절대 Z가 0에 가까워지면서 `u=f·X/Z+cx` 투영식이 폭발한다 — 버그가 아니라 실제로 존재하는
드문 극단적 오검출(대부분 `brics-odroid-009_cam1`=27mm 그룹에 집중, 이 그룹은 recall도 가장
낮았던 바로 그 카메라)을 있는 그대로 반영한 값이다. **평균보다 중앙값을 주된 대표값으로 보는
게 맞다.**

## 검증

- n_TP(1846) + n_FN(958) = **2804** = 전체 GT 인스턴스 수, 정확히 일치.
- FN 목록: `eval_per_instance_gigahands.csv`의 `matched==False`로 직접 필터링(역산 불필요,
  이미 존재).
- **MPJPE-p == MPJPE(recall 100% 가정 시) 엣지케이스**: 코드 로직상 `-p` 값은 `matched` 여부로
  분기해 같은 배열(df)에 pool한 뒤 `.mean()/.median()`을 계산하므로, FN이 0개면 `df`가
  `tp`와 완전히 같아져 정의상 `-p`==TP-only가 된다(별도 실험 없이 코드 구조로 보장됨).
- GO-p용 GT(`Rh`): 2,804개 중 71개(2.5%)는 `params/*.json`에 해당 프레임의 Rh가 없어(그 take의
  poses 배열 길이보다 frame_idx가 큰 경우) 계산 제외 — 위 GO/GO-p는 이 71개를 뺀 2,733개 기준.

## 산출물

- `coverage_aware_metrics_gigahands.csv` — 지표별 요약표(TP-only vs -p)
- `coverage_aware_per_instance_gigahands.csv` — 인스턴스별 상세(2,804행: ct/mpjpe/pa_mpjpe/epe/go)
- `gigahands_detection_frames.csv` (1,500행, FAcc/F1용), `gigahands_jitter.csv` (6행)
- `gigahands_pose_keypoints_cache.csv` — 이번에 재추론으로 캐싱한 raw keypoints_3d/mano_pose
- `canonical_mano_joints.npy` — placeholder 계산용 canonical MANO 21관절
