# 영상 → 초점거리

손 동작 영상에서 휴대전화 카메라의 초점거리를 추정하는 명령입니다. 판정(가설 검정), 편향 보정, 기준 초점거리와의 비교는 하지 않습니다.

```bash
source .venv/bin/activate
python HandCalib/run.py estimate-focal --config HandCalib/configs/video_to_focal.yaml --videos IMG_XXXX.MOV,IMG_YYYY.MOV
```
영상은 `configs/video_to_focal.yaml`의 `videos`에 적거나 `--videos`로 줍니다(`input_dir` 안의 파일 이름). 같은 카메라·렌즈로 찍은 영상끼리 한 번에 넣으세요.

## 하는 일
영상 규격 변환 → 15 fps 프레임 → mediapipe 손 검출(CPU) → 정면 창(손을 가만히 둔 1초)에서 평면 손 템플릿 → 기울인 프레임들로 f 추정(손목 제외 + `fixed_template`).

## 결과 (`runs/video_to_focal/<UTC>/`)
| 파일 | 내용 |
|---|---|
| `report.md` | 영상별 f, 품질 판정, 날카로움, 불확실성, 촬영 지표, 경고. 2개 이상이면 합친 f |
| `results.yaml`, `focal.csv` | 같은 값(기계가 읽는 형태) |
| `figures/<영상>_cost_curve.png` | f에 따른 비용 곡선과 추정값 |
| `results.json`, `metadata.json`, `log.txt` | 원 수치, 코드 commit·설정 사본, 로그 |

- f의 단위는 처리 영상(긴 변 1920 px)의 픽셀입니다.
- **품질 판정**(GOOD / ROUGH / REJECT)은 이 추정기의 자체 판정이며 정확도를 뜻하지 않습니다. 지금까지 모든 영상이 REJECT(1개 ROUGH)였습니다.
- **불확실성**: 부트스트랩 로그 SD(영상 안에서 블록을 다시 뽑은 f의 로그 표준편차)와 결합 불확실성. 영상이 길수록(기울기 구간 20초 이상) 작아집니다.
- f가 탐색 범위(긴 변의 0.35~1.6배) 경계에 붙으면 `값 없음`으로 적습니다.
- 경고(정면 창 없음, 가장자리 접촉 5% 초과, 손 검출 실패 20% 초과)는 옆에 적을 뿐 추정을 막지 않습니다.
- 합친 f는 영상별 비용 곡선을 더해 구한 값입니다(영상 3개 이상이면 영상 단위 부트스트랩 95% 구간도). 같은 카메라·렌즈일 때만 의미가 있고, 방법으로 검증된 것은 아닙니다.

## 옵션 (`compare`, 기본은 모두 꺼짐)
`keep`(손목 포함 f), `joint_delta`, `anycalib`(AnyCalib pinhole·gen, GPU 필요). 참고용이며 결과 f에는 쓰이지 않습니다. 손목 처리는 `candidate.wrist_mode`(exclude | keep)로 바꿉니다.

## 알아 둘 점
- 이 방법은 핀홀, 주점 중심, 왜곡 없음을 가정합니다. 초광각 렌즈는 왜곡이 커서 값을 믿기 어렵습니다.
- 지금까지의 검증(실험 25, 새 영상 6개): 영상 하나의 f는 기준 대비 중앙값 약 8% 벗어났고 일부는 40% 넘게 벗어났습니다. 영상은 길게(약 20초) 찍고 기울임을 크게(θ2d 최댓값 55° 이상) 넣는 것이 좋습니다.
- GPU가 필요한 것은 `compare.anycalib`뿐입니다.
