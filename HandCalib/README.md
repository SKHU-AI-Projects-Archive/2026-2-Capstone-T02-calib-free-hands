# HandCalib 연구 상태

## 프로젝트 개요

HandCalib은 손-물체 상호작용 영상의 카메라 보정 방법을 연구하기 위한 작업 공간입니다. 현재는 AnyCalib 공식 코드를 연결하고 GigaHands demo 데이터를 준비하는 단계이며, HandCalib 고유 모델과 전체 학습 파이프라인은 아직 구현 중입니다.

## 현재 상태

- [x] HandCalib 기본 폴더와 AnyCalib 연결 구조 정리
- [x] 실제 데이터셋과 Python dataloader 분리
- [x] GigaHands 데이터와 다운로드 cache의 Git 제외
- [x] 공식 GigaHands 5개 demo archive 다운로드 및 검증 도구 추가
- [x] 공식 archive와 현재 local dataset의 파일 단위 비교
- [x] CAM-EXP-001 과거 검증의 목적과 raw dataset 비수정 여부 확인
- [x] Linux에서 downloader unittest와 실제 공식 archive 설치 검증
CAM-EXP-001은 제공 camera와 제공 3D joints를 2D annotation에 재투영해 camera convention, distortion, frame 대응을 확인한 과거 검증입니다. 일부 `(0, 0)` 2D detection과 per-view outlier 패턴을 관찰했지만 raw GigaHands 파일을 수정한 실험은 아닙니다.

## 실험 로드맵

- [ ] 01번 실험 — 사전학습 AnyCalib 기본 성능 확인
- [ ] 02번 실험 — GigaHands를 이용한 AnyCalib 추가 학습
- [ ] 03번 실험 — 손 정보 활용을 위한 데이터 확인
- [ ] 04번 실험 — HandCalib 설계 및 구현
- [ ] 05번 실험 — HandCalib 학습 및 비교 평가

이번 작업에서는 01번 실험을 실행하지 않았습니다.

## 폴더 구성

```text
HandCalib/
├── AnyCalib/       공식 AnyCalib 코드
├── configs/        실험 설정 파일
├── datasets/       실제 local 데이터셋. GitHub에는 올리지 않음
├── dataloaders/    데이터셋을 Python에서 읽는 코드
├── data/            split과 다운로드 cache
├── models/          모델 연결과 HandCalib 모델 코드
├── utils/           공통 보조 기능
├── tools/           다운로드와 데이터 검사 도구
├── runs/            local 실험 결과. 기본적으로 Git 제외
├── train.py        학습 진입점
└── evaluate.py     평가 진입점
```

## GigaHands 데이터 준비

```bash
git submodule update --init --recursive
cd HandCalib
python tools/download_gigahands_demo.py
python tools/download_gigahands_demo.py --check
```

기본 데이터 위치는 `datasets/gigahands/`이며, 다운로드 cache는 `data/downloads/gigahands/`입니다. 두 위치 모두 Git에서 제외됩니다. 다른 검증 위치가 필요하면 `--target`과 `--cache-dir`를 사용할 수 있습니다.

공식 출처: [Brown IVL GigaHands](https://github.com/brown-ivl/GigaHands)

## 확인된 데이터

- sequence 5개
- RGB MP4 248개
- camera parameter row 245개
- 전체 파일 1,385개
- 공식 archive: 760,999,592 bytes, tar entries 1,746개
- 공식 archive SHA256: `4243B1F837B85EF3F0689502E83D911798E6F634BA6A35FA27816BDD61EFBBA4`
- 현재 `datasets/gigahands/`와 공식 archive 추출본은 1,385개 파일의 경로·크기·SHA256이 모두 일치

검증 과정은 raw dataset을 수정하지 않았습니다. 알려진 invalid 2D pattern, exact video stem 연결, distortion 적용은 loader와 validation 규칙으로만 다룹니다.

## 아직 남은 일

- AnyCalib 환경과 pretrained baseline 실행
- fine-tuning protocol, split, metric 확정
- HandCalib 모델과 학습/evaluation 코드 구현
