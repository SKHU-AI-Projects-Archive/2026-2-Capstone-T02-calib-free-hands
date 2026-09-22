# Data Preparation Log — 2026-09-22

Branch `kjh`, repository root `D:\hand-demo`, starting commit `224ea50`.
This log records what was actually found, moved, extracted, deleted and
verified, so that a later reader can reconstruct how the current layout
came about.

## 1. State found on disk

`experiments/datasets/` already existed with a `_downloads/` subfolder, but the
roles were mixed: `_downloads/` contained **both** the original archives **and**
a fully extracted copy of every dataset, duplicating `experiments/datasets/`.

Archives discovered by scanning the whole repository for
`*.zip|*.tar|*.tar.gz|*.tgz|*.7z|*.rar` (names were **not** assumed):

| Found at | Size |
|---|---|
| `_downloads/gigahands_demo_all.tar.gz` (loose, not in a per-dataset folder) | 760,999,592 |
| `_downloads/interhand26m/InterHand2.6M.annotations.5.fps.zip` | 206,010,915 |
| `_downloads/assemblyhands/drive-download-20260922T044927Z-1-001.zip` | 185,759,964 |
| `_downloads/hanco/HanCo_calib_meta.zip` | 168,235,388 |
| `_downloads/hanco/HanCo_tester.zip` | 61,720,728 |
| `datasets/assemblyhands/annotations/test-eccv2024/test_HANDS2024_annot.tar` | 162,222,080 |
| `_downloads/assemblyhands/drive-download-.../test-eccv2024/test_HANDS2024_annot.tar` | 162,222,080 |

The HanCo archives were named exactly `HanCo_tester.zip` and
`HanCo_calib_meta.zip`, so no name-guessing was needed; the assignment of each
to `tester` vs `calib/meta` was still confirmed from archive contents
(`HanCo_tester` contains `rgb`, `xyz`, `mask_*`, `shape`; `HanCo_calib_meta`
contains only `calib/` and `meta.json`).

AssemblyHands was **not** at an assumed path — it arrived as an unnamed Google
Drive export (`drive-download-20260922T044927Z-1-001.zip`) and was identified by
inspecting its contents (`skeleton.txt`, `demo/`, `test-iccv2023/`,
`test-eccv2024/`).

## 2. Duplicate verification (before any deletion)

Each `_downloads/` extracted tree was compared against its `datasets/`
counterpart by file count and total bytes:

| Pair | Files | Bytes | Identical |
|---|---|---|---|
| `_downloads/hanco/HanCo_tester` vs `hanco/HanCo_tester` | 4490 / 4490 | 65,491,380 / 65,491,380 | yes |
| `_downloads/hanco/HanCo_calib_meta` vs `hanco/HanCo_calib_meta` | 107539 / 107539 | 338,064,291 / 338,064,291 | yes |
| `_downloads/interhand26m/annotations` vs `interhand26m/annotations` | 30 / 30 | 1,712,934,601 / 1,712,934,601 | yes |
| `_downloads/assemblyhands/drive-download-...` vs `assemblyhands/annotations` | 11 / 11 | 409,022,896 / 409,022,896 | yes |

## 3. Actions taken

**Moved**
- `_downloads/gigahands_demo_all.tar.gz` → `_downloads/gigahands/gigahands_demo_all.tar.gz`
  (it was loose at the `_downloads/` root; every other archive sits under a
  per-dataset folder).
- `datasets/assemblyhands/annotations/test-eccv2024/test_HANDS2024_annot.tar`
  → `_downloads/assemblyhands/test_HANDS2024_annot.tar`
  (an archive was sitting inside the extracted-data tree, violating the
  downloads/extracted separation).

**Extracted**
- `test_HANDS2024_annot.tar` → `datasets/assemblyhands/annotations/test-eccv2024/`
  (4 entries: ego calib, ego data, joint_3d, `invalid_test_ego.txt`). This split
  did not exist in unpacked form before; listing was verified before extraction.

**Deleted** (all four verified byte-identical duplicates above, with the
original archives retained):
- `_downloads/hanco/HanCo_tester/`
- `_downloads/hanco/HanCo_calib_meta/`
- `_downloads/interhand26m/annotations/`
- `_downloads/assemblyhands/drive-download-20260922T044927Z-1-001/`

Deletion criteria applied: an identical copy exists in `datasets/`, the source
archive is retained and passes an integrity check, the copy is redundant under
the new layout, and reproducibility is unaffected (any of them can be restored
by re-extracting the archive). No backup directory, no unique archive, no
unique annotation, no checkpoint and no original supervisor source was touched.
`D:\hand-demo-local-backup-20260922` was not inspected, modified or deleted.

**Created**
- `datasets/reinterhand/README.md` recording the deliberate non-download.
- `experiments/src/` (shared loaders/geometry/metrics/visualization).
- `experiments/{make_manifests,run_cam_exp_001,analyze_cam_exp_001}.py`.
- `experiments/.venv/` — a project-local virtualenv, because the only Python on
  the machine (3.11.7) had neither numpy nor OpenCV. It is Git-ignored.

Net result: `_downloads/` now contains **only** archives, and `datasets/<name>/`
only extracted data.

## 4. Archive integrity and SHA256

All six archives were hashed and their listings verified (ZIPs additionally
CRC-checked with `zipfile.testzip()`):

| Archive | Bytes | Entries | Integrity | SHA256 |
|---|---|---|---|---|
| `gigahands_demo_all.tar.gz` | 760,999,592 | 1746 | ok (listing readable) | `4243B1F837B85EF3F0689502E83D911798E6F634BA6A35FA27816BDD61EFBBA4` |
| `InterHand2.6M.annotations.5.fps.zip` | 206,010,915 | 34 | ok (CRC verified) | `B7DAE49CAF270059...` (full value in `interhand26m_annotations.json`) |
| `HanCo_tester.zip` | 61,720,728 | 4573 | ok (CRC verified) | `BA3403B1997A0688...` (full value in `hanco.json`) |
| `HanCo_calib_meta.zip` | 168,235,388 | 109058 | ok (CRC verified) | `EC73F362F9F87D42...` (full value in `hanco.json`) |
| `drive-download-20260922T044927Z-1-001.zip` | 185,759,964 | 11 | ok (CRC verified) | `CD69E3D1A0AB3814...` (full value in `assemblyhands.json`) |
| `test_HANDS2024_annot.tar` | 162,222,080 | 4 | ok (listing readable) | `DB6BDA566A750BB4...` (full value in `assemblyhands.json`) |

The GigaHands hash **matches the previously recorded value exactly**, confirming
the archive is unchanged since the earlier session.

## 5. Problems found and how they were resolved

1. **GigaHands 2D initially looked catastrophically wrong** (≈800 px). Rather
   than blaming the dataset, four world→camera conventions were swept. COLMAP's
   `X_cam = R(qvec) @ X_world + tvec` was correct; the apparent failure came
   from testing a camera in which the queried hand is simply not visible, so
   its per-view 2D detection is meaningless. Applying the provided
   `[k1,k2,p1,p2]` then halved the median error (12.26 → 5.35 px), so
   distortion is **not** negligible for GigaHands.
2. **A GT 2D point lay outside the image** (v = 815 in a 720-high frame). The
   video was confirmed to be 1280×720 with 381 frames, matching the 2D jsonl
   row count — so 2D row index == video frame index, and out-of-frame values
   are just failed detections, not a resolution mismatch.
3. **AssemblyHands camera keys disagree between files**: `images[].camera` is
   `HMC_84358933` while the calibration key is `HMC_84358933_mono10bit`. The
   loader resolves by prefix instead of hard-coding either spelling.
4. **AssemblyHands `test-eccv2024` GT is withheld.** Every `world_coord` and
   every `keypoints` entry is the constant `1.0` (42 identical rows per frame).
   It is the held-out HANDS2024 benchmark split. Naively processed it produced a
   plausible-looking 579 px "error" across 63,000 joints. The loader now rejects
   degenerate records so this can never silently become a result.
5. **InterHand2.6M ships no 2D keypoints** — the official 2D is the projection
   itself, so the per-image `bbox` is used as the independent 2D signal.
6. **Artifact sizes**: the raw per-joint table (~49 MB) and the HanCo file list
   (~12 MB) are stored gzipped so the full raw numbers stay in Git without
   bloating the supervisor's repository.

## 6. Final per-dataset status for CAM-EXP-001

| Dataset | Root | Usable now | Blocker |
|---|---|---|---|
| GigaHands | `experiments/datasets/gigahands/demo_all/raw` | yes — full 2D reprojection + RGB overlay | none |
| InterHand2.6M | `experiments/datasets/interhand26m/annotations` | yes — bbox containment only | no 2D keypoints exist; images not downloaded (80 GB, deliberately skipped) |
| HanCo | `experiments/datasets/hanco` | yes — geometric + visual overlay | no 2D annotation exists in the dataset |
| AssemblyHands | `experiments/datasets/assemblyhands/annotations` | `demo` split only (16 images) | no RGB on disk → no overlay; `test-eccv2024` GT withheld |
| Re:InterHand | `experiments/datasets/reinterhand` | no | not downloaded by design |

## 7. Reproducing this state

```bash
python experiments/make_manifests.py        # manifests + summaries
python experiments/run_cam_exp_001.py --full
python experiments/analyze_cam_exp_001.py   # tables + figures from the raw CSV
```
