# HandCalib benchmark

This folder holds tools for building and scoring the hand-video benchmark described in the HandCalib data-collection guide (2026-11-14). The guide's content is kept in [GUIDE.md](GUIDE.md); the original HTML is no longer kept.

`calibrator.py` and `check_session.py` are copied from the guide unchanged. The other files only call them.

| File | Role |
|---|---|
| calibrator.py | Guide section 7, verbatim. Two modes: checkerboard (reference f) and hand (f estimate). |
| check_session.py | Guide section 7, verbatim. Prints f and error for each hand video in one session. |
| session_rules.py | The rules from the guide that a script can check: names, session.json, the 3 % checkerboard rule, at least 20 s, resolution, landscape. |
| validate_session.py | Runs session_rules on one session or on the whole data root. |
| measure_reference.py | Measures f_before, f_after and reference_f from the two checkerboard videos. With --write it saves them to session.json. |
| run_benchmark.py | Runs Calibrator("hand") on every hand video of every valid session and writes metrics against reference_f. |
| session_template.json | A session.json with all fields from the guide. |
| create_session.py | Builds a session folder and session.json from the 10 original camera files, using their metadata. |
| qt_meta.py | Reads QuickTime metadata (device, lens, 35 mm focal length, date) without exiftool; location keys are skipped. |
| GUIDE.md | The data-collection guide in text form. |
| checkerboard_8x8.html, checkerboard_8x8.svg | The guide's on-screen board: 8x8 squares, 7x7 inner corners, one-square white margin. |
| requirements.txt | Pinned environment; install with `pip install --no-deps -r requirements.txt`. |

The config is `configs/benchmark.json`.

## Environment

Use the venv `/home/juson/venvs/handcalib_bench` (`requirements.txt`, installed with `--no-deps`):

- Python 3.11
- rtmlib 0.0.15
- onnxruntime-gpu 1.22.0, with the pip `nvidia-*-cu12` libraries for CUDA 12 on a 12.4 driver
- opencv-python 5.0.0
- numpy 2.4

The Wholebody models (yolox_m and rtmw-dw-x-l, 316 MB) are cached in `~/.cache/rtmlib`.

Throughput on an RTX A5000 is about 0.03 s per frame. A 1080p checkerboard video takes a few minutes because `findChessboardCornersSB` runs on every frame, on the CPU.

```bash
PY=/home/juson/venvs/handcalib_bench/bin/python
cd benchmark
```

## Adding one session

The data root is `/public/data/handcalib_data`. Collector IDs are not assigned yet; s01 (repository owner) and s02 (teammate) are used for now.

1. Create the session from the original files. The folder name and session.json are filled from the metadata:

   | Field | Source |
   |---|---|
   | date | creation date |
   | device | model |
   | camera | rear/front, from the lens model |
   | lens | 35 mm equivalent focal length / 25, rounded to a camera-app label |
   | resolution | displayed width x height |
   | fps | median of the 10 videos |

   ```bash
   $PY create_session.py --collector s01 \
       --before IMG_a.MOV --after IMG_j.MOV \
       --twinkle IMG_b.MOV IMG_c.MOV --nod IMG_d.MOV IMG_e.MOV \
       --circle IMG_f.MOV IMG_g.MOV --mix IMG_h.MOV IMG_i.MOV \
       --stabilization off --focus-exposure-locked --shown-on "tablet screen" \
       --digital-zoom 1.0 --notes "..." [--dry-run]
   ```

   The script stops in these cases:
   - The ten files do not share device, lens model, 35 mm focal length and resolution.
   - A file was shot in cinematic mode.

   Fields that the metadata does not contain are left null and listed at the end. These are stabilization, focus/exposure lock, checkerboard display, digital zoom and notes.

   Each file is copied byte for byte, as the guide requires original files. The camera metadata, including location, stays in the copy.

   Each copy is verified by the sha256 of the whole file, which is recorded in session.json `sources`. Source files are never modified.

2. Measure the reference f and write it into session.json:

   ```bash
   $PY measure_reference.py /public/data/handcalib_data/<session> --write
   ```

   If the two values differ by more than 3 %, the script says so and exits with code 1. The whole session must then be reshot.

3. Check the session against the guide:

   ```bash
   $PY validate_session.py /public/data/handcalib_data/<session>
   ```

   The script also prints the visual checklist from guide section 8, which has to be checked by eye.

## Scoring the benchmark

```bash
$PY run_benchmark.py --config ../configs/benchmark.json
```

The script runs these steps:

1. Every session folder under `data_root` is validated first.
2. Sessions with guide errors are skipped and listed in the summary. Set `include_invalid: true` to score them anyway.
3. Each hand video is run through `Calibrator("hand").run()`, the same call as `check_session.py`.
4. If a video ends with an error (hand barely detected), it is recorded as `error` and the run continues.
5. The GPU thermal guard pauses at 70 °C and resumes below 65 °C.
6. Results are appended per video. `--resume runs/benchmark/<run>` continues an interrupted run.

The output goes to `runs/benchmark/<UTC time>/`:

| File | Contents |
|---|---|
| run.json | Config, git commit and dirty flag, package versions, and the validation result per session. |
| results.csv | One row per hand video: session, file, motion, device, lens, resolution, status, f, frames, detected frames, seconds. |
| summary.md | Per-video error against reference_f, plus aggregate metrics overall and by motion, session, device, lens and resolution. |

The aggregate metrics are:

- median and mean absolute error
- median signed error (bias)
- share of videos within 5 %, 10 % and 20 %

Videos that end in error count toward the video total but not toward scored videos.

Reference f values appear only in session.json and summary.md.

## Rules checked automatically and rules checked by eye

Checked by script:

- Session folder name and file names
- Exactly two checkerboard videos and the eight hand videos `hand_{twinkle,nod,circle,mix}_{01,02}`
- Every session.json field filled
- collector and date match the folder name
- stabilization is off and focus/exposure is locked
- f_before and f_after within 3 %
- reference_f equals their mean
- videos[] matches the files
- Each video is at least 20 s, landscape, and has the resolution in session.json

Checked by eye (guide section 8):

- One right hand only
- Same finger shape throughout
- Starts from the frontal pose and never edge-on
- No blur
- The hand from wrist to fingertips is always inside the frame
- Hand length is at least 1/3 of the short side of the frame
- No faces or other people
