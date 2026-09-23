# Progress report v1 — evidence package

**This package is the source of truth for the next progress report.**

It contains no new research. Everything here is read out of the completed runs
CAM-EXP-001 … CAM-EXP-004.1, checked for internal consistency, and fixed so that
the report can be written without going back to the raw results or
re-remembering which caveat belongs to which number.

The report prose itself has **not** been written yet, and no document or PDF has
been produced. That is the next step, after this package is reviewed.

---

## What to read first

| If you want to… | Open |
|---|---|
| know what may and may not be claimed | `tables/claim_evidence_ledger.md` |
| write a section | `evidence_index.md` — one entry per report section |
| find a number | `report_numbers.json` — 126 canonical entries |
| know where a number came from | `source_map/report_artifact_sources.csv` |
| avoid an overstatement | `source_map/report_wording_guardrails.md` |
| know which chapter uses which experiment | `source_map/experiment_to_report_section.md` |
| see what was corrected | `correction_log.md` |

## Rules this package enforces

1. **`report_numbers.json` wins.** If a number in the report disagrees with it,
   the report is wrong. Every entry names its source file, its source column,
   its statistical unit and its evidence level.
2. **The statistical unit is the view**, not the frame. 1400 frames means 175
   static cameras × 8 frames.
3. **Five evidence levels only**: `CONFIRMED_IN_CURRENT_ENVIRONMENT`,
   `ROBUST_BUT_SINGLE_DATASET`, `EXPLORATORY_INTERNAL_VALIDATION`,
   `PENDING_EXTERNAL_CONFIRMATION`, `OPEN_ISSUE`.
4. **E2 is a frozen exploratory baseline**, never "our method", and its
   performance is always quoted as three numbers: 6.14 % (selection set),
   6.46 % (independent re-run), 5.73–8.80 % (leave-one-sequence-out).
5. **GeoCalib numbers carry a run-to-run tolerance** of about ±0.35 pp.
6. **Every figure ships the numbers it plotted**, in `figures/main/data/`.

## Layout

```
README.md                      this file
evidence_index.md              per-section claims, numbers, artifacts, caveats
correction_log.md              metadata corrections, with old values kept
report_numbers.json            126 canonical numbers, each with its source
quality_check_report.json      output of scripts/quality_checks.py
tables/                        report-ready tables (.csv and .md of each)
figures/main/                  Fig01-Fig07, PNG at 300 dpi and PDF
figures/main/data/             the exact numbers behind each figure
figures/appendix/              empty by design - see tables/appendix_figure_index.csv
source_map/                    traceability and wording rules
scripts/                       everything here is regenerable
```

## Regenerating

```
python scripts/fix_anycam_provenance.py    # once; already applied
python scripts/build_report_numbers.py
python scripts/build_multiframe_curve.py   # the single-run 1..64 curve
python scripts/build_tables.py
python scripts/build_tables2.py
python scripts/build_source_maps.py
python scripts/make_figures.py
python scripts/quality_checks.py           # must print no FAIL
```

Run with `experiments/.venv`. No GPU, no model inference, no dataset access
beyond the manifests and result files already in the repository.

## Figures

| Figure | What it shows |
|---|---|
| `Fig01_existing_pipeline_and_camera_role` | where the camera enters the existing hand pipeline, and the 5000 px vs ~923 px gap |
| `Fig02_research_flow_with_findings` | the four experiments with one finding each, and what comes next |
| `Fig03_gigahands_validation_and_qc_flow` | how the data was validated and why the experiments use different subsets |
| `Fig04_focal_error_vs_absolute_depth_shift` | focal perturbation vs absolute depth displacement |
| `Fig05_single_frame_calibration_and_distortion` | single-frame methods, pinhole vs distortion-aware |
| `Fig06_multiframe_1_to_64` | median focal error for N = 1…64, recomputed from one run |
| `Fig07_bias_noise_and_ensemble_summary` | where the error lives, and why two model families help |

All are white-background, 300 dpi, and readable in black and white: series are
distinguished by marker, line style and hatch, never by colour alone.

## Known state

`scripts/quality_checks.py` passes all 24 checks, including: every source file
exists; the figure CSVs match the canonical summaries; no file presents 1400 as
an independent sample count; no dangerous phrasing is used assertively; the E2
selection-set figure never appears without the re-run and leave-one-sequence-out
numbers beside it; every file mentioning GeoCalib carries the
reproducibility caveat; AnyCam provenance is consistent everywhere; and no
historical raw or summary result file was modified.

The one open decision that blocks nothing in the report but shapes its
conclusion is whether to download an external image set for the final
confirmatory holdout — see `tables/report_future_work.csv`.
