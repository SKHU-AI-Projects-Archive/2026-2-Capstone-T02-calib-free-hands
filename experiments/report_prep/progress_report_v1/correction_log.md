# Correction log

Metadata corrections made while assembling this package. **No historical
numerical result was changed.** Every entry records the old value, the new
value, the evidence, and whether any number moved.

---

## 2026-09-23 — AnyCam provenance metadata

**What was wrong.** Three files disagreed about AnyCam. The provenance manifest
and its CSV table said the repository was never installed; the CAM-EXP-004.1
report, `config.json` and `anycam_static_stress_test.md` recorded the exact
commit and the downloaded checkpoint. The cause is mundane: the script that
generates the provenance manifest hard-codes AnyCam's entry, and it was written
before the applicability audit cloned the repository.

**Old values**

| File | Field | Old |
|---|---|---|
| `experiments/manifests/external_model_provenance_v1.json` | `models.AnyCam.exact_commit` | `null` |
| | `models.AnyCam.commit_resolution` | `NOT_INSTALLED` |
| | `models.AnyCam.used_in` | `[]` |
| `runs/CAM-EXP-004_1.../tables/external_model_provenance.csv` | `exact_commit` | `NOT_INSTALLED` |
| | `checkpoints`, `checkpoint_sha256_first16`, `used_in` | empty / `none` |
| `runs/CAM-EXP-004_1.../report.md` | provenance table cell | "fresh clone (not used for any result)" |

**New values**

| Field | New |
|---|---|
| repository | `https://github.com/Brummi/anycam` |
| exact commit | `e609cc8a9e4ee8f78cf2ce39ebeb86b35e82d10d` |
| commit resolution | `git rev-parse HEAD` of the clone at `experiments/cache/external_models/anycam` |
| checkpoint | HuggingFace `fwimbauer/anycam_v1_seq8`, resolved from the repository's own `hubconf.py` |
| `pytorch_model.bin` | 459,833,700 bytes, sha256 `4b2f723f6b6672a9ef3a6a4d1831c08418db498f2832c565d524e3d201c300af` |
| `config.yaml` | 11,240 bytes, sha256 `4c978ef6e23279b56d0f21ae7a452eafdafbb4521dc82174717a7a07774d908c` |
| `REPOSITORY_INSTALLED_SOURCE_AVAILABLE` | true |
| `CHECKPOINT_DOWNLOADED` | true |
| `END_TO_END_INFERENCE` | **`BLOCKED_ON_WINDOWS`** |
| `MECHANISTIC_IDENTIFIABILITY_TEST` | **`COMPLETED`** |
| official code actually executed | `anycam.trainer.make_proj_from_focal_length`, `anycam.trainer.induce_flow_dist` |
| benchmark status | `NOT_A_PERFORMANCE_BENCHMARK` — AnyCam produced no focal-accuracy number |

The Windows blocker is now recorded in the manifest itself rather than only in
prose: `fit_video.py` applies `torch.compile` at import (which fails on torch
≤ 2.5 on Windows) while the UniDepth backbone needs `xformers.components`, which
exists only in `xformers ≤ 0.0.28.post3` — a version with no Windows wheel that
itself requires torch 2.5.1. The two requirements cannot both be satisfied on
this platform without editing official source, which was not done.

**Evidence.** `git rev-parse HEAD` in the clone; the HuggingFace download paths
and file sizes; the recorded import errors; `runs/CAM-EXP-004_1.../results/
summary/anycam_identifiability.json`.

**Numerical impact: none.** No experiment result depends on any of these fields.
AnyCam never produced a number that appears in any table.

**Audit trail.** The manifest now carries a `corrections` array holding the old
values, so the `NOT_INSTALLED` string still exists there deliberately — that is
the record of what was corrected, not a stale value. `schema_version` is
preserved. Applied by
`experiments/report_prep/progress_report_v1/scripts/fix_anycam_provenance.py`.

---

## 2026-09-23 — QC label counts: two consistent presentations

**Not an error, but a discrepancy a reader will notice.**
`runs/CAM-EXP-001_3.../results/summary/qc_status_summary.csv` reports
`PASS_STRICT = 48,831` with no `PASS_SINGLE_HAND` row, while
`manifests/gigahands_demo_qc_v1.csv.gz` reports `PASS_STRICT = 32,826` and
`PASS_SINGLE_HAND = 16,005`.

These are the same thing: 32,826 + 16,005 = 48,831, and both add up to the same
108,240 total. The run summary was written before the strict class was split
into "both hands clean" and "this hand clean, partner hand not".

**Action:** nothing was edited. `report_numbers.json` reads the **manifest**,
which is the finer-grained and later version, and the report should quote the
four-way split. The CAM-EXP-001.3 report body already uses the four-way split.

---

## Deliberately not corrected

* **CAM-EXP-002's scope wording.** Where earlier documents paraphrase it as
  camera-calibration sensitivity, the raw results were left untouched and the
  scope is corrected here, in `tables/report_limitations.md` and in
  `source_map/report_wording_guardrails.md`. Rewriting a finished run's prose
  after the fact would be worse than recording the correction.
* **The CAM-EXP-004 N ≤ 8 numbers.** The single-run curve in `Fig06` recomputes
  every N from the CAM-EXP-004.1 predictions, which differ slightly from
  CAM-EXP-004's own run because GeoCalib is not deterministic. CAM-EXP-004's
  original numbers were not overwritten; they are shown beside the curve and
  tabulated in `tables/multiframe_run_agreement.csv` (largest difference
  0.52 pp; AnyCalib identical to the digit).
