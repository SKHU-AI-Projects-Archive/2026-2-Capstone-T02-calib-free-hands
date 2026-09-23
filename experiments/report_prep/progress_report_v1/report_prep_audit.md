# Report-preparation audit summary

What was found while consolidating CAM-EXP-001 … CAM-EXP-004.1 into a report
evidence package, what was changed, and what was deliberately left alone.

---

## Inconsistencies found

1. **AnyCam provenance disagreed across three files.** The provenance manifest
   and its CSV said `NOT_INSTALLED` with a null commit; the CAM-EXP-004.1
   report, `config.json` and `anycam_static_stress_test.md` recorded the exact
   commit and the downloaded checkpoint. Cause: the manifest-generating script
   hard-codes AnyCam's entry and predates the clone. **Corrected**, with the old
   values preserved in a `corrections` array inside the manifest and in
   `correction_log.md`. Numerical impact: none — AnyCam produced no number that
   appears in any table.

2. **The CAM-EXP-004.1 report described the AnyCam clone as "not used for any
   result".** That was inaccurate: two of its functions were executed for the
   identifiability test. **Corrected** in that one table cell; the surrounding
   prose already described the test correctly.

3. **QC label counts are presented two ways.** CAM-EXP-001.3's
   `qc_status_summary.csv` reports `PASS_STRICT = 48,831` with no
   `PASS_SINGLE_HAND`; the QC manifest reports 32,826 + 16,005, which is the
   same 48,831 split into "both hands clean" and "this hand clean, partner
   not". **Not an error, and nothing was edited.** `report_numbers.json` reads
   the manifest (the finer, later version) and the discrepancy is documented so
   a reader comparing the two files is not puzzled.

4. **The N ≤ 8 and N ≥ 8 parts of the frame-count story came from different
   runs**, which disagree slightly at N = 8 because GeoCalib is not
   deterministic. Splicing them would have put a step in the curve that is a run
   boundary. **Resolved by recomputing every N from the single CAM-EXP-004.1
   inference pass** — an aggregation of existing predictions, no new inference —
   and showing the CAM-EXP-004 originals as separate unjoined points
   (`tables/multiframe_run_agreement.csv`; largest difference 0.52 pp, AnyCalib
   identical to the digit).

## Metadata corrected

Only three files, all provenance, all logged in `correction_log.md`:
`experiments/manifests/external_model_provenance_v1.json`,
`runs/CAM-EXP-004_1.../tables/external_model_provenance.csv`,
and one table cell in `runs/CAM-EXP-004_1.../report.md`.

## Historical results NOT changed

No file under any run's `results/raw/` or `results/summary/` was modified — this
is checked automatically (`quality_checks.py`, check I). In particular:

* CAM-EXP-002's raw results were left alone despite its scope having been
  paraphrased too broadly elsewhere; the scope correction lives in this package
  and in `OPEN_ISSUES.md`, not in the old run.
* CAM-EXP-004's original N ≤ 8 numbers were left alone and are reported beside
  the recomputed curve rather than replaced by it.
* The `LEGACY_FRAME_BOOTSTRAP` intervals from CAM-EXP-003.1 remain, labelled.

## Canonical numbers

`report_numbers.json` holds **139** entries (see the second pass below; it was 126 before the CAM-EXP-002 usage audit). Each carries value, unit,
experiment, source file, source column, statistical unit, evidence level and
notes. Every source file was verified to exist.

## Figure generation status

Seven main figures, each as PNG (300 dpi) and PDF, each with its plotted values
written to `figures/main/data/FigNN.csv`. `figures/appendix/` is intentionally
empty: the appendix should reference the original run figures listed in
`tables/appendix_figure_index.csv` (9 KEEP, 3 DROP with reasons) rather than
duplicate them.

## Quality checks

24/24 pass at the time of the first pass; **37/37** after the second pass below. The interesting ones:

* **C2** — the Fig06 curve matches the CAM-EXP-004.1 summary at every N ≥ 8, so
  the recomputation did not silently change anything.
* **D** — no file presents 1400 as an independent sample count, except where it
  is explicitly being warned against.
* **F** — the E2 selection-set figure never appears alone: wherever it is
  written, the same-views repeat execution (6.46 %) and the
  leave-one-sequence-out range
  (5.73–8.80 %) are within a few lines of it.
* **G** — every file mentioning GeoCalib carries the reproducibility caveat.
* **H4** — no file still records AnyCam as `NOT_INSTALLED`, apart from the
  correction history, which is supposed to.
* **I** — no historical raw or summary result file was modified.

## Unresolved, and deliberately so

1. **The external confirmation decision.** No locally available dataset can
   confirm the static-camera multi-frame result: HanCo's released images are
   per-frame hand-centred crops, AssemblyHands and InterHand2.6M are
   annotation-only on disk. The options — download the InterHand2.6M images,
   download full HanCo RGB, or accept the claim as single-dataset — all involve
   downloads previously excluded on size grounds, so this is a decision to be
   made rather than assumed.
2. **GeoCalib non-determinism** is registered and quantified but not fixed.
3. **Hand-QC manual validation** is registered as a blocker for CAM-EXP-006, not
   for the report.
4. **CAM-EXP-002's scope** — the report must either narrow the claim to focal
   sensitivity of absolute depth, or a CAM-EXP-002.1 must be run. The package
   assumes the narrow claim.
5. **AnyCam's behavioural test** remains unrun; only the mechanism was tested.

None of these blocks writing the report. Items 1 and 4 shape how two sentences
are phrased, and both phrasings are already prepared in the claim ledger.

---

# Second pass — semantic corrections (2026-09-23)

A review of the package found three semantic problems that the first round of
checks did not catch. All three were corrected in the **generator scripts**, not
only in the output files, and four new check families were added so they cannot
recur.

## 1. CAM-EXP-002: eligible pool was being presented as the experiment's input

`report_dataset_usage` listed the bimanual-clean manifest (16,413 frames,
154 views, 40 cameras) in the `n_frames_input` position for CAM-EXP-002. That is
the pool the experiment was allowed to draw from, not what it ran on.

Read from CAM-EXP-002's own files (`_inference_meta.json`,
`inference_cache_index.csv`, `model_failures.csv`,
`hand_association_failures.csv`, `focal_sweep_per_hand.csv.gz`,
`focal_sweep_per_frame.csv.gz`):

| Quantity | Value |
|---|---|
| eligible bimanual-clean pool | 16,413 frames / 154 views |
| frames drawn (stratified, seed 20260922, target 1200) | 1,200 |
| inference successful | 1,189 (11 with no hand detected) |
| dropped for ambiguous hand association | 16 |
| **frames evaluated** | **1,173** |
| **hands evaluated** | **2,210** |
| views / physical cameras / sequences | 134 / 37 / 5 |
| frames with BOTH hands evaluated (bimanual metrics) | 1,037 |

2,210 matches `n_hands` in every CAM-EXP-002 summary table, which is now checked
automatically (J4). The usage table gained explicit `eligible_*` and `actual_*`
columns; Fig03 gained a separate "actually evaluated" box; Fig04's title now
names the evaluated sample rather than the subset.

## 2. "per-camera bias" was the wrong unit

The 98.1 % figure comes from decomposing the squared log focal error into a
per-(sequence, camera) **view** mean and a within-view residual. Calling it a
per-camera bias implies a physical-camera-level result, which is not what was
computed — physical camera appears in this project only as a *resampling* unit
in the CAM-EXP-004.1 robustness check.

Corrected in the claim ledger (C06), the main-results table, the evidence index
(Section 7 heading, claims and numbers), Fig02 and the report_numbers notes. The
principles table gained an explicit row separating the **decomposition unit**
from the **resampling unit**.

## 3. Experiment 1's tail was implied to be fully explained

The evidence index said the tail "is caused by" the all-zero pattern and hand
swaps. CAM-EXP-001.3 still contains 15,611 `BAD_2D_GEOMETRY` and 10,207
`UNRESOLVED_INSUFFICIENT_GEOMETRY` cases. The wording now says a *substantial
part* was traced to named causes and that the rest was conservatively labelled
REVIEW/EXCLUDE rather than explained; Fig03's diagnosis box is titled "diagnosis
of PART of the error tail" and prints the unexplained counts; and every verdict
class is now in `report_numbers.json` (previously the unresolved class was
omitted).

## New checks

| Check | What it enforces |
|---|---|
| J1–J5 | the CAM-EXP-002 pool and evaluated sample are recorded separately, agree with the raw files, and the pool is never presented as the sample |
| K1–K3 | the 98.1 % decomposition is never described as a physical-camera bias; no bare "per-camera bias" is asserted; the principles table separates decomposition unit from resampling unit |
| L1–L3 | the Experiment 1 tail is never claimed as fully explained, and the unexplained classes are present in the numbers file and the evidence index |
| M1–M2 | Fig03 and Fig04 data files distinguish the eligible pool from the evaluated sample |

24 checks → **37 checks, all passing**.

## Wording of "no new number"

The package previously said it contains "no new research". More precisely: it
contains **no new model inference and no new experimental evidence**; some
report-facing aggregates (the 1→64 curve) were deterministically regenerated
from existing frozen predictions with the frozen aggregation rules. README and
this audit now say that. The earlier commit message is left as written.

## Historical results

Unchanged, as before — checked automatically. No file under any run's
`results/raw/` or `results/summary/` was touched in this pass either, including
CAM-EXP-002's.

---

# Third pass — focal semantics and independence wording (2026-09-23)

Three terminology problems, all of which would have read as stronger claims than
the evidence supports. As before, fixed in the generators and locked with new
checks. No numerical value changed.

## 1. "physical focal" was the wrong term for both numbers

The package described the CAM-EXP-002 comparison as 5000 px "assumed" versus
922.77 px "physical". CAM-EXP-002's own `focal_usage_audit.md` says something
more specific, and both halves of that phrasing were wrong.

**5000 px is `PIPELINE_BASELINE_FOCAL`** — a *training-convention virtual focal*,
`FOCAL_LENGTH/IMAGE_SIZE * max(W,H)` = `1000/256 * 1280`, expressed in original
full-image pixels. The audit's own table rules out the alternatives explicitly:
it is not a physical focal in original pixels, not a resized-coordinate focal and
not a crop-coordinate focal, because no calibration is ever read and there is no
resize (`rgb_predictor.py:404-411`). It is a convention inside the
weak-perspective to camera-translation conversion.

**922.77 px is the median `GT_EFFECTIVE_FOCAL`** — the dataset-provided camera
intrinsic `fx` from GigaHands `optim_params.txt`. The audit *derives*
`GT_EFFECTIVE_FOCAL = GT_NATIVE_FX` exactly, because the pipeline focal and the
GigaHands intrinsics live in the same original 1280x720 pixel coordinate system
and the focal undergoes no resize or crop rescaling. It is a pixel intrinsic, not
a sensor/optical focal length.

So "physical focal" was wrong for 5000 px (it is virtual) and misleading for
922.77 px (it is a pixel intrinsic, not an optical focal). Both are now named by
their audited roles everywhere.

**Keys renamed, no aliases kept** (the package is pre-report, so the canonical
names are fixed now rather than deprecated later):

| Old | New |
|---|---|
| `cam002_gigahands_physical_focal_median_px` | `cam002_gt_effective_focal_median_px` |
| `cam002_physical_focal_root_error_median_mm` | `cam002_gt_effective_focal_root_error_median_mm` |
| `cam002_physical_focal_absolute_mpjpe_median_mm` | `cam002_gt_effective_focal_absolute_mpjpe_median_mm` |

Fig01's on-figure text now reads "pipeline focal convention: f = 5,000 px" and
"dataset-provided reference focal (median): f ≈ 923 px", with a footnote naming
each one's definition. Fig04's title already named the evaluated sample.
A new guardrail section (M) carries the safe sentence in English and Korean.

**What was also checked:** no artifact asserts that 5000 px is a "wrong",
"incorrect" or "unphysical" focal. What CAM-EXP-002 showed is narrower and is
what the tables now say: on the same cached predictions, substituting the
dataset-provided reference focal for the pipeline's focal convention changed the
median absolute root error from 2881.885 mm to 78.54 mm, while the root-aligned
MPJPE was identical at 34.404 mm.

## 2. E2's 6.46 % was called an "independent re-run"

It is not independent of anything: that execution used the same 175 GigaHands
benchmark views. The three E2 numbers are now labelled by what they are:

* **6.14 %** — selection-set result;
* **6.46 %** — a separate execution on the **same** 175 benchmark views
  (same-data repeat execution);
* **5.73–8.80 %** — retrospective leave-one-sequence-out internal validation.

None is an independent-dataset result. That is precisely what the sealed
external holdout is reserved for, and E2's status is unchanged: a frozen
exploratory ensemble, never a validated or proposed method.

## 3. "±0.35 pp tolerance" was an invented interval

Two executions do not make a confidence interval. The package now states the two
GeoCalib measurements separately and never merges them:

* **between-execution difference**: across the two full benchmark executions run
  here, the aggregate median differed by about **0.32 pp** (GeoCalib
  11.184 → 10.860 %, E2 6.138 → 6.460 %; AnyCalib 9.890 → 9.890 %, identical);
* **repeat diagnostic** (40 frames × 3 repeats on byte-identical images):
  median spread 1.18 %, p90 6.85 %, max 41.50 %, 0 % bit-identical.

## New checks

| Check | What it enforces |
|---|---|
| N1, N1b | "physical/actual/true focal" is never asserted; the focal key is renamed with no alias left behind |
| N2 | wherever 5000 px and 922 px appear together, both semantic roles are named |
| N2b | 5000 px is never asserted to be a wrong or unphysical camera intrinsic |
| N3 | E2's 6.46 % is never called an independent run, test, validation or dataset |
| N4 | 6.46 % always carries the "same benchmark views" context |
| N5 | no "±0.3x pp tolerance" is stated as an uncertainty interval |
| N6 | the repeat-spread diagnostic and the two-execution difference exist as separate numbers |

37 checks → **45 checks, all passing**. The 37 earlier checks still pass, including
the CAM-EXP-002 pool-versus-sample separation, the 2,210 / 1,173 / 134 / 37
counts, the per-view terminology, the Experiment 1 unresolved cases, the E2
context rule, AnyCam provenance, and the untouched historical results.

## Numerical impact

None. Every value in `report_numbers.json` is unchanged; only three keys were
renamed and the accompanying descriptions rewritten. No file under any run's
`results/raw/` or `results/summary/` was modified.

---

# Fourth pass — CAM-EXP-002 magnitude and causal wording (2026-09-23)

Two remaining overstatements, both in `scripts/build_tables.py` and therefore in
every table it generated. Fixed at the generator, locked with four new checks.
No numerical value changed.

## 1. "nearly two orders of magnitude" → the measured factor

The CAM-EXP-002 hypothesis row recorded the result as *"confirmed, by nearly two
orders of magnitude"*. Two orders of magnitude is ~100×. The measured ratio is

```
2881.885 mm / 78.54 mm = 36.69
```

— about a third of what the phrase implies. The figure is now derived in
`report_numbers.json` as `cam002_root_error_reduction_factor` (and
`cam002_absolute_mpjpe_reduction_factor` = 42.01 for the absolute MPJPE), so the
ratio is computed from the two canonical medians and cannot drift from them.

The row now reads: *"supported; the median root error was approximately 36.69×
lower under the dataset-provided reference focal condition (2881.885 mm →
78.54 mm)"*. Guardrail section O forbids the "orders of magnitude" phrasing and
requires the factor to be quoted with both medians.

## 2. "the camera, not the hand model" → an upstream-priority claim

The same row's `next_implication` said *"the camera, not the hand model, is the
first thing to fix"*. Nothing in the evidence exonerates the hand-pose model.
Under the dataset-provided reference focal the absolute error does not vanish:

* median root error **78.54 mm**
* root-aligned MPJPE **34.404 mm**

The row now reads: *"camera focal handling is a major upstream source of
absolute-placement error and is therefore a justified calibration priority
before interpreting residual hand-model error"*, and its
`remaining_uncertainty` states the residuals explicitly. Guardrail section P
carries the safe framing in English and Korean.

## 3. C01's causal wording

C01 said the focal difference *"alone displaces the hand by metres"*. It is now
phrased as what was actually done — an intervention on fixed predictions:

> Holding all cached hand predictions fixed, replacing the pipeline's focal
> convention with the dataset-provided reference focal reduced the median root
> error from 2881.885 mm to 78.54 mm.

Its scope line now says that substantial absolute error remains under the
reference focal condition, so the claim is not an attribution of the whole
absolute error to the camera. The `wording_to_avoid` cell picked up both
retired phrases.

## New checks

| Check | What it enforces |
|---|---|
| O1 | "orders of magnitude" is never used for the CAM-EXP-002 improvement |
| O2 | no dichotomy that clears the hand-pose model ("camera, not the hand model", "first thing to fix") |
| O3a | the stored reduction factor equals the ratio of the two canonical medians |
| O3b | every improvement factor quoted next to those medians matches the canonical one |
| O4 | wherever the reference-focal result is claimed, the residual error (78.54 mm root, 34.404 mm root-aligned) is stated too |

45 checks → **50 checks, all passing.**

## Numerical impact

None. Two derived ratio entries were added to `report_numbers.json` (both
computed from existing canonical medians); no experimental value changed, and no
file under any run's `results/raw/` or `results/summary/` was modified.
