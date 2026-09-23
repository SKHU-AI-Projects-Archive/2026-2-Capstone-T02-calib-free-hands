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

`report_numbers.json` holds **126** entries. Each carries value, unit,
experiment, source file, source column, statistical unit, evidence level and
notes. Every source file was verified to exist.

## Figure generation status

Seven main figures, each as PNG (300 dpi) and PDF, each with its plotted values
written to `figures/main/data/FigNN.csv`. `figures/appendix/` is intentionally
empty: the appendix should reference the original run figures listed in
`tables/appendix_figure_index.csv` (9 KEEP, 3 DROP with reasons) rather than
duplicate them.

## Quality checks

24/24 pass. The interesting ones:

* **C2** — the Fig06 curve matches the CAM-EXP-004.1 summary at every N ≥ 8, so
  the recomputation did not silently change anything.
* **D** — no file presents 1400 as an independent sample count, except where it
  is explicitly being warned against.
* **F** — the E2 selection-set figure never appears alone: wherever it is
  written, the independent re-run (6.46 %) and the leave-one-sequence-out range
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
