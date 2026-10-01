# The GigaHands focal confound

## The problem

GigaHands was captured on a single rig whose cameras have a **very narrow focal
distribution**. Earlier runs in this programme established that a constant is
already a strong predictor on this dataset.

That means a method can look good here for a reason that has nothing to do with
the method: shrinking every estimate toward the rig's typical focal.

## The diagnostic baselines

- **`RIG_MEDIAN_TRAIN_ONLY`** — the median reference focal of the TRAIN physical
  cameras, applied to the TEST camera. It never touches the test focal. It is a
  **`DATASET_CONFOUND_DIAGNOSTIC`, not a deployment method**: in a real
  deployment there is no rig median to borrow.
- **`REF_FOCAL_ORACLE`** — the test reference focal itself, used only as an
  `UPPER_DIAGNOSTIC` for absolute 3D. Not an estimator.

If `RIG_MEDIAN_TRAIN_ONLY` beats the real methods, that is reported plainly, not
hidden.

## The correction-direction audit

For every unit, `Delta log f = log(f_M3 / f_M0)` is compared against the
TRAIN-only rig median. If nearly all corrections move toward that median, the
gain is better explained as **dataset shrinkage** than as hand-specific
evidence. The fraction is reported in `results/summary/rig_confound_summary.json`.

## The standing limitation

Whatever this run finds, GigaHands alone cannot establish general camera
generalisation. A positive result here justifies a **broader-intrinsics
evaluation** - a dataset with genuinely varied focal lengths, or controlled
self-captured calibration sequences - not a claim about cameras in general.

The `FINAL_CONFIRMATORY_HOLDOUT` (InterHand2.6M / HanCo) is **not opened** by
this experiment.
