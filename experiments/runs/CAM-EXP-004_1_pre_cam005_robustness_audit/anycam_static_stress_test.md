# AnyCam supplementary static stress test

**Status: `SUPPLEMENTARY_STATIC_STRESS_TEST`.** This is not a benchmark and
AnyCam is not being ranked against the single-frame models. The question is
narrow: what does the official pretrained model do when the camera never moves?

CAM-EXP-004 answered that from source code alone
(`NOT_APPLICABLE_REQUIRES_CAMERA_MOTION`). This run tried to check it by
execution.

## What was attempted, and what happened

Official repository cloned at
`e609cc8a9e4ee8f78cf2ce39ebeb86b35e82d10d`, official checkpoint downloaded from
the official HuggingFace location the repo's own `hubconf.py` points at
(`fwimbauer/anycam_v1_seq8`, `pytorch_model.bin`, 459 833 700 bytes, plus
`config.yaml`), into an isolated `experiments/.venv-anycam`. No system CUDA or
driver was touched.

The end-to-end path (`torch.hub.load(..., 'AnyCam')` →
`model.process_video(frames)`) could **not** be executed on this machine. The
blocker is a genuine dependency deadlock on Windows, documented here in full
rather than worked around by editing the official code:

| Requirement | Consequence |
|---|---|
| `anycam/scripts/fit_video.py` applies `torch.compile` at import time | with torch 2.1.2 (our frozen calibration env) the import dies with `RuntimeError: Windows not yet supported for torch.compile`; it needs a much newer torch |
| AnyCam's depth backbone UniDepth imports `xformers.components.attention.NystromAttention` | that module was removed from modern xformers; it exists only up to `xformers<=0.0.28.post3`, which AnyCam's `requirements.txt` indeed pins |
| `xformers==0.0.28.post3` | has **no Windows wheel**; pip falls back to a source build, which fails, and it requires torch 2.5.1 |
| torch 2.5.1 on Windows | still fails `torch.compile` |

So the two requirements are mutually exclusive on this platform: new enough
torch for `torch.compile`, old enough xformers for UniDepth. Satisfying both
would require **editing the official source** (stripping `torch.compile`,
patching UniDepth's attention import) or a Linux machine. Editing the official
implementation and then reporting the output as "AnyCam" is exactly the
substitution this project forbids, so it was not done.

**Execution status: `BLOCKED_IMPLEMENTATION_ON_WINDOWS`** — recorded as a
platform limitation, not as a defect of AnyCam and not as a result about it.
`results/raw/anycam_static_predictions.csv.gz` is deliberately absent; no
placeholder numbers were fabricated.

## What *was* executed: the mechanism itself

The blocker sits in the depth backbone and the demo wrapper. The function that
actually decides AnyCam's focal length is plain PyTorch and imports fine, so the
mechanistic question could still be tested with **AnyCam's own code, unmodified**:

* `anycam.trainer.make_proj_from_focal_length` — builds the projection matrices
  for the focal candidates;
* `anycam.trainer.induce_flow_dist(depths, projs, rel_poses)` — the function
  `fit_video.py` uses to score those candidates before `proj_labels.argmax()`.

Using the package's own candidate grid (32 candidates, `focal_min = 0.1`,
`focal_max = 4.0`, `LOG_FOCAL_LENGTH_BIAS = 1.8`) with synthetic depth:

| Condition | max &#124;induced flow&#124; | spread across the 32 focal candidates | candidates distinguishable? |
|---|---|---|---|
| **static camera** (identity relative pose) | `1.79e-07` | **`2.73e-09`** | **no** |
| moving camera (0.3 translation over 8 frames) | `2.394` | `0.223` | yes |

Under a static camera the induced flow is zero to float precision and — the
decisive number — it is the **same** zero for every focal candidate. The
objective that selects the focal is exactly flat, so the argmax is decided by
nothing but the network's prior and numerical noise. With camera motion the same
function separates the candidates by eight orders of magnitude more.

`results/summary/anycam_identifiability.json`,
`results/raw/anycam_candidate_diagnostics.csv.gz`.

## Verdict

**The official AnyCam formulation is not identifiable in our static-camera
deployment condition.** The parameter it would need to estimate is not
constrained by the signal its method is built on. This is a statement about the
match between the method's assumptions and our setting, not about the method's
quality: for casual video with a moving camera, which is what the paper targets,
the same mechanism works.

Recorded as `METHOD_ASSUMPTION_MISMATCH`, consistent with CAM-EXP-004.

## What is still missing, honestly

The behavioural test — feed real static clips in and look at the numbers that
come out — was not run. The mechanistic test shows the selection objective is
degenerate; it does not show empirically what the model emits anyway, nor
whether the prior happens to land near the truth. The two planned conditions
(A: real static video with moving hands, B: the same frame repeated) remain
unexecuted, and the driver script `src/anycam_static_test.py` is kept so they can
be run unchanged on Linux. If a Linux environment becomes available, that is a
half-day of work and it should be done before any paper claims AnyCam was
evaluated.
