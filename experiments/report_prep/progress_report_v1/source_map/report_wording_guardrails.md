# Wording guardrails for the progress report

Each item is a phrase that is easy to write and would overstate what we
measured. The replacement is not softer for its own sake — it is what the
evidence actually supports.

---

## A. GigaHands 2D annotations

**Do not write:** "GT 2D", "ground-truth 2D keypoints".

**Write:** "provided 2D annotation", "dataset-provided 2D observation".

**Why:** the released 2D are per-view detections with confidence values, not
curated ground truth. CAM-EXP-001.1 / 001.2 found both an invalid all-zero
pattern and per-camera left/right hand swaps inside them.

## B. GigaHands 3D

**Do not write:** "ground truth 3D", "the true hand position", or anything
implying an independent external reference.

**Write:** "provided 3D", "reference 3D", "dataset-provided 3D".

**Why:** we never had an independent measurement of where the hands were.
CAM-EXP-001.3 reconstructed 3D from the released 2D and compared it with the
provided 3D; both come from the same dataset, so the result is
*self-consistency*, not accuracy.

## C. The all-zero 2D entries

**Do not write:** "the official sentinel value", "the documented missing-data
marker".

**Write:** "observed invalid all-zero pattern".

**Why:** no GigaHands documentation defines `(0,0)` with confidence 1.0 as a
sentinel. We observed the pattern and decided how to treat it; that is a
different kind of statement.

## D. CAM-EXP-002's scope

**Do not write:** "camera calibration sensitivity", "intrinsic sensitivity
analysis".

**Write:** "focal-length sensitivity of absolute hand depth".

**Why:** the experiment perturbed the focal length and measured absolute depth.
Principal point, anisotropic focal and distortion were never varied downstream.
Registered as an open issue; closing it means either narrowing the claim or
running CAM-EXP-002.1.

## E. The E2 ensemble

**Do not write:** "our method", "our final method", "the proposed ensemble".

**Write:** "frozen exploratory ensemble", "the most promising baseline found so
far", "frozen candidate baseline".

**Why:** E2 was chosen because it scored best among five candidates on the same
175 views it is reported on. It is frozen so later work cannot quietly reshape
it, not because it is a finished contribution.

## F. Quoting E2's performance

**Do not write:** "E2 achieves 6.14 %".

**Write:** all three numbers together — "6.14 % on the selection set, 6.46 % on
an independent re-run, and 5.73–8.80 % across leave-one-sequence-out folds".

**Why:** the single number looks like a measurement of the method. The three
together show what is actually known: the direction is solid, the exact value is
not yet pinned down.

## G. The frame-count result

**Do not write:** "multi-frame aggregation does not help", "8 frames are always
enough".

**Write:** "within the evaluated GigaHands static-camera condition, up to 64
frames, aggregating more frames did not improve focal accuracy".

**Why:** we measured one rig up to N = 64. The mechanism (bias, not noise) makes
the result plausible elsewhere, but it was not tested elsewhere.

## H. Statistical size

**Do not write:** "n = 1400", "1400 independent samples".

**Write:** "175 fixed-camera views, 8 frames per view, 1400 frames in total;
statistical unit = view".

**Why:** 8 frames of one static camera are 8 repeated observations of the same
quantity. CAM-EXP-004.1 showed the frame-level intervals were about 2.5× too
narrow.

## I. AnyCam

**Do not write:** "AnyCam performed poorly", "AnyCam failed", "we benchmarked
AnyCam".

**Write:** "AnyCam's official focal-selection mechanism relies on
camera-induced motion, so under our static-camera condition the focal candidates
were not identifiable."

**Why:** no focal-accuracy number for AnyCam exists, and none should. The end-to-end
inference could not run on Windows; what we tested was the identifiability of its
own scoring function. The finding is a mismatch between its assumptions and our
setting.

## J. GeoCalib numbers

**Do not write:** a bare GeoCalib or E2 figure with no tolerance.

**Write:** the number plus "GeoCalib은 동일 입력에서도 실행 간 변동이
관찰되었으나, 전체 중앙값 변화는 약 0.32 %p로 주요 결론을 변경하지 않았다."
(details in the appendix).

**Why:** GeoCalib is not run-to-run deterministic on byte-identical input.

## K. The ±5 % target

**Do not write:** "the required accuracy is ±5 %".

**Write:** "a ±5 % focal target, which corresponds to roughly 33 mm of depth
displacement in this working-distance regime (CAM-EXP-002)".

**Why:** the target is ours, derived from one sensitivity experiment on one
dataset. It is a working threshold, not a standard.

## L. Dataset sizes

**Do not write:** anything implying the subsets shrank because of failures.

**Write:** camera-clean, hand-clean and bimanual-clean are different subsets
because a camera experiment and a hand experiment need different things to be
valid.

**Why:** hand annotation quality never removed a camera-valid view; that was a
deliberate design rule.
