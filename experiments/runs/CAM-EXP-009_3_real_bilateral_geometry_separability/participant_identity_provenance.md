# Participant identity provenance

## The requirement

The experiment as specified compares SAME-SUBJECT against DIFFERENT-SUBJECT
pairings. That requires knowing which sequences belong to the same person, and
the spec forbids inferring it from a filename.

## What was actually found

**Status: `SAME_SUBJECT_PROVENANCE_UNRESOLVED`.**

The five local sequences are:

- `p36-tea-0010`
- `p41-boxing-0021`
- `p41-plant-0004`
- `p44-dog-0004`
- `p52-instrument-0034`

Each carries a `p<NN>` prefix that *looks* like a participant id, giving four
candidate groups (`p36`, `p41`, `p44`, `p52`), with `p41` appearing twice.

But:

- **262 shipped metadata files were scanned** (every `.json`, `.txt` and `.csv`
  in the GigaHands tree outside the keypoint and bbox payloads). **None**
  declares a participant, subject or person field.
- No manifest produced by CAM-EXP-001 through CAM-EXP-009.2 declares one
  either.
- The dataset preparation record (`experiments/manifests/gigahands_demo_all.json`)
  lists the five sequence names and nothing about who performed them.

So the only available basis for asserting that `p41-boxing-0021` and
`p41-plant-0004` are the same person is the filename - which is exactly what
the spec rules out.

## The one tempting shortcut, and why it is refused

`hand_pose/<seq>/params/*.json` contains a fitted MANO `shapes` vector per
sequence. Two sequences with near-identical shape vectors would be suggestive
evidence of the same performer.

It is **not used**, because MANO shape *is* hand anatomy. Defining subject
identity from fitted hand shape and then measuring whether same-subject hands
have similar shape would be circular: the answer would be built into the
labels.

## Consequence for this experiment

The contrast is renamed to what the evidence supports:

| specified | reported as |
| --- | --- |
| same-subject pairing | **WITHIN_SEQUENCE** - left and right of the same sequence and camera |
| cross-subject pairing | **CROSS_SEQUENCE** - left and right from different sequences, same camera |

and the single same-prefix pair (`p41-boxing-0021` / `p41-plant-0004`) is
**excluded from the primary cross set**, because it is precisely the pair whose
subject relationship is unverified. It is reported separately as the
cross-sequence same-candidate-participant diagnostic.

`tables/participant_identity_audit.csv` records every sequence with
`identity_confidence = LOW_UNVERIFIED` and
`same_subject_comparison_allowed = NO`.
