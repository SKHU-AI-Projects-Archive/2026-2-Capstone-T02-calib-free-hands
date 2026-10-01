# Official participant provenance

## Source

| field | value |
| --- | --- |
| title | GigaHands official repository README |
| URL | https://github.com/Kristen-Z/GigaHands |
| raw URL | https://raw.githubusercontent.com/Kristen-Z/GigaHands/main/README.md |
| access date | 2026-09-28 |
| local snapshot | `notes\gigahands_README_snapshot.md` |
| snapshot SHA-256 | `7082c858e8b8b00ba16977ea40f4f570fd07a51526c30cb60a45a77c71483a56` |

## The naming convention, verbatim

The README's own directory specification writes the take directory as:

```
p<participant id>-<scene>-<squence
```

and elsewhere:

```
p<participant id>-<scene>/
p<participant id>-<scene>_<squence
```

The README also states the dataset captures **56 subjects**.

## Why this is not filename intuition

CAM-EXP-009.3 refused to read `p41` as a participant because nothing in the
**locally shipped files** declared a participant field, and the spec forbids
inferring identity from a filename. That was the correct call on the evidence
available to it.

What is different here is that the dataset's own documentation **names the
field**. The README does not merely show example directory names — it writes the
path as `p<participant id>-<scene>-<squence id>/`, labelling the first token as
the participant id. That is a dataset-defined identifier whose semantics the
dataset documents.

The local layout matches that specification exactly:

```
demo_all/raw/hand_pose/p36-tea-0010/
                       ^^^ ^^^ ^^^^
                        |   |    +-- squence id
                        |   +------- scene
                        +----------- participant id
```

## Mapping

| sequence | participant | scene | sequence id |
| --- | --- | --- | --- |
| p36-tea-0010 | **p36** | tea | 0010 |
| p41-boxing-0021 | **p41** | boxing | 0021 |
| p41-plant-0004 | **p41** | plant | 0004 |
| p44-dog-0004 | **p44** | dog | 0004 |
| p52-instrument-0034 | **p52** | instrument | 0034 |

**5 sequences, 4 participants.** `p41` contributes two different
sessions/sequences.

## What was NOT used as an identity source

- **MANO `shapes` vectors** in `params/*.json`. Defining participant identity
  from fitted hand shape and then measuring hand-shape similarity would be
  circular; the answer would be built into the labels. Excluded, exactly as in
  CAM-EXP-009.3.
- Filename intuition.
- Any CAM-EXP-009.3 assumption.

## What this does not fix

Verifying identity does **not** fix statistical power. 4 participants remains
below the pre-set adequacy bar of 8, so
`SUBJECT_SEPARABILITY_UNDERPOWERED` / `PARTICIPANT_REANALYSIS_UNDERPOWERED`
still applies.
