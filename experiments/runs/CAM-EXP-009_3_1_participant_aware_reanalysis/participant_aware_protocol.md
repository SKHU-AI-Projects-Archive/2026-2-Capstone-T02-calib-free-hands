# Participant-aware protocol

## Metric continuity — nothing new was introduced

Exactly CAM-EXP-009.3's representation and primary distance:

- 20 connected bones
- `p_b = l_b / sum_b l_b` (absolute hand size removed)
- `z_b = log(p_b + 1e-8)`
- `D(A,B) = median_b |z_A,b - z_B,b|`

No alternative metric was tried and no winner was hunted.

## Weighting rules

Stated before the numbers were produced, and enforced in code:

1. **Camera views are not participants.** Distances aggregate frame →
   sequence-camera → sequence → participant.
2. **`p41` does not get double weight.** Its two sessions are reduced to one
   participant value before any participant-level summary, so each of the four
   participants carries one vote.
3. **Cross-participant distances are summarised per ORDERED PARTICIPANT PAIR
   first**, then pooled. Each participant pair gets equal weight regardless of
   how many sequence pairs it contributes. Raw pair tables are kept
   separately.
4. **`p41` ↔ `p41` never enters `D_CROSS_PARTICIPANT`.** It is by construction a
   same-participant comparison, and it is analysed only as the cross-session
   diagnostic.
5. **Both directions are run**: LEFT→RIGHT and RIGHT→LEFT.

## Three quantities, never merged

| quantity | same participant? | same session? |
| --- | --- | --- |
| `D_WITHIN_SESSION` | yes | yes |
| `D_SAME_PARTICIPANT_CROSS_SESSION` | yes | **no** |
| `D_CROSS_PARTICIPANT` | no | no |

plus `D_REPEAT` (same hand, deterministic A/B frame split) as the measurement
noise baseline and `D_VIEW_REPEAT` (same hand, two cameras).

## Statistics

- **Permutation**: exact over 4! = 24 **participant-block** relabelings, with
  `p41`'s two sessions always moving together. The floor is 1/24 = 0.0417 and
  there are 4 units, so it is coarse and **supporting only**. This corrects
  CAM-EXP-009.3's sequence-level 5! = 120 test, which let `p41`'s two sessions
  move independently.
- **Bootstrap**: cluster = participant, 4 clusters, 10,000 iterations.
  Supplemental and unstable; a narrow-looking interval here is not strong
  evidence.
- **Two sequence-level metrics renamed, not deleted.** CAM-EXP-009.3's 53.7 %
  figure is retained as `WITHIN_SESSION_PAIRING_TOP1` /
  `SEQUENCE_PAIRING_DISCRIMINATION`. It is **not** participant identification.
- **Cross-session participant identification is not estimable** on this subset:
  only one participant has multiple sessions, so it cannot be made into a
  4-participant benchmark and is not reported as a primary metric.

## Participant-level templates — secondary

Each participant's sequence-level templates are aggregated with equal weight per
sequence. This is a participant-level **descriptive diagnostic**, not a
session-independent proof — `p41`'s template is an average of two sessions that
the cross-session diagnostic shows disagree strongly.
