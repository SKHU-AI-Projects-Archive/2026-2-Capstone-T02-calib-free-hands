# CalibFreeHands UX / Result / Comparison Specification

**Status:** Frozen implementation contract for the next milestones
**Repository:** `D:\2026-2-Capstone-T02-calib-free-hands`  
**Application root:** `CalibFreeHands/`  
**Working branch:** `CalibFreeHands`

**Current implementation status**
- Milestone A — implemented and manually validated
- Milestone B — implemented and manually validated
- Milestone C-1 — implemented and manually approved after Fix Pass
- Milestone C-2A — implemented and automated tests pass — Metric Engine
- Milestone C-2B — foundation checkpoint saved; Presentation Redesign is next
- Then: Milestone D — Calibration model expansion
- Later: Milestone E — Performance / packaging

---

# C-2A implementation status

The Qt-independent reference metric engine aligns canonical results by source `frame_indices` and left/right hand slots, skips undetected/non-finite joint pairs, and reports spatial errors in millimetres. It implements per-joint error, wrist error, raw camera-space MPJPE, median/max/RMSE, depth bias, translation errors when compatible arrays exist, relative wrist error with its camera-to-wrist denominator, camera-space MPVPE with matching MANO faces, and supplemental PA-MPJPE/PA-MPVPE similarity alignment. Raw result arrays are not modified.

F@5/F@15, AUC-J/AUC-V, MPFJE/MPFVE, Jitter, and RTE remain `PROTOCOL_PENDING`: the official WiLoR project and paper report benchmark metrics, but this implementation has not established a reproducible protocol compatible with the application's saved camera-space Reference data. See the [official WiLoR repository](https://github.com/rolpotamias/WiLoR) and [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html).

---

# 1. Purpose

CalibFreeHands is a Windows research/demo application for comparing how different camera-calibration methods affect reconstructed 3D hands.

The user-facing research question is:

> If the same RGB media and the same hand-reconstruction model are used, how much does the resulting 3D hand differ when the camera focal/calibration source changes?

The application must support:

1. processing one or many media files;
2. processing one media file with multiple Calibration/Reconstruction configurations;
3. storing one user-facing `.cfh` result per configuration;
4. reopening several results from the same source;
5. selecting one result as a Reference;
6. comparing every other result against that Reference numerically and visually;
7. changing visible metrics while the Viewer is running;
8. showing any practical number of results, not only two or three;
9. preserving scientific raw coordinates while allowing flexible visualization.

---

# 2. Scientific language

## 2.1 Checkerboard is a Reference, not automatically true GT

The preferred terminology in code, UI, documentation, and files is:

```text
Reference
Checkerboard Reference
Reference 3D
```

A checkerboard-calibrated focal/result may be treated as an **assumed reference** for the experiment.

Do not automatically claim:

```text
true 3D Ground Truth
mocap Ground Truth
absolute Ground Truth
```

unless the experiment genuinely provides such data.

For informal presentation, the team may describe the checkerboard result as a practical GT/reference, but implementation terminology should remain `Reference`.

---

# 3. Preserve scientific contracts

The following existing behavior is frozen.

Focal propagation:

```text
tz_new = tz_source * target_focal / source_focal
```

- camera translation X unchanged
- camera translation Y unchanged
- camera translation Z rescaled by focal ratio

Final camera-space joints:

```text
keypoints_3d_relative + corrected_translation
```

Final camera-space MANO vertices:

```text
vertices_relative + corrected_translation
```

Coordinate convention:

```text
X = image right
Y = image down
Z = camera forward
unit = metre
```

`RAW_TO_VIEW` is display-only.

Missing hands:

```text
detected = false
coordinates = NaN
```

Do not mutate raw scientific data to make plots visually closer.

---

# 4. Internal reuse remains mandatory

Calibration and Reconstruction remain separated.

Example:

```text
Checkerboard Reference + Original WiLoR
Original AnyCalib      + Original WiLoR
Fine-tuned AnyCalib    + Original WiLoR
Original GeoCalib      + Original WiLoR
HandCalib              + Original WiLoR
```

For the same source media, the same Reconstruction backend/variant should be reused whenever scientifically valid.

A multi-calibration comparison must not rerun identical raw WiLoR inference once per Calibration configuration.

Conceptually:

```text
RGB media
 ├─ Calibration backend(s)
 │      ↓
 │  CalibrationResult(s)
 │
 └─ Reconstruction backend(s)
        ↓
     RawPoseResult(s)

CalibrationResult + RawPoseResult
             ↓
       CanonicalResult
             ↓
          .cfh
```

---

# 5. Input / Queue / Result behavior

These Milestone A/B decisions remain frozen.

## 5.1 Main controls

```text
Open...
Remove
Clear
Start Processing
View Results...
Cancel
```

## 5.2 Queue

`Start Processing` processes the full queue sequentially.

States:

```text
Waiting
Processing
Completed
Failed
Cancelled
```

## 5.3 Result unit

One user-facing result is:

> one source media + one Calibration model + one Reconstruction model = one `.cfh`

Examples:

```text
1_insert__Original-AnyCalib__Original-WiLoR.cfh
1_insert__Original-GeoCalib__Original-WiLoR.cfh
```

Checkerboard Reference includes its user-specified focal:

```text
1_insert__Checkerboard-Reference-1200px__Original-WiLoR.cfh
1_insert__Checkerboard-Reference-1280.5px__Original-WiLoR.cfh
```

Predicted focal values from learned Calibration models do not belong in filenames.

---

# 6. `.cfh` behavior

`.cfh` remains a ZIP-compatible user-facing result bundle.

Conceptual minimum contents:

```text
result.npz
metadata.json
```

Source media is referenced, not embedded.

Store at least:

```text
source_media_name
source_media_path
source_media_sha256
```

The SHA256 is the authoritative same-source identity.

If source media is missing, stored 3D data must still open.

---

# 7. Preset system

The Preset system remains:

```text
Built-in Preset
User Preset
Custom (unsaved)
```

User Presets store research-processing settings only:

- configuration rows
- row names
- Calibration model IDs
- Reconstruction model IDs
- Reference Focal where applicable
- Sampling
- Sample count
- Aggregation

They do not store:

- Theme
- Language
- Output path
- Viewer colors
- window geometry

Preset actions:

```text
Save Preset...
Manage Presets...
Import Preset...
Export Preset...
Set as Default
```

Modified User Presets remember their origin so the user may explicitly:

```text
Overwrite
Save As...
Cancel
```

Built-in presets cannot be overwritten.

---

# 8. Comparison Viewer — overall purpose

The Comparison Viewer is not only a visualization tool.

Its primary purpose is:

> **A demonstration/research dashboard that shows how multiple 3D reconstruction results differ from a selected Reference, both visually and quantitatively.**

The Viewer must support:

- one source video/image;
- N selected `.cfh` results from that same source;
- a runtime-selectable Reference result;
- runtime-selectable metrics;
- Overlay visualization;
- Split/adaptive multi-panel visualization;
- optional separate/detached windows;
- no inference rerun for display changes.

---

# 9. N-result requirement

Do not hard-code the Comparison Viewer to exactly:

```text
2 results
3 results
```

It must be architected for:

```text
N results
```

Examples include:

```text
2
3
4
5
```

and should remain functional for more, subject to practical UI limits.

The number of selected result files determines the number of result views.

---

# 10. Overlay mode

Overlay remains conceptually:

```text
┌────────────────────┬────────────────────────────┐
│ Original media     │ Overlaid 3D results       │
└────────────────────┴────────────────────────────┘
```

All selected result hands occupy the same scientific camera-space coordinate system.

Do not introduce arbitrary per-result translations or scales.

---

# 11. Split / Adaptive Grid mode

The source media must remain visible in Split mode.

The previous behavior where Split removed the source and showed only 3D views is not acceptable.

For a few results:

```text
Original media | Reference 3D | Result B 3D | Result C 3D
```

For more results, use an adaptive grid instead of forcing every panel into one extremely narrow row.

Example:

```text
┌───────────────┬───────────────┬───────────────┐
│ Original      │ Reference     │ Original      │
│ media         │ 3D            │ AnyCalib 3D   │
├───────────────┼───────────────┼───────────────┤
│ Fine-tuned    │ GeoCalib      │ HandCalib     │
│ 3D            │ 3D            │ 3D            │
└───────────────┴───────────────┴───────────────┘
```

Requirements:

- source media always occupies one panel;
- every selected result has one 3D panel;
- current frame remains synchronized;
- camera views remain synchronized by default;
- adding more results should not require new hard-coded UI branches.

---

# 12. Separate Windows / Detach

The default demo view is one large Comparison Viewer with multiple panels.

Also provide:

```text
Separate Windows
```

or:

```text
Detach
```

to open individual result/source views in separate windows.

Initial requirement:

- detaching does not rerun inference;
- frame/playback remains synchronized where practical;
- detached 3D cameras may remain synchronized;
- recombining can be deferred.

---

# 13. Runtime Reference selection

The Viewer must expose a Reference selector.

Example:

```text
Reference [Checkerboard Reference 1500 px ▼]
```

Behavior:

- if exactly one selected result uses Checkerboard Reference, choose it by default;
- if multiple Checkerboard Reference results exist, select deterministically and let the user change it;
- if none exists, select a sensible result and clearly indicate that there is no checkerboard reference;
- user may change Reference while Viewer is open;
- changing Reference recomputes/reselects metrics from loaded arrays only;
- no model inference rerun;
- no source-media reload unless already required by existing Viewer behavior.

The Reference result itself should display `REFERENCE` rather than misleading non-zero comparison values.

---

# 14. Metric categories

Metrics are grouped by scientific meaning.

Primary categories:

```text
Reference Accuracy
Pose / Mesh Shape
Temporal Stability
```

---

# 15. Reference Accuracy metrics — primary

These metrics directly compare candidate camera-space results with the selected Reference.

They are the most important metrics for the HandCalib camera-calibration experiment.

## 15.1 Per-Joint 3D Error

For corresponding joint `j`:

```text
e_j = || P_candidate[j] - P_reference[j] ||_2
```

Use canonical absolute camera-space coordinates.

Stored coordinates are metres.

Display in:

```text
mm
```

so:

```text
display_mm = error_m * 1000
```

Compute only for valid aligned detections.

The joint correspondence is the existing canonical 21-joint ordering.

## 15.2 Wrist Position Error

Joint 0 is the canonical wrist.

```text
Wrist Error =
|| wrist_candidate - wrist_reference ||_2
```

Display in mm.

## 15.3 Raw Camera-Space MPJPE

For valid corresponding joints:

```text
MPJPE =
mean_j || P_candidate[j] - P_reference[j] ||_2
```

No Procrustes alignment.

No translation removal.

No scale normalization.

Display in mm.

Suggested UI display name:

```text
Camera-space MPJPE
```

## 15.4 Median Joint Error

Median of valid per-joint Euclidean errors.

Display in mm.

## 15.5 Max Joint Error

Maximum valid per-joint Euclidean error.

Display in mm.

## 15.6 Joint Error RMSE

```text
sqrt(mean(e_j^2))
```

Display in mm.

---

# 16. Depth metrics

## 16.1 Wrist Depth Error

Signed:

```text
candidate_wrist_Z - reference_wrist_Z
```

Absolute:

```text
abs(candidate_wrist_Z - reference_wrist_Z)
```

Primary display should use absolute magnitude in mm.

## 16.2 Mean Joint Depth Error

```text
mean_j |Z_candidate[j] - Z_reference[j]|
```

Display in mm.

## 16.3 Signed Mean Depth Bias

```text
mean_j (Z_candidate[j] - Z_reference[j])
```

Display in mm.

---

# 17. Camera Translation metrics

If both results contain compatible camera translation:

## 17.1 Translation Vector Error

```text
|| t_candidate - t_reference ||_2
```

Display in mm.

## 17.2 Translation Z Error

```text
|tz_candidate - tz_reference|
```

Display in mm.

## 17.3 Translation XY Error

```text
sqrt((tx_c - tx_r)^2 + (ty_c - ty_r)^2)
```

Display in mm.

If unavailable, show `N/A`.

Do not fabricate translation from joints.

---

# 18. Relative / percentage metrics

Percentage metrics are secondary and require an explicit denominator.

## 18.1 Relative Wrist Error

```text
Relative Wrist Error (%) =
Wrist Position Error
/
Reference camera-to-wrist distance
* 100
```

with:

```text
Reference camera-to-wrist distance = || wrist_reference ||
```

The UI/help text must expose this denominator.

If invalid, return N/A.

---

# 19. Raw Mesh Error

If both results contain compatible MANO vertices:

## 19.1 Camera-space MPVPE

```text
MPVPE =
mean_v || V_candidate[v] - V_reference[v] ||_2
```

No Procrustes alignment.

Display in mm.

Only calculate when vertex counts/topology are compatible.

---

# 20. Procrustes-aligned metrics — supplemental

These are useful for local pose/shape agreement but are not primary calibration metrics.

## 20.1 PA-MPJPE

Apply a clearly defined similarity Procrustes alignment of candidate joints to Reference joints, then compute mean joint error.

Display in mm.

## 20.2 PA-MPVPE

Same concept for compatible MANO vertices.

Display in mm.

## 20.3 Interpretation

Do not use PA-MPJPE alone to claim improved camera calibration.

A result may have:

```text
low PA-MPJPE
high Camera-space MPJPE
```

which can indicate similar local articulation but incorrect global placement.

---

# 21. F@5 / F@15

WiLoR benchmark terminology includes:

```text
F@5
F@15
```

Do not invent an implementation merely because the names exist.

Before implementation, verify/reproduce the exact benchmark protocol for the available Reference representation.

Only expose when the protocol is explicitly defined and tested.

Until then:

```text
PROTOCOL_PENDING
```

---

# 22. AUC-J / AUC-V

WiLoR reports:

```text
AUC_J
AUC_V
```

under benchmark-specific protocols.

Implementation rule:

- verify exact evaluation curve/range/protocol;
- implement only when reproducible from current data;
- cover with deterministic tests;
- otherwise keep unavailable.

Do not manufacture a generic AUC definition.

---

# 23. Temporal Stability metrics

These answer:

> How stable is this reconstructed sequence over time?

They are separate from Reference Accuracy.

## 23.1 MPFJE

WiLoR describes mean per-frame Euclidean joint displacement between consecutive frames.

Conceptually:

```text
MPFJE =
mean over valid consecutive frames and joints
|| J[t] - J[t-1] ||_2
```

Units/scaling must be documented explicitly.

## 23.2 MPFVE

Analogous consecutive-frame vertex displacement:

```text
MPFVE =
mean over valid consecutive frames and vertices
|| V[t] - V[t-1] ||_2
```

Only when compatible vertices exist.

## 23.3 Jitter

WiLoR uses joint-motion jerk.

Exact derivative/discretization and units must be verified before exposing a number.

Until verified:

```text
PROTOCOL_PENDING
```

## 23.4 RTE

WiLoR describes RTE as a global Root Translation Error related to wrist displacement across frames.

Exact protocol/normalization must be verified.

Do not confuse it with static Wrist Position Error.

Until verified:

```text
PROTOCOL_PENDING
```

---

# 24. Metric implementation states

Metrics use explicit states:

```text
READY
PROTOCOL_PENDING
UNAVAILABLE_FOR_DATA
```

The UI only allows metrics valid for loaded data/protocol.

Never return fake zero values for unsupported metrics.

---

# 25. Frame alignment and hand matching

Reference-vs-candidate comparisons must align:

- source frame;
- hand side/identity;
- canonical joint index.

Requirements:

- respect frame indices/timestamps;
- respect `detected`;
- NaN is not zero;
- left compares with left;
- right compares with right;
- invalid pairs are skipped;
- expose valid sample count where useful.

---

# 26. Per-frame vs sequence summaries

## 26.1 Current-frame metrics

Suitable for live playback:

```text
Wrist Error
Camera-space MPJPE
Median Joint Error
Max Joint Error
Depth Error
Translation Error
Per-Joint Error
```

## 26.2 Sequence summaries

Optional:

```text
Mean over frames
Median over frames
RMSE over frames
95th percentile
Valid frame count
```

Always label summary scope clearly.

---

# 27. Per-Joint Error UI

Support all canonical 21 joint errors.

Optional panel:

```text
Joint                     Error
Wrist                     24.3 mm
Thumb 1                   ...
...
Pinky Tip                 ...
```

Use existing canonical joint ordering/naming.

Hide by default if necessary for demo clarity.

---

# 28. Joint error visualization

Optional toggle:

```text
Show Error Vectors
```

Draw residual lines from Reference joints to candidate joints.

Requirements:

- display-only;
- no raw-data mutation;
- valid paired joints only;
- avoid clutter with many results.

This may follow basic metric text/table support.

---

# 29. Runtime metric selector

Metrics must be selectable while Viewer is open.

Preferred:

```text
Metrics ▾
```

with checkable items grouped by category.

Conceptual menu:

```text
Reference Accuracy
☑ Wrist Error
☑ Camera-space MPJPE
☑ Wrist Depth Error
☐ Median Joint Error
☐ Max Joint Error
☐ Per-Joint Error
☐ Mean Joint Depth Error
☐ Camera Translation Error
☐ Relative Wrist Error

Pose / Mesh
☐ Camera-space MPVPE
☐ PA-MPJPE
☐ PA-MPVPE
☐ F@5
☐ F@15
☐ AUC-J
☐ AUC-V

Temporal Stability
☐ MPFJE
☐ MPFVE
☐ Jitter
☐ RTE
```

Protocol-pending metrics are disabled/omitted with explanation.

---

# 30. Default visible metrics

Do not enable everything by default.

Recommended default:

```text
Wrist Error
Camera-space MPJPE
Wrist Depth Error
```

Optional fourth default when available:

```text
Translation Z Error
```

---

# 31. Metric display

Every non-reference result should have an associated readout.

Example:

```text
Original AnyCalib
Wrist Error          24.3 mm
Camera-space MPJPE   18.7 mm
Wrist Depth Error    19.1 mm
```

Reference result:

```text
Checkerboard Reference 1500 px
REFERENCE
```

---

# 32. Runtime updates

When frame changes:

- current-frame metrics update immediately;
- no inference;
- no bundle reopen.

When selected metrics change:

- display updates immediately;
- no inference.

When Reference changes:

- recompute from loaded arrays;
- no inference.

---

# 33. Metric caching

Expensive sequence metrics may be cached.

Cache key should include:

```text
reference result ID
candidate result ID
metric ID
protocol/version
```

Display/layout/color changes must not invalidate scientific caches.

---

# 34. Result visibility controls

Each result uses:

```text
[standard checkbox] [color swatch] [result name]
```

Example:

```text
☑ ■ Checkerboard Reference
☑ ■ Original AnyCalib
```

Fix the current checkbox-indicator bug.

Do not color-style the checkbox indicator itself in a way that hides checked state.

---

# 35. Color semantics

Frozen mapping:

```text
Result identity      = base hue
Handedness           = deterministic variants of base hue
Detection Confidence = optional whole-hand color mapping or opacity input
```

Left/right variants must be distinguishable but related.

The existing result field named `confidence` stores the YOLO hand bounding-box detector score with shape `[frame, hand]` (`left`, then `right`). It is not emitted by the WiLoR reconstruction model. In user-facing text call it **Hand Detection Confidence**; compact controls may say **Detection confidence**.

Read-only inspection of three local Original WiLoR `.cfh` samples found shape `(570, 2)`, dtype `float32`, and finite values approximately `0.3284`–`0.8926`; their confidence arrays were identical across the inspected calibration variants. This is an observation about those files, not a universal schema constraint or a claim about every model run. The local WiLoR source tree does not pin a separate upstream revision.

This score is not WiLoR reconstruction confidence, 3D-pose correctness probability, joint correctness probability, joint confidence, or an uncertainty estimate. A high detector score does not guarantee an accurate 3D reconstruction. Never copy the hand score to 21 joints or fabricate per-joint values.

Detection confidence may be shown as one value/bar per hand, used as a whole-hand opacity input, or mapped to whole-hand color. These controls are display-only and must not alter result arrays or metric calculations. Color-by-confidence must remain visually distinct from result identity; prefer a perceptually ordered sequential scale over a rainbow scale.

---

# 36. User-selectable colors

Each result has a clickable base-color swatch.

Changing color is display-only.

It must not modify:

- `.cfh`;
- coordinates;
- detection confidence;
- metric values.

Persistence across sessions is optional initially.

---

# 37. Mesh opacity

Compact UI:

```text
Mesh opacity [────●────]
```

Mesh opacity controls mesh rendering only. **Use Detection Confidence as opacity** is a separate optional whole-hand visualization. The two values must not be conflated. Disable the mesh-opacity control when Mesh is off so the control cannot imply an active effect.

---

# 38. Existing Viewer controls

Preserve:

- Joint
- Bone
- Mesh
- mesh opacity
- Use Detection Confidence as opacity
- Left Hand
- Right Hand
- Grid
- Axis
- Front
- Side
- Top
- Fit Hands
- Reset View
- Previous
- Play / Pause
- Next
- frame slider
- metrics CSV
- metrics JSON

Comparison playback uses `CanonicalResult.timestamps` with a monotonic wall clock, skips stale frames when rendering is behind, and wraps at the end as in the current viewer. Invalid or non-increasing timestamps use the median positive timestamp interval when available, then recorded source FPS, then 30 fps. Offer compact playback speeds of 0.1×, 0.25×, 0.5×, 0.75×, 1.0×, 1.25×, 1.5×, and 2.0×; changing speed preserves the current source position. Pausing holds the displayed frame.

---

# 39. Metric export

Extend CSV/JSON metric export so records clearly identify:

- source identity;
- Reference result;
- candidate result;
- metric ID/name;
- protocol/version;
- frame index;
- hand side;
- joint index/name where applicable;
- value;
- unit;
- valid status.

Do not export ambiguous unlabeled numbers.

---

# 40. Milestone C-2A — Metric Engine

C-2A comes before major dashboard UI changes.

Goal:

> Make metric calculations scientifically correct and independently testable.

## 40.1 Required first metrics

Implement/test:

```text
Per-Joint 3D Error
Wrist Position Error
Camera-space MPJPE
Median Joint Error
Max Joint Error
Joint Error RMSE
Wrist Depth Error
Mean Joint Depth Error
Signed Mean Depth Bias
Camera Translation Vector Error
Translation Z Error
Relative Wrist Error
```

Where data permits:

```text
Camera-space MPVPE
PA-MPJPE
PA-MPVPE
```

## 40.2 Secondary metrics

Implement only after exact protocol verification:

```text
F@5
F@15
AUC-J
AUC-V
MPFJE
MPFVE
Jitter
RTE
```

Do not guess protocols.

## 40.3 Metric registry

Prefer a lightweight descriptor/registry containing:

```text
id
display name
category
unit
frame/sequence scope
data requirements
reference requirement
protocol version
implementation status
```

Do not build a generic plugin framework.

---

# 41. C-2A tests

Use synthetic data with obvious analytical answers.

## Exact match

All Reference errors = 0.

## Known +10 mm X translation

Expected:

```text
all joint errors = 10 mm
Wrist = 10 mm
MPJPE = 10 mm
Depth = 0
```

## Known +25 mm Z translation

Expected:

```text
MPJPE = 25 mm
Wrist Depth Error = 25 mm
Mean Joint Depth Error = 25 mm
```

## Single-joint outlier

Validate:

```text
mean
median
max
RMSE
```

## Missing/NaN

Invalid samples skipped; no zero substitution.

## Left/right

No cross-hand matching.

## PA test

Candidate differs only by known similarity transform.

Expected:

```text
Camera-space MPJPE > 0
PA-MPJPE ≈ 0
```

---

# 42. Milestone C-2B — Comparison Dashboard

After C-2A passes:

- N-result loading;
- runtime Reference selector;
- runtime metric multi-select;
- current-frame metrics;
- optional sequence summaries;
- per-joint table;
- Overlay;
- adaptive Split;
- source media always visible;
- checkbox visual fix;
- color swatches;
- handedness variants;
- optional Hand Detection Confidence value/bar/whole-hand color/opacity;
- compact Mesh opacity, separate from Detection Confidence visualization;
- Separate Windows / Detach.

---

# 43. Demo-first layout

Priority:

1. source video;
2. result identity;
3. selected Reference;
4. 3D comparison;
5. 2–4 selected metrics;
6. optional advanced tools.

Do not show every supported metric by default.

---

# 44. No inference for Viewer actions

The following must never rerun models:

```text
Overlay ↔ Split
adaptive grid changes
Reference selection
metric selection
color selection
Left/Right toggles
Joint/Bone/Mesh
Grid/Axis
Opacity
Detach
frame navigation
```

---

# 45. No scientific-data mutation

Metrics/visualization never rewrite:

- canonical arrays;
- raw joints;
- vertices;
- focal provenance;
- camera translation;
- source hashes.

---

# 46. Milestone D — Calibration model expansion

After C-2:

```text
Original AnyCalib
Fine-tuned AnyCalib
Original GeoCalib
HandCalib
Checkerboard Reference
```

Only verified runnable backends may be enabled.

Fine-tuned AnyCalib remains distinct from HandCalib.

GeoCalib requires upstream/license/runtime/API/focal-semantics verification before integration.

HandCalib remains unavailable until a real scientifically defined runtime model exists.

---

# 47. Milestone E — Performance / packaging

Later:

```text
long-video RAM usage
lazy media decoding
viewer frame cache
backend/model VRAM lifecycle
portable Windows build
release packaging
```

Realtime remains deferred unless reprioritized.

---

# 48. Current milestone status

```text
A     COMPLETE
B     COMPLETE
C-1   IMPLEMENTED + MANUALLY APPROVED
C-2A  IMPLEMENTED + AUTOMATED TESTS PASS
C-2B  PRESENTATION AND MEDIA HARDENING IMPLEMENTED; FINAL VALIDATION IN PROGRESS
D     LATER
E     LATER
```

If C-1 changes are still uncommitted, finalize their normal commit/push before beginning C-2.

---

# 49. Acceptance criteria — C-2A

C-2A is complete only when:

- Reference/candidate alignment is explicit;
- 21-joint correspondence preserved;
- Per-Joint Error implemented;
- Wrist Error implemented;
- Camera-space MPJPE implemented;
- Median/Max/RMSE implemented;
- Depth metrics implemented;
- translation metrics implemented when data exists;
- Relative Wrist denominator documented;
- MPVPE guarded by compatible topology;
- PA-MPJPE tested;
- PA-MPVPE guarded appropriately;
- unsupported benchmark protocols are not fabricated;
- missing/NaN behavior tested;
- left/right matching tested;
- units explicit;
- synthetic analytical tests pass;
- raw arrays unchanged.

---

# 50. Acceptance criteria — C-2B

C-2B is complete only when:

- source plus N results supported;
- Overlay works;
- Split always retains source media;
- Split uses adaptive N-panel layout;
- runtime Reference selector works;
- Checkerboard Reference auto-selected when unambiguous;
- metric multi-select works live;
- concise default metric set;
- current-frame values update during playback;
- Reference change needs no inference;
- Per-Joint table available;
- visibility checkbox visually correct;
- separate color swatch;
- selectable base hue;
- left/right variants distinguishable;
- Hand Detection Confidence is the YOLO box detector score and is stored as `[frame, hand]`;
- Detection Confidence may affect whole-hand display only, never per-joint confidence or metric weighting;
- compact Mesh opacity, separate from Detection Confidence visualization;
- Separate Windows / Detach works;
- Viewer-only actions rerun no inference;
- metric export works;
- missing-source results still viewable.

---

# 51. Frozen decisions

1. Checkerboard is `Reference`, not automatically true GT.
2. One source + one Calibration + one Reconstruction = one `.cfh`.
3. Same-source identity uses SHA256.
4. Internal intermediate files remain hidden.
5. No automatic preview video.
6. Existing focal propagation remains unchanged.
7. Existing raw coordinate convention remains unchanged.
8. Same Reconstruction backend is reused when scientifically valid.
9. Comparison supports N results.
10. Split always shows Original media.
11. Default Split is one adaptive multi-panel window.
12. Separate Windows is secondary.
13. Reference can change at runtime.
14. Metrics can change at runtime.
15. Metric changes never rerun inference.
16. Primary calibration metrics are raw camera-space Reference errors.
17. PA metrics are supplemental.
18. 21-joint errors are supported.
19. Wrist Error is a primary demo metric.
20. Percentage errors require explicit denominators.
21. Undefined benchmark protocols are never fabricated.
22. Configuration identity = base hue.
23. Handedness = hue variants.
24. The legacy `confidence` result field means YOLO Hand Detection Confidence `[frame, hand]`; opacity is only one optional visualization of that score.
25. Viewer colors do not mutate result files.
26. C-2A correctness precedes C-2B dashboard expansion.
27. New Calibration models belong to D.


---

# 52. Change control

Every implementation agent must:

1. read this specification first;
2. inspect current repository before editing;
3. preserve existing working features;
4. not silently change metric definitions;
5. document metric formulas and units;
6. distinguish Reference Accuracy, Pose/Mesh, and Temporal Stability;
7. never guess unknown metric protocols;
8. add analytical tests for implemented metrics;
9. avoid hard-coded result counts;
10. avoid inference in Viewer-only interactions;
11. preserve backward compatibility where practical;
12. update this specification with any explicitly approved frozen-decision change.

## C-2B UI Fix Pass presentation contract

- Comparison Viewer prioritizes the source and 3D viewports; it does not show a permanent metric table.
- Selected frame-level metrics appear in a compact, internally scrollable overlay grouped by result. The selected Reference is marked `REFERENCE` and does not show self-comparison metrics.
- Per-Joint Error opens its 21-joint Left/Right detail table on demand. Error vectors remain an independent visualization.
- View, Results, Reference, Metrics, Detection confidence, and color controls stay compact; result swatches remain separate from visibility controls.
- Split mode retains the source panel and lays out visible result panels in an adaptive grid. The Comparison Viewer minimum size is 1280×720; 1920×1080 is the visual review target.
- Checkerboard's Reference Focal column header remains the label; its row contains the value control without a duplicate label.
- This pass changes presentation only. Metric definitions, result arrays, and inference behavior remain unchanged.

## C-2B Presentation Redesign contract

The C-2B dashboard foundation is saved as a checkpoint. It is not the completion of C-2B. The next implementation pass is a presentation redesign and must preserve the existing scientific metric definitions, Reference behavior, result visibility, confidence provenance, exports, and no-reinference behavior.

### In-viewport HUD and annotations

- Metric and Detection Confidence HUDs are rendered over their associated 3D viewport as display layers. They must not be sibling layout cells, side panels, or scroll areas that reduce viewport geometry.
- The viewport remains the dominant surface. HUD content is compact, readable, high contrast, and must not obscure the hand unnecessarily. HUDs have no scroll area or scrollbar. Fit rows vertically by reducing their font from the selected size down to 7 pt, recomputing only when viewport geometry, row count, or selected size changes; never shrink the 3D view to fit more rows.
- Candidate HUDs identify the result and show selected aggregate metrics for the current frame. Overlay mode groups values by result. A Reference shows its identity and `REFERENCE` badge without self-comparison metric rows.
- Detection Confidence is shown at most once per detected hand and is explicitly named as detector confidence. Missing values remain unavailable; do not fabricate them.
- Joint annotation is a separate optional display layer with `Off`, `Error (mm)`, and `Joint name` modes; compatible modes may be combined. `Error (mm)` is the existing per-joint Euclidean distance between the candidate and selected Reference, not confidence. Draw labels near current projected joint positions without changing raw coordinates. Show only valid paired joints and identify the selected Reference context.
- The existing 21-row Per-Joint Details view remains available for exact inspection; it is distinct from in-viewport annotations.
- HUD size, background opacity, and corner/position may be user-adjustable. Defaults must preserve readability and hand visibility; do not freeze a specific text color independent of theme.

### Presentation mode and controls

- Provide Presentation Mode with a fullscreen option. While media is playing, automatically hide toolbar/control chrome after approximately two seconds without user input; keep media, 3D viewports, HUD, and annotations visible.
- Mouse movement, click, or keyboard input reveals controls. Pausing keeps controls visible. An open menu/dialog, active slider drag, or active panel reorder prevents auto-hide until the interaction ends.
- Keep existing compact, content-sized controls. Do not stretch buttons to fill unused width.
- Control semantics: View and Reference are single-select; Results and Metrics are multi-select; geometry visibility and Detection Confidence display options are independent toggles.
- Name the confidence menu **Detection confidence** with `Use as opacity`, `Show value`, and `Show bar` options. Name the color selector **Color by** with `Result` and `Detection confidence` choices.
- Keep `Mesh opacity` separate from the confidence-opacity toggle and disable it while Mesh is off.
- Make Error Vectors a clear on/off action with an explicit candidate target. Keep Detach in a secondary `More` menu so it does not displace primary viewing controls.
- In Split mode, keep Original media fixed in its source panel and allow result panels to be reordered using a visible drag handle. Reordering is presentation-only: result identity, color mapping, Reference ID, and data association stay attached to each result.
- Auto-hide, fullscreen, reordering, HUD controls, annotations, and control-menu changes must not rerun inference or mutate canonical results.

### Detection Confidence frozen decisions

1. The current `confidence` field is YOLO hand bounding-box detector confidence, sourced from `det.boxes.conf` and carried through `HandPrediction.score`.
2. Its current saved shape is `[frame, hand]`, with left/right hand slots; it is not `[frame, hand, joint]`.
3. It is not WiLoR reconstruction confidence, 3D correctness probability, per-joint confidence, or uncertainty.
4. Never duplicate a hand score across joints or invent joint-specific scores.
5. Whole-hand numeric/bar/opacity/color visualizations are permitted when labeled as Detection Confidence; do not use it to weight or correct metrics/results.
6. Genuine future joint confidence requires a backend that actually emits joint-specific values and a separately specified canonical field and provenance. It is outside the current C-2B scope.
7. Per-Joint Error annotations are error in millimetres and remain separate from confidence.

### C-2B status

The Presentation Redesign and media memory optimization are implemented. Report **C-2B READY FOR FINAL MANUAL APPROVAL**. Do not report C-2B complete before the user's final visual approval.

## C-2B media source loading status

Comparison and saved-result viewers share a source-keyed asynchronous frame provider. Video RGB frames are decoded only when requested and retained in a 384 MiB byte-bounded LRU cache; images use one static cached frame. Display images copy from the cached NumPy frame so cache eviction cannot invalidate a visible QImage. Missing or changed source media does not prevent viewing the saved 3D result. On the Windows 570-frame `4_assemble.mp4` source with three same-source results, the measured cache stayed at 379.7 MiB; process RSS was 594.5 MiB after decoding through frame 569 and 606.9 MiB after playback, versus the earlier approximately 3.7 GiB full-frame preload observation. This is a local benchmark, not a platform-wide memory guarantee.

C-2B Final Validation remains **READY FOR FINAL MANUAL APPROVAL**. Automated tests and local Windows visual review passed; the user's final visual approval is outstanding.


## C-2B Final Stabilization and Hardening

- Keep the source viewport and 3D viewport geometry fixed while control chrome hides and returns. Controls use compact outer padding and remain within the normal 1280x800 review target.
- Pin has an explicit checked appearance and a `Pinned` label; pinned controls do not auto-hide. Mouse movement restores hidden controls without a click.
- QToolButton menus open immediately on hover, remain open while the pointer moves between the button and menu, and close when the pointer leaves both. Clicking latches the menu until its action, outside click, Escape, or a second toggle closes it. QComboBox popups remain click-only.
- The default HUD is transparent text without a scroll area, scrollbar, panel border, background, or result-group separator. Optional subtle background remains an explicit setting. Font fitting is limited to viewport geometry, row-count, and selected-size changes; playback updates reuse label widgets.
- If a source frame is not in the asynchronous cache, keep the last valid source image visible until the currently requested frame is ready. Ignore ready notifications for stale frame requests.
- Viewer interactions never rerun inference or modify canonical results.

C-2B remains **READY FOR FINAL MANUAL APPROVAL** until the user completes the final visual review.
