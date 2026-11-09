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
- Left
- Right
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
- Historical Split contract (superseded by Phase D.1): Original media stayed fixed while result panels could be reordered. Phase D.1 allows Source Media to move with result panels.
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

The Presentation Redesign and media memory optimization are implemented. This status was superseded on 2026-11-03 when the user completed final Windows review and manually approved C-2B. See the final closure record below.

## C-2B media source loading status

Comparison and saved-result viewers share a source-keyed asynchronous frame provider. Video RGB frames are decoded only when requested and retained in a 384 MiB byte-bounded LRU cache; images use one static cached frame. Display images copy from the cached NumPy frame so cache eviction cannot invalidate a visible QImage. Missing or changed source media does not prevent viewing the saved 3D result. On the Windows 570-frame `4_assemble.mp4` source with three same-source results, the measured cache stayed at 379.7 MiB; process RSS was 594.5 MiB after decoding through frame 569 and 606.9 MiB after playback, versus the earlier approximately 3.7 GiB full-frame preload observation. This is a local benchmark, not a platform-wide memory guarantee.

C-2B Final Validation was manually approved by the user on 2026-11-03. Historical entries preserve which automated and local visual checks were individually recorded.


## C-2B Final Stabilization and Hardening

- Keep the source viewport and 3D viewport geometry fixed while control chrome hides and returns. Controls use compact outer padding and remain within the normal 1280x800 review target.
- Pin has an explicit checked appearance and a `Pinned` label; pinned controls do not auto-hide. Mouse movement restores hidden controls without a click.
- QToolButton menus and visible QComboBox selectors use the same transient-hover and click-latched policy. Hover opens immediately; leaving both control and popup closes a transient popup. A click latches it until outside click, Escape, second toggle, action selection, or normal dismissal. A latched popup has priority over other hover targets; clicking another dropdown dismisses the old popup and opens the new one.
- The default HUD is transparent text without a scroll area, scrollbar, panel border, background, or result-group separator. Optional subtle background remains an explicit setting. Font fitting is limited to viewport geometry, row-count, and selected-size changes; playback updates reuse label widgets.
- Comparison playback separates `requested_frame` from `presented_frame`. While a requested source frame is missing, the source image, 2D/3D results, mesh, error vectors, annotations, metrics/HUD, slider, and status retain the previous presented frame. A ready notification is applied only if it still matches the newest request. Playback time continues and can skip frames; pause invalidates pending playback presentation, while manual seek presents the exact requested frame when ready. Source images retain their last valid decoded frame.
- Dropdown-like control audit: Results (QToolButton/QMenu, multi-select), Reference (QComboBox, single-select), Metrics (QToolButton/QMenu, multi-select), Color by (QComboBox, single-select), Detection (QToolButton/QMenu, multi-option), Error vectors (QToolButton/QMenu), Joints (QToolButton/QMenu), HUD (QToolButton/QMenu), More (QToolButton/QMenu), and Speed (QComboBox, single-select).
- Viewer interactions never rerun inference or modify canonical results.

C-2B manual approval was recorded on 2026-11-03; see the authoritative closure entry at the end of this document.

## C-2B Final Sync & Menu Consistency Fix

- Comparison presentation uses a single `presented_frame` for RGB, 2D/3D, mesh, metrics/HUD, annotations, slider, and status. Async cache misses retain the prior full presentation; out-of-date completions are ignored.
- Playback clock time advances during decode and may drop intermediate frames. Pause discards pending playback presentation. A paused manual seek remains the newest exact target and presents when its source frame is ready.
- Results, Metrics, Detection, Vectors, Joints, HUD, and More use QToolButton/QMenu. Reference, Color by, and Speed remain QComboBox selectors. All visible dropdown controls open on immediate hover and follow the same transient/click-latched dismissal policy. A click on another control dismisses the old latch and opens the new control; passive hover does not steal a latched popup.
- This pass does not change canonical result data, metric definitions, inference, the bounded media cache, or the single-result/comparison viewer routing.
- Validation and normal Windows GUI evidence for this pass are recorded in `VALIDATION_CHECKLIST.md`; user approval is recorded below.

## C-2B final closure

**C-2B COMPLETE — USER MANUALLY APPROVED** (2026-11-03). The user confirmed final Windows behavior including source/2D/3D synchronization at 2×, pause/seek, popup and hover interactions, HUD, annotations, playback speed matrix, and a clean console after the QWindow event-filter hotfix. This overall acceptance does not mark specific layout cases as manually inspected unless their evidence is recorded individually in `VALIDATION_CHECKLIST.md`.

## C-2C Unified Result Viewer

Saved `.cfh` results use the modern Comparison Viewer engine whether the user opens one result or multiple results. One result enters Single Result mode; two or more results retain Comparison mode and require matching source-media SHA256 identities. The application also uses this path for automatic preview after processing.

Single Result mode retains the modern source/2D/3D presentation, Overlay and source-plus-result Split views, timestamp-based playback controls, bounded asynchronous source provider, and detector-confidence controls. It shows the configuration identity and omits Reference selection, reference-difference metric HUD/menu, error vectors, error-valued joint labels, comparison metric exports, and result reordering. It does not compute or display self-reference metrics. Missing or changed source media does not prevent viewing the saved 3D result. A single bundle without a source hash can still open; hash matching remains mandatory for multi-result comparison.

The legacy `ResultViewer` implementation remains available for existing NPZ/session compatibility. This pass changes the `.cfh` application route and does not change scientific result payloads, coordinates, metrics, confidence provenance, or inference.

## First-User UX & Stability Pass

**Status:** Specification only. Implementation has not started.

This is a post-C-2C follow-up based on a team member's first use of the application, from launch through media selection, configuration, processing, opening a result, playback, and 3D inspection. The pass addresses discoverability, layout hierarchy, terminology, interaction consistency, and reproduced stability problems. It adds no scientific or inference features.

### Project status and boundary

- C-2B remains **COMPLETE — USER MANUALLY APPROVED** (2026-11-03).
- C-2C Unified Result Viewer remains implemented and **READY FOR USER REVIEW**.
- This pass is separate from both milestones. Its specification is ready for review; its implementation and validation are pending.
- Preserve the C-2B and C-2C records and all scientific, bundle, preset, and backend compatibility contracts.

### P0 stability and functional requirements

#### Windows native media dialog

The reported native-dialog failure displayed `Windows media dialog unavailable / using the fallback media picker`, followed by `counting selected media failed (HRESULT 0x80004002)`. Closing the message opened the Qt fallback instead of the Windows Explorer-style picker. Treat this as a P0 functional bug.

On a normal supported Windows environment, the native Windows Common Item Dialog is the primary media picker. Selecting files or folders must not produce the reported HRESULT failure. Use the fallback only when the native dialog is genuinely unavailable. If fallback is needed, explain the reason in readable terms, keep the application alive, and allow the selection workflow to continue. Fallback use as a routine path on a normal Windows system is a failure.

The native picker shall support single-file selection, multiple-file selection, folder selection, selecting the current folder, locations on C: and D:, cancellation, closing and reopening the dialog, and repeated use.

#### Shared `MediaFrameProvider` ownership and release

Treat the reported source-by-source RAM accumulation as a P0 lifecycle and memory bug. Shared source decoding and the bounded per-provider cache remain required.

Provider ownership shall follow acquire/release semantics or an equivalent ownership model:

- Viewers using the same source concurrently share its provider.
- Closing one of several users keeps the provider available to the remaining users.
- Closing the final user removes that source from the registry, stops its decode worker, and releases its cache and other source resources.
- Application exit performs final cleanup of any providers still registered.
- Do not preload a full video. Keep each active provider's frame cache bounded to the existing 384 MiB limit.

After opening, playing, and closing result A, then B, then C from different source videos, closed sources must not retain their providers or full caches. Acceptance concerns active provider count, registry entries, worker count, and decoded cache bytes. The process RSS is not required to return exactly to its initial value because Python, Qt, and GPU allocators may retain memory.

#### Error Vectors candidate state

Treat the tiny unexplained candidate window and the repeated `Off → Candidate → Off → Candidate` failure as a P0/P1 functional interaction bug. Candidate selection shall remain within the Vectors control or its clearly associated menu; it shall not open an unexplained tiny popup/window. Repeatedly selecting Off and a candidate shall work. The selected UI state and rendered vectors must agree after every transition, and changing candidates must not leave stale vectors or state.

#### HUD size, layout stability, and identity text

HUD size is a functional state bug. Keep these values distinct:

- **Requested HUD size:** Small, Medium, or Large, selected by the user.
- **Effective HUD size:** the font size currently rendered after fitting the HUD into its viewport.

When the viewport has room, effective size shall match the requested size. When space is limited, the HUD may reduce its effective font size to fit. When room returns, it shall restore the requested size. Selecting Small once must not permanently cap the effective size or prevent a later Medium or Large selection.

HUD font fitting shall depend only on actual viewport bounds, requested HUD size, and current HUD content. Opening or closing Results/Detection popups, changing menu focus, or changing a menu check state shall not cause a transient shrink when the viewport and HUD content have not changed.

Reference identity text shall keep its label, focal value, unit, and badge visually associated. Prefer the compact form `Manual Focal 1500 px [REFERENCE]`. In narrow viewports, wrap semantic groups naturally; never strand `px` on a separate line from its number or split the badge from the identity in a way that makes the label ambiguous.

### MainWindow first-user UX

#### Preset and configuration controls

The Preset row shall contain preset-related controls, including the preset selector, Preset Actions, and Set as default. Remove Add Configuration from that row. Place a `+ Add` control below the configuration table, aligned naturally with the Name column. It shall remain below the last configuration row as rows are added and retain the existing Add Configuration row-creation behavior.

Order configuration columns as:

```text
Name | Calibration | Focal | Reconstruction | Action
```

Focal is a Calibration parameter and belongs immediately after Calibration.

#### Manual Focal terminology and field state

Use **Manual Focal** consistently throughout the user-facing display layer for the calibration option that directly accepts a focal length in pixels. This applies to the Main Window selector and configuration row, Viewer result/configuration names, and newly created user-facing configuration/result display names. Show its parameter as **Focal (px)**. Enable the numeric field only for Manual Focal; for other calibration methods, show it disabled or as an unambiguous dash/N/A state. A Viewer name with a focal value shall read **Manual Focal 1500 px**; keep the name, value, and `[REFERENCE]` badge as a natural semantic group in the HUD (for example, **Manual Focal 1500 px [REFERENCE]**).

Keep UI wording separate from internal identity. The existing `calibration_model_id = checkerboard_reference` may remain unchanged. Preserve old presets, `.cfh` results, metadata, and scientific schemas; a UI rename alone shall not trigger an internal ID or schema migration. Do not rewrite or migrate historical saved data or forcibly rename existing files. When displaying historical `.cfh` metadata whose saved display name says **Checkerboard Reference**, the Viewer may normalize the presentation to **Manual Focal** using the internal model ID or metadata context, but must not modify the file or its metadata.

#### Settings hierarchy and Output row

Swap the current placement of the Theme/Language area with the Sampling/Sample Count/Aggregate area. Keep Sampling and aggregation visually close to processing configuration; group Theme and Language with application preferences and Output. Do not add a new theme/settings framework.

Present Output path, Browse, and Collision policy in one compact logical row. Give the path field most of the horizontal stretch while keeping ordinary paths inspectable. Keep Browse and Collision on the same row where the available window width permits and minimize clipping at smaller widths.

### Viewer playback and mode controls

Give the playback footer balanced left/right horizontal margins and natural spacing consistent with the bottom margin; controls shall not sit against the viewport/window edge.

Replace the Speed dropdown with a stepper such as `▼ 1.0× ▲` (or equivalent). Keep the current speed visible, use the existing presets `0.1×, 0.25×, 0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 2.0×`, and make the lower/upper control select the adjacent preset. At a limit, disable the corresponding control or make it a no-op. Changing speed shall preserve the current source-time position and existing timestamp-based playback-clock semantics.

Replace Previous, Play/Pause, and Next text buttons with appropriate Qt native media icons or `QStyle` standard icons. The play control shall display Play while paused and Pause while playing. Preserve tooltips and accessible names. Do not add Stop, Fast Forward, or Rewind functionality.

Replace separate Overlay and Split buttons with an exclusive two-state segmented control or equivalent. Exactly one mode shall always be active and clearly indicated; both modes must not be switchable off.

Make Pin an explicit, clearly readable toggle. OFF means controls may auto-hide; ON means controls remain visible. The current state must be apparent without relying on stylesheet color alone.

### HUD direct manipulation

Dragging the HUD is the primary way to position it. On press within the HUD, the HUD captures the pointer interaction; while dragging, the 3D viewport must not orbit, pan, or change annotation hover in response to the same gesture. On release, snap the HUD to the nearest of the four allowed anchors: top-left, top-right, bottom-left, or bottom-right. Use HUD-center or equivalent geometry to determine the nearest corner.

Animate the snap over approximately 120–180 ms, with no bounce. The animation shall not block playback, source-frame presentation, or 3D rendering. HUD Size, Background, opacity, and equivalent visual settings may remain in the HUD menu; corner-position menu actions may be removed. Normal viewport interactions resume after the drag ends.

### Annotation terminology

Rename the result annotation control/menu from **Joints** to **Annotations**, because it controls annotation modes and label content. Keep the separate geometry visibility checkbox **Joint**; it controls joint rendering and is not the annotation menu.

### Terminology mapping

| Existing UI wording/control | First-user wording/control |
| --- | --- |
| Checkerboard Reference (calibration option) | Manual Focal |
| Reference Focal | Focal (px) |
| Joints (annotation menu) | Annotations |
| Separate Overlay and Split buttons | Exclusive Overlay/Split segmented toggle |
| Speed dropdown | Speed stepper |

`checkerboard_reference` remains a valid internal ID. Normalize the user-facing calibration/configuration/result name to **Manual Focal** across the Main Window and Viewer, including presentation-only normalization of historical display names; never rewrite saved metadata or filenames. The Viewer concept of a selected **Reference** and the `[REFERENCE]` badge remain unchanged and stay semantically grouped with the focal value (for example, **Manual Focal 1500 px [REFERENCE]**).

### First-user acceptance

After implementation, the team member who supplied this feedback shall repeat the complete flow without a preview explanation of the changes: launch, select media, configure, start processing, wait, open the result, play it, and inspect 3D. The goal is successful use without explanation. Prefer repeating the same task with a second team member who has not used the application; observe whether they can find the primary actions, understand Manual Focal, locate Add Configuration, and operate the result controls without coaching.

No new scientific feature, metric, inference behavior, calibration model, confidence type, or playback operation is part of this pass. Preserve same-source result behavior, `.cfh` compatibility, C-2B/C-2C viewer behavior, scientific data and confidence semantics, and all established coordinate/metric contracts.

## Phase C Physical / Hand 3D space clarification

The unified Viewer used for a single `.cfh` and multi-result comparison provides two presentation workspaces, defaulting to **Physical**. Physical uses fixed camera-space bounds X/Y [-1, 1] m and Z [0, 2] m. Its display convention is X right, Y up, and Z forward; display Y is the negation of raw camera Y. Canonical saved data remains X right, Y down, Z forward in metres and is never rewritten by viewing controls.

**Hand** uses cached robust whole-sequence bounds from detected finite geometry in the selected GT result, or the sole result in single-result mode. Candidate results do not influence those bounds. The space uses padded percentile extents, frame-center outlier filtering, a minimum span, and adaptive metric grid spacing. If no valid reference geometry exists, it uses Physical bounds. Neither workspace follows playback frames.

Numeric **Labels** visibility is independent of Grid and Axis visibility. Physical/Hand changes and GT selection retain the current frame and camera. **Focus GT** (shown as **Focus Hands** for a single result) applies whole-video reference hand framing; **Space Overview** frames the active metric workspace; **Fit All** frames finite visible current-frame geometry. Workspace and Labels state are shared by Overlay, Split, and existing detached result panels.

## Phase D.1 Unified Panel Ordering Foundation

- Stable presentation identities are `source-media`, `overlay-3d`, and `result:<configuration_id>`. Display names, colors, list positions, and current GT selection do not define identity.
- Overlay has two panels, Source Media and the composite Overlay 3D view. Its initial order is Source Media then Overlay 3D; users may swap them independently of result order.
- Split has Source Media and one panel for each result. Initial order places Source Media first, then the selected GT, then other results in stable configuration order. Source Media can subsequently move to any slot.
- Result ordering is shared by the Overlay HUD, Split result panels, and Results menu. Reordering in the HUD preserves Source Media's Split slot. Overlay's left/right panel order remains independent.
- GT-first is an initialization policy until the user explicitly reorders a result or Split panel. Before that point, changing GT updates the leading result. After a user reorder, later GT changes update semantics, badges, colors, and metrics without moving panels.
- Drag insertion uses the actual HUD group geometry or Split panel geometry. A click, same-order insertion, Escape, or drop outside the active surface leaves order unchanged. Hidden results retain their identity and order slot.
- Reordering reparents existing presentation surfaces while retaining the original source widgets, GL views, result arrays, provider, frame/playback state, camera, HUD, and annotations. It does not run inference or modify `.cfh` data.
- The older fixed-Source-Media Split requirement and HUD-only GT-first presentation are superseded by this section. D.2 individual Detach/Reattach and D.3 detached-workspace synchronization and source aspect-ratio work remain deferred.

## Phase C.1A Workspace Geometry and Controls clarification

Both Physical and Hand workspaces use a centered cube as the common metric definition for all three grid planes, axis lines, numeric tick labels, and workspace overview framing. Physical retains its fixed camera-space bounds; Hand derives robust whole-sequence bounds and expands them to a cube. Numeric labels use scale-relative offsets and zero is displayed as `0 cm`; workspace changes remove and regenerate labels so stale graphics do not accumulate. The Labels control is an accessible On/Off switch independent of Grid and Axis visibility. View, Space, Labels, and Pin are grouped in a palette-aware primary control strip. This clarification does not change canonical result arrays or established camera/frame behavior. Native Windows visual acceptance remains pending. Camera interaction changes and detached-view redesign belong to later work and are not implied by this clarification.

## Phase C.1A Fix Pass: Workspace Placement and Space Center

The earlier C.1A expectation that all Physical grid planes share the volume midpoint is superseded. Physical bounds remain X/Y [-1, 1] m and Z [0, 2] m, with volume center raw (0, 0, 1 m). The Physical XY plane is at Z=0; XZ is at Y=0; YZ is at X=0. Physical axes follow the global camera-space axes through raw origin (0, 0, 0). All reference planes remain 2 m × 2 m. Hand planes and equal-length axes remain centered on the selected-GT cube. On a Space change, the orbit center moves to the active workspace center while distance and orientation stay fixed; frame and playback state stay fixed. The primary controls container is transparent, with palette-aware individual controls. This pass does not redesign camera presets; that remains C.1B work. Native Windows review identified the previous Physical placement error, so the fix still requires user review.

## Phase C Final: Camera State, Controls, Contrast, and GT Switching

The Comparison Viewer retains one independent camera pose per 3D Space, including center, distance, azimuth, and elevation. The first Physical pose is the actual initial view. First entry to Hand initializes framing from that Space's policy; subsequent transitions restore each Space's last pose. Overlay and Split share this camera state. The earlier Phase C policy that a GT change should not move the camera is superseded by **Phase D.1 Camera Center Fix** below: choosing a different GT updates both saved Space poses using their separate framing policies while retaining each Space's own azimuth and elevation. Physical reference geometry remains fixed; Hand bounds follow the selected GT. Focus GT, Space Overview, and Fit All remain distinct camera actions.

Orientation controls are **Front**, **Left**, **Right**, and **Top**. They change orientation while preserving center and distance. Viewport-overlaid plain text uses a consistent light foreground in all application themes; the primary controls remain transparent.

Comparison metrics are cached by ordered `(reference result ID, candidate result ID)` pairs because metric direction is reference-dependent. Small uncached workloads are computed inline; larger GT changes use a single background worker. While a new GT's metrics are pending, stale metrics are cleared. Results from a superseded selection can populate the pair cache but cannot replace the current GT's displayed metrics. Metric formulas and canonical result data are unchanged.

Automated Phase C Final coverage is recorded in `VALIDATION_CHECKLIST.md`. Native Windows visual/playback acceptance remains pending; passing offscreen tests does not count as manual acceptance. Phase D panel workspace work remains deferred.

## Phase D.1 Fix Pass: Panel Interaction and Adaptive Layout

Top-level Source Media, Overlay 3D, and Split result panel headers and drag buttons are removed; the duplicate Split Source Media title is also removed. Existing source and GL widgets remain hosted by stable panel surfaces. Overlay panels swap through a compact, accessible control on the splitter divider; a swap exchanges panel contents while preserving the two slot sizes and existing viewer state.

In Split mode, a Shift + left-button drag that starts on Source Media or a result viewport moves that panel. The gesture is captured at press, so ordinary unmodified GL drags retain camera control. The hovered cell is the drop target; moving onto the source's own cell is a no-op, and releasing outside the Split workspace or pressing Escape cancels. Source and target cells receive immediate palette-highlight feedback, cleared on completion or cancellation. HUD result reordering keeps its existing drag behavior and shows an insertion marker at the rendered insertion boundary. These are static indicators; no Phase E animation is implied.

The Split grid is recalculated from currently visible panels and available viewport geometry after visibility changes and debounced resizes. Source Media is always included; hidden results retain their identity and order for restoration. Grid stretches, minimum constraints, and cell membership are updated without rebuilding result data or GL widgets. Native Windows review is still required before D.1 is user-approved. HUD snap margins/rebound, excess right-side HUD space, GT badge height, and reorder animations remain Phase E polish. D.2 and D.3 remain deferred.

### Phase D.1 Rendering Regression Fix & UI Stabilization (2026-11-07)

Native reproduction identified a Split surface being shown while parentless before insertion into its final grid. This temporarily exposed its `QOpenGLWidget` under a separate top-level context; reparenting then changed context/FBO ownership. The repair inserts surfaces into their owning grid before showing them and avoids hide/show cycles for retained surfaces during reflow. It does not reset or monkey-patch pyqtgraph's process-global shader cache. Native Windows checks with saved `.cfh` results observed valid GL framebuffers, stable contexts through reflow and interaction, and no GL errors or pyqtgraph draw warnings. Detach All now assigns a renderable initial window size. The Overlay splitter uses a 4 logical-pixel handle with a compact palette-aware Qt-painted double-arrow control over the divider. This records implementation and automated/native-harness evidence only; Phase D.1 remains subject to user visual and interaction acceptance. See `VALIDATION_CHECKLIST.md` for the detailed verification record and deferred work.

### Phase D.1 Final Fix: Adaptive Grid & Overlay Divider UX (2026-11-07)

Split layout evaluates candidate column counts from 1 through the visible panel count. For each candidate it computes required size using the 340×260 minimum cell, actual viewport, layout margins, and spacing. Its deterministic score weights vertical overflow 1.5 times more heavily than horizontal overflow to avoid extra stacked rows and vertical scrolling, then penalizes cell aspect-ratio distance from a moderately landscape 1.4:1 target, unused positions in the final row, and finally excess rows. This yields the expected 5/4/3/2/1 panel layouts of 3×2, 2×2, 2×2, 2×1, and 1×1 for the normal 1280×720 landscape Viewer, without count-specific branches. Larger counts and other viewport shapes use the same candidate scoring and retain scrolling when minimum cell sizes cannot fit. Row-major panel identity/order and existing widget instances remain stable across hide/restore, reorder, and resize.

Overlay uses an 8 logical-pixel horizontal `QSplitter` handle with exactly three vertically arranged 3 logical-pixel circular dots, spaced 5 logical pixels center-to-center. Paint the divider background from the palette `Window` role and dots from `WindowText`, without inherited native handle decoration, so the grip stays distinct across Light/Dark/System palettes and device-pixel ratios. Left-drag remains the native resize interaction. Right-click opens a small context menu with `Swap`; dismissing it does nothing. Swap reorders the existing Source Media and Overlay 3D surfaces and preserves splitter slot sizes and Viewer state. There is no floating button and no left-click swap action.

The current combined Detach All window is temporary. Phase D.2 is expected to provide one independent window per detached panel (Source Media and each result), individual Detach/Reattach, and safe GL ownership. This is explicitly deferred. D.3 synchronization/aspect-ratio work, Phase E HUD/animation polish, Phase F/G, and release packaging also remain deferred. Native test evidence and remaining user review are recorded in `VALIDATION_CHECKLIST.md`.

### Phase D.1 Stabilization: GT Camera Focus and Playback During Resize (2026-11-09)

Historical D.1 Stabilization behavior assigned the same whole-video hand target and distance to both saved Space poses. **That shared-center policy is superseded by Phase D.1 Camera Center Fix below.** The previous Phase C and Phase 2.1 wording that GT selection leaves the camera in place is also superseded.

Native Windows Qt measurements with the saved four-result `2_screw` set found repeated `QImage` conversion/copy work in Source Media painting when the decoded source frame was already cached. The canvas now retains its owned `QImage` for that source frame and rebuilds it only when the source frame changes. In a matched 2-second playback/resize harness, Source Media paint time during resize changed from 2.614 ms average / 3.216 ms max with the prior copy behavior to 1.600 ms / 1.821 ms with image reuse; source/3D presentation stayed on frames 59/59 in both runs. The production playback clock and speed are unchanged. Native Qt layout-resize measurements and remaining user acceptance are detailed in `VALIDATION_CHECKLIST.md`; this does not complete deferred phases or constitute user acceptance.

### Phase D.1 Camera Center Fix (2026-11-09)

GT selection updates each saved Space pose according to its own camera policy. **Physical Space** centers on the selected GT's robust whole-video hand-framing target and uses that framing's distance. Its grids and axes remain fixed at their camera-space coordinates. **Hand Space** centers on the intersection of the three axes actually rendered for its selected-GT workspace. Its distance fits the final display-space workspace corners using the active view's FOV and aspect ratio. This uses the final `WorkspaceGeometry`, including its cube expansion, Z adjustment, and deterministic fallback; it does not use the wrist target as a proxy for the Hand axis intersection.

The axis intersection is available as a distinct workspace property and is the same raw/display-space point used to build the rendered axes. GT changes and first entry to Hand apply the corresponding policy. Switching spaces without changing GT restores each saved user-adjusted pose, including its independent orientation. Focus GT intentionally targets the hands in either Space; Space Overview frames the current workspace; Fit All continues to frame finite visible current-frame result geometry. Result arrays and scientific coordinates remain unchanged.

### Phase D.1 Playback Resize and HUD Insertion Marker Fix

During active playback, Overlay splitter drag handling coalesces divider updates to at most 30 Hz; paused resizing remains uncoalesced. The final pointer position is applied on release so coalescing cannot leave the divider short of the user's drop location. This reduces resize-triggered OpenGL redraws while preserving live resizing and synchronized source/3D presentation. The HUD insertion marker and drop slots use the actual body-local result-group bounds, including variable-height groups; the marker is clipped within the HUD body and inset from its content edges. No saved result, camera policy, playback clock, or inference data is rewritten.

Warm-cache native measurements showed resize events falling from 82 to 31 and GL paints from 131 to 85 in matched 1.6 s continuous-drag runs; event-loop p95 fell from 21.57 to 18.19 ms and playback tick p95 from 31.01 to 25.67 ms. No frames were skipped in either run. Cold-cache Comparison Viewer playback still showed source-frame presentation stuck at frame 0 while its decoder worker waited for FFmpeg output; this also occurred without splitter interaction and remains unresolved. User visual/playback acceptance is pending. D.2, D.3, E, F, G, and release work remain deferred.

### Phase D.1 HUD Layout Stability Fix

An order-only change now moves the existing result-group widgets within the Overlay body layout. It preserves their identities, current fitted fonts, badge dimensions, card geometry, and fit signature. The order refresh skips HUD reposition/refit because content, settings, and viewport have not changed; it still refreshes annotations and raises the HUD above them. A configuration/reference change or a HUD/content setting change continues through the normal rebuild and fitting path. Native Qt paint/layout tracing verified stable appearance before, during, immediately after release, through queued events and idle, without pointer movement. Cold-cache playback remains a separate deferred investigation; this pass does not change playback, media, camera, or scientific result behavior. User visual confirmation remains pending.
