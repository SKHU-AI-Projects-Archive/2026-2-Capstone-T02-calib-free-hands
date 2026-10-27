# CalibFreeHands UX / Result Packaging Specification

**Status:** Frozen for next implementation milestone  
**Repository:** `D:\2026-2-Capstone-T02-calib-free-hands`  
**Application root:** `CalibFreeHands/`  
**Working branch:** `CalibFreeHands`

## 1. Purpose

This document freezes the intended user experience and result-packaging behavior for the next CalibFreeHands implementation milestone.

The current application already has a useful internal research architecture:

- calibration backends
- reconstruction backends
- `CalibrationResult`
- `RawPoseResult`
- `CanonicalResult`
- comparison configurations
- comparison sessions
- Overlay / Split comparison viewer
- confidence visualization
- reference metrics

These internal structures should remain reusable.

However, the current application exposes too much of its internal processing structure directly to the user, including `session.json`, `calibration/`, `pose/`, hashed result directories, and multiple intermediate `.npz` / `.json` files.

The target UX is:

> **One input media + one Calibration model + one Reconstruction model = one user-facing result file.**

## 2. Design principles

### 2.1 User-facing simplicity

Normal workflow:

```text
Open...
  ↓
Select file or folder
  ↓
Choose model configuration(s)
  ↓
Start Processing
  ↓
Completed
  ↓
Viewer opens automatically
```

The user should not need to understand raw-pose NPZ files, calibration cache JSON files, hashed output directories, or session internals.

### 2.2 Preserve scientific correctness

Do not change the current scientific contracts during UX work.

```text
tz_new = tz_source * target_focal / source_focal
```

- camera translation X unchanged
- camera translation Y unchanged
- only Z is rescaled by focal ratio

Final camera-space joints:

```text
keypoints_3d_relative + corrected_translation
```

Final MANO vertices:

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

`RAW_TO_VIEW` remains display-only. Missing hands remain `detected=false` with NaN coordinates.

### 2.3 Preserve internal reuse

The separation between Calibration and Reconstruction must remain.

Example:

```text
Checkerboard Reference + Original WiLoR
Original AnyCalib      + Original WiLoR
Fine-tuned AnyCalib    + Original WiLoR
Original GeoCalib      + Original WiLoR
```

must not require Original WiLoR inference four times when the same raw pose can be reused.

## 3. Input UX

### 3.1 One Open button

Remove the separate user-facing buttons:

```text
Browse files
Browse folder
```

Replace them with:

```text
Open...
```

The user should be able to add either:

- one or more supported media files
- one folder

If files are selected, add those supported files only.

If a folder is selected, add all supported media files directly inside that folder.

Default folder discovery remains non-recursive.

### 3.2 Queue controls

Minimum controls:

```text
Open...
Remove
Clear
Start Processing
Cancel
```

`Remove` removes selected queue entries only.  
`Clear` removes all queue entries only.  
Neither operation deletes source media from disk.

### 3.3 Queue processing

`Start Processing` means:

> Process the whole current queue sequentially.

Each media item should have an understandable state:

```text
Waiting
Processing
Completed
Failed
Cancelled
```

## 4. Completion behavior

### 4.1 Single media

When one media item is processed successfully:

```text
Processing
  ↓
Completed
  ↓
Viewer automatically opens
```

The main UI must clearly show `Completed`.

### 4.2 Batch input

For multiple media files:

- process sequentially
- do not open one new viewer per item during the batch
- update each item state
- after the whole batch ends, show `Completed`
- then present/open one result-selection workflow

## 5. Output directory UX

The application must show a real effective output path at all times.

Target form:

```text
Output    D:\example\CalibFreeHandsResults    [Browse...]
```

Requirements:

- actual path always visible
- clicking Browse opens a folder picker
- selected path persists in application settings
- no placeholder-only output field
- if no saved output exists, select and display a deterministic concrete default

## 6. User-facing result model

The user-facing result unit is:

> **one source media + one Calibration model + one Reconstruction model = one result file**

Examples:

```text
1_insert__Original-AnyCalib__Original-WiLoR.cfh
1_insert__FineTuned-AnyCalib__Original-WiLoR.cfh
1_insert__Original-GeoCalib__Original-WiLoR.cfh
```

A multi-configuration run therefore creates multiple result files, while internal shared inference may still be reused.

## 7. `.cfh` result bundle

Introduce:

```text
.cfh
```

as the normal user-facing result format.

Recommended implementation:

> `.cfh` is a ZIP-compatible container created with the Python standard library.

Minimum conceptual contents:

```text
result.npz
metadata.json
```

### 7.1 NPZ role

`result.npz` remains the canonical numeric-array payload and may contain:

- frame indices
- timestamps
- 2D keypoints
- 3D keypoints
- detected mask
- focal values
- MANO vertices
- MANO faces
- confidence
- bbox
- MANO pose
- MANO shape
- camera translation

### 7.2 Metadata role

`metadata.json` should contain provenance and human-readable metadata, including where available:

- schema version
- application version
- source media name
- source media path
- source media SHA256
- image/video dimensions
- Calibration model ID and display name
- Reconstruction model ID and display name
- final focal
- sampling mode
- sample count
- aggregation
- Calibration provenance
- checkpoint hash
- runtime metadata
- coordinate-system description
- unit
- whether principal point was applied
- whether distortion was applied

Do not claim parameters were applied when they were not.

## 8. Source media

Do not embed the original video/image in `.cfh` by default.

Store:

```text
source_media_name
source_media_path
source_media_sha256
```

If the source still exists, the viewer can show it.

If the source is missing, show a clear missing-source state while still allowing the stored 3D result to open.

## 9. Result filenames

Default:

```text
{source-stem}__{calibration-display-name}__{reconstruction-display-name}.cfh
```

Example:

```text
1_insert__Original-AnyCalib__Original-WiLoR.cfh
```

Unsafe filesystem characters must be sanitized.

Reuse the existing collision concepts:

```text
Ask
Overwrite
Auto rename
Skip
```

An Auto rename example:

```text
1_insert__Original-AnyCalib__Original-WiLoR_1.cfh
```

## 10. Internal intermediate files

Keep internal structures such as:

```text
CalibrationResult
RawPoseResult
session/cache metadata
```

but do not normally expose these in the user output directory as:

```text
calibration/
pose/
results/<hash>/
```

Use an internal working/cache location such as a temporary directory or:

```text
%LOCALAPPDATA%\CalibFreeHands\cache\
```

The exact cache path may be chosen during implementation.

Do not remove inference deduplication/reuse just to achieve a one-file UX.

## 11. Preview video

Do not generate `preview.mp4` automatically for every result by default.

Preferred future workflow:

```text
Viewer
  ↓
Export Preview Video...
```

Keep the existing preview encoder for explicit export.

## 12. Result opening UX

Rename:

```text
Open saved result
```

to:

```text
View Results...
```

The dialog should open in the most recently used/result output directory.

Normal selectable type:

```text
CalibFreeHands Result (*.cfh)
```

Multiple selection must be supported.

## 13. Opening results

### One `.cfh`

Open a single Result Viewer.

### Multiple `.cfh`

1. read metadata from each selected result
2. compare source-media identity
3. if they came from the same source, open Comparison Viewer
4. if they came from different sources, do not silently compare them

Initial acceptable mixed-source behavior:

```text
Selected results were generated from different source media.
```

### Same-source identity

Use:

```text
source_media_sha256
```

as the authoritative identity.

Do not rely only on filenames.

## 14. Comparison Viewer

Preserve existing:

```text
Overlay
Split
```

### Overlay

Preferred concept:

```text
Original media | overlaid 3D results
```

Method/configuration identity remains a stable color.

Confidence remains based on the actual stored frame × hand confidence scalar, using alpha/opacity.

Do not invent joint-level confidence.

### Split

Preferred concept:

```text
Original media | Result A 3D | Result B 3D | Result C 3D
```

Show fewer result columns when fewer results are selected.

## 15. Separate windows

Add a way to detach results into individual windows.

Suggested label:

```text
Separate Windows
```

or:

```text
Detach
```

Conceptually:

```text
Original media window
Result A window
Result B window
Result C window
```

The first implementation only needs detaching. Recombining may be deferred.

Overlay/Split must not be removed.

## 16. Viewer synchronization

Keep synchronization for:

- frame index
- timestamp
- playback state

Split 3D cameras remain synchronized for:

- orbit
- zoom
- pan
- Front
- Side
- Top
- Fit Hands
- Reset View

Changing view modes must not rerun inference, rebuild scientific data, mutate raw coordinates, or unnecessarily reset the current frame.

## 17. Checkerboard Reference focal input

The current always-visible disabled `1000.0 px` control is confusing.

New behavior:

If Calibration is not Checkerboard Reference:

> hide the Reference Focal control.

If Calibration is Checkerboard Reference:

show:

```text
Reference Focal    [ 1000.0 px ]
```

and allow editing.

## 18. Calibration model roadmap

Intended model list:

```text
Original AnyCalib
Fine-tuned AnyCalib
Original GeoCalib
HandCalib
Checkerboard Reference
```

Only verified runnable models may be enabled.

### 18.1 Original AnyCalib

Current intended status:

```text
available
```

The repository contains official AnyCalib as:

```text
HandCalib/AnyCalib
```

but the app currently imports:

```python
from anycalib import AnyCalib
```

Therefore runtime provenance should verify the actual imported module location rather than assuming it.

### 18.2 Fine-tuned AnyCalib

Treat as distinct from HandCalib:

```text
Original AnyCalib != Fine-tuned AnyCalib != HandCalib
```

Do not label a fine-tuned AnyCalib checkpoint as HandCalib.

### 18.3 Original GeoCalib

Target repository layout:

```text
HandCalib/
├─ AnyCalib/
└─ GeoCalib/
```

Before integration:

- verify official upstream
- verify license
- verify Python/PyTorch requirements
- verify pretrained weight mechanism
- verify inference API
- verify focal semantics

Prefer repository-managed dependency handling such as a Git submodule if appropriate.

Do not hard-code personal absolute paths.

### 18.4 HandCalib

Current status:

```text
unavailable
```

until a real runtime adapter and checkpoint are scientifically defined.

Do not fake HandCalib by renaming/reusing another backend or returning constant focal values.

## 19. Reconstruction model roadmap

Current default:

```text
Original WiLoR
```

Future variants may use the existing reconstruction registry.

The result filename must always include the reconstruction model name.

## 20. Default configuration

Default remains:

```text
Calibration: Original AnyCalib
Reconstruction: Original WiLoR
Sampling: Uniform
Sample count: 8
Aggregation: Median
```

Users may persist their own defaults through the settings system.

## 21. Legacy compatibility

Existing historical:

```text
results.npz
run.json
session.json
```

results should not be intentionally broken.

Legacy loading may remain available for migration/development.

Normal future user interaction should use `.cfh`.

Internal raw-pose/cache NPZ files must not be presented as normal user results.

## 22. Invalid-result protection

The previous error:

```text
KeyError: 'keypoints_3d is not a file in the archive'
```

occurred because an internal raw-pose NPZ was opened as if it were a final canonical result.

The new workflow should structurally prevent this.

`View Results...` should expose `.cfh` files, not arbitrary NPZ files.

Any legacy NPZ importer should validate the schema and show a human-readable error instead of an uncaught traceback.

## 23. Processing warnings

Warnings such as:

- Lightning checkpoint auto-upgrade notice
- MANO 10 shape coefficients warning
- `torch.cross` deprecation warning

are not automatically processing failures.

Report the actual exception that caused a failure.

## 24. Implementation milestones

### Milestone A — Input / Queue / Output / Completion UX

Implement:

- one `Open...` entry point
- file/folder input
- `Remove`
- `Clear`
- queue-wide sequential processing
- real visible Output path
- Output Browse
- clear item/job states
- `Completed`
- auto-open Viewer after single-media processing
- sensible batch completion behavior
- hide Reference Focal unless Checkerboard Reference is selected

Do not redesign scientific result formats here.

### Milestone B — `.cfh` Result Bundle / View Results

Implement:

- `.cfh` writer
- `.cfh` reader
- `result.npz`
- `metadata.json`
- readable filename convention
- source SHA256 identity
- internal temp/cache separation
- disable automatic preview by default
- rename to `View Results...`
- multiple `.cfh` selection
- same-source validation
- single-result opening
- comparison opening from multiple result files
- schema validation
- legacy compatibility where practical

### Milestone C — Comparison Viewer UX

Implement/refine:

- Original media column in Split
- existing Overlay
- existing Split
- Separate Windows / detach
- frame synchronization
- camera synchronization
- result visibility
- current confidence semantics
- reference metric compatibility

Do not alter scientific coordinates for layout convenience.

### Milestone D — Calibration model expansion

After UX/result packaging is stable:

- audit actual AnyCalib runtime source
- add Fine-tuned AnyCalib adapter if validated
- add official GeoCalib dependency
- add Original GeoCalib adapter
- keep HandCalib unavailable until real runtime support exists
- add presets only for genuinely available models

## 25. Immediate non-goals

Not required for Milestones A/B:

- full long-video streaming rewrite
- realtime processing
- true multi-frame calibration network
- localization redesign
- installer/release packaging
- arbitrary plugin system
- database
- cloud processing
- fake HandCalib support
- Ground Truth claims

## 26. Acceptance criteria

A user should be able to:

```text
1. Open CalibFreeHands.
2. Click Open...
3. Select a video or folder.
4. Remove an accidentally added item if needed.
5. See/change the actual Output directory.
6. Choose one or more model configurations.
7. Click Start Processing.
8. See Processing.
9. See Completed.
10. Have the Viewer open automatically for a single-media run.
11. See readable .cfh result files in the Output directory.
12. Later click View Results...
13. Select one .cfh → single result opens.
14. Select multiple same-source .cfh files → comparison opens.
15. Switch Overlay / Split.
16. Optionally detach results into separate windows.
```

The normal workflow must not require manually navigating into:

```text
calibration/
pose/
results/<hash>/
```

## 27. Frozen decisions

The following are frozen for the next implementation milestone unless explicitly changed by the user:

1. One user-facing `Open...` entry point.
2. File or folder input.
3. Folder input adds supported media directly inside the folder.
4. Queue includes `Remove` and `Clear`.
5. Start Processing processes the whole queue sequentially.
6. Output path is always visibly resolved.
7. One source + one Calibration + one Reconstruction = one `.cfh`.
8. `.cfh` packages numeric NPZ + metadata JSON.
9. Source media is referenced, not embedded by default.
10. Source SHA256 is the authoritative same-source identity.
11. Intermediate calibration/pose files are hidden from normal output.
12. Automatic `preview.mp4` generation is disabled by default.
13. Finished single-media processing automatically opens the Viewer.
14. `Open saved result` becomes `View Results...`.
15. `View Results...` supports multiple `.cfh` files.
16. Multiple same-source `.cfh` files open in Comparison Viewer.
17. Existing Overlay / Split remain.
18. Separate-window viewing is added.
19. Reference focal is visible only for Checkerboard Reference.
20. Original AnyCalib + Original WiLoR remains the default.
21. Fine-tuned AnyCalib and HandCalib remain distinct.
22. Only real, verified model backends may be enabled.
23. Scientific camera-space data must not be modified for display convenience.
24. Existing internal calibration/reconstruction reuse remains.

## 28. Change control

This specification is the implementation reference for the next UX/result milestones.

Any implementation agent should:

1. read this file before changing the workflow;
2. inspect current repository code first;
3. reuse existing architecture where possible;
4. not silently reinterpret frozen decisions;
5. report unavoidable conflicts before replacing architecture;
6. preserve backward compatibility where reasonably possible;
7. never fake unavailable scientific/model functionality.

If a frozen decision changes later, update this document first or in the same change set.
