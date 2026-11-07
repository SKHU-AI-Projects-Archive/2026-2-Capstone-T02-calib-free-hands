# Implementation checklist

- [x] Python package and PySide6 application shell
- [x] Non-recursive supported-media discovery
- [x] ffprobe metadata boundary with explicit missing-tool error
- [x] Main-window drag and drop input
- [x] FFmpeg subprocess decoder boundary with frame timestamps
- [x] Project-local bundled FFmpeg option via imageio-ffmpeg
- [x] Backend-independent frame processing service and canonical result integration
- [x] Preview encoder boundary with optional audio mapping
- [x] Queue start/cancel UI bound to QThread lifecycle
- [x] One Open button with Windows Common Item Dialog file/folder selection, a warned Qt fallback, canonical queue paths, duplicate prevention, Remove/Clear, and visible item states
- [x] Sequential whole-queue processing with per-media failure handling, cancellation propagation, and immutable run settings
- [x] Batch output writability preflight and per-item error details in queue tooltips
- [x] Persistent resolved output directory with folder picker and single-media automatic result viewer
- [x] Checkerboard Reference focal input hidden for other calibration backends
- [x] Sampling and focal aggregation contract
- [x] Canonical result model with NaN/mask representation
- [x] NPZ and `run.json` persistence
- [x] Saved-result viewer with timeline and interactive canvas controls
- [x] Synthetic multi-frame result fixture tests
- [x] CSV/JSON export
- [x] Persistent application settings
- [x] Processing job lifecycle with cancellation
- [x] QThread wrapper for non-blocking job execution
- [x] AnyCalib API adapter and focal extraction/translation regression contract
- [x] Original WiLoR asset/config boundary adapter
- [x] Reproducible official WiLoR asset preparation script (MANO intentionally excluded)
- [x] Project-local CUDA PyTorch runtime and AnyCalib package installation path
- [x] Core automated tests
- [x] FFmpeg-backed decoding and sequential frame fallback when ffprobe is unavailable
- [x] Original AnyCalib backend adapter
- [x] Original pretrained WiLoR asset/config adapter boundary
- [x] End-to-end camera-space inference pipeline
- [x] Result viewer and exports
- [x] Main-window saved-result reopen action without rerunning inference
- [x] P0 Windows launcher uses project-local Python and explicit PYTHONPATH
- [x] P0 authoritative OpenPose hand topology shared by 2D preview and viewer
- [x] P0 camera translation applied once to joints and MANO vertices
- [x] P0 optional MANO triangle faces persisted with legacy-NPZ compatibility
- [x] P0 real pyqtgraph OpenGL result viewer with mesh/skeleton/grid/axis toggles
- [x] P0.1 QPainter float/NaN-safe overlay rendering and guaranteed painter cleanup
- [x] P0.1 stage progress and cooperative cancellation callbacks in processing pipeline
- [x] P0.2 GL item data validation and persistent per-hand item updates
- [x] P0.2 Qt OpenGL shader-cache reset across repeated viewer contexts
- [x] P1-A viewer mesh-state synchronization, camera-space references, fit and view presets
- [x] P1-A.1 viewer mesh opacity, expanded axes and click-context keyboard controls
- [x] P1-A.2 opaque/translucent mesh state switching and compact sample-count controls
- [x] P1-B MANO parameter preservation and calibration provenance trace
- [x] P1-C mesh/skeleton mutual-exclusion visibility controls and compact opacity layout
- [x] Packaged JSON model and preset configuration with validation
- [x] Calibration/reconstruction registries for configured backend creation
- [x] Original AnyCalib + Original WiLoR preset and persistent user default selection
- [x] GUI model/preset controls with unavailable HandCalib and AnyHand WiLoR entries
- [x] Selected model/preset provenance retained in `run.json`
- [x] Explicit comparison configuration rows with add/remove, stable IDs, duplicate model selection, and reference focal input
- [x] Comparison session manifest with shared calibration/raw-pose outputs and per-row canonical result/preview
- [x] Multi-configuration presets and persistent full-row default snapshots with legacy settings fallback
- [x] Bounded per-model focal progress, preview cancellation checks, single-configuration legacy output paths, and file-type-based saved-result viewer routing
- [x] Original WiLoR raw-pose path preserving source focal and relative MANO geometry before focal application
- [x] Comparison viewer Overlay/Split modes with per-configuration visibility, stable colors, and shared frame control
- [x] Linked split cameras, confidence alpha, reference metrics, and CSV/JSON metric export
- [x] C-2B runtime Reference selection and cached current-frame metric dashboard
- [x] C-2B source-retaining adaptive Split grid, native visibility checkboxes, configurable swatches, and left/right hue variants
- [x] C-2B confidence opacity/value/bar/color controls, single-candidate error vectors, and Detach All frame/camera synchronization
- [x] C-2B true in-viewport metric HUD with adjustable corner, size, and background opacity
- [x] C-2B projected Reference-to-candidate per-joint error/name annotations; raw arrays remain unchanged
- [x] C-2B Detection Confidence terminology and whole-hand display; mesh opacity remains independent
- [x] C-2B play-time control auto-hide, activity restore, pause visibility, and F11/Escape fullscreen handling
- [x] C-2B compact semantic controls, More/Detach menu, and drag-to-reorder result panels with runtime-only order
- [x] C-2B current-Reference CSV/JSON export and per-joint detail table
- [x] C-2B timestamp-driven monotonic playback, speed presets, frame dropping, and active-view-only updates
- [x] C-2B in-place HUD value updates, widget reuse, and stylesheet/layout churn removal
- [x] C-2B optional playback statistics and reusable OpenGL geometry items; no inference or CUDA work during playback
- [x] C-2B frame-value HUD updates preserve widget identities; geometry changes update the active view once
- [x] Comparison viewer session opening without inference rerun; source frames decoded once and shared
- [x] Milestone B bundle format: atomic ZIP-compatible `.cfh` with canonical `result.npz` and validated `metadata.json`
- [x] Safe result filenames, SHA256 source identity, model/checkpoint/MANO provenance, and file-scoped collision handling
- [x] Temporary per-media processing workspace; normal Output contains only user-facing `.cfh` results
- [x] Automatic preview generation disabled while retaining the audio preference for future explicit preview export
- [x] `View Results...` opens one or multiple `.cfh` files and rejects mixed source SHA256 selections
- [x] ResultViewer and ComparisonViewer load bundles without extracting them; missing/changed source media does not block 3D viewing
- [x] Canonical legacy NPZ/session loading remains available; raw-pose NPZ gives a readable validation error
- [x] Windows-native media picker, stable configuration headings, aligned settings, persistent input/result directories, and no unused Preserve audio control
- [x] Milestone B Offscreen Qt, bundle validation, atomic cleanup, output cleanliness, source identity, and auto-viewer tests
- [x] Milestone C-1 native media dialog, compact controls, Checkerboard focal filenames, validated persistent/importable User Presets, and unique WiLoR reuse
- [x] C-1 Fix Pass shared configuration grid, compact local control groups, preset origin tracking, and explicit overwrite/Save As
- [x] C-1 Fix Pass manually approved in the Windows GUI
- [x] C-2A Qt-independent metric registry and canonical frame/hand alignment
- [x] C-2A per-joint, wrist, raw camera-space, depth, translation, relative wrist, MPVPE, and Procrustes metrics
- [x] C-2A explicit protocol-pending states for unreproduced WiLoR benchmark/temporal metrics
- [x] C-2B viewport-first presentation: compact controls, selected metric overlays, Reference badge, on-demand 21-joint details, and no permanent metric table
- [x] C-2B duplicate Checkerboard Reference Focal row label removed while retaining the header and value control
- [x] C-2B presentation stabilization: fixed viewport overlay chrome, pin, mouse restore, compact menus, text-only HUD, bounded annotation modes, and pointer-based panel reorder
- [x] C-2B shared asynchronous MediaFrameProvider with lazy FFmpeg frames, source identity sharing, and a 384 MiB byte-bounded LRU cache
- [x] C-2B viewer frame requests remain non-blocking; copied QImage display storage survives cache eviction
- [x] C-2B static image sources use one cached RGB frame; missing and changed source media remain non-blocking for 3D results
- [x] C-2B automated and Windows GUI validation completed; user manually approved final acceptance on 2026-11-03


## C-2B Final Stabilization and Hardening

- [x] QWindow-safe activity filtering and QComboBox-safe menu close handling
- [x] Unified dropdown hover/click behavior for QToolButton menus and visible QComboBox selectors; button/popup transit, outside click, second toggle, action close, and Escape
- [x] Checked Pin presentation, pinned visibility, and mouse-movement control restore
- [x] HUD has no scroll area or result separators; adaptive font fit, transparent default, and reusable per-frame labels
- [x] Comparison presentation tracks requested and presented frames separately; source misses retain the full previous presentation and stale frame-ready events are discarded
- [x] Comparison frame-ready transition updates source, 2D, 3D, HUD/metrics, annotations, slider, and status from one presented-frame index
- [x] Playback clock continues while decoding; pause invalidates pending playback presentation and paused manual seek remains exact
- [x] Targeted viewer/presentation/media-provider tests pass; full suite result recorded in VALIDATION_CHECKLIST.md
- [x] Actual three-result Windows Comparison Viewer inspected in Overlay and Split without inference rerun; source frames, pose overlays, metrics, pin state, and playback updates observed
- [x] C-2B complete; user manually approved final acceptance on 2026-11-03

## C-2C Unified Result Viewer

- [x] Saved `.cfh` files use the same modern ComparisonViewer shell for one result and for 2+ same-source results; automatic post-processing preview follows the same route
- [x] Single-result mode has a Result title and result identity HUD, retains Overlay/Split and the shared timestamp-based playback/provider path, and hides reference selection, comparison metrics, error vectors, comparison exports, and result reordering
- [x] Single-result mode does not calculate reference metrics or expose self-reference error labels; joint-name annotations and detector-confidence display remain available
- [x] MANO topology in a single result enables the Mesh toggle independently of the comparison controls
- [x] Single-result bundles can be opened without a source hash; multi-result source SHA256 matching remains required
- [x] Single-result static images use one cached RGB frame and remain seek-safe across result frames
- [x] Single-result playback probe measured 29.75 source fps at 1x and 59.75 source fps at 2x on the local 570-frame sample; playback uses the shared 384 MiB bounded provider
- [x] QApplication shutdown closes the shared media providers so decoder workers stop before Qt objects are torn down
- [x] Legacy ResultViewer and NPZ/session loading implementation remain present for compatibility

## First-User UX & Stability Pass

### P0 Stability / functional bugs

- [x] Fix Windows Common Item Dialog initialization/selection failure (HRESULT 0x80004002); native picker remains the normal Windows path and fallback is reserved for genuine unavailability
- [x] Add shared MediaFrameProvider acquire/release ownership; same-source viewers share providers, final release removes registry entry, stops worker, and releases cache/resources
- [x] Preserve bounded per-provider cache and prohibit full-video preload while ensuring unused source caches do not accumulate across A/B/C open-play-close sequence
- [x] Fix Error Vectors candidate selection UI; no unexplained tiny popup, repeated Off/Candidate transitions work, and selected/rendered state stays synchronized
- [x] Separate requested HUD size (Small/Medium/Large) from effective fitted font size and restore requested size when viewport space permits
- [x] Prevent transient HUD shrink from popup, menu focus, or check-state changes when viewport/content are unchanged
- [x] Keep Reference identity, focal value/unit, and badge semantically together with natural narrow-viewport wrapping

### Main Window first-user UX

- [x] Keep Preset selector/actions/Set as default in Preset row; move Add Configuration out of Preset row
- [x] Add a `+ Add` control below the configuration table, aligned with Name and kept beneath the last row
- [x] Reorder configuration columns to Name, Calibration, Focal, Reconstruction, Action
- [x] Use Manual Focal consistently in the selector, configuration row, Viewer result/configuration names, and newly created user-facing names; retain `checkerboard_reference` and normalize historical display names in presentation only, without rewriting metadata or renaming files
- [x] Label the numeric parameter Focal (px); enable it only for Manual Focal and show a clear disabled/N/A state for other calibration choices
- [x] Relocate Sampling/Sample Count/Aggregate toward processing configuration and Theme/Language toward application preferences and Output
- [x] Compact Output path, Browse, and Collision into one logical row with stretch priority on the path field

### Viewer first-user UX

- [x] Add balanced horizontal margins to playback footer controls
- [x] Replace Speed dropdown with adjacent-preset stepper; preserve preset sequence, current source time, and timestamp clock behavior
- [x] Replace Previous/Play/Pause/Next text buttons with Qt-native media icons and retain tooltips/accessibility names
- [x] Replace Overlay/Split buttons with an exclusive two-state segmented control
- [x] Make Pin a clear ON/OFF toggle with explicit auto-hide semantics
- [x] Simplify comparison toolbar start to `Layout [switch] [pin icon]`, with state tooltip and palette-aware icon-only Pin
- [x] Rename the comparison toolbar group to `View`; give the icon-only Pin a visible neutral button surface and border in OFF/ON states
- [x] Reduce the Pin button to 26-by-26 px; use a subdued green ON surface with a white vector pin while retaining the gray OFF state
- [x] Support HUD drag, pointer ownership, nearest-corner snap, and non-blocking 120–180 ms snap animation
- [x] Keep HUD visual settings available; remove corner menu actions if drag becomes the primary position control
- [x] Rename annotation menu Joints to Annotations while retaining the separate Joint geometry checkbox
- [x] Keep Manual Focal identity, focal number/unit, and Reference badge naturally grouped and wrapped (for example, `Manual Focal 1500 px [REFERENCE]`)

All implementation items above are complete. Remaining user and Windows GUI acceptance is tracked in `VALIDATION_CHECKLIST.md`; implementation completion alone does not establish overall readiness.

## Phase 2 Fixed Metric Workspace and Shared Camera

- [x] Keep the three grid planes, positive raw camera-space axes, and major numeric labels fixed to X/Y [-1, 1] m and Z [0, 2] m with 0.1 m minor and 0.5 m major spacing; Grid also controls numeric labels and Axis controls axis names/lines.
- [x] Keep Metric Overview and its fixed workspace-centered 20-degree elevation / 45-degree azimuth framing available; Phase 2.1 adds whole-video GT Hand View as the initial camera and Reset behavior.
- [x] Make Fit All explicit and fit only finite geometry from the current frame and visible hands/results, without moving or scaling workspace references or mutating canonical arrays.
- [x] Share camera parameters between Overlay, visible Split panels, and detached comparison panels; preserve the active camera pose when switching Overlay/Split.

## Phase 2.1 Whole-Video GT Hand Framing

- [x] Compute the selected GT (or sole result) whole-video detected finite wrist median and robust all-joint 2.5–97.5 percentile extent once per loaded result/GT selection; apply only the camera transform.
- [x] Use the selected GT framing for initial Hand View; candidate results do not affect the default framing, and playback/seek never recenter the camera.
- [x] Keep Metric Overview at the fixed Phase 2 workspace camera; Fit All remains based on current-frame visible geometry; Reset restores Hand View.
- [x] Recompute/cache a new GT framing on selection while preserving current camera pose; apply it only when Hand View is chosen.
- [x] Keep Front/Side/Top orientation-only relative to current center/distance, and synchronize Hand View / Metric Overview across Overlay and Split.

## Phase C Physical / Hand 3D Space

- [x] Add shared Physical/Hand runtime mode to the unified result/comparison Viewer; default remains fixed X/Y [-1, 1] m and Z [0, 2] m Physical workspace.
- [x] Present physical camera Y as display-up without changing raw camera-space arrays; use real metric bounds and adaptive 1/2/5 grid spacing in Hand Space.
- [x] Cache robust whole-video Hand bounds from selected GT (or sole result), using detected finite joints and MANO vertices, percentile extents, frame-center outlier filtering, padding, and a 0.2 m minimum span; fall back to Physical bounds without valid geometry.
- [x] Separate numeric scale Labels from Grid and Axis visibility. Physical/Hand changes and GT selection preserve the active camera and frame.
- [x] Expose Focus GT / Focus Hands, Space Overview, and Fit All as separate camera actions; synchronize workspace and labels across Overlay, Split, and existing detached views.
- [x] Correct README's built-in Original preset description to all-frame sampling, matching the Phase B implementation.

## Phase C.1A Workspace Geometry and Controls Stabilization

- [x] Use one centered cubic workspace model for all three grid planes, axis lines, numeric labels, and workspace framing in both Physical and Hand modes.
- [x] Preserve robust whole-video Hand bounds while expanding them to a cube; keep all raw camera-space results unchanged and retain Physical fallback for missing/invalid geometry.
- [x] Derive label offsets from workspace scale, show actual metric ticks with normalized zero labels, and remove stale numeric label items when changing modes.
- [x] Replace the Labels checkbox with an accessible On/Off switch; consolidate View, Space, Labels, and Pin in the palette-aware primary control strip and remove duplicate state text.
- [x] Preserve camera/frame behavior and keep Grid, Axis, and numeric Labels visibility independent.
- [ ] Native Windows visual acceptance and manual playback review remain pending; Phase C.1B camera behavior and detached-view redesign are deferred.

## Phase C.1A Fix Pass: Workspace Placement and Space Center (2026-11-07)

- [x] Correct Physical reference planes to the global coordinate planes: XY at raw Z=0, XZ at raw Y=0, and YZ at raw X=0. Plane sizes remain 2 m × 2 m; the Physical volume center remains (0, 0, 1 m).
- [x] Keep Physical axes on the raw coordinate axes through (0, 0, 0); Hand planes and equal-length axes remain centered on the selected-GT cube.
- [x] On Space changes, move only the orbit center to the active workspace center while preserving distance/orientation and synchronizing Overlay/Split cameras. Frame and playback state remain unchanged.
- [x] Make the primary control container transparent; retain its compact View, Space, Labels, Pin order and palette-aware individual controls.
- [x] Focused and full automated validation pass (100 viewer/comparison tests; 281 total tests), with import, compileall, and dependency checks passing.
- [ ] Native Windows review of Physical plane placement, orbit center, and transparent toolbar remains pending. Camera preset semantics remain deferred to C.1B.

## Phase C Final Camera, Contrast, and GT Switching

- [x] Store separate Physical/Hand camera poses and restore each pose across space changes in Overlay and Split; preserve the initial Physical pose and rebase retained Hand offsets when GT changes.
- [x] Provide deterministic Front/Left/Right/Top orientation presets and reliable Focus GT, Space Overview, and current-frame Fit All recovery actions.
- [x] Use a consistent light foreground for plain viewport chrome labels without adding an opaque primary-controls panel.
- [x] Cache metrics by ordered Reference/Candidate result IDs; remove duplicate GT-change render/panel refreshes and run large uncached metric batches in a single background worker with stale-selection protection.
- [x] Keep metric formulas, canonical result arrays, `.cfh` data, and frame/playback behavior unchanged.
- [ ] Native Windows visual contrast, camera interaction, and playback acceptance remains pending; Phase D is not started.

## Phase D.1 Unified Panel Ordering Foundation

- [x] Stable Source Media, Overlay 3D, and configuration-ID result panel identities; Split and Overlay orders use existing widget instances.
- [x] Shared result ordering across Overlay HUD, Split panels, and Results menu; Source Media Split position and Overlay left/right order are independent.
- [x] Initial GT-first policy yields to explicit user ordering; later GT changes preserve the custom order.
- [x] HUD drag insertion uses rendered HUD group geometry; Split panels and Overlay Source/3D can be reordered with compact handles.
- [x] Preserve original GL/source widgets, result arrays, providers, frame/playback, camera, HUD, annotations, and saved data during reordering.
- [x] Focused automated regression coverage added for ID/order behavior, mouse-driven HUD and Split drags, source placement, Overlay swapping, and frame/camera/provider/array preservation.
- [ ] Native Windows D.1 visual and interaction review remains pending.
- [ ] D.2 individual Detach/Reattach and D.3 detached synchronization/aspect-ratio work are deferred.

Phase C Final native Windows testing and acceptance were reported by the user as completed. This is user-reported acceptance, not a claim that Codex independently performed the native checks.
