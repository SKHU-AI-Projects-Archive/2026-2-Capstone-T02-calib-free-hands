# Validation checklist

## Milestone C-2B

- [x] Full pytest suite: 177 passed; includes runtime Reference/checkerboard/fallback selection, N-result adaptive source grid, current-frame metric cache, missing-hand N/A, data-unavailable gating, and selected-Reference export.
- [x] Offscreen dashboard tests cover per-joint values, confidence opacity/value/bar/color controls, optional error vectors, display-color raw-array immutability, and detached frame synchronization.
- [x] `compileall` for `src` and `tests`; `git diff --check`.
- [x] Offscreen captures generated for Overlay + 2, Split + 2, Split + 5 and layout structure inspected.
- [ ] HUMAN VISUAL REVIEW REQUIRED in the normal Windows GUI; offscreen captures have tofu text and a blank OpenGL viewport in this environment.

- [x] Branch is `CalibFreeHands`
- [x] Folder discovery is non-recursive
- [x] Focal sampling/aggregation unit tests pass
- [x] NPZ round-trip and missing-value mask test pass
- [x] ffprobe missing-tool error test pass
- [x] FFmpeg decoder missing-tool contract test pass
- [x] Bundled FFmpeg preview encoding test pass
- [x] Bundled FFmpeg RGB decode smoke test pass (two frames, 8x8 RGB)
- [x] Pipeline service missing-hand and camera-space integration test pass
- [x] Preview encoder missing-tool contract test pass
- [x] Offscreen Qt startup smoke test pass
- [x] Saved-result viewer load and widget construction test pass
- [x] MainWindow offscreen construction includes saved-result reopen action
- [x] CSV/JSON export test pass
- [x] Settings persistence test pass
- [x] Queue completion/cancellation test pass
- [x] Queue operation callback executes actual work boundary (no GUI sleep loop)
- [x] Worker implementation keeps processing outside the GUI event loop by QThread boundary
- [x] focal extraction and camera-translation rescale regression test pass
- [ ] Windows GUI launch (requires PySide6)
- [x] Raw unannotated media end-to-end test (`local_test_media/5_fold.mp4`)
- [x] AnyCalib/WiLoR integration test with local model assets
- [x] AnyCalib CUDA model instantiate and focal inference smoke test
- [x] AnyCalib application adapter focal inference smoke test
- [x] Original WiLoR checkpoint loads through MANO with licensed assets
- [x] WiLoR runtime identity recorded: Python 3.11.7, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 Ti
- [x] Real NPZ/run.json validation: 502 frames, MANO mesh, masks, focal, provenance
- [x] Real preview validation: 2D overlay plus original AAC audio preserved
- [x] Mean aggregation and All-frame sampling smoke path with real model backends
- [x] MainWindow `_process_media` real pipeline reaches completion and writes result/preview/run.json
- [x] P0 project-local Windows launcher path and preflight runtime diagnostics
- [x] P0 OpenPose topology regression and non-sequential bone check
- [x] P0 vertex translation numerical regression test
- [x] P0 real OpenGL viewer offscreen construction test
- [x] P0 real sample rerun: 502 frames, 2x1538 MANO faces, NaN/mask and mesh finite checks
- [x] P0.1 float32/NaN QPainter regression test and offscreen repaint
- [x] P0.1 actual Uniform/8/Median short-video run with calibration/reconstruction/preview progress
- [x] P0.1 cooperative cancellation regression test
- [x] P0.2 existing NPZ structural validation for 1_insert and 5_fold_p0
- [x] P0.2 native Windows minimal GL smoke test with grid/axis/line/scatter/mesh
- [x] P0.2 repeated ResultViewer lifecycle: 5_fold_p0 ??1_insert ??5_fold_p0
- [x] P0.2 shared display-transform regression test
- [x] P1-A viewer mesh/grid/axis toggle, auto-fit and raw-data immutability tests
- [x] P1-A native Qt viewer construction for `1_insert` and `5_fold_p0` results
- [x] P1-A.1 viewer opacity, axis-length and click-context keyboard routing tests
- [x] P1-A.1 native Qt interaction checks for `1_insert` and `5_fold_p0` results
- [x] P1-A.2 mesh render-mode/sample-count control regression tests
- [x] P1-A.2 native Qt checks for mesh render-state switching and sample-count enablement
- [x] P1-B synthetic MANO NPZ round-trip and calibration provenance trace regression tests
- [x] P1-B legacy NPZ reload without MANO parameters
- [x] P1-C mesh/skeleton visibility transitions and compact opacity layout regression tests
- [x] Model config validation, original preset values, and registry factory tests
- [x] Model/preset GUI controls, unavailable model state, custom default persistence, and stale-ID fallback tests
- [x] Selected backend wiring and model/preset provenance regression tests
- [x] Comparison session calibration and raw-pose deduplication tests
- [x] Shared raw pose with distinct focal values produces independent canonical NPZ outputs
- [x] Session JSON/NPZ round-trip, cancellation state, and unavailable backend validation
- [x] Configuration row add/remove, stable IDs, duplicate selection, and reference focal UI tests
- [x] Full pytest suite and offscreen Qt tests (86 passed)
- [x] Milestone A queue workflow regressions: Open menu, file/folder input, path identity, remove/clear, serial processing, partial failure, cancellation, control lock, output persistence, automatic Viewer routing, and focal visibility
- [x] Full pytest and offscreen Qt suite after Milestone A (102 passed)
- [x] Output-base write preflight stops the batch as `Failed`; per-media failure details remain on the queue item
- [x] Python compileall for `src` and `tests`
- [x] Bounded real queue smoke: 1-second local video, cached Original AnyCalib + WiLoR/MANO, output written, `Completed`, GUI-thread ResultViewer construction
- [x] Regression tests for focal progress bounds, cancellation during preview, and single-configuration output compatibility
- [x] Multi-row preset loading, persistent configuration snapshots, malformed snapshot fallback, and saved-result routing independent of file dialog filter
- [x] Comparison viewer construction from a saved three-result session and Overlay/Split frame retention
- [x] Split camera initialization, preset, orbit/pan/zoom state synchronization tests
- [x] Configuration hue, frame/hand confidence alpha, NaN fallback, and visibility tests
- [x] Reference joint/depth/translation metrics and CSV/JSON numerical export test
- [x] Shared media source is decoded once and its frame list is reused across configurations
- [x] Official WiLoR checkpoint and detector downloaded outside protected source tree
- [x] Official model asset setup script with network access
- [x] Interactive Windows GUI inspection performed with actual same-source results; overall C-2B user approval recorded in the final-closure section
- [ ] C-2B normal Windows visual review: Overlay + 2 results, Split + 2, Split + 5
- [ ] P1-B new original WiLoR run confirming MANO parameter persistence in a fresh real result
- [x] Actual AnyCalib + WiLoR/MANO smoke on first 3 frames of existing `4_assemble.mp4`; single and two-configuration outputs validated without network downloads
- [ ] `git diff` final review

Specific Windows layout cases not individually recorded: Split + 2 and Split + 5. Actual Overlay + 3 and Split + 3 layouts were inspected; the user separately approved overall C-2B acceptance.

## Milestone B ??UX Polish / CFH Result Bundle / Result Loading

- [x] Windows-native file/folder selection boundary opens directly from a menu-free Open button; multi-file/direct-folder queue semantics tested
- [x] Last input and result directories persist; drag/drop and non-recursive discovery remain covered
- [x] Configuration headings, input widths, sequential default naming, aligned Theme/Language and separate Output row tested
- [x] Preserve audio control removed from normal processing UI; automatic preview is false while preview encoder remains
- [x] `.cfh` round-trip preserves canonical arrays and metadata; archive contains exactly `result.npz` and `metadata.json`
- [x] Invalid ZIP, missing/extra members, malformed metadata/schema, malformed SHA, and raw-pose NPZ rejected
- [x] Atomic write failure leaves no partial result or temporary file
- [x] Safe filename and Ask/Overwrite/Auto rename/Skip result-file collision behavior tested
- [x] Processing writes one or more `.cfh` files to Output and cleans the internal temporary workspace
- [x] Same-source comparison uses SHA256 even with different source filenames; mixed-source selection displays a readable error
- [x] Single and comparison viewers load bundles directly; missing source still permits 3D viewing
- [x] Legacy canonical NPZ and session APIs remain covered; raw pose NPZ has a readable error
- [x] Offscreen Qt application suite: 125 passed
- [x] Real one-frame Original AnyCalib + Original WiLoR smoke produced one `.cfh`; bundle load and ResultViewer construction passed
- [x] Real two-configuration smoke produced two same-source `.cfh` files, only bundles in Output, shared one WiLoR pose model, and constructed ComparisonViewer
- [x] Actual AnyCalib module provenance recorded from `HandCalib/AnyCalib/anycalib/__init__.py`
- [x] No automatic preview was created; temporary internal workspace was removed
- [x] C-1 preset schema validation, persistence, import/export, built-in protection, custom-state detection, and default reset tests
- [x] C-1 Checkerboard focal filename formatting and collision behavior tests
- [x] C-1 native dialog boundary, cancellation, readable fallback warning, and Queue routing tests
- [x] C-1 Windows Common Item Dialog smoke: initialized on C: and D:, selected current folder, and returned the selected folder path
- [x] User-reported Windows check: native dialog file selection and D: navigation work
- [x] C-1 comparison run constructs the shared WiLoR reconstruction backend once across rows
- [x] C-1 focused pytest suite: 82 passed
- [x] C-1 full pytest suite: 153 passed
- [x] C-1 Python compileall for `src` and `tests`
- [x] C-1 `git diff --check` and final changed-file/scope review
- [x] C-1 Fix Pass preset origin/prefill, stable-ID overwrite, Save As identity, built-in reservation, and Custom item regressions
- [x] C-1 Fix Pass targeted tests: 118 passed
- [x] C-1 Fix Pass full pytest suite: 163 passed
- [x] C-1 Fix Pass Python compileall for `src` and `tests`
- [x] C-1 Fix Pass offscreen MainWindow screenshots reviewed at 980x560 and 1180x720: one Original row and Original + Checkerboard Reference; shared grid columns, compact settings groups, and expanding Output row
- [x] C-1 Fix Pass `git diff --check`
- [x] C-1 Fix Pass user-reported final visual approval in the normal Windows GUI
- [x] User-reported interactive Windows GUI check and approval for C-1 Fix Pass
- [x] C-2A analytical metric tests: exact match, 10 mm lateral offset, 25 mm depth offset, missing/NaN joints, left/right and frame-index alignment
- [x] C-2A translation availability, topology-gated MPVPE, PA-MPJPE/PA-MPVPE similarity alignment, and raw-array immutability tests
- [x] C-2A protocol registry keeps unverified WiLoR benchmark and temporal metrics pending
- [ ] C-2A saved `.cfh` smoke test (no `.cfh` result bundle is present in the repository)
- [x] Startup hotfix fresh-process `import calibfreehands.app.main` regression test
- [x] Original startup ImportError reproduced before fix; root cause: `JOINT_NAMES` cross-import during eager package initialization
- [x] Export path, pipeline result, and visualization topology imports stay lightweight in a fresh process
- [x] Offscreen MainWindow construction smoke test (`MAINWINDOW_OK`)
- [x] Final full diff review

## C-2B UI / Presentation Fix Pass

- [x] Targeted Comparison Viewer and focal-row regression tests: 16 passed
- [x] Viewport-child HUD geometry, immediate metrics, Reference badge, and HUD controls covered
- [x] Synthetic joint error values, Reference exclusion, handedness visibility, and joint-name mode covered
- [x] Detection confidence terminology/hand-level scope, separate Mesh opacity, and Mesh-off disabled state covered
- [x] Runtime-only panel reorder preserves IDs/raw arrays and updates Results menu order
- [x] Playback auto-hide, mouse restore, pause visibility, and F11/Escape behavior covered
- [x] Full pytest suite: 182 passed
- [x] Startup import, compileall, and `git diff --check` for Presentation Redesign
- [x] Offscreen captures created and inspected for Overlay + 3 (visible/hidden), Split + 3, Split + 5, and joint annotations
- [x] Offscreen review confirms HUD overlay, compact metric rows, Original Media first, and usable Split 3/5 layout
- [x] Offscreen joint annotation capture confirms projected labels; coincident synthetic joints stack vertically as expected
- [ ] HUMAN VISUAL REVIEW REQUIRED in the normal Windows GUI. Headless OpenGL capture does not reliably render hand meshes/source imagery and can corrupt HUD text in the controls-visible Overlay capture.

## C-2B Playback Performance Optimization Pass

- [x] PlaybackClock fake-monotonic tests cover 30 fps, irregular timestamps, invalid/duplicate fallback, speed continuity, pause/resume, and frame skipping
- [x] Offscreen ComparisonViewer checks cover 100-frame HUD reuse, scroll preservation, active mode, hidden results, disabled annotations/vectors, and speed controls
- [x] Full Windows pytest suite: 193 passed
- [x] Startup import, `compileall -q src tests`, and `git diff --check`
- [x] Real Windows `4_assemble.mp4` (1080×1920, 30 fps) with three same-source `.cfh` results: five six-second scenarios; source clock measured 6.0 source seconds per 6.0 wall seconds
- [x] Frame updates / mean `_set_frame` ms / p95 ms: Overlay 29.99 fps / 0.61 / 0.67; Confidence 29.96 / 6.12 / 10.85; Annotations 29.99 / 0.61 / 0.72; Full 29.99 / 6.56 / 13.70; Split 3 29.88 / 0.33 / 0.36
- [x] Process RAM sampled 3.72–3.74 GB, within the prior 3.9 GB observation; shared decoded frame cache remains unchanged
- [x] HUD stylesheet/layout loop fix reduced sampled process CPU from 98.8% idle / 101.6% playback to 4.7% idle / 27.1% playback in matched 3-second probes
- [x] OpenGL renderer identified as Intel UHD Graphics 750; per-process Windows GPU Engine 3D samples were 0–5.64% (one-second sampling, with invalid-data warnings on some samples); NVIDIA utilization was system-wide and not used to infer Intel rendering
- [x] Windows GUI playback smoke: 0.1×/1×/2× selections, continuous playback, pause, source synchronization, auto-hide, and Split mode; the full eight-speed final matrix still needs user review

## C-2B Presentation UX Stabilization
- [x] Full pytest suite: 201 passed; includes fixed viewport geometry, HUD interactions, hover menus, annotation modes, and reorder lifecycle
- [x] Windows GUI visual check at 1280x800 with six synthetic test-only result panels; drag reorder changed Method 2/Method 1 ordering
- [x] Windows GUI visual review for Split + 3 layout; hovered-joint readability remains for user review

## C-2B Media Memory Optimization
- [x] Provider tests: lazy sequential decode, repeated request coalescing, cache hit/eviction/budget, reverse seek, latest request replacement, image buffer ownership, missing/changed source, and source sharing
- [x] Full Windows pytest suite: 207 passed; `import calibfreehands.app.main`, compileall, and diff check pass
- [x] Windows 4_assemble.mp4 (570 frames) + three same-source CFH results: one provider shared by all rows; frame 569 decoded; 931 decoded-frame insertions including rereads after eviction; cache 379.7 MiB under 384 MiB cap
- [x] Measured process RSS: 220.7 MiB after opening results, 594.5 MiB after decoding through frame 569, and 606.9 MiB after playback; previous full-frame preload observation was about 3.7 GiB
- [x] Seek latency at frames 0/60/150/300/569: 0/310/506/834/1458 ms
- [x] 4-second playback probes: 1x 29.99 source fps at 19.1% process CPU; 2x 59.95 source fps at 30.8% process CPU

## C-2B Final Validation
- [x] C-2B COMPLETE — USER MANUALLY APPROVED (2026-11-03); detailed evidence remains limited to the individual scenarios recorded below.
- [x] Windows GUI opened three actual same-source CFH results without inference; Overlay and Split + 3 inspected at the 1280x720 minimum window; 1280x800 layout geometry also covered offscreen
- [x] Real source video, synchronized result poses, and current-frame metrics updated during playback; controls auto-hid while playing and returned on mouse activity
- [x] Windows UI speed selector accepted 0.1x and 2x; separate 4-second probes verified 1x/2x playback rates
- [x] Final C-2B hardening suite: 211 passed; application import, compileall, and diff check passed
- [x] User's final Windows review approved auto-hide, Pin, mouse restore, source loading stability, 2× source/2D/3D synchronization, pause/seek, dropdown and popup behavior, HUD, annotation modes, speed matrix, and clean console after the QWindow hotfix.
- [ ] Specific historical scenarios not separately recorded (for example Split + 2 and Split + 5) remain unclaimed; this does not override the user's overall C-2B acceptance.


## C-2B Final Stabilization and Hardening

- [x] Regression tests first reproduced QWindow `isAncestorOf` TypeError, QComboBox `.menu()` AttributeError, delayed hover behavior, HUD scroll-area presence, and source-frame placeholder flicker
- [x] Targeted presentation/viewer/provider suite after final hardening: 60 passed
- [x] Offscreen Qt checks cover QWindow events, combo close safety, immediate hover-open, button/menu transit, click latch, outside-click dismissal, second toggle, Escape/action dismissal, no-scroll HUD, fit bounds, source-frame retention, and stale-ready filtering
- [x] Windows app opened the three actual same-source `.cfh` results without inference; Overlay and Split + 3 were inspected, including playback frame/metric updates and pin state
- [x] Existing Phase 3 local media benchmark remains: 379.7 MiB cache under 384 MiB, 606.9 MiB process RSS after playback, 1x/2x playback probes at 29.99/59.95 source fps
- [x] Full pytest after final edits: 211 passed; startup import `IMPORT_OK`; compileall and `git diff --check` pass
- [x] 1280x800 offscreen layout checks keep the HUD inside the viewport; actual Windows window inspected at 1282x752 (minimum content target 1280x720)
- [x] User manually approved the overall C-2B result on 2026-11-03. This does not assert unrecorded individual layout scenarios.

## C-2B Final Sync & Menu Consistency Fix
- [x] Reproduced the desynchronization path: ComparisonViewer advanced its slider, 2D/3D, and HUD immediately while the shared MediaFrameProvider could still be decoding the requested RGB frame.
- [x] Added delayed-source tests for held presentation, atomic ready transition, stale completion discard, pause invalidation, and exact manual seek.
- [x] Added hover-open/click-latch tests for Reference, Speed, and Color by combo selectors; retained QToolButton menu transition, outside-click, second-toggle, action, and Escape coverage.
- [x] Targeted comparison viewer tests after final sync changes: 34 passed.
- [x] Full Windows pytest suite after final sync changes: 218 passed; application import, compileall, and `git diff --check` passed.
- [x] Normal Windows GUI opened three actual same-source `.cfh` results without inference; 2× playback advanced source, overlay, 3D poses, and current-frame metrics together; pause retained the displayed frame.
- [x] Results and Detection menus opened; Reference changed from 1300 to 1500 and metrics refreshed. Speed selector accepted 2×. Automated Qt tests cover hover-open, transient close, click latch, second toggle, outside click, Escape, and action dismissal across dropdown controls.
- [ ] Actual Windows pointer hover-open behavior was not directly exercised through the available UI automation controls. Final user visual review remains pending.

## C-2B Final QWindow EventFilter Hotfix
- [x] Regression reproduced: with an active Reference combo, `_is_active_hover_menu(QWindow)` raised `QWidget.isAncestorOf(QWindow)` TypeError before the fix.
- [x] QWidget ancestry is checked only for QWidget targets; active combo/menu popup windows are matched by their QWidget `windowHandle()` identities. Unrelated QWindow targets return false without calling QWidget APIs.
- [x] Regression coverage: QWidget root/child, unrelated QWindow, live QComboBox popup QWindow, live QMenu popup QWindow, and eventFilter Hide/MouseButtonPress handling for both active dropdown types.
- [x] Targeted comparison viewer tests: 35 passed; full pytest: 219 passed; app import `IMPORT_OK`, compileall, and `git diff --check` passed.
- [x] Windows GUI smoke: three existing same-source `.cfh` results opened without inference; Reference, Speed (2×), Results, and Detection popups exercised; 2× playback ran during a 22-second observation interval and advanced the frame; no traceback, Python override error, TypeError, or AttributeError in captured stderr (all counts 0).
- [ ] Direct hover-only testing was limited by the available Windows UI automation API; dropdown entry/leave was exercised through pointer movement accompanying control clicks. Final user approval remains pending.

## C-2C Unified Result Viewer

- [x] Single-result mode acceptance covers Result title, no Reference/metrics/error vectors/comparison export/results menu, no self-reference metric cache or badge, one-result Split, source-missing 3D availability, static-image one-frame cache, MANO mesh toggle, eight speed presets, and exact frame seek
- [x] Main-window processing auto-open routes one result to ComparisonViewer single mode and 2 results to ComparisonViewer comparison mode
- [x] Manual `View Results...` routes one or multiple `.cfh` bundles to ComparisonViewer; multi-result same-source enforcement remains covered
- [x] Full pytest: 224 passed; Windows actual single `.cfh` inspected in Overlay and Split; Windows three-result `.cfh` inspected in Overlay and Split
- [x] Current single and multi Windows windows showed source media and saved 2D/3D data without rerunning inference; single mode omitted comparison-only controls while multi mode retained Reference and comparison HUD
- [x] Single-result playback probe on the 570-frame sample measured 29.75 source fps at 1x and 59.75 source fps at 2x over 4 seconds each; Windows 2x playback visually advanced source, 2D, and 3D together
- [x] During 24 seconds of single-result 1x playback, provider cache reached 398,131,200 of 402,653,184 bytes and evicted 656 frames; cache remained within its 384 MiB bound
- [x] The QApplication shutdown hook closed providers after active single-result playback; the 7-second shutdown smoke produced no stderr

## First-User UX & Stability Pass

All entries below are pending implementation and validation. No acceptance item has been run for this pass.

### First-user flow

- [ ] Original feedback provider repeats launch → media selection → configuration → Start Processing → completion → result viewer → playback → 3D inspection without being briefed on the changes
- [ ] A second, previously inexperienced team member repeats the flow where available; observe Manual Focal meaning, Add location, and Viewer-control discoverability without coaching

### P0 stability

- [ ] Windows GUI picker matrix: single file, multiple files, folder, current folder, C: and D: locations, cancel, close/reopen, and repeated use; native picker is used normally and no HRESULT 0x80004002 occurs
- [ ] Genuine native-picker failure shows a readable reason, opens fallback, does not crash, and allows selection to continue; fallback is not the routine path on a normal Windows installation
- [ ] Open/play/close different-source results A, B, C and record process RSS, active provider count, registry entries, active worker count, and decoded cache bytes at each boundary
- [ ] Verify same-source simultaneous Viewers share a provider; closing one retains it, closing the final user removes the registry entry, stops the worker, and releases cache/resources
- [ ] Verify per-provider cache remains bounded and unused source caches do not accumulate; exact process RSS return to baseline is not required
- [ ] Repeat Error Vectors Off/Candidate transitions several times; no tiny unexplained window, stale vectors, or UI/rendered-state mismatch
- [ ] Exercise HUD Small → Medium → Large with sufficient and constrained viewport space; effective size may fit down and must restore the requested size
- [ ] Change Results/Detection popup values, focus, and popup visibility; HUD does not transiently shrink unless viewport bounds, requested size, or content changed
- [ ] Check Manual Focal in the Main Window selector/configuration row and Viewer result/configuration name/HUD (including `Manual Focal 1500 px [REFERENCE]`) at wide and narrow viewports; number and `px` remain associated and wrap is meaningful
- [ ] Open a historical `.cfh` with `Checkerboard Reference` in its saved display name; Viewer presents `Manual Focal` from model/metadata context while file metadata and filename remain unchanged
- [ ] Close the application with active source decoding/playback; no worker, Qt signal, or shutdown errors in console

### MainWindow first-user layout

- [ ] Preset row contains preset controls only; Add Configuration is no longer in that row
- [ ] `+ Add` appears below the configuration table, aligns with Name, and remains below the final row after repeated additions
- [ ] Column order is Name, Calibration, Focal, Reconstruction, Action
- [ ] Manual Focal enables Focal (px); every other calibration choice disables it or displays an unambiguous N/A state
- [ ] Sampling/Sample Count/Aggregate and Theme/Language occupy the requested hierarchy locations
- [ ] Output path, Browse, and Collision appear together in a compact row; path has primary stretch and remains inspectable
- [ ] Inspect default Original, one added configuration, multiple rows, Manual Focal row, a smaller desktop window, and 1280×800-class layout

### Single-result Viewer

- [ ] Playback footer has balanced horizontal breathing room
- [ ] Speed stepper shows current preset; lower/upper actions select adjacent speeds and respect endpoints
- [ ] Speed changes preserve playback position and existing timestamp-clock behavior
- [ ] Previous, Play/Pause, and Next use appropriate native icons; icon changes with playback state; tooltip and accessible name remain available
- [ ] Overlay/Split segmented control always has exactly one selected state
- [ ] Pin toggle visibly distinguishes ON/OFF and ON prevents auto-hide while OFF permits it
- [ ] Drag HUD to all four corners; it snaps to nearest anchor after release with smooth, non-blocking animation
- [ ] HUD drag does not orbit/pan the 3D view or trigger annotation hover; viewport interaction resumes after release
- [ ] HUD requested/effective Small/Medium/Large behavior and size recovery work at roomy and constrained viewport sizes
- [ ] Detection, Results where applicable, Annotations, reference identity wrapping, console stability, fullscreen, and existing auto-hide behavior regressions pass

### Multi-result Viewer and regressions

- [ ] Repeat the Viewer interaction checks with multiple same-source results
- [ ] Reference, Metrics, Error Vectors Off/Candidate repeated toggles, result visibility, and Detection retain correct state and rendering
- [ ] C-2B atomic source/2D/3D presentation, 2× synchronization, pause/seek, bounded cache, auto-hide, Pin, fullscreen, and menu hover/latch regressions pass
- [ ] C-2C unified single/multi routing, missing-source 3D, static image source, MANO mesh, and same-source validation regressions pass
- [ ] `.cfh` schema, presets, internal model IDs, focal provenance, confidence semantics, metrics, coordinates, and all scientific contracts remain unchanged
- [ ] Full test/import/compile/diff validation for implementation is recorded; no console traceback, Qt signal error, TypeError, or AttributeError remains in tested flows

## First-User UX & Stability Pass — implementation checks (2026-11-03)

### Automated

- [x] Full pytest: `236 passed` on the project Python 3.11 virtual environment
- [x] App import smoke: `import calibfreehands.app.main` printed `IMPORT_OK`
- [x] `compileall -q src tests` completed successfully
- [x] `git diff --check` completed successfully
- [x] Speed stepper endpoints/sequence and timestamp-clock continuity are covered by viewer/playback tests
- [x] HUD animation duration, all four nearest-corner destinations, final anchor position, and retained requested size are covered by offscreen Qt tests
- [x] Single/comparison mode controls, Pin icon state, annotations wording, and vectors state transitions are covered by offscreen Qt tests

### Windows GUI observations

- [x] User-reported manual validation at the starting checkpoint: Windows native media picker, Main Window first-user layout, provider A/B/C lifetime and RAM behavior, and console cleanliness.

- [x] Refreshed Main Window displayed the revised column order, `+ Add` below the row, processing/settings/output hierarchy, and path/Browse/Collision row. Added a second configuration, selected Manual Focal and saw its enabled focal field, then removed the temporary row.
- [x] Native Common Item Dialog opened normally with C: as its initial location; navigated to D:, selected one MP4, observed it enter the queue, cancelled and reopened the picker, and cancelled again. No `0x80004002` failure appeared in this exercised single-file path.
- [x] Opened a historical single `.cfh` through View Results without running inference. The Viewer presented `Manual Focal 1500 px`, source video, 2D overlay, 3D skeleton, Overlay/Split, media icons, and speed stepper.
- [x] Geometry visibility regression: single and multi-result viewers keep menu action checks, render flags, and GL item visibility synchronized through repeated mesh/joint/bone switching; automated tests pass.
- [x] Native Windows GUI opened three saved same-source `.cfh` results in Comparison Viewer; toggling Geometry > Mesh changed the actual 3D display from skeleton to MANO meshes.
- [x] In the single-result Viewer, changed speed from 1.0× to 1.25×, played and paused, and observed the source frame and 2D/3D pose advance together. Pin showed explicit On state and kept controls visible during playback.
- [x] Dragged the single-result HUD from its initial corner to bottom-right; the saved `.cfh` was read only.
- [x] Opened a two-result Viewer from two existing same-source `.cfh` bundles without inference. Reference and Manual Focal labeling, metric HUD, result colors, and rendered candidate error vectors were visible. This smoke instantiated the Viewer from saved bundles directly; it did not exercise the Main Window's multi-file result selection flow.

### Still pending for full Windows/user acceptance

- [ ] Native picker multi-file selection, folder/current-folder selection, repeated use across the full C:/D: matrix, and fallback/error-path behavior
- [ ] Main Window manual layout review at 1280×800, including several added rows and removal at that exact window size
- [ ] End-to-end multi-file selection from Main Window, plus full comparison Results/Reference/Metrics/Detection/Annotations/vector-repeat interaction matrix
- [ ] Manual HUD drag to each of the four corners, size/background/opacity menu review, and constrained-viewport recovery
- [ ] Source A/B/C provider registry, worker, cache-byte, and RSS measurements; application console review during a complete close/shutdown cycle
- [ ] Original feedback provider and second first-time user repeat the complete flow without coaching
- [x] The user later confirmed fresh AnyCalib/WiLoR inference; the Viewer checks in this section used existing saved results.


### Viewer Final Polish & HUD Stabilization (2026-11-04)

- [x] Offscreen regression tests cover the single Overlay/Split switch, theme-aware playback icons, coalesced HUD fitting across result/Detection/reference/metric changes, requested Large size retention, and same-joint annotation staggering.
- [x] Windows saved-result Viewer review: dark-theme frame/playback icons remained visible; Overlay/Split switch toggled 20 times; Large HUD, Results visibility, Detection Numeric/Bar, Key-joint and Worst-5 annotations rendered without the previously reported transient shrink.
- [ ] Reference and metric popup changes, Hovered/All annotation modes, light-theme native GUI, constrained viewport HUD recovery, and 1x/2x playback performance remain for user/manual follow-up.
- [x] Fresh AnyCalib/WiLoR inference was later user-confirmed.
- [ ] Original/second first-user rerun remains pending until the user selects a date.


## Final Viewer Micro-Polish (2026-11-04)

- [x] Single Overlay/Split switch retains OFF=Overlay and ON=Split; compact neutral track/thumb colors adapt to light/dark palette and selected mode label is emphasized.
- [x] HUD stable update invalidates card/body/nested row layouts and updates geometry before layout activation, font fitting, final card sizing, and repaint.
- [x] No-pointer Bar ON/OFF test verifies immediate HUD size reflow, no change after a later Leave event, and ten repeated toggle cycles without geometry drift.
- [x] Focused viewer regression tests pass for switch mode/theme presentation and Detection Bar reflow.
- [x] Windows dark-theme switch screenshot visually confirms a neutral track and high-contrast thumb.
- [x] Windows Qt runtime smoke with three saved `.cfh` results: Detection Bar ON measured 316 px HUD height; OFF immediately measured 268 px without pointer movement; a later Leave event retained 268 px. No inference ran.
- [ ] Light-theme switch visual review remains pending.
- [x] Fresh AnyCalib/WiLoR inference was later user-confirmed.
- [ ] First-user rerun remains pending until the user selects a date.


## Phase 1.1 GT Terminology and Pin Contrast (2026-11-05)

- [x] The user-facing metric tooltip says `Camera-space differences from the selected GT.` without a contradictory ground-truth disclaimer.
- [x] The metric menu presents the internal `Reference Accuracy` category as `GT Differences`; metric registry IDs/categories and export field names remain unchanged.
- [x] The export dialog title is `Export GT differences`; regression coverage verifies the dialog title and the existing reference/candidate result ID fields.
- [x] Pin stays 26-by-26 px with existing OFF presentation and `#33FF33` ON background/border; the active pin silhouette is `#111111` in both palettes. Offscreen sampling verifies at least 4.5:1 contrast.
- [x] Native Windows Viewer smoke with saved `.cfh` bundles visually confirmed GT-first HUD, `Differences` / `GT Differences` wording, Pin OFF, and the dark Pin icon against the green active background.
- [x] Targeted regression selection: 3 passed; full pytest: 250 passed. App import, compileall, `pip check`, and `git diff --check` passed.
- [ ] Phase 2 grid/camera implementation remains in progress at this checkpoint; fresh inference was later user-confirmed.


## Final Toolbar Simplification (2026-11-04)

- [x] Replace visible Overlay/Split labels with a stable `Layout` label; switch remains OFF=Overlay and ON=Split, with current mode in its tooltip and accessible description.
- [x] Move Pin beside the mode switch as a 28-by-28 icon-only control; palette-aware OFF/ON appearance, hover/pressed feedback, focus, tooltip, accessibility state, and Space toggling are covered.
- [x] Offscreen toolbar geometry confirms Layout, switch, and Pin are one compact group before Results; regression confirms Pin preserves the existing auto-hide behavior.
- [x] Focused viewer tests: 4 passed; full pytest: 243 passed.
- [ ] Native Windows GUI review of dark/light Pin states, toolbar composition, auto-hide, and keyboard behavior is pending; the desktop is currently at the Windows lock screen, so the running app cannot be visually inspected without unlocking the user's session.
- [x] Fresh AnyCalib/WiLoR inference was later user-confirmed.
- [ ] First-user rerun remains pending until the user selects a date.


## Phase 2 Fixed Metric Workspace and Shared Camera (2026-11-05)

- [x] Offscreen tests verify fixed workspace bounds, planes, axis ranges, label values/positions, Grid/Axis visibility, and geometry-independent default camera framing.
- [x] Offscreen tests verify Fit All uses finite currently visible geometry, leaves metric references and canonical keypoint/MANO arrays unchanged, and leaves the camera unchanged when geometry is empty.
- [x] Offscreen comparison tests verify shared camera across Overlay/Split and visible Split views, pose retention when switching modes, detached-view camera linking, and fixed reference geometry after fitting.
- [x] Windows native read-only smoke opened three saved same-source `.cfh` results; the fixed workspace remained in view while the 15000 px prediction extended outside it at default framing.
- [x] Viewer/comparison tests: 81 passed; full pytest: 255 passed; visualization imports, compileall, `pip check`, and `git diff --check` passed.
- [ ] Native menu interaction for Fit All/Reset and user first-run rerun remain pending; no new inference was run for this phase.


## View/Pin Final Micro-Polish (2026-11-04)

- [x] Rename the visible group label to `View`; the switch's accessible name is `View mode`, its description explains Overlay/Split views, and its tooltip reports `View: Overlay` or `View: Split`.
- [x] Pin is a 26-by-26 checkable custom vector control; OFF retains the gray surface and muted pin, while ON uses a subdued green surface and white pin.
- [x] Offscreen dark/light assertions cover OFF/ON background, border, and icon colors. Tests cover button click, keyboard Space, accessible name, toolbar order/gap, auto-hide pin behavior, and global Space playback.
- [ ] Native Windows dark/light visual and interaction review remains pending because the desktop is at the Windows lock screen; inspecting the app requires unlocking the user's session.
- [x] Fresh AnyCalib/WiLoR inference was later user-confirmed.
- [ ] Original/second first-user rerun remains pending until the user selects a date.


## Pin Size and Active Color Refinement (2026-11-04)

- [x] Reduce the custom Pin control from 28-by-28 to 26-by-26 px and scale the vector icon down proportionally.
- [x] Offscreen dark/light rendering confirms the gray OFF state, green `#5B7925` ON background, white ON icon with at least 4.5:1 contrast, and visible button border.
- [x] Full pytest: 244 passed; app import, compileall, and `git diff --check` passed.
- [ ] Native Windows visual review has not been repeated for this color/size adjustment.


## Pin Active Color Contrast Refinement (2026-11-04)

- [x] Replace the overly bright lime with muted green `#5B7925`; retain the white pin and darker green border.
- [x] Active icon/background contrast assertion verifies at least 4.5:1 in offscreen dark and light palette cases.
- [ ] Native Windows visual review of the revised active color remains pending.


## Phase 1 GT and Confidence Viewer UX (2026-11-05)

- [x] User-facing comparison baseline is labeled `GT`; the internal reference identifiers, saved result metadata, and metric/export schema remain unchanged.
- [x] A unique checkerboard reference remains the default GT; otherwise the existing saved-id and fallback selection behavior is preserved.
- [x] GT uses `#33FF33`; the HUD shows GT first without reordering Split panels. Switching GT refreshes result colors, metrics, annotations, and HUD presentation.
- [x] Candidate base colors remain separate from the effective GT display color; custom candidate colors restore after GT changes. Joints and meshes use result identity color; bones keep the hue at 80% intensity.
- [x] Confidence color mode is removed. Confidence opacity, numeric value, and bar default off; confidence affects alpha only when requested. Its tooltip identifies the stored YOLO hand detector bounding-box confidence.
- [x] Comparison controls use `Visible (n/n)` and `Differences`; protocol-pending and currently unavailable metrics are omitted and availability is rebuilt when GT changes. Error vectors remain implemented but are absent from the normal toolbar.
- [x] Pin remains 26-by-26 px, gray while off, and uses `#33FF33` with a white icon while on. Offscreen sampling shows the white-on-green contrast is about 1.36:1; the requested icon color was retained, and native screenshot review also showed the white pin is difficult to distinguish at this brightness.
- [x] Offscreen Qt regression coverage includes A→B→C GT changes, candidate color restoration, annotation and metric refresh, joint/bone/mesh identity colors, confidence alpha, metric availability pruning, toolbar labels, hidden vectors, and existing Pin behavior.
- [x] Native Windows smoke with three existing same-source `2_screw` `.cfh` files (Manual Focal 1655, 2500, and 15000 px) verified GT HUD ordering and selection, old-color restoration, refreshed comparison values, Confidence opacity/numeric controls, and Overlay/Split views. No inference ran and the saved files were opened read-only.
- [x] Focused comparison viewer tests: 59 passed; full pytest: 249 passed; application import, compileall, `pip check`, and `git diff --check` passed.
- [x] Fresh AnyCalib/WiLoR inference was later user-confirmed.
- [ ] First-user rerun remains pending until the user selects a date.


## Final Viewer Control Styling (2026-11-04)

- [x] Custom checkable QAbstractButton mode switch is 42-by-22 device-independent pixels, uses a neutral rounded track and circular near-white thumb, animates over 120 ms, and retains strong keyboard focus/accessibility.
- [x] Overlay/Split labels share the same palette foreground; selected mode uses font weight only. Light/dark palette rendering and thumb travel are covered by offscreen image assertions.
- [x] Speed label, decrement, fixed-width value, and increment are grouped with fixed sizing and compact spacing; playback group is separate, and the stretch follows both groups.
- [x] Offscreen geometry and playback tests pass at 1280-by-720; speed value width accommodates the widest preset without moving the arrows.
- [x] Windows saved-result comparison Viewer inspected at 1280-by-720: switch toggled Overlay/Split; the speed controls displayed 0.75x and 1.25x in place with no clipping or large gap.
- [ ] Light-theme native Windows visual review remains pending; palette rendering is covered offscreen.
- [x] Fresh AnyCalib/WiLoR inference was later user-confirmed.
- [ ] First-user rerun remains pending until the user selects a date.


## Phase 2.1 Whole-Video GT Hand Framing (2026-11-06)

- [x] Whole-video framing uses the selected GT's detected finite wrist median; finite detected joints provide the documented fallback and robust 2.5th-97.5th percentile extent, with a minimum camera radius.
- [x] Historical Phase 2.1 validation: GT changes cached framing without moving the camera. This camera policy is superseded by Phase D.1 Stabilization below; frame changes and playback still do not recalculate or recenter the camera.
- [x] Single-result and comparison viewers provide Hand View, fixed Metric Overview, and Fit All; Overlay and Split share the selected GT framing. Front/Side/Top preserve current camera center and distance.
- [x] Offscreen read-only smoke opened the existing three same-source `.cfh` results (Manual Focal 1655, 2500, and 15000 px), verified selected-GT Hand View and frame navigation stability, and restored Metric Overview/Hand View. No inference ran and no result files were modified.
- [x] Viewer/comparison tests: 86 passed; full pytest: 260 passed; application import, compileall, `pip check`, and `git diff --check` passed.
- [ ] Native Windows visual/menu interaction review remains pending because the Computer Use native pipe was unavailable; first-user inference rerun remains pending until the user selects a date.


## Phase A Viewer Menu and HUD Cleanup (2026-11-07)

- [x] Toolbar labels are `Error Types`, `Confidence Display`, `Annotations`, `HUD`, and `More`; metric IDs, availability, formulas, confidence options/source, and metric categories are unchanged.
- [x] Annotation menu order is All joints, Hovered joint, Key joints, Worst 5, Off, separator, Label; the default remains Off. HUD menu order is HUD opacity, Size, Background.
- [x] HUD is a checkable split button: its body performs actual panel hide/show and the arrow opens settings. Visibility stays synchronized across Overlay/Split; anchor, size, opacity, and background survive toggling.
- [x] Removed the Auto-hide controls while playing setting; Pin is the only policy switch. Playback with Pin OFF can auto-hide after the existing delay, Pin ON restores/keeps controls visible, and stopping playback shows controls.
- [x] More menu action is Playback Stats; existing source FPS, frame-update FPS, speed, skipped-frame display and toggle behavior remain.
- [x] Focused comparison viewer tests: 68 passed; full pytest: 263 passed; app import, compileall, `pip check`, and `git diff --check` passed.
- [ ] Native Windows visual smoke was unavailable: Computer Use reported that its native pipe could not connect.


## Phase A.1 HUD Menu Row Height Polish (2026-11-07)

- [x] Size the custom opacity QWidgetAction from the displayed `Background` submenu row geometry after QMenu polish/adjustSize; this uses the active Qt style's actual row height rather than a fixed pixel guess.
- [x] Remove vertical layout margins and vertically center the opacity label and slider. Preserve the existing 8 px left padding and 100 px slider width.
- [x] Offscreen layout tests compare opacity/Size/Background row heights (1 px tolerance), slider and label centers (1 px), and menu width (280 px maximum) under dark/light palettes for Fusion, Windows, Windows Vista, and Windows 11 styles. All passed.
- [x] HUD slider remains 20-100, defaults to 100, and updates opacity while hidden; presentation settings persist across HUD visibility and Overlay/Split changes.
- [x] Focused comparison viewer tests: 72 passed; full pytest: 267 passed; app import, compileall, `pip check`, and `git diff --check` passed.
- [ ] Native Windows visual smoke remains unavailable: Computer Use native pipe connection failed twice, including after the documented 2-second retry.

## Phase C Physical / Hand 3D Space (2026-11-07)

- [x] Offscreen tests cover fixed Physical bounds, positive display-Y orientation and labels, unchanged raw arrays, robust whole-video Hand bounds, missing-geometry fallback, adaptive spacing, and `.cfh` single-result/comparison synchronization.
- [x] Offscreen comparison tests cover independent Labels visibility, frame and camera preservation during mode/GT changes, detached workspace propagation, and distinct Focus GT / Space Overview / Fit All behavior.
- [x] Existing same-source saved results remain available read-only under `D:\hand-demo\samples`, including Manual Focal 1655/2500/15000 px and Original results.
- [ ] Native Windows visual acceptance and manual playback check remain pending; offscreen tests do not establish visual acceptance.

## Phase C.1A Workspace Geometry and Controls Stabilization (2026-11-07)

- [x] Offscreen viewer tests validate the shared centered cube, equal plane/axis lengths, metric tick placement, zero formatting, robust Hand bounds, Physical fallback, and stale-label cleanup across repeated mode changes.
- [x] Comparison tests validate the accessible Labels switch, palette-aware compact primary control strip, toolbar order, and unchanged frame/camera behavior; detached views inherit the Labels state.
- [x] Viewer/comparison test modules: 100 passed. Full pytest suite: 281 passed.
- [ ] Native Windows visual acceptance and manual playback review remain pending; no native visual claim is made from offscreen testing.

## Phase C.1A Fix Pass: Workspace Placement and Space Center (2026-11-07)

- [x] Update regression coverage to distinguish the Physical volume center (0, 0, 1 m) from the coordinate origin (0, 0, 0); verify Physical XY/XZ/YZ planes at their global coordinate planes and Physical axes through the origin.
- [x] Verify Hand planes/axes remain centered on the selected-GT cube, while Physical↔Hand updates only the camera center and preserves distance, azimuth, elevation, current frame, and active playback timer across Overlay/Split.
- [x] Verify the primary control container does not auto-fill an opaque background; switch and toolbar behavior remains covered.
- [x] Focused viewer/comparison modules: 100 passed. Full pytest: 281 passed; app import, compileall, `pip check`, and `git diff --check` passed.
- [ ] Native Windows visual review and playback check remain pending; previous native review identified incorrect Physical plane placement and is not counted as acceptance.

## Phase C Final Camera, Contrast, and GT Switching (2026-11-07)

- [x] Offscreen regression tests verify distinct Physical/Hand camera poses and restoration in Overlay/Split, deterministic orientation angles, and Fit All recovery through Focus GT and Space Overview.
- [x] Offscreen tests verify light foreground styles for viewport chrome, ordered directed metric-pair reuse, one immediate coherent GT panel refresh, asynchronous large-workload calculation, and stale-result rejection.
- [x] Playback/frame preservation and canonical result immutability remain covered by the existing Viewer suite.
- [x] Viewer/comparison tests: 108 passed; full pytest: 289 passed. App import, compileall, `pip check`, and `git diff --check` pass after the final source changes.
- [x] Read-only `2_screw` set (4 `.cfh` results, 364 frames): previous uncached GT change took 0.880 s total (0.831 s metric comparison); repeated uncached reference switch took 0.871 s. Updated measurement: uncached selection handler returned in 0.031 s, background metric work took 0.902 s and completed in 0.942 s total; switching to the cached GT took 0.053 s. Initial Viewer load was 1.374 s.
- [ ] Native Windows visual review of camera presets/space restoration, Light/Dark chrome, GT changes during playback, and any visible playback skip remains pending; Computer Use native pipe was unavailable, so no manual acceptance is claimed.

## Phase D.1 Unified Panel Ordering Foundation

- [x] Baseline before changes: Viewer/comparison 108 passed; full suite 289 passed.
- [x] Regression tests cover stable panel IDs, initial GT priority and custom-order precedence, HUD insertion by mouse using HUD row geometry, Split panel drag, Source Media first/middle/last, Overlay swap, and frame/playback/camera/provider/result-array preservation.
- [x] Focused Viewer/comparison suite: 120 passed; comparison viewer suite: 96 passed.
- [x] Full pytest suite: 301 passed; application import prints `IMPORT_OK`; `compileall` and `pip check` pass.
- [x] `git diff --check` passes after the final documentation and test updates.
- [ ] Native Windows D.1 interaction review remains pending; no native acceptance is inferred from offscreen tests.
- [ ] D.2 individual Detach/Reattach and D.3 detached-state synchronization/Source Media aspect-ratio work remain deferred.

Phase C Final native Windows testing and acceptance were reported by the user as completed. This is user-reported acceptance, not an assertion that Codex independently performed the native checks.

### Phase D.1 Fix Pass: Panel Interaction and Adaptive Layout

- [x] Added offscreen regression coverage for header-free surfaces, Overlay swapping and state/slot-size preservation, Split Shift drags from source and GL content, ordinary unmodified drag, target feedback, Escape/outside cancellation, HUD insertion marker cleanup, visibility-driven 5-to-1 panel reflow, restore, and resize.
- [x] Focused comparison Viewer suite: 106 passed; full pytest suite: 311 passed.
- [x] Application import prints `IMPORT_OK`; `compileall` and `pip check` pass; `git diff --check` passes.
- [ ] Native Windows interaction/visual review remains pending; offscreen coverage does not count as user acceptance.

### Phase D.1 Rendering Regression Fix & UI Stabilization (2026-11-07)

- [x] Reproduced the native Windows failure with the four existing `2_screw` `.cfh` results: GL draw errors occurred during Split viewer construction at `_sync_split_layout`, when a parentless GL surface was shown before grid insertion.
- [x] Confirmed lifecycle cause: showing the parentless surface exposed its `QOpenGLWidget` in a temporary top-level context; subsequent reparenting into the Viewer changed context/FBO ownership. The process-global pyqtgraph shader cache was not reset by the Fix Pass and was not independently identified as the trigger; it makes context churn unsafe, so the correction stabilizes widget ownership/lifetime without shader monkey-patching.
- [x] Native real-context regression test verifies framebuffers contain rendered pixels, GL errors are absent, and context identities survive visibility/order/resize/mode/swap changes. `QT_QPA_PLATFORM=windows`: 1 passed.
- [x] Native direct Qt run with the saved four-result `.cfh` set exercised single-result and comparison views, Overlay/Split, Shift-drag, visibility/reflow, repeated swaps, resize, Physical/Hand, GT, seek/play/pause, camera preset, mesh, and Detach All. Captured 16 valid GL framebuffers; 0 GL errors and 0 pyqtgraph draw warnings.
- [x] Detach All regression checked: detached GL views receive renderable geometry and produce valid framebuffers.
- [x] Overlay divider is 4 logical px; palette-aware painted double-arrow overlays it at 28×28 px. Repeated swaps preserve slot widths and interaction state; ordinary splitter resizing remains enabled.
- [x] Focused comparison viewer suite: 108 passed, 1 skipped (native OpenGL case skips only under offscreen; separately run and passed on Windows platform).
- [x] Full pytest suite: 313 passed, 1 skipped. App import, compileall, `pip check`, and `git diff --check` passed.
- [ ] Computer Use native pipe was unavailable. Native Qt/window/framebuffer verification was performed directly, but no human visual acceptance is claimed. Phase D.1 remains unapproved until user confirmation.
- [ ] D.2 individual Detach/Reattach, D.3 detached synchronization/aspect ratio, Phase E HUD polish/animations, Phase F/G, and release packaging remain deferred.

### Phase D.1 Final Fix: Adaptive Grid & Overlay Divider UX (2026-11-07)

- [x] Starting baseline: branch `CalibFreeHands`, HEAD/origin `090453291c4822e341ad9b7110c0635e32b4ebd2`, clean tree; full suite 313 passed, 1 skipped.
- [x] Grid regression tests assert actual `QGridLayout` row/column placements at 1280×720: 5→3×2, 4→2×2, 3→2×2, 2→2×1, 1→1×1. Five visible panels have no horizontal/vertical scrollbar and retain 340×260 minimum panel dimensions.
- [x] Twelve-result grid tests cover narrow portrait and wider landscape resizing, row-major placement, scrollable minimum-size behavior, hidden result reorder/restore, and persistent panel widget identities.
- [x] Splitter interaction tests exercise left-click no-op, actual left-drag resize, right-click `Swap` menu, dismiss-without-selection, repeated menu swaps, stable panel identity/width/state, 8 px handle, horizontal-resize cursor, and palette-dependent three-dot grip rendering.
- [x] Native Windows (`QT_QPA_PLATFORM=windows`) OpenGL regression test: 1 passed; native direct run with four saved `2_screw` `.cfh` bundles verified Overlay and Split, five-panel 3×2 placement, right-click swap, visibility reflow, resize, playback/camera, and existing Detach All. GL framebuffer checks passed; GL errors=0, pyqtgraph draw warnings=0.
- [x] Focused comparison Viewer suite: 109 passed, 1 skipped. Full pytest suite: 314 passed, 1 skipped. Application import, compileall, and `pip check` passed.
- [ ] User has confirmed the preceding rendering recovery. This final layout/divider pass still awaits user review; no final Phase D.1 approval is claimed.
- [ ] Phase D.2 independent Source/3D windows and individual Detach/Reattach remain deferred. D.3 detached-state synchronization and Source Media aspect ratio, Phase E HUD/animation polish, Phase F/G, and release packaging remain deferred.

### Phase D.1 Final UI Fix: Overlay Splitter Grip (2026-11-09)

- [x] Reproduced weak grip visibility from the actual 8 px handle: 1 px radius circles used low-contrast `Mid` color, while base native handle painting was also underneath.
- [x] Pixel-level grip test confirms three distinct high-contrast dot runs, equal logical spacing, clear gaps, Light/Dark palettes, and 100%, 125%, 150%, and 200% device-pixel ratios.
- [x] Native Windows Qt tests exercise the actual handle grab, right-click Swap, left-drag resize, repeated swaps, preserved widths/state, and OpenGL rendering: 3 passed.
- [x] Focused Viewer suite: 110 passed, 1 skipped. Full pytest: 315 passed, 1 skipped. App import, compileall, and `pip check` passed.
- [ ] Final user confirmation that the dots are visually clear remains pending; no Phase D.1 approval is inferred.

### Phase D.1 Stabilization: GT Camera Focus and Playback During Resize (2026-11-09)

- [x] Replaced the superseded Phase C/2.1 “GT change leaves camera in place” behavior. Regression tests verify new-GT whole-video target/distance in Physical and Hand, preserved per-Space azimuth/elevation, repeated A→B→A changes without drift, invalid-geometry finite fallback, and no pending rebase state.
- [x] Tests cover camera synchronization across Overlay, Split, and existing detached views; during playback a GT change preserves frame, active timer, color semantics, and user panel order. Physical workspace bounds remain fixed; Hand bounds follow the selected GT.
- [x] Deterministic resize/playback test verifies the requested, presented, Source Media, and Overlay 3D frame IDs stay equal through repeated splitter resizes.
- [x] Source Media paint no longer rebuilds an owned `QImage` for an unchanged cached source frame. Regression test verifies the QImage cache key remains stable across repaints.
- [x] Native Windows Qt harness used the four saved `2_screw` `.cfh` results (364 frames; no inference). A resident 61-frame decoded window allowed equivalent 2-second, 1x playback runs. With the previous per-paint copy behavior vs image reuse, resize-run Source Media paint average/max was 2.614/3.216 ms vs 1.600/1.821 ms; source paint count was 77 vs 81, GL paint count 117 vs 122, and Source/3D both ended at frame 59. No playback frame was skipped in either resize run.
- [x] In the same resize harness, 43 vs 44 GL resize events were measured; 10 ms event timer intervals averaged 22.90 ms (33.68 ms max) vs 21.92 ms (29.66 ms max). Playback remained frame-synchronized; this shows the remaining layout/GL repaint cost and does not establish that every visible hitch is eliminated. Normal playback intervals averaged 11.45 ms (22.67 ms max) with the previous paint path and 11.43 ms (26.44 ms max) with image reuse.
- [x] Native GL paint profiling during a separate 2-second cached-window run measured 60 paints at 3.710 ms average / 5.450 ms max without resizing, versus 110 paints at 4.071 ms / 7.147 ms while resizing; the resize run skipped 2 clock frames while its displayed Source/Overlay frames remained aligned. This confirms extra GL redraw work is still present.
- [x] A native QtTest handle-drag run delivered 120 splitter-move events while playback was active; Source Media and Overlay displayed the same frame throughout sampling. The run lasted about 3.9 s and advanced beyond the 61-frame preloaded decoder window, so subsequent decode contention and QtTest event injection make its skipped-frame counts unsuitable as a before/after performance comparison.
- [ ] Native automated evidence includes both repeated splitter size changes and QtTest handle-drag events; a user's manual continuous-playback review and confirmation of perceived smoothness remain pending. No user acceptance is claimed.

### Phase D.1 Camera Center Fix (2026-11-09)

- [x] Synthetic GT result deliberately separates the robust wrist framing target from the final Hand workspace's rendered axis intersection; tests verify Physical uses the former and Hand uses the latter.
- [x] Tests derive Hand camera distance from the final display-space workspace corners, GL view FOV, and viewport aspect, and verify the three rendered axis segments meet at the camera center.
- [x] Tests verify GT changes update both independent Space policies, preserve each orientation, restore custom poses when GT is unchanged, retain finite fallback behavior, and synchronize Overlay/Split/detached and single-result views.
- [x] No media, playback, decoder, splitter, pipeline, canonical result, or inference code was changed in this pass.
- [x] Native Windows OpenGL smoke test `test_split_reflow_keeps_native_gl_contexts_and_paints_frames` passed. A native Qt harness loaded all four saved `2_screw__*.cfh` results (15000, 1655, 2500, Original), without inference; in each, Physical and Hand centers matched their separate policies, distances were finite/positive, independent orientations survived switching, and Overlay/Split Hand axes met at the camera center. User visual confirmation remains pending.
- [ ] Playback stutter during Overlay splitter resize remains outstanding for its separate task.

### Phase D.1 Playback Resize and HUD Insertion Marker Fix (2026-11-09)

- [x] Playback drag updates are coalesced to 30 Hz only while the playback timer is active; paused drag remains unthrottled and release applies the final pointer location. The native Qt regression verifies timer continuity, synchronized source/3D frames, unchanged camera/GT and no decoder misses in the warm-cache case.
- [x] HUD marker/drop geometry now derives from the rendered result-group containers rather than header positions. Offscreen coverage includes variable-height groups, different HUD dimensions, first/between/last slots, hidden rows, resize during drag, refresh, cancellation, and order preservation.
- [x] Matched native warm-cache 1.6 s drag benchmark: control vs coalesced had event-loop 5 ms probe avg/p95/max 15.45/21.57/24.24 vs 11.46/18.19/22.34 ms; playback timer avg/p95/max 20.09/31.01/33.96 vs 17.32/25.67/30.80 ms; GL paints 131 vs 85; GL resizes 82 vs 31; skipped frames 0 vs 0. GL draw time per paint remained about 3.5 ms, supporting reduced redundant resize/redraw work as the warm-cache benefit.
- [x] Native Windows OpenGL smoke coverage was run for the resize and camera paths; the native handle-drag test passes and no inference or `.cfh` reload occurs.
- [x] Cold-cache investigation: the earlier apparent frame-0 stall was not reproduced using the normal native Qt event loop. The earlier freeze observation used repeated Python-side `QTest.qWait()`/`processEvents()` polling; this starved the worker in that harness and is not evidence of a product deadlock. This supersedes the initial unresolved observation above.
- [x] Deterministic fresh-provider regression blocks frame 0 with a synchronization event, changes the target to frame 2 while decoding is pending, then releases frame 0. The stale completion is ignored; frame 2 eventually presents in Source Media and Overlay 3D; the provider decodes exactly the two requested frames. Focused test passed.
- [x] Native Windows Qt event-loop runs with the actual saved 2_screw `.cfh` files (no inference): 3 fresh Overlay 1× immediate-play repeats and 1 wait-for-frame-zero-then-play run advanced; first advancing presentation was 0.25–0.50 s after Play. Viewer open-to-Play was 1.20–1.73 s with frame zero already ready at visible Viewer; this is not an isolated decoder latency. Fresh Split 1.5× advanced and synchronized. Newly created providers had empty in-memory caches; OS filesystem cache remained untouched.
- [x] Native paused seek presented frame 120 in Source and 3D within 0.75 s. Pause/resume during a pending seek recovered from the last presented frame and advanced. Playback mode and GT changes retained Source/3D synchronization. Closing with active decode stopped the provider worker and terminated the captured FFmpeg processes. Standalone FFmpeg decoded 10 frames from the same source in 0.22 s.
- [ ] User visual/playback review remains required; no user acceptance or full Phase D.1 approval is claimed. D.2/D.3/E/F/G and release packaging remain deferred.
- [ ] User visual acceptance of HUD insertion positions and continuous playback remains pending. D.2/D.3/E/F/G and release packaging remain deferred.

### Phase D.1 HUD Layout Stability Fix (2026-11-09)

- [x] Starting state: `CalibFreeHands`, HEAD/origin `7c565feae25afb3287f0f566ce4eb1b33dc29241`, clean tree; baseline full pytest `329 passed, 1 skipped`.
- [x] Pre-fix mouse-driven trace reproduced an order-only refresh replacing live groups with default-size widgets (new group geometry 100×30 and GT badge geometry 640×480 immediately after release), followed by delayed layout/fitting geometry changes. `_refresh_ordered_presentations` also initiated an unnecessary HUD `_position`/fit through annotation refresh. This confirmed the rebuild/refit sequence as the layout instability source.
- [x] Fix reorders the existing Overlay group widgets without recreating their labels, metrics, confidence bars, drag handles, or GT badge; it preserves the prior fit signature and card anchor/size. Actual GT/content/settings changes still rebuild and fit normally. Split panels, menus, annotation refresh, and stacking remain synchronized.
- [x] New mouse-driven regression captures widget paint and geometry state before/during/release/after event-loop and idle with no pointer movement, checks widget identity, font sizes, label wrapping/size hints, group and badge sizes, and confirms a later mouse move is not required. Four cases cover GT first/middle/last, all HUD sizes, all four anchors, confidence numeric/bars, metric rows, and playing/paused states.
- [x] Focused HUD reorder/marker/settings/cancellation tests: 13 passed. Native Windows Qt (`QT_QPA_PLATFORM=windows`) HUD paint/layout and GL context smoke tests: 6 passed. Full pytest: 334 passed, 1 skipped. App import printed `IMPORT_OK`; `compileall` succeeded; `pip check` reported no broken requirements.
- [x] Native Windows Qt run opened the four saved `2_screw` `.cfh` results (Manual Focal 1655/2500/15000 px and Original), without inference. During an actual mouse reorder, card geometry, badge geometry, and fonts were unchanged through release and idle; group height changed by at most 1 px from Qt layout rounding. Existing group widget identities were retained.
- [ ] User visual confirmation that no enlargement appears during actual Windows interaction remains pending. Cold-cache playback investigation is complete, while overall playback visual acceptance remains pending; D.2/D.3/E/F/G and Release remain deferred.


### Phase D.2 Independent Detach, Reattach & Composable Multi-Window Workspace (2026-11-09)

- [x] Starting tree was on `CalibFreeHands`, at the expected `78010a461d6cdfc9238589b02fb81f10ad7cb009` baseline with a clean worktree; existing baseline was 335 passed, 1 skipped.
- [x] Workspace model and Viewer regression suite: `tests/test_workspace_model.py` plus `tests/test_comparison_viewer.py`: 146 passed, 1 skipped.
- [x] Full suite after implementation: 351 passed, 1 skipped. Import check printed `IMPORT_OK`; `compileall` succeeded; `pip check` reported no broken requirements; `git diff --check` passed.
- [x] Native Windows Qt OpenGL test `test_split_reflow_keeps_native_gl_contexts_and_paints_frames`: passed. It verified shared contexts and framebuffer drawing through Detach All, result overlay, Source+3D composition, extraction, close-to-reattach, and Reattach All, including mesh/joint/bone/grid/axis rendering without GL errors or draw warnings.
- [x] Native harness opened the four existing read-only `D:\hand-demo\samples_screw__*.cfh` bundles (Manual Focal 1655/2500/15000 px and Original), without inference. Detach All produced five independent windows. Result windows retained distinct unlinked camera poses; merging results into a Source-only target retained the result group's pose; Source and 3D composition stayed on frame 4 after asynchronous decode; Source/3D frame IDs matched. Extraction and Reattach All preserved group camera poses; enabling Link Cameras synchronized active viewports to the selected GT. Qt context sharing was valid.
- [x] The full suite initially had one transient Windows `PermissionError` replacing a temp session manifest; the isolated test passed on retry and the subsequent full suite passed.
- [ ] User must still manually review native drag-out/drop feedback, independent window movement/resizing, Source controls, camera linking, X reattach, and playback interaction. No user acceptance is inferred. D.3, E, F/G, and release packaging remain deferred.
