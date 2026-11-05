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
- [ ] Fresh AnyCalib/WiLoR inference smoke; existing saved results were used for Viewer inspection


### Viewer Final Polish & HUD Stabilization (2026-11-04)

- [x] Offscreen regression tests cover the single Overlay/Split switch, theme-aware playback icons, coalesced HUD fitting across result/Detection/reference/metric changes, requested Large size retention, and same-joint annotation staggering.
- [x] Windows saved-result Viewer review: dark-theme frame/playback icons remained visible; Overlay/Split switch toggled 20 times; Large HUD, Results visibility, Detection Numeric/Bar, Key-joint and Worst-5 annotations rendered without the previously reported transient shrink.
- [ ] Reference and metric popup changes, Hovered/All annotation modes, light-theme native GUI, constrained viewport HUD recovery, and 1x/2x playback performance remain for user/manual follow-up.
- [ ] Fresh AnyCalib/WiLoR inference and original/second first-user rerun remain pending.


## Final Viewer Micro-Polish (2026-11-04)

- [x] Single Overlay/Split switch retains OFF=Overlay and ON=Split; compact neutral track/thumb colors adapt to light/dark palette and selected mode label is emphasized.
- [x] HUD stable update invalidates card/body/nested row layouts and updates geometry before layout activation, font fitting, final card sizing, and repaint.
- [x] No-pointer Bar ON/OFF test verifies immediate HUD size reflow, no change after a later Leave event, and ten repeated toggle cycles without geometry drift.
- [x] Focused viewer regression tests pass for switch mode/theme presentation and Detection Bar reflow.
- [x] Windows dark-theme switch screenshot visually confirms a neutral track and high-contrast thumb.
- [x] Windows Qt runtime smoke with three saved `.cfh` results: Detection Bar ON measured 316 px HUD height; OFF immediately measured 268 px without pointer movement; a later Leave event retained 268 px. No inference ran.
- [ ] Light-theme switch visual review remains pending.
- [ ] Fresh AnyCalib/WiLoR inference and first-user reruns remain pending.


## Final Toolbar Simplification (2026-11-04)

- [x] Replace visible Overlay/Split labels with a stable `Layout` label; switch remains OFF=Overlay and ON=Split, with current mode in its tooltip and accessible description.
- [x] Move Pin beside the mode switch as a 28-by-28 icon-only control; palette-aware OFF/ON appearance, hover/pressed feedback, focus, tooltip, accessibility state, and Space toggling are covered.
- [x] Offscreen toolbar geometry confirms Layout, switch, and Pin are one compact group before Results; regression confirms Pin preserves the existing auto-hide behavior.
- [x] Focused viewer tests: 4 passed; full pytest: 243 passed.
- [ ] Native Windows GUI review of dark/light Pin states, toolbar composition, auto-hide, and keyboard behavior is pending; the desktop is currently at the Windows lock screen, so the running app cannot be visually inspected without unlocking the user's session.
- [ ] Fresh AnyCalib/WiLoR inference and first-user reruns remain pending.


## View/Pin Final Micro-Polish (2026-11-04)

- [x] Rename the visible group label to `View`; the switch's accessible name is `View mode`, its description explains Overlay/Split views, and its tooltip reports `View: Overlay` or `View: Split`.
- [x] Pin is a 26-by-26 checkable custom vector control; OFF retains the gray surface and muted pin, while ON uses a subdued green surface and white pin.
- [x] Offscreen dark/light assertions cover OFF/ON background, border, and icon colors. Tests cover button click, keyboard Space, accessible name, toolbar order/gap, auto-hide pin behavior, and global Space playback.
- [ ] Native Windows dark/light visual and interaction review remains pending because the desktop is at the Windows lock screen; inspecting the app requires unlocking the user's session.
- [ ] Fresh AnyCalib/WiLoR inference and original/second first-user reruns remain pending.


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
- [ ] Fresh AnyCalib/WiLoR inference and first-user reruns remain pending.


## Final Viewer Control Styling (2026-11-04)

- [x] Custom checkable QAbstractButton mode switch is 42-by-22 device-independent pixels, uses a neutral rounded track and circular near-white thumb, animates over 120 ms, and retains strong keyboard focus/accessibility.
- [x] Overlay/Split labels share the same palette foreground; selected mode uses font weight only. Light/dark palette rendering and thumb travel are covered by offscreen image assertions.
- [x] Speed label, decrement, fixed-width value, and increment are grouped with fixed sizing and compact spacing; playback group is separate, and the stretch follows both groups.
- [x] Offscreen geometry and playback tests pass at 1280-by-720; speed value width accommodates the widest preset without moving the arrows.
- [x] Windows saved-result comparison Viewer inspected at 1280-by-720: switch toggled Overlay/Split; the speed controls displayed 0.75x and 1.25x in place with no clipping or large gap.
- [ ] Light-theme native Windows visual review remains pending; palette rendering is covered offscreen.
- [ ] Fresh AnyCalib/WiLoR inference and first-user reruns remain pending.
