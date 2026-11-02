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
- [ ] Interactive viewer/manual validation
- [ ] C-2B normal Windows visual review: Overlay + 2 results, Split + 2, Split + 5
- [ ] P1-B new original WiLoR run confirming MANO parameter persistence in a fresh real result
- [x] Actual AnyCalib + WiLoR/MANO smoke on first 3 frames of existing `4_assemble.mp4`; single and two-configuration outputs validated without network downloads
- [ ] `git diff` final review

Known remaining validation gap: interactive Windows GUI and visual viewer controls
still require a human manual check; automated offscreen construction and saved
result reopening pass.

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
- [ ] HUMAN VISUAL REVIEW REQUIRED in the normal Windows GUI for speed selection, continuous playback, pause, and mode switching

## C-2B Presentation UX Stabilization
- [x] Full pytest suite: 201 passed; includes fixed viewport geometry, HUD interactions, hover menus, annotation modes, and reorder lifecycle
- [x] Windows GUI visual check at 1280x800 with six synthetic test-only result panels; drag reorder changed Method 2/Method 1 ordering
- [ ] Windows GUI visual review for hovered-joint readability and Split + 3 layout

## C-2B Media Memory Optimization
- [x] Provider tests: lazy sequential decode, repeated request coalescing, cache hit/eviction/budget, reverse seek, latest request replacement, image buffer ownership, missing/changed source, and source sharing
- [x] Full Windows pytest suite: 207 passed; `import calibfreehands.app.main`, compileall, and diff check pass
- [x] Windows 4_assemble.mp4 + three same-source CFH results: provider shared by all rows; 863 decoded-frame insertions including rereads after eviction; cache 379.7 MiB under 384 MiB cap
- [x] Measured process RSS: 215.1 MiB after open, 601.4 MiB after full source decode; previous full-frame preload observation was about 3.7 GiB
- [x] Seek latency at frames 0/60/150/300/501: 78/332/502/829/1120 ms
- [x] 4-second playback probes: 1x 29.99 source fps at 22.3% process CPU; 2x 59.72 source fps at 29.7% process CPU

## C-2B Final Validation
- [x] Windows GUI opened three actual same-source CFH results without inference; Overlay and Split + 3 results inspected at 1280x800
- [x] Real source video, synchronized result poses, and current-frame metrics updated during playback; controls auto-hid while playing and returned on mouse activity
- [x] Windows UI speed selector accepted 0.1x and 2x; separate 4-second probes verified 1x/2x playback rates
- [x] Full final pytest suite: 207 passed; application import, compileall, and diff check passed
- [ ] HUMAN FINAL REVIEW REQUIRED: hovered-joint annotation readability, all annotation modes, and final approval. Automated tests and agent visual inspection do not count as user approval.
