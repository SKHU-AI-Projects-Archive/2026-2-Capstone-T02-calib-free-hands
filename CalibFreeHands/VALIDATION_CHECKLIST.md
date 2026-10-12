# Validation checklist

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
- [x] Official WiLoR checkpoint and detector downloaded outside protected source tree
- [x] Official model asset setup script with network access
- [ ] Interactive viewer/manual validation
- [ ] `git diff` final review

Known remaining validation gap: interactive Windows GUI and visual viewer controls
still require a human manual check; automated offscreen construction and saved
result reopening pass.
