from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


def main(argv=None) -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CalibFreeHands saved-video capstone demo")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=project,
                        help="Directory containing WiLoR, rgb_predictor.py and mano_data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "CalibFreeHands" / "capstone-runs")
    parser.add_argument("--estimator", choices=("handcalib", "anycalib"), default="handcalib")
    parser.add_argument("--device", default="cuda", help="WiLoR and AnyCalib PyTorch device")
    parser.add_argument("--handcalib-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--calibration-hz", type=float, help="Maximum number of sampled calibration frames per second")
    parser.add_argument("--work-seconds", type=float, default=5.0,
                        help="Maximum duration of the following reconstruction segment")
    parser.add_argument("--headless", action="store_true", help="Run the same pipeline without the Windows viewer")
    options = parser.parse_args(argv)
    if options.headless and options.video is None:
        parser.error("--headless requires an input video")
    options.project_root = project
    options.runtime_root = options.runtime_root.resolve()
    options.output_dir = options.output_dir.resolve()
    sys.path.insert(0, str(project / "CalibFreeHands" / "src"))
    from .app_run import process

    if options.headless:
        def progress(stage, done, total):
            if done in (0, 1, total) or done % 20 == 0:
                print(f"{stage} {done}/{total}", flush=True)

        result_path, report = process(options.video, options, progress,
                                      lambda focal: print(f"f = {focal:.4f} px", flush=True))
        print(json.dumps({"result": str(result_path), **report}, ensure_ascii=True), flush=True)
        return 0 if report["frames_with_hands"] else 2

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from .window import DemoWindow

    app = QApplication(sys.argv[:1])
    window = DemoWindow(options)
    window.show()
    if options.video is not None:
        QTimer.singleShot(0, lambda: window.open_video(options.video))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
