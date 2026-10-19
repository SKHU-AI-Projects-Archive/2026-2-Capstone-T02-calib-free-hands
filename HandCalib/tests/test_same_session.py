"""Experiment 18 tests (unittest).  python -m unittest HandCalib/tests/test_same_session.py -v"""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import run_same_session as R18                    # noqa: E402
from hand_focal import anycalib_ref as AC         # noqa: E402
from hand_focal import frames as FR               # noqa: E402
from hand_focal import normalize as NZ            # noqa: E402
from same_session import checkerboard as CB       # noqa: E402
from same_session import cut as CUT               # noqa: E402
from same_session import segment as SG            # noqa: E402
from synth_board import poses, render             # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/18_same_session_check.yaml").read_text())
FAKE_TAGS = {"location": "+37.5665+126.9780/", "make": "ZZTESTMAKE", "model": "ZZTESTMODEL"}


def board_frames(K, W, H, n, tilt, seed=0):
    rng = np.random.default_rng(seed)
    return [{"index": i, "time_s": i / 10, "gray": render(K, W, H, r, t)} for i, (r, t) in enumerate(poses(n, tilt, rng))]


class Segmentation(unittest.TestCase):
    def test_split_and_warnings(self):
        t = np.arange(0, 20, 0.2)
        hand = t < 9
        board = (t >= 11) & (t <= 18)
        s = SG.split(t, hand, board, 0.5)
        self.assertAlmostEqual(s["hand"][1], 10.5, places=6)
        self.assertEqual(s["board"], [11.0, 18.0])
        self.assertEqual(s["warnings"], [])
        hand2 = hand.copy()
        hand2[(t > 12) & (t < 13)] = True                          # a hand inside the board segment
        board2 = board.copy()
        board2[(t > 3) & (t < 3.3)] = True                          # an early stray board detection
        s2 = SG.split(t, hand2, board2, 0.5)
        self.assertAlmostEqual(s2["board"][0], 3.2, places=6)       # the first detection opens the board segment
        self.assertTrue(any("손 검출" in w for w in s2["warnings"]))
        self.assertIn("체커보드가 검출되지 않았습니다", SG.split(t, hand, np.zeros_like(board), 0.5)["warnings"])


class Checkerboard(unittest.TestCase):
    W, H, f = 1080, 1920, 1700.0
    K = np.array([[1700.0, 0, 540.0], [0, 1700.0, 960.0], [0, 0, 1.0]])

    def test_recovers_f_with_tilt(self):
        r = CB.analyse(board_frames(self.K, self.W, self.H, 50, (30, 50)), CFG["checkerboard"], self.W, self.H)
        self.assertLess(abs(r["f_reference_px"] / self.f - 1), 0.01)
        self.assertLess(abs(r["f_model_E_px"] / self.f - 1), 0.01)

    def test_small_tilt_not_good(self):
        r = CB.analyse(board_frames(self.K, self.W, self.H, 40, (0, 5), seed=1), CFG["checkerboard"], self.W, self.H)
        self.assertNotEqual(r["verdict"], "GOOD")
        self.assertIn("tilt_p90", r["failed_good"])


def ffv1_testsrc(path, seconds, fps=30, size="320x240"):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}", "-t", str(seconds),
                    "-c:v", "ffv1", str(path)], check=True)


class CutAndResolution(unittest.TestCase):
    def test_cut_keeps_frame_numbers_and_times(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ffv1_testsrc(d / "a.mkv", 4)
            info = CUT.cut(d / "a.mkv", d / "b.mkv", 1.0, 2.5, 30.0)
            fa, _ = FR.extract(d / "a.mkv", d / "fa", 30.0, None, 30)
            fb, _ = FR.extract(d / "b.mkv", d / "fb", 30.0, None, 30)
            self.assertEqual(len(fb), 45)
            self.assertEqual(info["first_frame"], 30)
            for i, r in enumerate(fb):
                self.assertEqual(r["sha256"], fa[i + 30]["sha256"])
                self.assertAlmostEqual(r["time_s"] + info["t0"], fa[i + 30]["time_s"], places=6)
            self.assertAlmostEqual(NZ.probe(d / "b.mkv")["fps"], 30.0, places=3)

    def test_board_f_in_normalised_pixels(self):
        """A 3840x2160 recording with f = 3400 px is normalised to 1920x1080; the board f must come out in that resolution (~1700)."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            W, H, f = 3840, 2160, 3400.0
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            for fr in board_frames(K, W, H, 30, (30, 50), seed=2):
                cv2.imwrite(str(d / f"s_{fr['index']:04d}.png"), fr["gray"])
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10", "-i", str(d / "s_%04d.png"), "-c:v", "libx264", "-crf", "8",
                            "-pix_fmt", "yuv420p", str(d / "v.mp4")], check=True)
            n = NZ.normalize(d / "v.mp4", d / "n.mkv", CFG["pipeline"])
            self.assertEqual([n["output"]["display_w"], n["output"]["display_h"]], [1920, 1080])
            fr = R18.board_frames(d / "n.mkv", n["output"]["fps"], [0.0, 100.0], CFG["checkerboard"], d)
            r = CB.analyse(fr, CFG["checkerboard"], 1920, 1080)
            self.assertEqual(r["image_size"], [1920, 1080])
            self.assertLess(abs(r["f_reference_px"] / 1700.0 - 1), 0.01)


class EndToEnd(unittest.TestCase):
    def test_no_tags_offline_anycalib_and_segments(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vdir = d / "v"
            vdir.mkdir()
            W, H, f = 1920, 1080, 1600.0
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            for fr in board_frames(K, W, H, 40, (30, 50), seed=3):
                cv2.imwrite(str(d / f"s_{fr['index']:04d}.png"), cv2.cvtColor(fr["gray"], cv2.COLOR_GRAY2BGR))
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-t", "7", "-i", f"testsrc=size={W}x{H}:rate=10",
                            "-framerate", "10", "-i", str(d / "s_%04d.png"), "-filter_complex", "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]concat=n=2:v=1[v]",
                            "-map", "[v]", "-c:v", "libx264", "-crf", "12", str(d / "tmp.mp4")], check=True)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(d / "tmp.mp4"), "-c", "copy"]
            for k, v in FAKE_TAGS.items():
                cmd += ["-metadata", f"{k}={v}"]
            subprocess.run(cmd + [str(vdir / "clip.mov")], check=True)
            c = copy.deepcopy(CFG)
            c.update({"videos": ["clip.mov"], "input_dir": str(vdir), "detectors": ["mediapipe_hands"]})
            c["output"]["root"] = str(d / "runs")
            c["pipeline"]["output"]["root"] = str(d / "runs")
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(c, allow_unicode=True))
            before = AC.hub_snapshot(CFG["pipeline"]["anycalib"]["torch_hub_dir"])
            root, vids = R18.main(["--config", str(cp), "--out", str(d / "o")])
            self.assertEqual(AC.hub_snapshot(CFG["pipeline"]["anycalib"]["torch_hub_dir"]), before)
            self.assertEqual(vids[0]["hand"]["anycalib"]["hub_check"]["new_files"], [])
            s = vids[0]["segments"]
            self.assertTrue(6.5 <= s["board"][0] <= 7.3, s)
            self.assertLess(abs(vids[0]["checkerboard"]["f_reference_px"] / f - 1), 0.01)
            texts = "".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            self.assertIn("summary.md", [p.name for p in root.iterdir()])


if __name__ == "__main__":
    unittest.main()
