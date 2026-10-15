"""Experiment 16 tests (unittest).  python -m unittest HandCalib/tests/test_detector_swap.py -v"""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detector_swap" / "workers"))

import common as WC                                              # noqa: E402  (worker helpers, numpy only)
import run_detector_swap as R16                                  # noqa: E402
from detector_swap import analysis16 as A16                      # noqa: E402
from detector_swap import detect as DT                           # noqa: E402
from hand_focal import anycalib_ref as AC                        # noqa: E402
from hand_focal.detectors import adapters as DA                  # noqa: E402
from hand_focal.frames import sha256                             # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/16_detector_swap.yaml").read_text())
GIGA = Path("/public/data/gigahands/demo_all/raw/hand_pose/p44-dog-0004/rgb_vid/brics-odroid-009_cam1/brics-odroid-009_cam1_1725202053203954.mp4")
VID = Path("/home/juson/data/hand_focal_videos/IMG_7141.MOV")
FAKE_TAGS = {"location": "+37.5665+126.9780/", "make": "ZZTESTMAKE", "model": "ZZTESTMODEL"}
REF14 = ROOT / CFG["reuse_14"]["detection_cache"]


def frames_from_giga(d: Path, n=3, dup_first=False):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(GIGA), "-vf", "select=not(mod(n\\,40))", "-vsync", "0",
                    "-frames:v", str(n), str(d / "f_%06d.png")], check=True)
    pngs = sorted(d.glob("f_*.png"))
    if dup_first:
        pngs = [pngs[0], pngs[0]] + pngs[1:]
    return [{"index": i, "png": str(p), "sha256": sha256(p), "frame": i, "time_s": i / 15} for i, p in enumerate(pngs)]


def cfg_tmp(root: Path, detectors):
    c = copy.deepcopy(CFG)
    c["detectors"] = detectors
    c["output"]["root"] = str(root / "runs")
    c["pipeline"]["output"]["root"] = str(root / "runs")
    return c


class Helpers(unittest.TestCase):
    def test_normalised_to_pixels(self):
        xy = np.array([[0.0, 0.0], [1.0, 1.0], [0.25, 0.5]])
        px = WC.to_pixels(xy, 1080, 1920)
        self.assertTrue(np.allclose(px, [[0, 0], [1080, 1920], [270, 960]]))

    def test_remap(self):
        k = np.arange(21 * 2, dtype=float).reshape(1, 21, 2)
        c = np.arange(21, dtype=float)[None]
        order = list(range(20, -1, -1))
        k2, c2 = WC.remap(k, c, order)
        self.assertTrue(np.allclose(k2[0, 0], k[0, 20]) and c2[0, 0] == 20)

    def test_order_assignment(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 500, (30, 21, 2))
        b = a + rng.normal(0, 2, a.shape)
        m = np.ones((30, 21), bool)
        self.assertTrue(A16.order_assignment(a, b, m)["identity"])
        bp = b.copy()
        bp[:, [9, 13]] = b[:, [13, 9]]
        self.assertFalse(A16.order_assignment(a, bp, m)["identity"])

    def test_edge_text(self):
        hi = 3072.0
        self.assertEqual(R16.f_text({"f": 3072.0, "at_grid_edge": True}, hi), "값 없음 (상한)")
        self.assertEqual(R16.f_text({"f": 672.0, "at_grid_edge": True}, hi), "값 없음 (하한)")
        self.assertEqual(R16.f_text({"f": 1650.0, "at_grid_edge": False}, hi, 0.06), "1650.0 px (±6.0%)")
        self.assertEqual(R16.spread_text(0.0, True), "계산 불가")
        self.assertEqual(R16.spread_text(0.012, False), "1.20%")


@unittest.skipUnless(GIGA.exists() and REF14.exists(), "GigaHands frames and 14's detection cache needed")
class Workers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hf16t_"))
        cls.frames = frames_from_giga(cls.tmp)
        cls.cfg = cfg_tmp(cls.tmp, ["wilor_yolo", "mediapipe_hands", "rtmpose_hand"])
        cls.out = DT.run_all(cls.frames, cls.cfg, cls.tmp / "cache", log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_npz_format_matches_14(self):
        ref = np.load(sorted(REF14.glob("detections_*.npz"))[0])
        for name, d in self.out.items():
            self.assertNotIn("skipped", d, f"{name}: {d.get('skipped')}")
            z = np.load(d["npz"])
            self.assertEqual(set(z.files), set(ref.files), name)
            for k in ref.files:
                if k == "info":
                    continue
                self.assertEqual(z[k].dtype.kind, ref[k].dtype.kind, f"{name}.{k}")
                self.assertEqual(z[k].shape[1:], ref[k].shape[1:], f"{name}.{k}")
                if z[k].dtype.kind == "f":
                    self.assertEqual(z[k].dtype, ref[k].dtype, f"{name}.{k}")
            n = len(z["boxes"])
            self.assertEqual(int(z["offsets"][-1]), n)
            if n:                                            # pixel coordinates inside the PNG (not normalised)
                self.assertGreater(float(np.nanmax(z["kxy"])), 2.0)

    def test_same_png_list(self):
        for name, d in self.out.items():
            z = DA._load(d["npz"])
            self.assertEqual([int(x) for x in z["frame_index"]], [r["index"] for r in self.frames], name)
            self.assertEqual([str(x) for x in z["sha256"]], [r["sha256"] for r in self.frames], name)

    def test_rtmpose_uses_yolo_boxes_only(self):
        y, t = np.load(self.out["wilor_yolo"]["npz"]), np.load(self.out["rtmpose_hand"]["npz"])
        self.assertTrue(np.allclose(y["boxes"], t["boxes"]) and (y["offsets"] == t["offsets"]).all())
        self.assertFalse(np.allclose(y["kxy"], t["kxy"]))

    def test_mediapipe_image_mode_repeatable(self):
        d = self.tmp / "dup"
        d.mkdir()
        fr = frames_from_giga(d, n=2, dup_first=True)
        c = cfg_tmp(d, ["mediapipe_hands"])
        o = DT.run_all(fr, c, d / "cache", log=lambda *a: None)
        z = np.load(o["mediapipe_hands"]["npz"])
        off = z["offsets"]
        a, b = z["kxy"][off[0]:off[1]], z["kxy"][off[1]:off[2]]
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.allclose(a, b))
        self.assertEqual(json.loads(str(z["info"]))["running_mode"], "IMAGE")


@unittest.skipUnless(VID.exists(), "IMG_7141 needed")
class ReproduceAndPrivacy(unittest.TestCase):
    def test_wilor_only_reproduces_14(self):
        with tempfile.TemporaryDirectory() as d:
            root, res = R16.main(["--config", str(ROOT / "configs/16_detector_swap.yaml"), "--video", str(VID), "--detectors", "wilor_yolo",
                                  "--no-anycalib", "--out", str(Path(d) / "o")])
            r16 = json.loads((root / "IMG_7141" / "wilor_yolo" / "results.json").read_text())
            r14 = json.loads((ROOT / CFG["reuse_14"]["official"]["IMG_7141.MOV"] / "results.json").read_text())
            self.assertEqual(r16["verdict"], r14["verdict"])
            self.assertEqual(r16["f_hand_at_grid_edge"], r14["f_hand_at_grid_edge"])
            self.assertAlmostEqual(r16["f_hand_px"], r14["f_hand_px"], places=6)
            self.assertAlmostEqual(r16["f_joint_delta_px"], r14["f_joint_delta_px"], places=6)
            self.assertTrue(res[0]["frames"]["same_as_14"])
            self.assertIn("값 없음 (상한)", (root / "report.md").read_text())

    def test_no_location_device_tags_and_offline_anycalib(self):
        from tests.test_hand_template_focal import make_video                    # 14's generator of tagged test videos
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vdir = d / "v"
            vdir.mkdir()
            v = vdir / "clip.mov"
            make_video(v, 640, 360, 6, 30, tags=FAKE_TAGS)
            before = AC.hub_snapshot(CFG["pipeline"]["anycalib"]["torch_hub_dir"])
            c = cfg_tmp(d, ["mediapipe_hands"])
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(c))
            root, res = R16.main(["--config", str(cp), "--input-dir", str(vdir), "--out", str(d / "o")])
            self.assertNotIn("skipped", res[0]["detectors"]["mediapipe_hands"])       # an empty detection file is valid
            self.assertEqual(AC.hub_snapshot(CFG["pipeline"]["anycalib"]["torch_hub_dir"]), before)
            self.assertEqual(res[0]["anycalib"]["hub_check"]["new_files"], [])
            self.assertIsNone(res[0]["anycalib"]["reused_from_14"])
            texts = "".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            self.assertFalse(list(root.rglob("f_[0-9][0-9][0-9][0-9][0-9][0-9].png")) + list(root.rglob("*.mkv")))


if __name__ == "__main__":
    unittest.main()
