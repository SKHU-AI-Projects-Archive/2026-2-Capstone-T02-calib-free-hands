"""Experiment 17 tests (unittest).  python -m unittest HandCalib/tests/test_protocol_rerun.py -v"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_protocol_rerun as R17                       # noqa: E402
from hand_focal import anycalib_ref as AC              # noqa: E402
from protocol_rerun import conditions as CO            # noqa: E402
from protocol_rerun import ranking as RK               # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/17_protocol_rerun.yaml").read_text())
C = CFG["conditions"]
VID = Path(CFG["input_dir"]) / "IMG_7141.MOV"
RUN16 = ROOT / "runs/16_detector_swap/20261015T053136Z"
FAKE_TAGS = {"location": "+37.5665+126.9780/", "make": "ZZTESTMAKE", "model": "ZZTESTMODEL"}


def trajectory(W=1920, H=1080, box_ratio=0.4, x0=700, theta_amp=30.0, theta_noise=3.0, seed=0):
    """15 fps, 3 s still front then a circular tilt of about theta_amp (deg)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 8, 1 / 15)
    n = len(t)
    side = box_ratio * min(W, H)
    box = np.tile([x0, 300, x0 + side, 300 + side], (n, 1)).astype(float)
    theta = np.where(t < 3.0, rng.uniform(0, 4, n), theta_amp + rng.normal(0, theta_noise, n))
    role = np.where(t >= 3.0, "tilt", "front")
    return t, np.ones(n, bool), box, theta, role


def metrics(**kw):
    t, p, box, th, role = trajectory(**{k: v for k, v in kw.items() if k in ("box_ratio", "x0", "theta_amp", "theta_noise")})
    return CO.metrics_from_arrays(t, p, box, 1920, 1080, th, role, kw.get("transition", 3.1), 0.003, kw.get("axis_bins", 6), kw.get("ext", 0.98))


class Conditions(unittest.TestCase):
    def test_pass(self):
        r = CO.check(metrics(), C)
        self.assertTrue(r["guide"], r["reasons"])

    def test_each_threshold_fails(self):
        cases = {"box_short_ratio": dict(box_ratio=0.6), "edge_rate": dict(x0=0), "transition": dict(transition=1.0),
                 "theta_p95": dict(theta_amp=40, theta_noise=6), "theta_tilt_median": dict(theta_amp=15, theta_noise=1),
                 "axis_bins": dict(axis_bins=4), "finger_extension": dict(ext=0.85)}
        for item, kw in cases.items():
            r = CO.check(metrics(**kw), C)
            self.assertFalse(r["items"][item], item)
            self.assertFalse(r["guide"], item)

    def test_no_transition_fails(self):
        r = CO.check(metrics(transition=None), C)
        self.assertFalse(r["items"]["transition"])
        self.assertIn("자동 전환 없음", "; ".join(r["reasons"]))


def det(verdict="REJECT", edge=False, sharp=1.1, unc=0.1, rms=0.01, rate=0.9):
    return {"verdict": verdict, "fixed_template": {"f": 1500.0, "at_grid_edge": edge, "side": "상한" if edge else None},
            "joint_delta": {"f": 1500.0, "at_grid_edge": edge, "side": "상한" if edge else None, "delta_deg": 0.0},
            "f_rel_uncertainty": unc, "reproj_rms_rel": rms, "detect_rate": rate, "profiles": {"flat17": {"official": {"sharpness": sharp}}}}


class Ranking(unittest.TestCase):
    R = CFG["ranking"]

    def test_verdict_priority_and_edge_below_reject(self):
        r = RK.rank({"a": det("REJECT", sharp=2.0), "b": det("ROUGH", sharp=1.01), "c": det("GOOD", sharp=1.0), "d": det("REJECT", edge=True, sharp=5.0),
                     "e": {"verdict": "REJECT", "failed": "no_template"}}, self.R)
        self.assertEqual([r[k]["rank"] for k in "cbade"], [1, 2, 3, 4, 5])
        self.assertEqual(r["d"]["category"], "EDGE")

    def test_keys_in_order_and_ties(self):
        r = RK.rank({"a": det(sharp=1.2, unc=0.5), "b": det(sharp=1.2, unc=0.1), "c": det(sharp=1.3, unc=0.9)}, self.R)
        self.assertEqual((r["c"]["rank"], r["b"]["rank"], r["a"]["rank"]), (1, 2, 3))
        r = RK.rank({"a": det(), "b": det(), "c": det(sharp=0.5)}, self.R)
        self.assertEqual((r["a"]["rank"], r["b"]["rank"], r["c"]["rank"]), (1.5, 1.5, 3))

    def test_mean_over_guide_videos(self):
        pv = {"v1": RK.rank({"a": det(sharp=2), "b": det(sharp=1)}, self.R), "v2": RK.rank({"a": det(sharp=1), "b": det(sharp=2)}, self.R),
              "v3": RK.rank({"a": det(sharp=2), "b": det(sharp=1)}, self.R)}
        s = RK.summary(pv, {"v1": True, "v2": True, "v3": False})
        self.assertEqual((s["a"]["mean_rank_guide"], s["a"]["n_guide_videos"]), (1.5, 2))


class Selection(unittest.TestCase):
    def test_videos_from_yaml_and_override(self):
        c = copy.deepcopy(CFG)
        c["videos"] = ["IMG_7141.MOV"]
        self.assertEqual([v.name for v in R17.select_videos(c)], ["IMG_7141.MOV"])
        self.assertEqual([v.name for v in R17.select_videos(c, ["IMG_7142.MOV"])], ["IMG_7142.MOV"])
        c["videos"] = []
        self.assertGreaterEqual(len(R17.select_videos(c)), 2)
        c["videos"] = ["NOPE.MOV"]
        with self.assertRaises(SystemExit):
            R17.select_videos(c)

    def test_edge_text(self):
        line = R17.det_line("x", det(edge=True))
        self.assertIn("값 없음 (상한)", line)
        self.assertNotIn("1500", line)


def cfg_tmp(root: Path, **kw):
    c = copy.deepcopy(CFG)
    c["output"]["root"] = str(root / "runs")
    c["pipeline"]["output"]["root"] = str(root / "runs")
    c.update(kw)
    return c


@unittest.skipUnless(VID.exists() and RUN16.exists(), "IMG_7141 and the 16 run needed")
class Runs(unittest.TestCase):
    def test_yaml_detector_and_video_selection_reproduces_16(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg_tmp(d, detectors=["wilor_yolo"], videos=["IMG_7141.MOV"]), allow_unicode=True))
            root, vids = R17.main(["--config", str(cp), "--no-anycalib", "--out", str(d / "o")])
            self.assertEqual([v["video"] for v in vids], ["IMG_7141.MOV"])
            self.assertEqual(list(vids[0]["detectors"]), ["wilor_yolo"])
            r17 = json.loads((root / "IMG_7141" / "wilor_yolo" / "results.json").read_text())
            r16 = json.loads((RUN16 / "IMG_7141" / "wilor_yolo" / "results.json").read_text())
            self.assertEqual(r17["verdict"], r16["verdict"])
            self.assertEqual(r17["f_hand_at_grid_edge"], r16["f_hand_at_grid_edge"])
            self.assertAlmostEqual(r17["f_hand_px"], r16["f_hand_px"], places=6)
            self.assertAlmostEqual(r17["f_joint_delta_px"], r16["f_joint_delta_px"], places=6)
            rep = (root / "report.md").read_text()
            self.assertIn("값 없음 (상한)", rep)
            self.assertIn("계산 불가", rep)
            self.assertIn(R17.RANK_NOTE, rep)
            self.assertTrue(vids[0]["previous"])

    def test_no_tags_and_offline_anycalib(self):
        from tests.test_hand_template_focal import make_video
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vdir = d / "v"
            vdir.mkdir()
            make_video(vdir / "clip.mov", 640, 360, 6, 30, tags=FAKE_TAGS)
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg_tmp(d, detectors=["mediapipe_hands"], videos=[], input_dir=str(vdir)), allow_unicode=True))
            before = AC.hub_snapshot(CFG["pipeline"]["anycalib"]["torch_hub_dir"])
            root, vids = R17.main(["--config", str(cp), "--out", str(d / "o")])
            self.assertEqual(AC.hub_snapshot(CFG["pipeline"]["anycalib"]["torch_hub_dir"]), before)
            self.assertEqual(vids[0]["anycalib"]["hub_check"]["new_files"], [])
            texts = "".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)


if __name__ == "__main__":
    unittest.main()
