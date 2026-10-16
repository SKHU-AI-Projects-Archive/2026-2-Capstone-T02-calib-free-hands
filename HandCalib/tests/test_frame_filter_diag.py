"""17 frame-filter diagnostic tests.  python -m unittest HandCalib/tests/test_frame_filter_diag.py -v"""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol_rerun import frame_filter as FF          # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/17_frame_filter_diag.yaml").read_text())
C17 = yaml.safe_load((ROOT / CFG["pipeline_config"]).read_text())


class Mask(unittest.TestCase):
    def test_rule(self):
        t = np.arange(10) * 0.5                               # 0 .. 4.5 s
        present = np.ones(10, bool)
        theta = np.array([60, 2, 3, 4, 5, 5, 30, 50, 40, np.nan], float)
        box = np.tile([100.0, 100, 300, 300], (10, 1))
        box[8] = [0, 100, 200, 300]                           # touches the left border
        k = FF.keep_mask(t, present, theta, box, 1920, 1080, 3.0, CFG["filter"])
        self.assertTrue(k[0])                                 # before the tilt start: untouched
        self.assertTrue(k[6])                                 # 30 deg kept
        self.assertFalse(k[7])                                # 50 deg dropped
        self.assertFalse(k[8])                                # edge dropped
        self.assertFalse(k[9])                                # no theta dropped


SRC = ROOT / CFG["sources"]["IMG_7147.MOV"] / "rtmpose_hand"


@unittest.skipUnless(SRC.exists(), "17 run needed")
class NoFilterReproduces17(unittest.TestCase):
    def test_disabled_filter_gives_official_f(self):
        pc = copy.deepcopy(C17["pipeline"])
        pc["tracking"]["conf_min"] = float(C17["detector_settings"]["rtmpose_hand"]["conf_min"])
        r = FF.run_one(SRC, pc, {"theta_max_deg": 1e9, "exclude_edge": False}, methods_full=())
        off = json.loads((SRC / "results.json").read_text())["analysis"]["solutions"]
        self.assertTrue(r["same_template"])
        self.assertEqual(r["n_tilt_kept"], r["n_tilt_before"])
        self.assertAlmostEqual(r["solutions"]["fixed_template"]["f"], off["fixed_template"]["f"], places=6)


if __name__ == "__main__":
    unittest.main()
