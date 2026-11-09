"""optical_flow 의 사본이 원본(handfocal_report.py, run.py)과 같은 결과를 내는지 확인하는 테스트.

입력 영상은 쓰지 않는다. 무늬가 있는 '손' 다각형을 흰 바탕 위에 그리고, 알려진 작은 호모그래피로 조금씩 옮긴 합성 프레임을 만든다.
실행: (HandCalib 폴더에서) python -m unittest optical_flow/test_equivalence.py -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import handfocal as hf                             # noqa: E402
import run                                         # noqa: E402
import optical_flow as of                          # noqa: E402

# 실험 56 의 tracking 설정
KCFG = {"fps": 30, "dilate_frac": 0.03, "max_corners": 400, "quality": 0.01, "min_distance": 7, "block": 7, "win": 21, "levels": 3, "fb_px": 1.0, "threshold_px": 1.5, "min_inliers": 20}
W, H, N = 640, 480, 12


def _scene(tmp: Path):
    """합성 프레임 N 장(PNG)과 프레임마다의 관절(21개)·상자. 손 = 무늬 있는 다각형, 바탕 = 흰 종이. 프레임마다 회전 1.5°, 이동 (3, 1) px."""
    rng = np.random.default_rng(0)
    tex = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (5, 5), 1.2)
    ang = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    P0 = np.c_[320 + 110 * np.cos(ang) * (1 + 0.3 * np.sin(3 * ang)), 240 + 70 * np.sin(ang)]
    pngs, kps = [], []
    for i in range(N):
        a = np.radians(1.5 * i)
        Hm = np.array([[np.cos(a), -np.sin(a), 3.0 * i], [np.sin(a), np.cos(a), 1.0 * i], [0, 0, 1]])
        c = np.array([320.0, 240.0])
        T = np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]]) @ Hm @ np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
        hand = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hand, cv2.convexHull(P0.astype(np.float32)).astype(np.int32), 255)
        img = np.full((H, W, 3), 245, np.uint8)
        img[hand > 0] = tex[hand > 0]
        img = cv2.warpPerspective(img, T, (W, H), borderValue=(245, 245, 245))
        p = tmp / f"{i:03d}.png"
        cv2.imwrite(str(p), img)
        pngs.append(str(p))
        kps.append(cv2.perspectiveTransform(P0.reshape(-1, 1, 2), T).reshape(-1, 2))
    kps = np.array(kps)
    boxes = np.c_[kps.min(1), kps.max(1)]
    return pngs, kps, boxes


def _same_H(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all((a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and np.array_equal(a[k], b[k])) for k in a)


class TestEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.pngs, cls.kps, cls.boxes = _scene(Path(cls._tmp.name))
        cls.order = list(range(N))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_helpers(self):
        self.assertTrue(np.array_equal(of.box_long(self.boxes), hf.box_long(self.boxes)))
        self.assertTrue(np.array_equal(of.edge_flags(self.boxes, np.ones(N, bool), W, H), hf.edge_flags(self.boxes, np.ones(N, bool), W, H)))
        self.assertTrue(np.array_equal(of.kt_hand_mask(self.kps[0], self.boxes[0], (H, W), 0.03), hf.kt_hand_mask(self.kps[0], self.boxes[0], (H, W), 0.03)))
        pres = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], bool)
        self.assertTrue(np.array_equal(of.kt_carry(pres, self.boxes[:, 0]), hf.kt_carry(pres, self.boxes[:, 0])))
        self.assertEqual(of.rt_windows(0, N - 1, pres, 5), hf.rt_windows(0, N - 1, pres, 5))

    def test_chain(self):
        a = of.kt_chain(self.pngs, self.kps, self.boxes, self.order, KCFG)
        b = hf.kt_chain(self.pngs, self.kps, self.boxes, self.order, KCFG)
        self.assertEqual(a["breaks"], 0)
        self.assertTrue(_same_H(a["H"], b["H"]))
        self.assertEqual(a["inl"], b["inl"])

    def test_chain_points_and_track_hand(self):
        ch = hf.kt_chain(self.pngs, self.kps, self.boxes, self.order, KCFG)
        for x, y in zip(of.chain_points(self.kps[0], ch["H"], N, W, H), run.pp_points(self.kps[0], ch["H"], N, W, H)):
            self.assertTrue(np.array_equal(x, y, equal_nan=True))
        t = of.track_hand(self.pngs, self.kps, np.ones(N, bool), self.boxes, (W, H), KCFG)
        ref = t["ref"]
        fwd = hf.kt_chain(self.pngs, self.kps, self.boxes, list(range(ref, N)), KCFG)
        bwd = hf.kt_chain(self.pngs, self.kps, self.boxes, list(range(ref, -1, -1)), KCFG)
        kp, tb, valid, edge = run.pp_points(self.kps[ref], {**bwd["H"], **fwd["H"]}, N, W, H)
        self.assertTrue(np.array_equal(t["kp"], kp, equal_nan=True))
        self.assertTrue(np.array_equal(t["use"], valid & ~edge))

    def test_variants(self):
        a = of.kt_chain_log(self.pngs, self.kps, self.boxes, self.order, KCFG)
        b = hf.kt_chain_log(self.pngs, self.kps, self.boxes, self.order, KCFG)
        self.assertTrue(_same_H(a["H"], b["H"]))
        for k in ("s", "t", "res"):
            self.assertTrue(np.array_equal(a[k], b[k], equal_nan=True))
        a = of.bc_chain(self.pngs, self.kps, self.boxes, self.order, KCFG, [0, 1, 5, 9, 13, 17])
        b = hf.bc_chain(self.pngs, self.kps, self.boxes, self.order, KCFG, [0, 1, 5, 9, 13, 17])
        self.assertTrue(_same_H(a["H"], b["H"]))
        a = of.wt_track(self.pngs, self.boxes, self.order, KCFG, self.kps[0][0], 0.15, 8)
        b = hf.wt_track(self.pngs, self.boxes, self.order, KCFG, self.kps[0][0], 0.15, 8)
        self.assertEqual(a["breaks"], b["breaks"])
        for k in a["p"]:
            self.assertTrue((a["p"][k] is None and b["p"][k] is None) or np.array_equal(a["p"][k], b["p"][k]))
        ec = {"paper_s_max": 40, "paper_v_min": 90, "edge_frac": 0.1}
        for ex in (False, True):
            a = of.eg_chain(self.pngs, self.kps, self.boxes, self.order, KCFG, ec, ex)
            b = hf.eg_chain(self.pngs, self.kps, self.boxes, self.order, KCFG, ec, ex)
            self.assertTrue(_same_H(a["H"], b["H"]))
            self.assertEqual(a["n_edge"], b["n_edge"])
        g0 = cv2.cvtColor(cv2.imread(self.pngs[0]), cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cv2.imread(self.pngs[1]), cv2.COLOR_BGR2GRAY)
        a, b = of.kv_step(g0, g1, self.kps[0], self.boxes[0], KCFG), hf.kv_step(g0, g1, self.kps[0], self.boxes[0], KCFG)
        self.assertTrue(np.array_equal(a["H"], b["H"]))
        self.assertTrue(np.array_equal(a["score"], b["score"]))


if __name__ == "__main__":
    unittest.main()
