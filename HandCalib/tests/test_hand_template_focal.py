"""Experiment 14 tests (unittest).  python -m unittest HandCalib/tests/test_hand_template_focal.py -v"""
import copy
import json
import os
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

import run_hand_focal as RUN                                         # noqa: E402
from hand_focal import analysis as A                                 # noqa: E402
from hand_focal import anycalib_ref as AC                            # noqa: E402
from hand_focal import normalize as NZ                               # noqa: E402
from hand_focal import phases as PH                                  # noqa: E402
from hand_focal import synthetic as S                                # noqa: E402
from hand_focal import template as TP                                # noqa: E402
from hand_focal import track as TR                                   # noqa: E402
from hand_focal import uncertainty as U                              # noqa: E402
from hand_focal.detectors import adapters as DA                      # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/14_hand_template_focal.yaml").read_text())
GIGA = Path("/public/data/gigahands/demo_all/raw/hand_pose/p44-dog-0004/rgb_vid/brics-odroid-009_cam1/brics-odroid-009_cam1_1725202053203954.mp4")
FAKE_TAGS = {"location": "+37.5665+126.9780/", "make": "ZZTESTMAKE", "model": "ZZTESTMODEL"}
GPU = shutil.which("nvidia-smi") is not None


def cfg_copy(**kw):
    c = copy.deepcopy(CFG)
    for k, v in kw.items():
        c[k].update(v)
    return c


def synth(cond_over=None, seed=0, scfg=None):
    c = copy.deepcopy(S.conditions(CFG["synthetic"])[0])
    c.update(cond_over or {})
    return S.generate(c, scfg or CFG["synthetic"], np.random.default_rng(seed))


def analyze_synth(g, methods_full=(), **kw):
    return A.analyze(g["kp"], g["mask"], g["present"], g["box"], g["cls"], g["times"], g["size"], kw.pop("cfg", CFG), front_idx=g["front_idx"],
                     tilt_start=g["tilt_start"], methods_full=methods_full, **kw)


def hand_track(n, times, W, H, f, dist, rng, tilt_from=3.0):
    """Planar-hand keypoint track for a mock video: still and frontal before tilt_from, then tilting."""
    P = S.hand_points(0.0)
    P[:, 1] += 95.0
    ids = [0] + list(range(5, 21))
    kp = np.full((n, 21, 2), np.nan)
    for i, t in enumerate(times):
        if t < tilt_from:
            R, tt = S.pose(0.0, 0.0, 0.0), np.array([0.0, 0.0, dist])
        else:
            R = S.pose(20 + 20 * np.sin(1.3 * t) ** 2, (70 * t) % 360, 10 * np.sin(t))
            tt = np.array([10 * np.sin(t), 5 * np.cos(t), dist])
        kp[i, ids] = S.project(P[ids], R, tt, f, W / 2, H / 2) + rng.normal(0, 0.5, (17, 2))
    kp[:, 1:5] = kp[:, [5]] + np.array([-40.0, 20.0])        # thumb points (not used by flat17), for the order check
    return kp


class Geometry(unittest.TestCase):
    def test_f_recovery_within_1pct(self):
        g = synth({"noise": {"px": 1.0}, "tilt_range_deg": [25, 50], "delta_deg": 0}, seed=3)
        r = analyze_synth(g)
        err = r["solutions"]["fixed_template"]["f"] / g["f_true"] - 1
        self.assertLess(abs(err), 0.01, err)

    def test_template_alignment(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(17, 2)) * 50
        frames = []
        for k in range(8):
            a = rng.uniform(-0.3, 0.3)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            frames.append((rng.uniform(0.8, 1.2) * (R @ X.T)).T + rng.uniform(-100, 100, 2))
        t = TP.build(np.array(frames), 4)
        s, R, tr = TP.similarity_align(t["template"], X)
        self.assertLess(np.abs((s * (R @ t["template"].T)).T + tr - X).max(), 1e-6)
        self.assertLess(t["jitter_rms_px"], 1e-6)

    def test_theta2d_matches_synthetic_tilt(self):
        P = S.hand_points(0.0)[[0] + list(range(5, 21))]
        P[:, 1] += 95.0
        T = P[:, :2]
        for th in (10, 25, 40, 55):
            for az in (0, 60, 135):
                q = S.project(P, S.pose(th, az, 15), np.array([0, 0, 20000.0]), 1650 * 20000 / 350, 960, 540)
                theta, axis = TP.affine_tilt(T, q)
                self.assertAlmostEqual(theta, th, delta=0.5)

    def test_small_tilt_not_good(self):
        g = synth({"tilt_range_deg": [3, 8]}, seed=2)
        r = analyze_synth(g, methods_full=("fixed_template",), n_boot=4)
        self.assertNotEqual(r["per_method"]["fixed_template"]["verdict"]["verdict"], "GOOD")

    def test_estimator_switch(self):
        g = synth({"delta_deg": 10}, seed=4)
        r = analyze_synth(g, methods_full=("fixed_template", "joint_delta"), n_boot=2)
        self.assertEqual(r["solutions"]["fixed_template"]["delta_deg"], 0.0)
        self.assertGreater(r["solutions"]["joint_delta"]["delta_deg"], 0.0)
        self.assertEqual(set(r["per_method"]), {"fixed_template", "joint_delta"})
        c = cfg_copy(estimate={"estimator": "fixed_template"})
        self.assertEqual(RUN.resolve_estimator(c)[0], "fixed_template")
        self.assertEqual(RUN.resolve_estimator(c, "joint_delta")[0], "joint_delta")


class PhasesTracking(unittest.TestCase):
    def test_phase_split_and_transition(self):
        t = np.arange(0, 8, 1 / 15)
        n = len(t)
        present = np.ones(n, bool)
        kp = np.zeros((n, 21, 2)) + np.linspace(0, 100, 21)[None, :, None]
        moving = (t < 1.2) | (t > 2.4)
        kp[moving] += (np.arange(n)[moving] * 3.0)[:, None, None]
        mask = np.ones((n, 21), bool)
        box = np.tile([0, 0, 100, 100.0], (n, 1))
        fw = PH.front_window(t, present, kp, mask, box, CFG["phases"])
        self.assertGreaterEqual(fw["t0"], 1.2 - 1e-9)
        self.assertLessEqual(fw["t1"], 2.4)
        theta = np.where(t >= 3.4, 30.0, 2.0)
        self.assertAlmostEqual(PH.auto_transition(t, theta, present, CFG["phases"]), t[np.argmax(t >= 3.4)])
        self.assertTrue((t[PH.tilt_indices(t, present, 3.0)] >= 3.0).all())

    def test_bootstrap_reproducible(self):
        g = synth(seed=5)
        r1 = analyze_synth(g, methods_full=("fixed_template",), n_boot=4, seed=7)
        r2 = analyze_synth(g, methods_full=("fixed_template",), n_boot=4, seed=7)
        self.assertEqual(r1["per_method"]["fixed_template"]["bootstrap"]["f"], r2["per_method"]["fixed_template"]["bootstrap"]["f"])

    def _det(self, boxes_per_frame, kps_per_frame, kconf=None):
        offs, boxes, kxy, kc, cls, bc = [0], [], [], [], [], []
        for bs, ks in zip(boxes_per_frame, kps_per_frame):
            for b, k in zip(bs, ks):
                boxes.append(b)
                kxy.append(k)
                kc.append(np.ones(21) if kconf is None else kconf)
                cls.append(1)
                bc.append(0.9)
            offs.append(len(boxes))
        return {"offsets": np.array(offs), "boxes": np.array(boxes, float), "kxy": np.array(kxy, float), "kconf": np.array(kc, float),
                "cls": np.array(cls), "box_conf": np.array(bc)}

    def test_masks(self):
        k = np.tile([[50.0, 50.0]], (21, 1))
        k[3] = [-5, 50]
        kc = np.ones(21)
        kc[7] = 0.2
        det = self._det([[[0, 0, 100, 100]]], [[k]], kconf=kc)
        tr = TR.track(det, np.array([0.6]), (200, 200), CFG["tracking"], front_start_s=0.5)
        self.assertFalse(tr["mask"][0, 3])
        self.assertFalse(tr["mask"][0, 7])
        self.assertEqual(int(tr["mask"][0].sum()), 19)

    def test_tracking_two_hands(self):
        n = 20
        times = 0.5 + np.arange(n) / 15
        boxes, kps = [], []
        for i in range(n):
            big = [100 + 5 * i, 100, 400 + 5 * i, 400]          # target: larger, moving right
            small = [800, 100, 900, 200]
            order = [small, big] if i % 2 else [big, small]
            boxes.append(order)
            kps.append([np.full((21, 2), b[0] + 1.0) for b in order])
        tr = TR.track(self._det(boxes, kps), times, (1920, 1080), CFG["tracking"], front_start_s=0.5)
        self.assertTrue(tr["present"].all())
        self.assertTrue(np.allclose(tr["box"][:, 0], 100 + 5 * np.arange(n)))

    def test_order_check(self):
        rng = np.random.default_rng(0)
        kp = hand_track(5, np.zeros(5), 1920, 1080, 1650, 350, rng)
        m = np.ones((5, 21), bool)
        self.assertTrue(TR.order_check(kp, m)["ok"])
        bad = kp.copy()
        bad[:, [9, 17]] = kp[:, [17, 9]]
        self.assertFalse(TR.order_check(bad, m)["ok"])


class Detection(unittest.TestCase):
    def test_frame_alignment(self):
        frames = [{"index": i, "sha256": f"h{i}"} for i in range(3)]
        DA.check_alignment(frames, {"frame_index": np.arange(3), "sha256": np.array(["h0", "h1", "h2"])})
        with self.assertRaises(DA.DetectorError):
            DA.check_alignment(frames, {"frame_index": np.array([0, 2, 1]), "sha256": np.array(["h0", "h1", "h2"])})
        with self.assertRaises(DA.DetectorError):
            DA.check_alignment(frames, {"frame_index": np.arange(3), "sha256": np.array(["h0", "hX", "h2"])})

    def test_worker_failure_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "a.png"
            png.write_bytes(b"x")
            frames = [{"index": 0, "png": str(png), "sha256": "0"}]
            dc = dict(CFG["detector"], python="/bin/false", cache=False)
            with self.assertRaises(DA.DetectorError) as cm:
                DA.wilor_yolo(frames, dc, Path(d) / "cache", log=lambda *a: None)
            self.assertIn("YOLO worker failed", str(cm.exception))


def make_video(path: Path, W, H, seconds, fps, rotate=None, tags=None, marker=None):
    src = path.with_suffix(".tmp.mp4")
    vf = "drawbox=x={}:y={}:w={}:h={}:color=white:t=fill".format(*marker) if marker else "null"
    lav = f"color=c=black:s={W}x{H}:r={fps}" if marker else f"testsrc=size={W}x{H}:rate={fps}"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", lav, "-t", str(seconds), "-vf", vf, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-c", "copy"]
    if rotate is not None:
        cmd += ["-metadata:s:v:0", f"rotate={rotate}"]
    for k, v in (tags or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    subprocess.run(cmd + [str(path)], check=True)
    src.unlink()


class Normalization(unittest.TestCase):
    def test_coordinates_after_resize_and_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            v = d / "rot.mov"
            make_video(v, 640, 360, 1, 30, rotate=90, marker=(40, 30, 60, 40))
            tag = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream_tags=rotate", "-of", "csv=p=0", str(v)], text=True).strip()
            cw = int(tag)                                               # display = rotate this many degrees clockwise
            c = cfg_copy(normalize={"long_side": 320})
            r = NZ.normalize(v, d / "n.mkv", c)
            self.assertEqual(r["output"]["rotation_deg"], 0)
            fr, k = __import__("hand_focal.frames", fromlist=["x"]).extract(d / "n.mkv", d / "f", r["output"]["fps"], None, 15)
            import cv2
            im = cv2.imread(fr[0]["png"], cv2.IMREAD_GRAYSCALE)
            self.assertEqual(im.shape[::-1], (180, 320) if cw % 180 else (320, 180))
            ys, xs = np.nonzero(im > 128)
            x0, y0 = 40 + 30 - 0.5, 30 + 20 - 0.5                       # marker centre in the stored frame (pixel centres)
            W0, H0 = 640, 360
            if cw % 360 == 90:
                ex, ey = H0 - 1 - y0, x0
            elif cw % 360 == 270:
                ex, ey = y0, W0 - 1 - x0
            else:
                ex, ey = W0 - 1 - x0, H0 - 1 - y0
            s = 0.5
            self.assertAlmostEqual(xs.mean(), (ex + 0.5) * s - 0.5, delta=1.5)
            self.assertAlmostEqual(ys.mean(), (ey + 0.5) * s - 0.5, delta=1.5)

    def test_no_upscale_even_sides(self):
        self.assertEqual(NZ.target_dims(1279, 719, 1920), (1280, 720))
        self.assertEqual(NZ.target_dims(3840, 2160, 1920), (1920, 1080))
        self.assertEqual(NZ.target_dims(1080, 1920, 1920), (1080, 1920))


@unittest.skipUnless(GPU and GIGA.exists(), "GPU and a GigaHands frame are needed")
class AnyCalibRef(unittest.TestCase):
    def test_half_resize_halves_f(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.png"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(GIGA), "-frames:v", "1", str(p)], check=True)
            im = AC.load_rgb(p)
            half = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            sys.path.insert(0, str(ROOT))
            before = AC.hub_snapshot(CFG["anycalib"]["torch_hub_dir"])
            with AC.offline(CFG["anycalib"]["offline_env"]):
                f1 = AC.predict_frames("anycalib_pinhole", "pinhole", [im], "cuda:0")[0, :2].mean()
                f2 = AC.predict_frames("anycalib_pinhole", "pinhole", [half], "cuda:0")[0, :2].mean()
            self.assertEqual(AC.hub_snapshot(CFG["anycalib"]["torch_hub_dir"]), before)
            self.assertAlmostEqual(f2 / f1, 0.5 * (im.shape[1] // 2) / (im.shape[1] / 2), delta=0.01)

    def test_offline_guard_blocks_download(self):
        import torch.hub
        with AC.offline(CFG["anycalib"]["offline_env"]):
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            with self.assertRaises(AC.DownloadBlocked):
                torch.hub.download_url_to_file("https://example.invalid/x.pt", "/nonexistent/x.pt")


class EndToEnd(unittest.TestCase):
    """Mock detector on a generated video (with fake location / device tags) through the real runner."""

    def _run(self, anycalib: bool, estimator: str):
        d = Path(tempfile.mkdtemp(prefix="hf14test_"))
        self.addCleanup(shutil.rmtree, d, True)
        W, H, fps, sec = 640, 360, 30, 7
        vdir = d / "videos"
        vdir.mkdir()
        v = vdir / "clip.mov"
        make_video(v, W, H, sec, fps, tags=FAKE_TAGS)
        tags = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(v)], text=True))["format"]["tags"]
        self.assertEqual(tags.get("make"), FAKE_TAGS["make"])           # the fake tags are really in the input
        n = len(range(0, fps * sec, 2))
        times = np.arange(n) * 2 / fps
        rng = np.random.default_rng(0)
        kp = hand_track(n, times, W, H, 1650 * W / 1920, 600, rng)
        lo, hi = np.nanmin(kp, 1), np.nanmax(kp, 1)
        boxes = np.c_[lo - 10, hi + 10]
        npz = d / "mock.npz"
        np.savez(npz, frame_index=np.arange(n), offsets=np.arange(n + 1), boxes=boxes, box_conf=np.full(n, 0.9), cls=np.ones(n, int),
                 kxy=kp, kconf=np.ones((n, 21)), info=json.dumps({"mock": True}))
        c = copy.deepcopy(CFG)
        c["detector"].update({"name": "mock", "mock_npz": str(npz)})
        c["anycalib"]["enabled"] = anycalib
        c["uncertainty"]["n_bootstrap"] = 4
        c["output"]["root"] = str(d / "runs")
        c["estimate"]["estimator"] = estimator
        cp = d / "cfg.yaml"
        cp.write_text(yaml.safe_dump(c))
        hub_before = AC.hub_snapshot(CFG["anycalib"]["torch_hub_dir"])
        args = ["--config", str(cp), "estimate", "--input-dir", str(vdir), "--out", str(d / "out")] + ([] if anycalib else ["--no-anycalib"])
        out_root, rows = RUN.main(args)
        self.assertEqual(AC.hub_snapshot(CFG["anycalib"]["torch_hub_dir"]), hub_before)
        return d, out_root / "clip", rows, v

    def _check_common(self, d, out, rows, v):
        for f in ("results.json", "result.yaml", "report.md", "per_frame.csv", "metadata.json", "log.txt", "detections.npz"):
            self.assertTrue((out / f).exists(), f)
        self.assertFalse(list(out.rglob("*.mkv")) + list(out.rglob("f_[0-9][0-9][0-9][0-9][0-9][0-9].png")) + list(out.rglob("*.mov")))
        self.assertTrue(v.exists())
        texts = "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
        texts += (out.parent / "summary.md").read_text()
        for val in FAKE_TAGS.values():
            self.assertNotIn(val, texts)
        for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"', "location"):
            self.assertNotIn(key, texts)
        self.assertIn("정답 데이터와 비교하지 않았다", (out / "report.md").read_text())
        self.assertFalse(list(Path(tempfile.gettempdir()).glob("hf14_*")))
        res = json.loads((out / "results.json").read_text())
        self.assertIn(res["verdict"], ("GOOD", "ROUGH", "REJECT"))
        return res

    def test_mock_end_to_end_without_anycalib(self):
        d, out, rows, v = self._run(False, "fixed_template")
        res = self._check_common(d, out, rows, v)
        self.assertEqual(res["estimator"], "fixed_template")
        self.assertIsNone(res["anycalib"])
        self.assertLess(abs(res["f_hand_px"] / (1650 * 640 / 1920) - 1), 0.05)
        self.assertTrue(out.parent.name == "out")

    @unittest.skipUnless(GPU, "GPU needed")
    def test_mock_end_to_end_with_anycalib(self):
        d, out, rows, v = self._run(True, "joint_delta")
        res = self._check_common(d, out, rows, v)
        self.assertEqual(res["estimator"], "joint_delta")
        self.assertEqual(res["anycalib"]["hub_check"]["new_files"], [])
        self.assertIn("anycalib_pinhole", res["anycalib"]["models"])


if __name__ == "__main__":
    unittest.main()
