"""Tests of handfocal.py, run.py and detector_worker.py (consolidated from the 14/16/17/18 test files; 16's command-line tests were dropped with its command).

    python -m unittest HandCalib/tests/test_handfocal.py -v
"""
import copy
import json
import os
import shutil
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

import detector_worker as WC                       # noqa: E402  (worker helpers; numpy only at import)
import handfocal as hf                             # noqa: E402
import run                                         # noqa: E402

CFG17 = yaml.safe_load((ROOT / "configs/17_protocol_rerun.yaml").read_text())
CFGFF = yaml.safe_load((ROOT / "configs/17_frame_filter_diag.yaml").read_text())
CFG18 = yaml.safe_load((ROOT / "configs/18_same_session_check.yaml").read_text())
CFG20 = yaml.safe_load((ROOT / "configs/20_phase_segmentation.yaml").read_text())
CFG21 = yaml.safe_load((ROOT / "configs/21_guide_revision.yaml").read_text())
PIPE = CFG17["pipeline"]                           # the 14 pipeline settings (identical in 16, 17 and 18)
SYN = {"f_px": 1650, "image": [1920, 1080], "distance_mm": 350, "front_frames": 10, "tilt_frames": 40}     # 14's synthetic camera
BASE_COND = {"name": "base", "delta_deg": 0, "noise": {"px": 1.0}, "heavy_tail": False, "tilt_range_deg": [25, 50], "arch_mm": 0,
             "bias_rel": 0.0, "front_frames": 10}
GIGA = Path("/public/data/gigahands/demo_all/raw/hand_pose/p44-dog-0004/rgb_vid/brics-odroid-009_cam1/brics-odroid-009_cam1_1725202053203954.mp4")
VID = Path(CFG17["input_dir"]) / "IMG_7141.MOV"
FAKE_TAGS = {"location": "+37.5665+126.9780/", "make": "ZZTESTMAKE", "model": "ZZTESTMODEL"}
GPU = shutil.which("nvidia-smi") is not None


# ====================================================================================================
# synthetic checkerboard frames (known pinhole K)
# ====================================================================================================

def board_texture(inner=(7, 10), px=60):
    nx, ny = inner[0] + 1, inner[1] + 1
    img = np.zeros((ny * px + 2 * px, nx * px + 2 * px), np.uint8) + 255
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                img[px + j * px:px + (j + 1) * px, px + i * px:px + (i + 1) * px] = 0
    return img, px


def render(K, W, H, rvec, tvec, inner=(7, 10), square=15.0):
    tex, px = board_texture(inner)
    s = square / px                                       # mm per texture pixel
    # texture pixel (u, v) -> board point (X, Y, 0) mm, board origin at the first inner corner
    A = np.array([[s, 0, -2 * square], [0, s, -2 * square], [0, 0, 1.0]])   # first inner corner = board origin
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    Hb = K @ np.c_[R[:, :2], np.asarray(tvec, float)]     # board (X, Y, 1) -> image
    M = Hb @ A
    img = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=128)
    return img


def poses(n, tilt_range, rng, dist=600.0):
    out = []
    for _ in range(n):
        th = np.radians(rng.uniform(*tilt_range))
        az = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(az), np.sin(az), 0.0])
        r = axis * th + np.array([0, 0, rng.uniform(-0.3, 0.3)])
        R, _ = cv2.Rodrigues(r)
        centre = np.array([5 * 15 / 2, 8 * 15 / 2, 0.0])          # roughly the board centre (mm)
        t = np.array([rng.uniform(-60, 60), rng.uniform(-60, 60), dist * rng.uniform(0.9, 1.1)]) - R @ centre
        out.append((r, t))
    return out

# ====================================================================================================
# core: geometry, phases, tracking, detection, normalisation, AnyCalib, end to end (from 14)
# ====================================================================================================

GIGA = Path("/public/data/gigahands/demo_all/raw/hand_pose/p44-dog-0004/rgb_vid/brics-odroid-009_cam1/brics-odroid-009_cam1_1725202053203954.mp4")
FAKE_TAGS = {"location": "+37.5665+126.9780/", "make": "ZZTESTMAKE", "model": "ZZTESTMODEL"}
GPU = shutil.which("nvidia-smi") is not None


def cfg_copy(**kw):
    c = copy.deepcopy(PIPE)
    for k, v in kw.items():
        c[k].update(v)
    return c


def synth(cond_over=None, seed=0, scfg=None):
    c = copy.deepcopy(BASE_COND)
    c.update(cond_over or {})
    return hf.synth_generate(c, scfg or SYN, np.random.default_rng(seed))


def analyze_synth(g, methods_full=(), **kw):
    return hf.analyze(g["kp"], g["mask"], g["present"], g["box"], g["cls"], g["times"], g["size"], kw.pop("cfg", PIPE), front_idx=g["front_idx"],
                     tilt_start=g["tilt_start"], methods_full=methods_full, **kw)


def hand_track(n, times, W, H, f, dist, rng, tilt_from=3.0):
    """Planar-hand keypoint track for a mock video: still and frontal before tilt_from, then tilting."""
    P = hf.synth_hand_points(0.0)
    P[:, 1] += 95.0
    ids = [0] + list(range(5, 21))
    kp = np.full((n, 21, 2), np.nan)
    for i, t in enumerate(times):
        if t < tilt_from:
            R, tt = hf.synth_pose(0.0, 0.0, 0.0), np.array([0.0, 0.0, dist])
        else:
            R = hf.synth_pose(20 + 20 * np.sin(1.3 * t) ** 2, (70 * t) % 360, 10 * np.sin(t))
            tt = np.array([10 * np.sin(t), 5 * np.cos(t), dist])
        kp[i, ids] = hf.synth_project(P[ids], R, tt, f, W / 2, H / 2) + rng.normal(0, 0.5, (17, 2))
    kp[:, 1:5] = kp[:, [5]] + np.array([-40.0, 20.0])        # thumb points (not used by flat17), for the order check
    return kp


class CoreGeometry(unittest.TestCase):
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
        t = hf.template_build(np.array(frames), 4)
        s, R, tr = hf.similarity_align(t["template"], X)
        self.assertLess(np.abs((s * (R @ t["template"].T)).T + tr - X).max(), 1e-6)
        self.assertLess(t["jitter_rms_px"], 1e-6)

    def test_theta2d_matches_synthetic_tilt(self):
        P = hf.synth_hand_points(0.0)[[0] + list(range(5, 21))]
        P[:, 1] += 95.0
        T = P[:, :2]
        for th in (10, 25, 40, 55):
            for az in (0, 60, 135):
                q = hf.synth_project(P, hf.synth_pose(th, az, 15), np.array([0, 0, 20000.0]), 1650 * 20000 / 350, 960, 540)
                theta, axis = hf.affine_tilt(T, q)
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


class CorePhasesTracking(unittest.TestCase):
    def test_phase_split_and_transition(self):
        t = np.arange(0, 8, 1 / 15)
        n = len(t)
        present = np.ones(n, bool)
        kp = np.zeros((n, 21, 2)) + np.linspace(0, 100, 21)[None, :, None]
        moving = (t < 1.2) | (t > 2.4)
        kp[moving] += (np.arange(n)[moving] * 3.0)[:, None, None]
        mask = np.ones((n, 21), bool)
        box = np.tile([0, 0, 100, 100.0], (n, 1))
        fw = hf.front_window(t, present, kp, mask, box, PIPE["phases"])
        self.assertGreaterEqual(fw["t0"], 1.2 - 1e-9)
        self.assertLessEqual(fw["t1"], 2.4)
        theta = np.where(t >= 3.4, 30.0, 2.0)
        self.assertAlmostEqual(hf.auto_transition(t, theta, present, PIPE["phases"]), t[np.argmax(t >= 3.4)])
        self.assertTrue((t[hf.tilt_indices(t, present, 3.0)] >= 3.0).all())

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
        tr = hf.track(det, np.array([0.6]), (200, 200), PIPE["tracking"], front_start_s=0.5)
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
        tr = hf.track(self._det(boxes, kps), times, (1920, 1080), PIPE["tracking"], front_start_s=0.5)
        self.assertTrue(tr["present"].all())
        self.assertTrue(np.allclose(tr["box"][:, 0], 100 + 5 * np.arange(n)))

    def test_order_check(self):
        rng = np.random.default_rng(0)
        kp = hand_track(5, np.zeros(5), 1920, 1080, 1650, 350, rng)
        m = np.ones((5, 21), bool)
        self.assertTrue(hf.order_check(kp, m)["ok"])
        bad = kp.copy()
        bad[:, [9, 17]] = kp[:, [17, 9]]
        self.assertFalse(hf.order_check(bad, m)["ok"])


class CoreDetection(unittest.TestCase):
    def test_frame_alignment(self):
        frames = [{"index": i, "sha256": f"h{i}"} for i in range(3)]
        hf.check_alignment(frames, {"frame_index": np.arange(3), "sha256": np.array(["h0", "h1", "h2"])})
        with self.assertRaises(hf.DetectorError):
            hf.check_alignment(frames, {"frame_index": np.array([0, 2, 1]), "sha256": np.array(["h0", "h1", "h2"])})
        with self.assertRaises(hf.DetectorError):
            hf.check_alignment(frames, {"frame_index": np.arange(3), "sha256": np.array(["h0", "hX", "h2"])})

    def test_worker_failure_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "a.png"
            png.write_bytes(b"x")
            frames = [{"index": 0, "png": str(png), "sha256": "0"}]
            dc = dict(CFG17["detector_settings"]["wilor_yolo"], python="/bin/false", cache=False)
            with self.assertRaises(hf.DetectorError) as cm:
                hf.detector_wilor_yolo(frames, dc, Path(d) / "cache", log=lambda *a: None)
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


class CoreNormalization(unittest.TestCase):
    def test_coordinates_after_resize_and_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            v = d / "rot.mov"
            make_video(v, 640, 360, 1, 30, rotate=90, marker=(40, 30, 60, 40))
            tag = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream_tags=rotate", "-of", "csv=p=0", str(v)], text=True).strip()
            cw = int(tag)                                               # display = rotate this many degrees clockwise
            c = cfg_copy(normalize={"long_side": 320})
            r = hf.normalize(v, d / "n.mkv", c)
            self.assertEqual(r["output"]["rotation_deg"], 0)
            fr, k = hf.extract_frames(d / "n.mkv", d / "f", r["output"]["fps"], None, 15)
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
        self.assertEqual(hf.target_dims(1279, 719, 1920), (1280, 720))
        self.assertEqual(hf.target_dims(3840, 2160, 1920), (1920, 1080))
        self.assertEqual(hf.target_dims(1080, 1920, 1920), (1080, 1920))


@unittest.skipUnless(GPU and GIGA.exists(), "GPU and a GigaHands frame are needed")
class CoreAnyCalib(unittest.TestCase):
    def test_half_resize_halves_f(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.png"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(GIGA), "-frames:v", "1", str(p)], check=True)
            im = hf.load_rgb(p)
            half = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            sys.path.insert(0, str(ROOT))
            before = hf.hub_snapshot(PIPE["anycalib"]["torch_hub_dir"])
            with hf.offline(PIPE["anycalib"]["offline_env"]):
                f1 = hf.predict_frames("anycalib_pinhole", "pinhole", [im], "cuda:0")[0, :2].mean()
                f2 = hf.predict_frames("anycalib_pinhole", "pinhole", [half], "cuda:0")[0, :2].mean()
            self.assertEqual(hf.hub_snapshot(PIPE["anycalib"]["torch_hub_dir"]), before)
            self.assertAlmostEqual(f2 / f1, 0.5 * (im.shape[1] // 2) / (im.shape[1] / 2), delta=0.01)

    def test_offline_guard_blocks_download(self):
        import torch.hub
        with hf.offline(PIPE["anycalib"]["offline_env"]):
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            with self.assertRaises(hf.DownloadBlocked):
                torch.hub.download_url_to_file("https://example.invalid/x.pt", "/nonexistent/x.pt")


class CoreEndToEnd(unittest.TestCase):
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
        c = copy.deepcopy(PIPE)
        c["detector"].update({"name": "mock", "mock_npz": str(npz), "weights": "mock"})
        c["anycalib"]["enabled"] = anycalib
        c["uncertainty"]["n_bootstrap"] = 4
        c["output"]["root"] = str(d / "runs")
        hub_before = hf.hub_snapshot(PIPE["anycalib"]["torch_hub_dir"])
        (d / "out").mkdir()
        row = run.process_video(v, c, yaml.safe_dump(c), "test", d / "out", estimator, "test", None, None, anycalib)
        self.assertEqual(hf.hub_snapshot(PIPE["anycalib"]["torch_hub_dir"]), hub_before)
        return d, d / "out" / "clip", [row], v

    def _check_common(self, d, out, rows, v):
        for f in ("results.json", "result.yaml", "report.md", "per_frame.csv", "metadata.json", "log.txt", "detections.npz"):
            self.assertTrue((out / f).exists(), f)
        self.assertFalse(list(out.rglob("*.mkv")) + list(out.rglob("f_[0-9][0-9][0-9][0-9][0-9][0-9].png")) + list(out.rglob("*.mov")))
        self.assertTrue(v.exists())
        texts = "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
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

# ====================================================================================================
# detector workers and per-detector processing (16 code used by 17)
# ====================================================================================================

REF14 = ROOT / CFG17["reuse_14"]["detection_cache"]


def frames_from_giga(d: Path, n=3, dup_first=False):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(GIGA), "-vf", "select=not(mod(n\\,40))", "-vsync", "0",
                    "-frames:v", str(n), str(d / "f_%06d.png")], check=True)
    pngs = sorted(d.glob("f_*.png"))
    if dup_first:
        pngs = [pngs[0], pngs[0]] + pngs[1:]
    return [{"index": i, "png": str(p), "sha256": hf.sha256(p), "frame": i, "time_s": i / 15} for i, p in enumerate(pngs)]


def cfg_tmp(root: Path, detectors):
    c = copy.deepcopy(CFG17)
    c["detectors"] = detectors
    c["output"]["root"] = str(root / "runs")
    c["pipeline"]["output"]["root"] = str(root / "runs")
    return c


class SwapHelpers(unittest.TestCase):
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
        self.assertTrue(hf.order_assignment(a, b, m)["identity"])
        bp = b.copy()
        bp[:, [9, 13]] = b[:, [13, 9]]
        self.assertFalse(hf.order_assignment(a, bp, m)["identity"])

    def test_edge_text(self):
        hi = 3072.0
        self.assertEqual(run.f_text({"f": 3072.0, "at_grid_edge": True}, hi), "값 없음 (상한)")
        self.assertEqual(run.f_text({"f": 672.0, "at_grid_edge": True}, hi), "값 없음 (하한)")
        self.assertEqual(run.f_text({"f": 1650.0, "at_grid_edge": False}, hi, 0.06), "1650.0 px (±6.0%)")
        self.assertEqual(run.spread_text(0.0, True), "계산 불가")
        self.assertEqual(run.spread_text(0.012, False), "1.20%")


@unittest.skipUnless(GIGA.exists() and REF14.exists(), "GigaHands frames and 14's detection cache needed")
class SwapWorkers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hf16t_"))
        cls.frames = frames_from_giga(cls.tmp)
        cls.cfg = cfg_tmp(cls.tmp, ["wilor_yolo", "mediapipe_hands", "rtmpose_hand"])
        cls.out = hf.swap_run_all(cls.frames, cls.cfg, cls.tmp / "cache", log=lambda *a: None)

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
            z = hf.load_detections(d["npz"])
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
        o = hf.swap_run_all(fr, c, d / "cache", log=lambda *a: None)
        z = np.load(o["mediapipe_hands"]["npz"])
        off = z["offsets"]
        a, b = z["kxy"][off[0]:off[1]], z["kxy"][off[1]:off[2]]
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.allclose(a, b))
        self.assertEqual(json.loads(str(z["info"]))["running_mode"], "IMAGE")


# ====================================================================================================
# protocol rerun (17)
# ====================================================================================================

C = CFG17["conditions"]
RUN17 = ROOT / "runs/17_protocol_rerun/20261016T053345Z"       # official 17 run that includes IMG_7141


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
    return hf.metrics_from_arrays(t, p, box, 1920, 1080, th, role, kw.get("transition", 3.1), 0.003, kw.get("axis_bins", 6), kw.get("ext", 0.98))


class ProtocolConditions(unittest.TestCase):
    def test_pass(self):
        r = hf.conditions_check(metrics(), C)
        self.assertTrue(r["guide"], r["reasons"])

    def test_each_threshold_fails(self):
        cases = {"box_short_ratio": dict(box_ratio=0.6), "edge_rate": dict(x0=0), "transition": dict(transition=1.0),
                 "theta_p95": dict(theta_amp=40, theta_noise=6), "theta_tilt_median": dict(theta_amp=15, theta_noise=1),
                 "axis_bins": dict(axis_bins=4), "finger_extension": dict(ext=0.85)}
        for item, kw in cases.items():
            r = hf.conditions_check(metrics(**kw), C)
            self.assertFalse(r["items"][item], item)
            self.assertFalse(r["guide"], item)

    def test_no_transition_fails(self):
        r = hf.conditions_check(metrics(transition=None), C)
        self.assertFalse(r["items"]["transition"])
        self.assertIn("자동 전환 없음", "; ".join(r["reasons"]))


def det(verdict="REJECT", edge=False, sharp=1.1, unc=0.1, rms=0.01, rate=0.9):
    return {"verdict": verdict, "fixed_template": {"f": 1500.0, "at_grid_edge": edge, "side": "상한" if edge else None},
            "joint_delta": {"f": 1500.0, "at_grid_edge": edge, "side": "상한" if edge else None, "delta_deg": 0.0},
            "f_rel_uncertainty": unc, "reproj_rms_rel": rms, "detect_rate": rate, "profiles": {"flat17": {"official": {"sharpness": sharp}}}}


class ProtocolRanking(unittest.TestCase):
    R = CFG17["ranking"]

    def test_verdict_priority_and_edge_below_reject(self):
        r = hf.rank({"a": det("REJECT", sharp=2.0), "b": det("ROUGH", sharp=1.01), "c": det("GOOD", sharp=1.0), "d": det("REJECT", edge=True, sharp=5.0),
                     "e": {"verdict": "REJECT", "failed": "no_template"}}, self.R)
        self.assertEqual([r[k]["rank"] for k in "cbade"], [1, 2, 3, 4, 5])
        self.assertEqual(r["d"]["category"], "EDGE")

    def test_keys_in_order_and_ties(self):
        r = hf.rank({"a": det(sharp=1.2, unc=0.5), "b": det(sharp=1.2, unc=0.1), "c": det(sharp=1.3, unc=0.9)}, self.R)
        self.assertEqual((r["c"]["rank"], r["b"]["rank"], r["a"]["rank"]), (1, 2, 3))
        r = hf.rank({"a": det(), "b": det(), "c": det(sharp=0.5)}, self.R)
        self.assertEqual((r["a"]["rank"], r["b"]["rank"], r["c"]["rank"]), (1.5, 1.5, 3))

    def test_mean_over_guide_videos(self):
        pv = {"v1": hf.rank({"a": det(sharp=2), "b": det(sharp=1)}, self.R), "v2": hf.rank({"a": det(sharp=1), "b": det(sharp=2)}, self.R),
              "v3": hf.rank({"a": det(sharp=2), "b": det(sharp=1)}, self.R)}
        s = hf.rank_summary(pv, {"v1": True, "v2": True, "v3": False})
        self.assertEqual((s["a"]["mean_rank_guide"], s["a"]["n_guide_videos"]), (1.5, 2))


class ProtocolSelection(unittest.TestCase):
    def test_videos_from_yaml_and_override(self):
        c = copy.deepcopy(CFG17)
        c["videos"] = ["IMG_7141.MOV"]
        self.assertEqual([v.name for v in run.select_videos(c)], ["IMG_7141.MOV"])
        self.assertEqual([v.name for v in run.select_videos(c, ["IMG_7142.MOV"])], ["IMG_7142.MOV"])
        c["videos"] = []
        self.assertGreaterEqual(len(run.select_videos(c)), 2)
        c["videos"] = ["NOPE.MOV"]
        with self.assertRaises(SystemExit):
            run.select_videos(c)

    def test_edge_text(self):
        line = run.det_line("x", det(edge=True))
        self.assertIn("값 없음 (상한)", line)
        self.assertNotIn("1500", line)


def cfg17_tmp(root: Path, **kw):
    c = copy.deepcopy(CFG17)
    c["output"]["root"] = str(root / "runs")
    c["pipeline"]["output"]["root"] = str(root / "runs")
    c.update(kw)
    return c


@unittest.skipUnless(VID.exists() and RUN17.exists(), "IMG_7141 and the official 17 run needed")
class ProtocolRuns(unittest.TestCase):
    def test_yaml_detector_and_video_selection_reproduces_17(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg17_tmp(d, detectors=["wilor_yolo"], videos=["IMG_7141.MOV"]), allow_unicode=True))
            root, vids = run.protocol_main(["--config", str(cp), "--no-anycalib", "--out", str(d / "o")])
            self.assertEqual([v["video"] for v in vids], ["IMG_7141.MOV"])
            self.assertEqual(list(vids[0]["detectors"]), ["wilor_yolo"])
            r17 = json.loads((root / "IMG_7141" / "wilor_yolo" / "results.json").read_text())
            ref = json.loads((RUN17 / "IMG_7141" / "wilor_yolo" / "results.json").read_text())
            self.assertEqual(r17["verdict"], ref["verdict"])
            self.assertEqual(r17["f_hand_at_grid_edge"], ref["f_hand_at_grid_edge"])
            self.assertAlmostEqual(r17["f_hand_px"], ref["f_hand_px"], places=6)
            self.assertAlmostEqual(r17["f_joint_delta_px"], ref["f_joint_delta_px"], places=6)
            rep = (root / "report.md").read_text()
            self.assertIn("값 없음 (상한)", rep)
            self.assertIn("계산 불가", rep)
            self.assertIn(run.RANK_NOTE, rep)
            self.assertTrue(vids[0]["previous"])

    def test_no_tags_and_offline_anycalib(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vdir = d / "v"
            vdir.mkdir()
            make_video(vdir / "clip.mov", 640, 360, 6, 30, tags=FAKE_TAGS)
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg17_tmp(d, detectors=["mediapipe_hands"], videos=[], input_dir=str(vdir)), allow_unicode=True))
            before = hf.hub_snapshot(CFG17["pipeline"]["anycalib"]["torch_hub_dir"])
            root, vids = run.protocol_main(["--config", str(cp), "--out", str(d / "o")])
            self.assertEqual(hf.hub_snapshot(CFG17["pipeline"]["anycalib"]["torch_hub_dir"]), before)
            self.assertEqual(vids[0]["anycalib"]["hub_check"]["new_files"], [])
            texts = "".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)

# ====================================================================================================
# frame-filter diagnostic (17)
# ====================================================================================================


class FrameFilterMask(unittest.TestCase):
    def test_rule(self):
        t = np.arange(10) * 0.5                               # 0 .. 4.5 s
        present = np.ones(10, bool)
        theta = np.array([60, 2, 3, 4, 5, 5, 30, 50, 40, np.nan], float)
        box = np.tile([100.0, 100, 300, 300], (10, 1))
        box[8] = [0, 100, 200, 300]                           # touches the left border
        k = hf.ff_keep_mask(t, present, theta, box, 1920, 1080, 3.0, CFGFF["filter"])
        self.assertTrue(k[0])                                 # before the tilt start: untouched
        self.assertTrue(k[6])                                 # 30 deg kept
        self.assertFalse(k[7])                                # 50 deg dropped
        self.assertFalse(k[8])                                # edge dropped
        self.assertFalse(k[9])                                # no theta dropped


SRC = ROOT / CFGFF["sources"]["IMG_7147.MOV"] / "rtmpose_hand"


@unittest.skipUnless(SRC.exists(), "17 run needed")
class NoFilterReproduces17(unittest.TestCase):
    def test_disabled_filter_gives_official_f(self):
        pc = copy.deepcopy(CFG17["pipeline"])
        pc["tracking"]["conf_min"] = float(CFG17["detector_settings"]["rtmpose_hand"]["conf_min"])
        r = hf.ff_run_one(SRC, pc, {"theta_max_deg": 1e9, "exclude_edge": False}, methods_full=())
        off = json.loads((SRC / "results.json").read_text())["analysis"]["solutions"]
        self.assertTrue(r["same_template"])
        self.assertEqual(r["n_tilt_kept"], r["n_tilt_before"])
        self.assertAlmostEqual(r["solutions"]["fixed_template"]["f"], off["fixed_template"]["f"], places=6)

# ====================================================================================================
# same-recording check (18)
# ====================================================================================================


def synth_board_frames(K, W, H, n, tilt, seed=0):
    rng = np.random.default_rng(seed)
    return [{"index": i, "time_s": i / 10, "gray": render(K, W, H, r, t)} for i, (r, t) in enumerate(poses(n, tilt, rng))]


class SessionSegmentation(unittest.TestCase):
    def test_split_and_warnings(self):
        t = np.arange(0, 20, 0.2)
        hand = t < 9
        board = (t >= 11) & (t <= 18)
        s = hf.segment_split(t, hand, board, 0.5)
        self.assertAlmostEqual(s["hand"][1], 10.5, places=6)
        self.assertEqual(s["board"], [11.0, 18.0])
        self.assertEqual(s["warnings"], [])
        hand2 = hand.copy()
        hand2[(t > 12) & (t < 13)] = True                          # a hand inside the board segment
        board2 = board.copy()
        board2[(t > 3) & (t < 3.3)] = True                          # an early stray board detection
        s2 = hf.segment_split(t, hand2, board2, 0.5)
        self.assertAlmostEqual(s2["board"][0], 3.2, places=6)       # the first detection opens the board segment
        self.assertTrue(any("손 검출" in w for w in s2["warnings"]))
        self.assertIn("체커보드가 검출되지 않았습니다", hf.segment_split(t, hand, np.zeros_like(board), 0.5)["warnings"])


class SessionCheckerboard(unittest.TestCase):
    W, H, f = 1080, 1920, 1700.0
    K = np.array([[1700.0, 0, 540.0], [0, 1700.0, 960.0], [0, 0, 1.0]])

    def test_recovers_f_with_tilt(self):
        r = hf.board_analyse(synth_board_frames(self.K, self.W, self.H, 50, (30, 50)), CFG18["checkerboard"], self.W, self.H)
        self.assertLess(abs(r["f_reference_px"] / self.f - 1), 0.01)
        self.assertLess(abs(r["f_model_E_px"] / self.f - 1), 0.01)

    def test_small_tilt_not_good(self):
        r = hf.board_analyse(synth_board_frames(self.K, self.W, self.H, 40, (0, 5), seed=1), CFG18["checkerboard"], self.W, self.H)
        self.assertNotEqual(r["verdict"], "GOOD")
        self.assertIn("tilt_p90", r["failed_good"])


def ffv1_testsrc(path, seconds, fps=30, size="320x240"):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}", "-t", str(seconds),
                    "-c:v", "ffv1", str(path)], check=True)


class SessionCutAndResolution(unittest.TestCase):
    def test_cut_keeps_frame_numbers_and_times(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ffv1_testsrc(d / "a.mkv", 4)
            info = hf.cut_segment(d / "a.mkv", d / "b.mkv", 1.0, 2.5, 30.0)
            fa, _ = hf.extract_frames(d / "a.mkv", d / "fa", 30.0, None, 30)
            fb, _ = hf.extract_frames(d / "b.mkv", d / "fb", 30.0, None, 30)
            self.assertEqual(len(fb), 45)
            self.assertEqual(info["first_frame"], 30)
            for i, r in enumerate(fb):
                self.assertEqual(r["sha256"], fa[i + 30]["sha256"])
                self.assertAlmostEqual(r["time_s"] + info["t0"], fa[i + 30]["time_s"], places=6)
            self.assertAlmostEqual(hf.probe(d / "b.mkv")["fps"], 30.0, places=3)

    def test_board_f_in_normalised_pixels(self):
        """A 3840x2160 recording with f = 3400 px is normalised to 1920x1080; the board f must come out in that resolution (~1700)."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            W, H, f = 3840, 2160, 3400.0
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            for fr in synth_board_frames(K, W, H, 30, (30, 50), seed=2):
                cv2.imwrite(str(d / f"s_{fr['index']:04d}.png"), fr["gray"])
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10", "-i", str(d / "s_%04d.png"), "-c:v", "libx264", "-crf", "8",
                            "-pix_fmt", "yuv420p", str(d / "v.mp4")], check=True)
            n = hf.normalize(d / "v.mp4", d / "n.mkv", CFG18["pipeline"])
            self.assertEqual([n["output"]["display_w"], n["output"]["display_h"]], [1920, 1080])
            fr = run.session_board_frames(d / "n.mkv", n["output"]["fps"], [0.0, 100.0], CFG18["checkerboard"], d)
            r = hf.board_analyse(fr, CFG18["checkerboard"], 1920, 1080)
            self.assertEqual(r["image_size"], [1920, 1080])
            self.assertLess(abs(r["f_reference_px"] / 1700.0 - 1), 0.01)


class SessionEndToEnd(unittest.TestCase):
    def test_no_tags_offline_anycalib_and_segments(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vdir = d / "v"
            vdir.mkdir()
            W, H, f = 1920, 1080, 1600.0
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            for fr in synth_board_frames(K, W, H, 40, (30, 50), seed=3):
                cv2.imwrite(str(d / f"s_{fr['index']:04d}.png"), cv2.cvtColor(fr["gray"], cv2.COLOR_GRAY2BGR))
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-t", "7", "-i", f"testsrc=size={W}x{H}:rate=10",
                            "-framerate", "10", "-i", str(d / "s_%04d.png"), "-filter_complex", "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]concat=n=2:v=1[v]",
                            "-map", "[v]", "-c:v", "libx264", "-crf", "12", str(d / "tmp.mp4")], check=True)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(d / "tmp.mp4"), "-c", "copy"]
            for k, v in FAKE_TAGS.items():
                cmd += ["-metadata", f"{k}={v}"]
            subprocess.run(cmd + [str(vdir / "clip.mov")], check=True)
            c = copy.deepcopy(CFG18)
            c.update({"videos": ["clip.mov"], "input_dir": str(vdir), "detectors": ["mediapipe_hands"]})
            c["output"]["root"] = str(d / "runs")
            c["pipeline"]["output"]["root"] = str(d / "runs")
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(c, allow_unicode=True))
            before = hf.hub_snapshot(CFG18["pipeline"]["anycalib"]["torch_hub_dir"])
            root, vids = run.session_main(["--config", str(cp), "--out", str(d / "o")])
            self.assertEqual(hf.hub_snapshot(CFG18["pipeline"]["anycalib"]["torch_hub_dir"]), before)
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


# ====================================================================================================
# phase segmentation (20)
# ====================================================================================================

RUN17_7158 = ROOT / CFG20["sources"]["IMG_7158.MOV"]
EST_IDS = list(PIPE["joints"]["sets"][PIPE["joints"]["estimate_set"]])


def pseg_seq(tilt_fn, seconds=8.0, fps=15.0, noise=0.3, seed=0, f=1650.0, W=1920, H=1080, dist=600.0, spike=None):
    """Planar-hand keypoint track at 15 fps: tilt_fn(t) -> (theta deg, azimuth deg); spike = (t0, t1, theta) overrides the tilt in that interval."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / fps)
    P = hf.synth_hand_points(0.0)
    P[:, 1] += 95.0
    ids = [0] + list(range(5, 21))
    kp = np.full((len(t), 21, 2), np.nan)
    for i, ti in enumerate(t):
        th, az = tilt_fn(ti)
        if spike is not None and spike[0] <= ti < spike[1]:
            th, az = spike[2], 20.0
        kp[i, ids] = hf.synth_project(P[ids], hf.synth_pose(th, az, 0.0), np.array([0.0, 0.0, dist]), f, W / 2, H / 2) + rng.normal(0, noise, (17, 2))
    kp[:, 1:5] = kp[:, [5]] + np.array([-40.0, 20.0])
    n = len(t)
    mask, present = np.ones((n, 21), bool), np.ones(n, bool)
    box = np.c_[np.nanmin(kp, 1) - 10, np.nanmax(kp, 1) + 10]
    return t, kp, mask, present, box


def pseg_detect(seq, p=None):
    """The transition steps of hf.pseg_analyze without the (slow) f estimates: front window, template, linear_d transition."""
    t, kp, mask, present, box = seq
    fw = hf.front_window(t, present, kp, mask, box, PIPE["phases"])
    fr = [int(i) for i in fw["idx"] if present[i] and mask[i][EST_IDS].all()]
    T = hf.template_build(kp[fr][:, EST_IDS], int(PIPE["joints"]["template_align_iters"]))["template"]
    return hf.pseg_transition(t, present, kp, mask, box, T, EST_IDS, fw["idx"], p or CFG20["phase_segmentation"])


class PsegSynthetic(unittest.TestCase):
    def test_still_sequence_has_no_transition(self):
        r = pseg_detect(pseg_seq(lambda t: (0.0, 0.0), noise=0.5, seed=1))
        self.assertIsNone(r["t_transition"])
        self.assertGreater(r["d_thr"], 0.0)

    def test_ramp_from_t0_is_found_within_0_3_s(self):
        """Tilt rising from 0 to 30 deg over 0.4 s from t0: the transition is found within 0.3 s of t0 (theta has to reach ~15-20 deg first, then at most
        one 15 fps frame of delay), for several azimuths and noise seeds."""
        t0 = 4.0
        for az in (0.0, 70.0, 110.0, 160.0):
            for seed in (2, 5):
                r = pseg_detect(pseg_seq(lambda t: (float(np.clip(30.0 * (t - t0) / 0.4, 0.0, 30.0)), az), seed=seed))
                self.assertIsNotNone(r["t_transition"], (az, seed))
                self.assertGreaterEqual(r["t_transition"], t0 - 1e-6)
                self.assertLessEqual(r["t_transition"], t0 + 0.3, (az, seed, r["t_transition"]))

    def test_slow_ramp_needs_the_motion_condition_off(self):
        """A 1 s ramp has little keypoint speed at its start: with the default motion condition it can be missed; without it the transition is found
        where d crosses its threshold (a documented property of the rule, not a tuning target)."""
        t0 = 4.0
        sq = pseg_seq(lambda t: (float(np.clip(30.0 * (t - t0) / 1.0, 0.0, 30.0)), 20.0), seed=2)
        r = pseg_detect(sq, dict(CFG20["phase_segmentation"], use_motion=False))
        self.assertIsNotNone(r["t_transition"])
        self.assertTrue(t0 + 0.4 <= r["t_transition"] <= t0 + 0.8, r["t_transition"])

    def test_short_spike_is_not_a_transition(self):
        r = pseg_detect(pseg_seq(lambda t: (0.0, 0.0), seed=3, spike=(5.0, 5.2, 30.0)))
        self.assertIsNone(r["t_transition"])

    def test_motion_condition_rejects_a_drift_without_motion_and_can_be_switched_off(self):
        """A tilt change that happens with (almost) no keypoint motion at its start is rejected by the motion condition and found without it."""
        seq = pseg_seq(lambda t: (0.0, 0.0), seed=4)
        t, kp, mask, present, box = seq
        base = pseg_detect(seq)
        self.assertIsNone(base["t_transition"])
        p = dict(CFG20["phase_segmentation"], use_motion=True, k_v=1e6)          # an unreachable speed threshold
        sq = pseg_seq(lambda t: (float(np.clip(30.0 * (t - 4.0) / 0.5, 0.0, 30.0)), 20.0), seed=2)
        self.assertIsNone(pseg_detect(sq, p)["t_transition"])
        self.assertIsNotNone(pseg_detect(sq, dict(p, use_motion=False))["t_transition"])


class PsegFunctions(unittest.TestCase):
    def test_d_is_one_minus_cos_theta(self):
        rng = np.random.default_rng(0)
        T = rng.normal(size=(17, 2)) * 60
        for th in (0.0, 10.0, 25.0, 50.0):
            c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
            Rm = np.array([[c, -s], [s, c]])
            P = (1.7 * (Rm @ np.diag([1.0, np.cos(np.radians(th))]) @ T.T)).T + [300.0, 200.0]        # tilt = compression along one axis, plus a similarity
            self.assertAlmostEqual(1.0 - hf.affine_ratio(T, P), 1.0 - np.cos(np.radians(th)), places=9)

    def test_median_filter_removes_short_spikes_only(self):
        t = np.arange(0, 4, 1 / 15)
        x = np.zeros(len(t))
        x[20:23] = 1.0                                     # 0.2 s
        self.assertLessEqual(int((hf.pseg_median_filter(t, x, 0.3) > 0.5).sum()), 3)      # a 0.2 s spike stays <= 3 samples (< the 0.5 s sustain)
        self.assertEqual(int((hf.pseg_median_filter(t, np.r_[x[:20], np.zeros(len(x) - 20)], 0.3) > 0.5).sum()), 0)
        x1 = np.zeros(len(t))
        x1[20] = 1.0                                                                    # a single-frame spike is removed completely
        self.assertEqual(float(hf.pseg_median_filter(t, x1, 0.3).max()), 0.0)
        y = np.zeros(len(t))
        y[20:35] = 1.0                                     # 1 s
        self.assertEqual(hf.pseg_median_filter(t, y, 0.3)[27], 1.0)
        self.assertEqual(hf.pseg_median_filter(t, y, 0.3)[5], 0.0)

    def test_tilt_keep_and_wrist_cfg(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        present = np.ones(5, bool)
        theta = np.array([60.0, 30.0, 50.0, 40.0, np.nan])
        keep = hf.pseg_tilt_keep(t, present, theta, 3.0, 45.0)
        self.assertEqual(keep.tolist(), [True, True, False, True, True])            # only tilt-phase frames above 45 deg are dropped
        c = hf.pseg_wrist_cfg(PIPE)
        self.assertTrue(all(0 not in ids for ids in c["joints"]["sets"].values()))
        self.assertIn(0, PIPE["joints"]["sets"]["flat17"])                          # the original settings are untouched


@unittest.skipUnless(RUN17_7158.exists(), "the official 17 run of IMG_7158 is needed")
class PsegReal(unittest.TestCase):
    """17's own detections: legacy_theta and the default f / verdict reproduce 17 exactly (options off)."""

    @classmethod
    def setUpClass(cls):
        c20 = dict(CFG20["phase_segmentation"], variants={})
        cls.job = hf.pseg_job(("IMG_7158.MOV", "mediapipe_hands", str(RUN17_7158), PIPE, c20, CFG20["conditions"], 0.5, "legacy_theta"))

    def test_legacy_theta_transition_and_default_f_match_17(self):
        s = self.job["same_as_17"]
        self.assertTrue(s["legacy_transition"] and s["f_fixed_template"] and s["f_joint_delta"] and s["verdict"], s)
        self.assertAlmostEqual(self.job["legacy_theta"]["t_transition"], 2.5333333333, places=6)
        self.assertEqual(self.job["t_selected"], self.job["legacy_theta"]["t_transition"])

    def test_legacy_guide_matches_17(self):
        r17 = json.loads((RUN17_7158.parent / "results.json").read_text())
        ref = [v for v in r17["videos"] if v["video"] == "IMG_7158.MOV"][0]["conditions"]["by_detector"]["mediapipe_hands"]
        self.assertEqual(self.job["legacy_theta"]["guide"], ref["guide"])
        self.assertEqual(self.job["legacy_theta"]["reasons"], ref["reasons"])

    def test_both_rules_and_diagnostics_are_reported(self):
        for k in ("t_transition", "d0", "d_thr", "v0", "v_thr", "guide"):
            self.assertIn(k, self.job["linear_d"])
        self.assertIn("t_transition", self.job["linear_d_no_motion"])
        self.assertEqual(self.job["variants"], {})


@unittest.skipUnless(RUN17_7158.exists(), "the official 17 run of IMG_7158 is needed")
class PsegEndToEnd(unittest.TestCase):
    def test_outputs_and_no_location_or_device_tags(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG20)
            cfg["phase_segmentation"]["variants"] = {"max_tilt": {"exclude_large_tilt": True}}
            cfg["workers"] = 1
            cfg["output"]["root"] = str(d / "runs")
            cfg["output"]["results_yaml"] = str(d / "r20.yaml")                     # never the tracked results file
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, jobs = run.pseg_main(["--config", str(cp), "--videos", "IMG_7158.MOV", "--detectors", "mediapipe_hands", "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            self.assertTrue((root / "figures" / "IMG_7158_mediapipe_hands.png").exists())
            y = run.pseg_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r20.yaml")
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            self.assertIn("수동 기준이 없다", (root / "report.md").read_text())          # the manual slots are empty and are never filled in
            self.assertNotIn("1680", Path(y).read_text())                              # the comparison with the reference f is in summary.md only
            self.assertIn("1680", (root / "summary.md").read_text())
            self.assertEqual(sorted(jobs[0]["variants"]), ["max_tilt"])


# ====================================================================================================
# guide revision (21)
# ====================================================================================================

def cjob21(cfg=CFG21, modes=("keep", "exclude", "downweight"), **over):
    c = {k: copy.deepcopy(cfg[k]) for k in ("phase_segmentation", "conditions", "f_estimation", "reference_f_px", "compare_20", "transition_method") if k in cfg}
    c["f_estimation"] = dict(c.get("f_estimation") or {}, compare_wrist_modes=list(modes))
    c.update(over)
    return c


def synth_wrist(wrist_xy=None, seed=3, f=1650.0, noise=0.3, tilt_frames=40):
    """Synthetic planar hand (17 points, tilt 30 deg, noise 0.3 px); optionally a wrist point replaced by wrist_xy (px) in every frame."""
    g = synth({"noise": {"px": noise}, "tilt_range_deg": [30, 30]}, seed=seed, scfg=dict(SYN, f_px=f, tilt_frames=tilt_frames))
    if wrist_xy is not None:
        g["kp"][:, 0] = np.asarray(wrist_xy, float)
        g["mask"][:, 0] = True
    return g


class GrevWristModes(unittest.TestCase):
    def test_synthetic_hand_recovers_f_within_1_percent_in_all_three_modes(self):
        """120 tilt frames (the random poses and distances of a draw already move f by up to ~1% with 40 frames, for 17's keep mode too), seeds 0-2."""
        for seed in (0, 1, 2):
            g = synth_wrist(seed=seed, tilt_frames=120)
            for mode, kw in (("keep", {}), ("exclude", {"cfg": hf.pseg_wrist_cfg(PIPE)}), ("downweight", {"wrist_weight": 0.3})):
                a = analyze_synth(g, methods_full=(), **kw)
                self.assertLess(abs(a["solutions"]["fixed_template"]["f"] / g["f_true"] - 1.0), 0.01, (seed, mode, a["solutions"]["fixed_template"]["f"]))

    def test_keep_is_the_old_analysis_and_weight_one_equals_keep(self):
        g = synth_wrist()
        a, b = analyze_synth(g, methods_full=("fixed_template",)), analyze_synth(g, methods_full=("fixed_template",), wrist_weight=1.0)
        for m in ("fixed_template", "joint_delta"):
            self.assertEqual(a["solutions"][m]["f"], b["solutions"][m]["f"])
        pm_a, pm_b = a["per_method"]["fixed_template"], b["per_method"]["fixed_template"]
        self.assertEqual(pm_a["f_rel_uncertainty"], pm_b["f_rel_uncertainty"])
        self.assertEqual(pm_a["diagnostics"]["sharpness"], pm_b["diagnostics"]["sharpness"])
        self.assertEqual(pm_a["verdict"]["verdict"], pm_b["verdict"]["verdict"])

    def test_exclude_never_uses_the_wrist_point(self):
        """The wrist replaced by two different garbage points gives bit-identical results in exclude mode (template, cost, jointsets, bootstrap, consistency), but not in keep."""
        ce = hf.pseg_wrist_cfg(PIPE)
        r = {}
        for name, xy in (("a", (50.0, 60.0)), ("b", (1700.0, 900.0))):
            g = synth_wrist(xy)
            r[name] = (analyze_synth(g, methods_full=("fixed_template",), cfg=ce), analyze_synth(g, methods_full=("fixed_template",)))
        ea, eb = r["a"][0], r["b"][0]
        self.assertEqual(ea["template"], eb["template"])
        for m in ("fixed_template", "joint_delta"):
            self.assertEqual(ea["solutions"][m]["f"], eb["solutions"][m]["f"])
        pa, pb = ea["per_method"]["fixed_template"], eb["per_method"]["fixed_template"]
        self.assertEqual(pa["f_rel_uncertainty"], pb["f_rel_uncertainty"])
        self.assertEqual(pa["jointset_range"], pb["jointset_range"])
        self.assertEqual(pa["diagnostics"]["sharpness"], pb["diagnostics"]["sharpness"])
        self.assertNotIn(0, pa["diagnostics"]["per_joint_rms_px"])
        self.assertNotEqual(r["a"][1]["solutions"]["fixed_template"]["f"], r["b"][1]["solutions"]["fixed_template"]["f"])      # keep does see the wrist

    def test_downweight_shrinks_the_wrist_outlier_effect(self):
        g = synth_wrist((960.0, 700.0))                    # a wrist point far from where it belongs, in every frame
        f = {m: analyze_synth(g, methods_full=(), **kw)["solutions"]["fixed_template"]["f"] for m, kw in (("keep", {}), ("down", {"wrist_weight": 0.3}))}
        self.assertNotEqual(f["keep"], f["down"])


class GrevTransition(unittest.TestCase):
    def test_front_search_range_is_respected(self):
        """Keypoint jitter falls steadily over the first 3 s: the lowest-motion second is the latest one. Searching to 3.0 s ends it at 3.0, searching to 2.5 s ends it at 2.5."""
        t = np.arange(0, 8.0, 1 / 15)
        rng = np.random.default_rng(0)
        seq = pseg_seq(lambda t_: (0.0, 0.0), noise=0.0, seed=0)
        _, kp, mask, present, box = seq
        kp = kp + rng.normal(0, 1.0, kp.shape) * np.clip(3.2 - t, 0.05, 3.0)[:, None, None]
        box = np.c_[np.nanmin(kp, 1) - 10, np.nanmax(kp, 1) + 10]
        wide = hf.grev_front_window(t, present, kp, mask, box, PIPE["phases"], [0.5, 3.0])
        narrow = hf.grev_front_window(t, present, kp, mask, box, PIPE["phases"], [0.5, 2.5])
        self.assertGreater(wide["t1"], 2.5)
        self.assertLessEqual(narrow["t1"], 2.5 + 1e-9)
        self.assertGreaterEqual(narrow["t0"], 0.5 - 1e-9)
        self.assertEqual(hf.front_window(t, present, kp, mask, box, PIPE["phases"])["t1"], wide["t1"])      # [0.5, 3.0] = 17's window

    def test_d0_cap_limits_the_threshold_and_warns(self):
        sq = pseg_seq(lambda t: (0.0, 0.0) if t < 2.0 else (float(min(25.0, 25.0 * (t - 2.0) / 1.0)), 20.0), seed=1)
        t, kp, mask, present, box = sq
        fw = hf.front_window(t, present, kp, mask, box, PIPE["phases"])
        fr = [int(i) for i in fw["idx"]]
        T = hf.template_build(kp[fr][:, EST_IDS], int(PIPE["joints"]["template_align_iters"]))["template"]
        tilted = [i for i in range(len(t)) if 2.6 <= t[i] < 3.6]                          # a "front window" that already contains the tilt
        p = CFG21["phase_segmentation"]
        capped = hf.grev_transition(t, present, kp, mask, box, T, EST_IDS, tilted, p)
        cap = 1 - np.cos(np.radians(p["d0_cap_theta_deg"]))
        self.assertTrue(capped["d0_capped"])
        self.assertGreater(capped["d0"], cap)
        self.assertAlmostEqual(capped["d_thr"], max(p["k_d"] * cap, 1 - np.cos(np.radians(p["theta_min_deg"]))), places=12)
        self.assertLess(capped["d_thr"], p["k_d"] * capped["d0"])
        self.assertIn("정면 창에 기울임이 섞였을 수 있음", "; ".join(capped["warnings"]))
        free = hf.grev_transition(t, present, kp, mask, box, T, EST_IDS, tilted, {k: v for k, v in p.items() if k != "d0_cap_theta_deg"})
        self.assertFalse(free["d0_capped"])
        self.assertAlmostEqual(free["d_thr"], p["k_d"] * free["d0"], places=12)           # no cap key = 20's rule
        calm = hf.grev_transition(t, present, kp, mask, box, T, EST_IDS, fr, p)
        self.assertFalse(calm["d0_capped"])
        self.assertEqual(calm["warnings"], [])

    def test_without_the_motion_condition_a_slow_ramp_is_found(self):
        sq = pseg_seq(lambda t: (float(np.clip(30.0 * (t - 4.0) / 1.0, 0.0, 30.0)), 20.0), seed=2)
        t, kp, mask, present, box = sq
        fw = hf.grev_front_window(t, present, kp, mask, box, PIPE["phases"], CFG21["phase_segmentation"]["front_search_s"])
        T = hf.template_build(kp[[int(i) for i in fw["idx"]]][:, EST_IDS], int(PIPE["joints"]["template_align_iters"]))["template"]
        r = hf.grev_transition(t, present, kp, mask, box, T, EST_IDS, fw["idx"], CFG21["phase_segmentation"])
        self.assertFalse(CFG21["phase_segmentation"]["use_motion"])
        self.assertTrue(4.4 <= r["t_transition"] <= 4.9, r["t_transition"])


def g21_metrics(box_ratio=0.4, n_tilt=60, theta_amp=30.0, n_extra_high=0, axis_bins=6, ext=0.98, transition=3.1, x0=700):
    """Metrics of a trajectory: n_tilt tilt-phase frames around theta_amp (deg), plus n_extra_high frames at 70 deg."""
    n_tot = n_tilt + n_extra_high + 45
    t = np.arange(n_tot) / 15.0
    present = np.ones(n_tot, bool)
    side = box_ratio * 1080
    box = np.tile([x0, 300, x0 + side, 300 + side], (n_tot, 1)).astype(float)
    theta = np.r_[np.full(45, 1.0), theta_amp + np.sin(np.arange(n_tilt)) * 3.0, np.full(n_extra_high, 70.0)]
    role = np.array(["front"] * 45 + ["tilt"] * (n_tilt + n_extra_high), dtype=object)
    return hf.grev_metrics(t, present, box, 1920, 1080, theta, role, transition, 0.003, axis_bins, ext, 45.0)


class GrevGuide(unittest.TestCase):
    C21 = CFG21["conditions"]

    def test_pass(self):
        r = hf.grev_conditions_check(g21_metrics(), self.C21)
        self.assertTrue(r["guide"], r["reasons"])
        self.assertEqual(sorted(r["items"]), sorted(["box_short_ratio", "edge_rate", "transition", "tilt_frames", "theta_tilt_median", "axis_bins", "finger_extension"]))
        self.assertNotIn("theta_p95", r["items"])

    def test_each_item_fails_as_intended(self):
        cases = {"box_short_ratio": dict(box_ratio=0.65), "edge_rate": dict(x0=0), "transition": dict(transition=1.0), "tilt_frames": dict(n_tilt=29),
                 "theta_tilt_median": dict(theta_amp=15.0), "axis_bins": dict(axis_bins=4), "finger_extension": dict(ext=0.85)}
        for item, kw in cases.items():
            r = hf.grev_conditions_check(g21_metrics(**kw), self.C21)
            self.assertFalse(r["items"][item], item)
            self.assertFalse(r["guide"], item)
            self.assertTrue(all(v for k, v in r["items"].items() if k != item), (item, r["items"]))

    def test_box_limit_is_0_6_not_0_5_and_no_transition_fails(self):
        m = g21_metrics(box_ratio=0.55)
        self.assertTrue(hf.grev_conditions_check(m, self.C21)["items"]["box_short_ratio"])
        self.assertFalse(hf.grev_conditions_check(m, CFG20["conditions"])["items"]["box_short_ratio"])
        r = hf.grev_conditions_check(g21_metrics(transition=None), self.C21)
        self.assertFalse(r["items"]["transition"])
        self.assertIn("자동 전환 없음", "; ".join(r["reasons"]))

    def test_large_tilt_frames_do_not_count_and_do_not_fail_the_guide(self):
        """Frames above 45 deg neither count toward the 30 frames nor into the median (the old p95 limit is gone); only the frames <= 45 deg decide."""
        m = g21_metrics(n_tilt=30, n_extra_high=40)
        self.assertEqual(m["tilt_frames_le_max"], 30)
        self.assertEqual(m["tilt_frames"], 70)
        self.assertTrue(hf.grev_conditions_check(m, self.C21)["guide"])
        self.assertFalse(hf.grev_conditions_check(m, CFG20["conditions"])["guide"])            # 20's p95 rule rejects the same metrics
        self.assertFalse(hf.grev_conditions_check(g21_metrics(n_tilt=29, n_extra_high=40), self.C21)["items"]["tilt_frames"])

    def test_tilt_rule_default_is_the_old_check(self):
        m = g21_metrics()
        self.assertEqual(hf.grev_conditions_check(m, CFG20["conditions"]), hf.conditions_check(m, CFG20["conditions"]))


def adoption_jobs(keep, mode):
    """keep / mode: lists of 15 fixed_template f (None = grid edge), pairs ordered video-major over 5 videos x 3 detectors."""
    jobs = []
    for i in range(15):
        j = {"video": f"V{i // 3}.MOV", "detector": f"d{i % 3}", "wrist": {}}
        for name, fv in (("keep", keep[i]), ("mode", mode[i])):
            j["wrist"][name] = {"solutions": {"fixed_template": {"f": 3072.0 if fv is None else fv, "at_grid_edge": fv is None}}}
        jobs.append(j)
    return jobs


class GrevAdoption(unittest.TestCase):
    R = CFG21["wrist_adoption"]

    def run_rules(self, keep, mode):
        return hf.grev_adoption(adoption_jobs(keep, mode), 1000.0, self.R, "mode")

    def test_all_rules_pass_and_each_can_fail(self):
        keep = [1300.0] * 15
        mode = [1100.0] * 15
        a = self.run_rules(keep, mode)
        self.assertTrue(a["candidate"], a["rules"])
        self.assertEqual(a["all"]["improved"], 15)
        b = self.run_rules(keep, [1100.0] * 9 + [1400.0] * 6)                           # 9 improved < 10
        self.assertFalse(b["rules"]["2_improved_pairs"])
        self.assertFalse(b["candidate"])
        c = self.run_rules(keep, [1250.0] * 15)                                          # median error 25% vs 30%: improvement too small
        self.assertTrue(c["rules"]["2_improved_pairs"])
        self.assertFalse(c["rules"]["3_median_drop"])
        d = self.run_rules([1300.0] * 14 + [None], [1100.0] * 14 + [None])               # edge count equal passes; more edges fails
        self.assertTrue(d["rules"]["1_edge_not_more"])
        e = self.run_rules(keep, [1100.0] * 14 + [None])
        self.assertFalse(e["rules"]["1_edge_not_more"])

    def test_leave_one_out_and_worst_video_rules(self):
        keep = [1100.0] * 15
        mode = [1000.0] * 15                                   # mode is better everywhere ...
        mode[0:3] = [1500.0] * 3                               # ... except one video where all three detectors get 30 pp worse
        a = self.run_rules(keep, mode)
        self.assertEqual(a["worse_videos"], ["V0.MOV"])
        self.assertFalse(a["rules"]["5_no_video_all_worse"])
        self.assertFalse(a["candidate"])
        keep2 = [1500.0] * 4 + [1100.0] * 11
        mode2 = [1000.0] * 4 + [1200.0] * 11                   # improvement comes from one video only: dropping it breaks the direction
        f = self.run_rules(keep2, mode2)
        self.assertEqual(f["leave_one_out"]["V0.MOV"]["direction_2"], False)

    def test_value_lost_counts_as_worse_and_value_gained_as_better(self):
        a = self.run_rules([1100.0, None] + [1100.0] * 13, [None, 1100.0] + [1100.0] * 13)
        self.assertEqual((a["all"]["improved"], a["all"]["worsened"]), (1, 1))


class GrevOverlayText(unittest.TestCase):
    def test_time_and_frame_number_match_the_frame_list(self):
        r17 = RUN17_7158 / "mediapipe_hands" / "results.json"
        if not r17.exists():
            self.skipTest("17 run needed")
        res = json.loads(r17.read_text())
        fps = res["normalization"]["output"]["fps"]
        for fr in (res["frames"]["list"][0], res["frames"]["list"][37], res["frames"]["list"][-1]):
            lines = run.grev_overlay_lines(fr, 0.0456, 0.101, 2.93, 5.87, 2.53)
            self.assertAlmostEqual(fr["frame"] / fps, fr["time_s"], places=9)
            self.assertIn(f"t {fr['time_s']:.2f} s", lines[0])
            self.assertIn(f"frame {fr['frame']} (#{fr['index']})", lines[0])
            self.assertEqual(lines[1], "d 0.046   thr 0.101")
            self.assertEqual(lines[2], "new 2.93 s   20 5.87 s   legacy 2.53 s")
        self.assertEqual(run.grev_overlay_lines(res["frames"]["list"][0], np.nan, None, None, None, None)[1:], ["d -   thr -", "new none   20 none   legacy none"])


@unittest.skipUnless(RUN17_7158.exists() and (Path(CFG17["input_dir"]) / "IMG_7158.MOV").exists(), "the 17 run and the input video of IMG_7158 are needed")
class GrevReal(unittest.TestCase):
    """17's own detections: with 20's settings the new code reproduces 20's transition, thresholds, guide and the default f; the keep mode equals 17."""

    @classmethod
    def setUpClass(cls):
        c20 = {"phase_segmentation": CFG20["phase_segmentation"], "conditions": CFG20["conditions"], "f_estimation": {"compare_wrist_modes": ["keep"]},
               "transition_method": "linear_d"}
        cls.job20 = hf.grev_job(("IMG_7158.MOV", "mediapipe_hands", str(RUN17_7158), PIPE, c20, 0.5, None))
        cls.ref20 = hf.pseg_job(("IMG_7158.MOV", "mediapipe_hands", str(RUN17_7158), PIPE, dict(CFG20["phase_segmentation"], variants={}), CFG20["conditions"], 0.5, "linear_d"))

    def test_20_settings_reproduce_20(self):
        a, b = self.job20["transition"], self.ref20["linear_d"]
        for k in ("t_transition", "d0", "d_thr", "v0", "v_thr", "guide", "reasons", "skipped_no_motion_s"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(self.job20["legacy_theta"]["t_transition"], self.ref20["legacy_theta"]["t_transition"])
        self.assertEqual(self.job20["legacy_theta"]["guide"], self.ref20["legacy_theta"]["guide"])
        self.assertEqual(self.job20["transition_no_motion"]["t_transition"], self.ref20["linear_d_no_motion"]["t_transition"])
        self.assertEqual(self.job20["default"]["solutions"], self.ref20["default"]["solutions"])
        self.assertEqual(self.job20["default"]["verdict"], self.ref20["default"]["verdict"])
        self.assertNotIn("transition_20", self.job20)

    def test_keep_mode_equals_17(self):
        s = self.job20["same_as_17"]
        self.assertTrue(s["f_fixed_template"] and s["f_joint_delta"] and s["verdict"] and s["keep_equals_default"], s)


@unittest.skipUnless(RUN17_7158.exists() and (Path(CFG17["input_dir"]) / "IMG_7158.MOV").exists(), "the 17 run and the input video of IMG_7158 are needed")
class GrevEndToEnd(unittest.TestCase):
    def test_outputs_overlay_and_no_location_or_device_tags(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG21)
            cfg["workers"] = 1
            cfg["videos"] = ["IMG_7158.MOV"]
            cfg["output"]["root"] = str(d / "runs")
            cfg["output"]["results_yaml"] = str(d / "r21.yaml")                     # never the tracked results file
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, jobs = run.grev_main(["--config", str(cp), "--detectors", "mediapipe_hands", "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            self.assertTrue((root / "figures" / "IMG_7158_mediapipe_hands.png").exists())
            y = run.grev_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r21.yaml")
            yy = yaml.safe_load(Path(y).read_text())
            e = yy["videos"]["IMG_7158.MOV"]
            self.assertIsNone(e["manual_transition_s"])
            self.assertEqual(sorted(e["detectors"]["mediapipe_hands"]["wrist"]), ["downweight", "exclude", "keep"])
            made = run.grev_main(["--config", str(cp), "overlay", "--run", str(root), "--videos", "IMG_7158.MOV"])
            cap = cv2.VideoCapture(made[0])
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok and float(frame.std()) > 5.0)
            self.assertEqual(n_video, jobs[0]["n_frames"])
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            self.assertIn("수동 전환 시각(manual_transition_s)이 비어 있다", (root / "report.md").read_text())
            self.assertNotIn("1680", Path(y).read_text())                              # the comparison with the reference f is in summary.md only
            self.assertNotIn("f_signal", Path(y).read_text())
            self.assertIn("1680", (root / "summary.md").read_text())
            self.assertNotIn("1680", (root / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
