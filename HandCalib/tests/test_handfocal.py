"""Tests of handfocal.py, run.py and detector_worker.py (consolidated from the 14/16/17/18 test files; 16's command-line tests were dropped with its command).

    python -m unittest HandCalib/tests/test_handfocal.py -v
"""
import copy
import json
import os
import re
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
CFG22 = yaml.safe_load((ROOT / "configs/22_f_response.yaml").read_text())
CFG23 = yaml.safe_load((ROOT / "configs/23_error_factors.yaml").read_text())
CFG25 = yaml.safe_load((ROOT / "configs/25_long_video_check.yaml").read_text())
CFG26 = yaml.safe_load((ROOT / "configs/26_generalization.yaml").read_text())
CFG27 = yaml.safe_load((ROOT / "configs/27_main_camera_extension.yaml").read_text())
CFG29 = yaml.safe_load((ROOT / "configs/29_wide_60cm.yaml").read_text())
CFG30 = yaml.safe_load((ROOT / "configs/30_jointset_sets.yaml").read_text())
CFG31 = yaml.safe_load((ROOT / "configs/31_verdict_calibration.yaml").read_text())
CFG34 = yaml.safe_load((ROOT / "configs/34_printed_hand.yaml").read_text())
CFG35 = yaml.safe_load((ROOT / "configs/35c_hand_markers_tracking.yaml").read_text())
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


# ====================================================================================================
# f response (22)
# ====================================================================================================

def blob_image(W, H, centres, sigma=6.0):
    """Gaussian blobs at continuous pixel-index positions (x, y)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.zeros((H, W))
    for cx, cy in centres:
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)


def centroid(img, box):
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((sub * xx).sum() / sub.sum()), float((sub * yy).sum() / sub.sum())


class FrespZoom(unittest.TestCase):
    W, H = 1920, 1080

    def test_centre_crop_keeps_the_principal_point_and_scales_offsets(self):
        """A blob on the optical axis (pixel-index centre (W-1)/2, (H-1)/2) stays there; a blob d away from it ends up s * d away."""
        cx, cy = (self.W - 1) / 2, (self.H - 1) / 2
        img = np.dstack([blob_image(self.W, self.H, [(cx, cy), (cx + 300, cy - 120)])] * 3)
        for s in (1.1, 1.2):
            z = hf.fresp_zoom_image(img, s, cv2.INTER_CUBIC)[..., 0]
            x, y = centroid(z, (int(cx) - 40, int(cy) - 40, int(cx) + 41, int(cy) + 41))
            self.assertAlmostEqual(x, cx, delta=0.05)
            self.assertAlmostEqual(y, cy, delta=0.05)
            x, y = centroid(z, (int(cx + 300 * s) - 40, int(cy - 120 * s) - 40, int(cx + 300 * s) + 41, int(cy - 120 * s) + 41))
            self.assertAlmostEqual(x, cx + 300 * s, delta=0.1)
            self.assertAlmostEqual(y, cy - 120 * s, delta=0.1)

    def test_zoom_matrix_fixes_the_centre_and_zoom_one_is_the_identity(self):
        M = hf.fresp_zoom_matrix(1.2, self.W, self.H)
        c = np.array([(self.W - 1) / 2, (self.H - 1) / 2, 1.0])
        np.testing.assert_allclose(M @ c, c[:2], atol=1e-9)
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (self.H, self.W, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(hf.fresp_zoom_image(img, 1.0, cv2.INTER_CUBIC), img))

    def test_zoom_one_frames_are_the_same_files_content(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(1)
            rows = []
            for i in range(3):
                p = d / f"f_{i:06d}.png"
                cv2.imwrite(str(p), rng.integers(0, 256, (108, 192, 3), dtype=np.uint8))
                rows.append({"index": i, "frame": 2 * i, "time_s": 2 * i / 30.0, "png": str(p), "sha256": hf.sha256(p)})
            z1 = hf.fresp_zoom_frames(rows, 1.0, d / "z", cv2.INTER_CUBIC)
            self.assertEqual([r["sha256"] for r in z1], [r["sha256"] for r in rows])
            self.assertEqual([(r["index"], r["frame"], r["time_s"]) for r in z1], [(r["index"], r["frame"], r["time_s"]) for r in rows])
            z2 = hf.fresp_zoom_frames(rows, 1.2, d / "z2", cv2.INTER_CUBIC)
            self.assertNotEqual([r["sha256"] for r in z2], [r["sha256"] for r in rows])


class FrespScaleEquivariance(unittest.TestCase):
    def test_scaled_track_moves_points_about_the_centre(self):
        kp = np.array([[[960.0, 540.0], [1060.0, 440.0]]])
        box = np.array([[900.0, 400.0, 1100.0, 600.0]])
        kp2, b2 = hf.fresp_scale_track(kp, box, 1.2, (960.0, 540.0))
        np.testing.assert_allclose(kp2[0, 0], [960.0, 540.0])
        np.testing.assert_allclose(kp2[0, 1], [960.0 + 120.0, 540.0 - 120.0])
        np.testing.assert_allclose(b2[0], [960 - 60 * 1.2, 540 - 140 * 1.2, 960 + 140 * 1.2, 540 + 60 * 1.2])

    def test_synthetic_hand_f_scales_with_the_picture(self):
        """Keypoints and boxes enlarged by s about the image centre: with the f grid moved by s too f is exactly s * f (to 1e-6); with the unmoved grid (the pipeline's
        way) the parabola on the profile gives it within 0.2% on these draws."""
        g = synth_wrist(seed=0, tilt_frames=120)
        W, H = g["size"]
        kw = dict(front_idx=g["front_idx"], tilt_start=g["tilt_start"])
        modes = ["keep", "exclude"]
        base = hf.fresp_f_pair(g["kp"], g["mask"], g["present"], g["box"], g["cls"], g["times"], g["size"], PIPE, modes, 0, **kw)
        for s in (1.1, 1.2):
            kp2, box2 = hf.fresp_scale_track(g["kp"], g["box"], s, (W / 2, H / 2))
            plain = hf.fresp_f_pair(kp2, g["mask"], g["present"], box2, g["cls"], g["times"], g["size"], PIPE, modes, 0, **kw)
            moved = hf.fresp_f_pair(kp2, g["mask"], g["present"], box2, g["cls"], g["times"], g["size"], PIPE, modes, 0, ratio_range_scale=s, **kw)
            for m in modes:
                f0 = base[m]["fixed_template"]["f"]
                self.assertAlmostEqual(moved[m]["fixed_template"]["f"] / f0 / s, 1.0, delta=1e-6)
                self.assertAlmostEqual(plain[m]["fixed_template"]["f"] / f0 / s, 1.0, delta=0.002)


class FrespHelpers(unittest.TestCase):
    def test_edge_threshold_drops_a_zoom_but_never_the_control(self):
        self.assertFalse(hf.fresp_zoom_excluded(1.0, 0.5, 0.05))
        self.assertFalse(hf.fresp_zoom_excluded(1.1, 0.05, 0.05))
        self.assertTrue(hf.fresp_zoom_excluded(1.1, 0.0501, 0.05))
        self.assertTrue(hf.fresp_zoom_excluded(1.2, 0.2, 0.05))
        box = np.array([[0.0, 100.0, 300.0, 400.0]] * 3 + [[500.0, 100.0, 800.0, 400.0]] * 7)       # 3 of 10 frames touch the left border
        self.assertAlmostEqual(hf.fresp_edge_rate(box, np.ones(10, bool), 1920, 1080), 0.3)

    def test_ratio_status(self):
        f1, f2 = {"f": 1000.0, "at_grid_edge": False}, {"f": 1190.0, "at_grid_edge": False}
        st = hf.fresp_ratio_status(f2, f1, 1.2, 0.05)
        self.assertTrue(st["within"])
        self.assertAlmostEqual(st["ratio"], 1.19)
        self.assertFalse(hf.fresp_ratio_status({"f": 1300.0, "at_grid_edge": False}, f1, 1.2, 0.05)["within"])
        self.assertIsNone(hf.fresp_ratio_status({"f": 3072.0, "at_grid_edge": True}, f1, 1.2, 0.05)["ratio"])

    def test_target_videos_range_and_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for n in (7141, 7147, 7148, 7155, 7184):
                (d / f"IMG_{n}.MOV").write_bytes(b"x")
            (d / "notes.txt").write_text("x")
            rows = hf.fresp_target_videos(d, CFG22["target_videos"], CFG22["used_in_21"])
            by = {r["number"]: r for r in rows}
            self.assertEqual(sorted(by), [7141, 7147, 7148, 7155, 7184])
            self.assertEqual([n for n, r in by.items() if r["included"]], [7148, 7184])
            self.assertIn("체커보드", by[7155]["reason"])
            self.assertIn("미만", by[7147]["reason"])
            self.assertTrue(by[7148]["used_in_21"])
            self.assertFalse(by[7184]["used_in_21"])
            rows2 = hf.fresp_target_videos(d, dict(CFG22["target_videos"], number_range=[7148, 7150], exclude=[]), [])
            self.assertEqual([r["number"] for r in rows2 if r["included"]], [7148])

    def test_thermal_guard_does_not_wait_when_cool_or_without_nvidia_smi(self):
        self.assertEqual(hf.fresp_thermal_wait(hi=1000.0, lo=999.0, poll_s=0.01), 0.0)


def lens_jobs(f_one_x, f_wide, vids=None):
    """f_one_x: {video: f or None (grid edge)} for the 1x videos; f_wide: f of IMG_7184 or None, for modes exclude (candidate) and keep (same values)."""
    def job(v, f):
        sol = {"f": 3072.0 if f is None else f, "at_grid_edge": f is None, "side": "상한" if f is None else None}
        return {"video": v, "original": {m: {"solutions": {"fixed_template": sol}} for m in ("exclude", "keep")}}
    return [job(v, f) for v, f in f_one_x.items()] + [job("IMG_7184.MOV", f_wide)]


class FrespLens(unittest.TestCase):
    META = CFG22["discrimination"]["videos"]
    RULES = CFG22["discrimination"]["rules"]

    def run_judgment(self, f_one_x, f_wide):
        return hf.fresp_lens_judgment(lens_jobs(f_one_x, f_wide), self.META, self.RULES, "exclude", ["exclude", "keep"])

    def test_response_within_15_percent_of_the_reference_ratio(self):
        ones = {"IMG_7148.MOV": 1600.0, "IMG_7157.MOV": 1700.0, "IMG_7158.MOV": 1800.0, "IMG_7176.MOV": None}      # median of the 3 values = 1700
        ref = 880.0 / 1680.0
        lo, hi = ref * 0.85, ref * 1.15
        a = self.run_judgment(ones, 1700.0 * ref)
        x = a["by_video"]["IMG_7184.MOV"]["exclude"]
        self.assertEqual(a["one_x_videos"], list(ones))
        self.assertAlmostEqual(x["f_1x_median_px"], 1700.0)
        self.assertEqual((x["n_1x_values"], x["n_1x_videos"]), (3, 4))
        self.assertAlmostEqual(x["ratio"], ref)
        self.assertEqual(x["verdict"], "f 변화에 반응함")
        self.assertAlmostEqual(x["range"][0], 0.4452, places=3)
        self.assertAlmostEqual(x["range"][1], 0.6024, places=3)
        self.assertEqual(self.run_judgment(ones, 1700.0 * hi * 0.999)["by_video"]["IMG_7184.MOV"]["exclude"]["verdict"], "f 변화에 반응함")
        self.assertEqual(self.run_judgment(ones, 1700.0 * hi * 1.01)["by_video"]["IMG_7184.MOV"]["exclude"]["verdict"], "f 변화에 반응하지 않음")
        self.assertEqual(self.run_judgment(ones, 1700.0 * lo * 0.99)["by_video"]["IMG_7184.MOV"]["exclude"]["verdict"], "f 변화에 반응하지 않음")
        self.assertEqual(self.run_judgment(ones, 1700.0)["by_video"]["IMG_7184.MOV"]["exclude"]["verdict"], "f 변화에 반응하지 않음")        # f did not change at all

    def test_only_the_candidate_is_judged(self):
        a = self.run_judgment({"IMG_7148.MOV": 1700.0}, 890.0)["by_video"]["IMG_7184.MOV"]
        self.assertTrue(a["exclude"]["judged"] and a["exclude"]["verdict"] is not None)
        self.assertFalse(a["keep"]["judged"])
        self.assertIsNone(a["keep"]["verdict"])
        self.assertIsNotNone(a["keep"]["ratio"])                      # computed anyway

    def test_unavailable_when_the_wide_f_is_on_the_grid_edge_or_no_1x_value(self):
        a = self.run_judgment({"IMG_7148.MOV": 1700.0}, None)["by_video"]["IMG_7184.MOV"]["exclude"]
        self.assertEqual(a["verdict"], "판정 불가")
        self.assertIn("값 없음", a["unavailable"])
        b = self.run_judgment({"IMG_7148.MOV": None}, 900.0)["by_video"]["IMG_7184.MOV"]["exclude"]
        self.assertEqual(b["verdict"], "판정 불가")
        self.assertIn("1배 추정값 없음", b["unavailable"])


class FrespStats(unittest.TestCase):
    def test_excluded_video_is_not_in_the_points_or_the_statistics(self):
        def j(v, box, f):
            return {"video": v, "hand_box_short_ratio_front": box, "original": {"exclude": {"solutions": {"fixed_template": {"f": f, "at_grid_edge": False}}}}}
        jobs = [j("A.MOV", 0.5, 1500.0), j("B.MOV", 0.6, 1600.0), j("C.MOV", 0.7, 1700.0), j("IMG_7184.MOV", 0.3, 900.0)]
        pts = run.fresp_a3_points(jobs, "exclude", ["IMG_7184.MOV"])
        self.assertEqual([p[0] for p in pts], ["A.MOV", "B.MOV", "C.MOV"])
        st = run.fresp_stats([p[1] for p in pts], [p[2] for p in pts])
        self.assertAlmostEqual(st["spearman"], 1.0)
        self.assertAlmostEqual(st["slope_px"], 1000.0, places=6)
        self.assertEqual(run.fresp_stats([0.5, 0.6], [1500.0, 1600.0])["spearman"], None)             # fewer than 3 videos
        self.assertEqual(len(run.fresp_a3_points(jobs, "exclude")), 4)


VID7158 = Path(CFG22["target_videos"]["input_dir"]) / "IMG_7158.MOV"
VID7184 = Path(CFG22["target_videos"]["input_dir"]) / "IMG_7184.MOV"


@unittest.skipUnless(VID7158.exists() and shutil.which("ffmpeg"), "the input video of IMG_7158 is needed")
class FrespDetectMany(unittest.TestCase):
    """mediapipe on the same frame lists with n_workers = 1 and > 1: identical keypoints, and the control list is detected once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="t22_"))
        c = copy.deepcopy(CFG22)
        c["a2"]["zoom_factors"] = [1.0, 1.1, 1.2]
        prep = hf.fresp_prepare(("IMG_7158.MOV", str(VID7158.parent), c["pipeline"], c, str(cls.tmp)))
        cls.prep = prep
        n = 14
        cls.lists = {z: fr[:n] for z, fr in prep["zoom"].items()}
        cls.orig = prep["frames"][:n]
        cls.dset = c["detector_settings"]["mediapipe_hands"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tasks(self):
        return [(("v", "orig"), self.orig)] + [(("v", z), fr) for z, fr in self.lists.items()]

    def test_zoom_one_frames_have_the_original_pixels(self):
        """The PNG bytes differ (another encoder), the decoded pixels are identical, and index / frame / time are the same."""
        for a, b in zip(self.orig, self.lists["1.0"]):
            self.assertTrue(np.array_equal(cv2.imread(a["png"]), cv2.imread(b["png"])))
            self.assertEqual((a["index"], a["frame"], a["time_s"]), (b["index"], b["frame"], b["time_s"]))

    def test_parallel_detection_equals_sequential(self):
        seq = hf.fresp_detect_many(self.tasks(), self.dset, self.tmp / "cache_seq", 1)
        par = hf.fresp_detect_many(self.tasks(), self.dset, self.tmp / "cache_par", 4)
        self.assertNotEqual(seq[("v", "1.1")], seq[("v", "1.2")])
        for key in seq:
            a, b = np.load(seq[key]), np.load(par[key])
            self.assertEqual(sorted(a.files), sorted(b.files))
            for f in a.files:
                if a[f].dtype.kind in "fiub":
                    self.assertTrue(np.array_equal(a[f], b[f], equal_nan=a[f].dtype.kind == "f"), (key, f))
        dets = hf.load_detections(par[("v", "1.2")])
        hf.check_alignment(self.lists["1.2"], dets)                                   # frame order = list order
        a, b = np.load(par[("v", "orig")]), np.load(par[("v", "1.0")])                # the 1.0 control gives the original frames' keypoints
        for f in a.files:
            if a[f].dtype.kind in "fiub":
                self.assertTrue(np.array_equal(a[f], b[f], equal_nan=a[f].dtype.kind == "f"), f)


def cfg22_tmp(d: Path, videos_note=None):
    cfg = copy.deepcopy(CFG22)
    cfg["workers"] = 2
    cfg["a2"]["zoom_factors"] = [1.0, 1.2]
    cfg["a2"]["n_workers"] = 2
    cfg["a2"]["gpu_ids"] = [1, 2]
    cfg["output"]["root"] = str(d / "runs")
    cfg["output"]["results_yaml"] = str(d / "r22.yaml")
    return cfg


@unittest.skipUnless(VID7158.exists() and VID7184.exists() and RUN17_7158.exists() and GPU, "input videos of IMG_7158 / IMG_7184, the 17 run and a GPU are needed")
class FrespEndToEnd(unittest.TestCase):
    def test_outputs_21_reproduction_lens_judgment_and_no_tags(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = cfg22_tmp(d)
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, jobs = run.fresp_main(["--config", str(cp), "--videos", "IMG_7158.MOV,IMG_7184.MOV", "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt", "per_frame.csv"):
                self.assertTrue((root / f).exists(), f)
            for f in ("a2_zoom_ratio.png", "a3_box_vs_f.png"):
                self.assertTrue((root / "figures" / f).exists(), f)
            self.assertFalse((root / "_frames").exists())                              # the enlarged frames are deleted
            by = {j["video"]: j for j in jobs}
            self.assertTrue(by["IMG_7158.MOV"]["same_as_21"]["all"], by["IMG_7158.MOV"]["same_as_21"])        # the 21 settings reproduce 21's keep / exclude f and verdicts
            self.assertEqual(by["IMG_7184.MOV"]["detections"], "new mediapipe run on the normalised frames")
            e = by["IMG_7158.MOV"]["a2"]
            self.assertEqual(e["1.0"]["hand"]["keep"]["fixed_template"]["f"], by["IMG_7158.MOV"]["original"]["keep"]["solutions"]["fixed_template"]["f"])      # control = original
            self.assertEqual(e["1.0"]["hand"]["exclude"]["fixed_template"]["f"], by["IMG_7158.MOV"]["original"]["exclude"]["solutions"]["fixed_template"]["f"])
            self.assertIn("anycalib_pinhole", e["1.2"]["anycalib"])
            y = run.fresp_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r22.yaml")
            yy = yaml.safe_load(Path(y).read_text())
            self.assertIn("IMG_7184.MOV", yy["lens_series"]["videos"])
            self.assertEqual(yy["lens_series"]["videos"]["IMG_7184.MOV"]["modes"]["exclude"]["judged"], True)
            self.assertFalse(yy["lens_series"]["videos"]["IMG_7184.MOV"]["modes"]["keep"]["judged"])
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            ytext = Path(y).read_text()
            rtext = (root / "report.md").read_text()
            for t_ in (ytext, rtext):
                self.assertNotIn("1680", t_)                                          # the comparison with the reference f is in summary.md only
                self.assertNotIn("880", t_)
            summ = (root / "summary.md").read_text()
            self.assertIn("1680", summ)
            self.assertIn("880", summ)
            sec2 = summ.split("## 2.")[1].split("## 3.")[0]
            self.assertNotIn("7184", sec2)                                            # the 0.5x video is not in the |error| medians
            self.assertNotIn("7184", rtext.split("손 박스 크기와 추정 f의 관계")[1].split("산점도")[0].split("| 방식 |")[1])


# ====================================================================================================
# error factors (23)
# ====================================================================================================

IDS16 = list(range(5, 21))                          # the candidate's joint set (flat17 without the wrist)


def hand_pts16(spread_deg=0.0, wrist_shift_mm=0.0):
    """Synthetic planar hand (21 x 3, hand frame, long axis = y); optionally the index and pinky fingers rotated outward in the palm plane about their MCP joints
    (spread_deg each) and the wrist shifted sideways (mm)."""
    P = hf.synth_hand_points(0.0)
    P[:, 1] += 95.0
    for mcp, sign in ((5, 1.0), (17, -1.0)):
        a = np.radians(sign * spread_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for j in range(mcp + 1, mcp + 4):
            P[j, :2] = P[mcp, :2] + R @ (P[j, :2] - P[mcp, :2])
    P[0, 0] += wrist_shift_mm
    return P


def project_hand(P, theta, azim, f=3000.0, dist=1500.0, W=1920, H=1080, noise=0.0, rng=None):
    pts = hf.synth_project(P, hf.synth_pose(theta, azim, 0.0), np.array([0.0, 0.0, dist]), f, W / 2, H / 2)
    return pts + (rng.normal(0, noise, pts.shape) if noise else 0.0)


def frontal_template(rng=None, noise=0.0):
    P = hand_pts16()
    pts = np.array([project_hand(P, 0.0, 0.0, noise=noise, rng=rng)[IDS16] for _ in range(6)])
    return hf.template_build(pts, 4)["template"]


class FerrTiltKinds(unittest.TestCase):
    FC = CFG23["factors"]

    def kind_of(self, theta, azim):
        T = frontal_template()
        P = project_hand(hand_pts16(), theta, azim)[IDS16]
        th, _ = hf.affine_tilt(T, P)
        return hf.ferr_tilt_class(T, P, th, self.FC, IDS16)

    def test_rotation_about_the_long_axis_is_roll_and_about_the_width_axis_is_pitch(self):
        for th in (20.0, 35.0):
            r = self.kind_of(th, 0.0)                         # rotation axis (0, 1, 0) = the hand's long axis
            self.assertEqual(r["kind"], "굴림", r)
            self.assertLess(r["axis_angle_deg"], 5.0)
            p = self.kind_of(th, 90.0)                        # rotation axis (-1, 0, 0) = the width axis
            self.assertEqual(p["kind"], "숙임", p)
            self.assertGreater(p["axis_angle_deg"], 85.0)
        self.assertEqual(self.kind_of(30.0, 45.0)["kind"], "혼합")

    def test_small_tilt_is_not_classified(self):
        T = frontal_template()
        P = project_hand(hand_pts16(), 3.0, 0.0)[IDS16]
        th, _ = hf.affine_tilt(T, P)
        self.assertEqual(hf.ferr_tilt_class(T, P, th, self.FC, IDS16)["kind"], "미분류")


class FerrShape(unittest.TestCase):
    def shape(self, spread=0.0, shift=0.0, theta=30.0, azim=70.0, rng=None, noise=0.0):
        T = frontal_template()
        P = project_hand(hand_pts16(spread, shift), theta, azim, noise=noise, rng=rng)
        return hf.ferr_shape(T, IDS16, P[IDS16], P[0])

    def test_rigid_hand_has_near_zero_change_and_spread_fingers_have_a_large_one(self):
        rigid = self.shape()
        base = self.shape(theta=0.0, azim=0.0)
        self.assertLess(rigid["bone_ratio"], 0.02)
        self.assertLess(abs(rigid["spread_deg"] - base["spread_deg"]), 2.5)
        self.assertLess(abs(rigid["wrist_deg"] - base["wrist_deg"]), 2.5)
        spread = self.shape(spread=-12.0)                                                   # index and pinky turned 12 deg outward each: the angle between them grows by 24 deg
        self.assertGreater(spread["spread_deg"] - base["spread_deg"], 18.0)
        self.assertGreater(spread["spread_deg"] - rigid["spread_deg"], 18.0)
        bent = self.shape(shift=30.0)
        self.assertGreater(abs(bent["wrist_deg"] - base["wrist_deg"]), 10.0)

    def test_noise_keeps_the_rigid_change_small(self):
        rng = np.random.default_rng(0)
        r = [self.shape(rng=rng, noise=0.3, azim=a) for a in (10.0, 80.0, 130.0)]
        self.assertTrue(all(x["bone_ratio"] < 0.03 for x in r), r)

    def test_overlap_and_bin_helpers(self):
        kp = np.zeros((21, 2))
        mask = np.ones(21, bool)
        for k, a in enumerate((5, 9, 13, 17)):
            for q in range(4):
                kp[a + q] = (100.0 * k + 3.0 * q, 50.0 * q)
        self.assertAlmostEqual(hf.ferr_overlap(kp, mask, 200.0), 100.0 / 200.0, places=6)
        rows = hf.ferr_bin_table([5.0, 15.0, 16.0, 26.0], [1000.0, 1100.0, 1300.0, 900.0], [False, False, True, False], edges=[0.0, 10.0, 20.0, 30.0])
        self.assertEqual([(r["n"], r["n_edge"], r["n_with_f"], r["median_f"]) for r in rows], [(1, 0, 1, 1000.0), (2, 1, 1, 1100.0), (1, 0, 1, 900.0)])
        rows = hf.ferr_bin_table(["굴림", "숙임", "굴림"], [1000.0, 1200.0, 1400.0], [False, False, False], labels=["숙임", "굴림", "혼합"])
        self.assertEqual([(r["label"], r["n"], r["median_f"]) for r in rows], [("숙임", 1, 1200.0), ("굴림", 2, 1200.0), ("혼합", 0, None)])


class FerrCurves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = synth_wrist(seed=1, tilt_frames=60)
        cls.a = analyze_synth(cls.g, methods_full=("fixed_template",))
        cls.pb, cls.ids = hf.ferr_problem(cls.a, cls.g["kp"], cls.g["mask"], cls.g["box"], PIPE, cls.g["size"])
        cls.grid = np.array(cls.a["grid"])

    def test_mean_of_the_frame_curves_is_the_ft_curve(self):
        curves = hf.ferr_frame_curves(self.pb, self.grid)
        prof = np.array(self.a["solutions"]["fixed_template"]["profile_cost"])
        self.assertEqual(curves.shape, (len(self.pb.keep), len(self.grid)))
        np.testing.assert_allclose(curves.mean(0), prof, rtol=1e-9)
        self.assertAlmostEqual(hf.cost(self.pb, self.a["solutions"]["fixed_template"]["f"]), self.a["solutions"]["fixed_template"]["cost"], places=9)

    def test_subset_with_every_frame_gives_the_ft_f(self):
        role = self.a["role"]
        front = [i for i in self.pb.keep if role[i] == "front"]
        tilt = [i for i in self.pb.keep if role[i] == "tilt"]
        r = hf.ferr_subset_ft(self.pb, front, tilt, self.grid)
        self.assertEqual(r["f"], self.a["solutions"]["fixed_template"]["f"])
        self.assertEqual(r["n_tilt"], len(tilt))
        half = hf.ferr_subset_ft(self.pb, front, tilt[:len(tilt) // 2], self.grid)
        self.assertEqual(half["n_tilt"], len(tilt) // 2)
        self.assertNotEqual(half["f"], r["f"])

    def test_preferred_f_of_a_frame_and_its_sharpness(self):
        curves = hf.ferr_frame_curves(self.pb, self.grid)
        k = self.pb.keep[-1]
        pr = hf.ferr_preferred(curves[-1], self.grid, self.pb.subset([k]))
        self.assertGreater(pr["f_pref"], 0.0)
        self.assertGreaterEqual(pr["sharpness"], 1.0 - 1e-9)
        self.assertEqual(pr["at_edge"], bool(int(np.argmin(curves[-1])) in (0, len(self.grid) - 1)))


class FerrCachedDetection(unittest.TestCase):
    def test_missing_cache_raises_instead_of_detecting(self):
        with tempfile.TemporaryDirectory() as d:
            frames = [{"index": 0, "sha256": "ab" * 32}]
            with self.assertRaises(FileNotFoundError):
                hf.ferr_cached_npz(frames, CFG23["detector_settings"]["mediapipe_hands"], d)
            p = Path(d) / "x.npz"
            key_path = Path(d) / f"mediapipe_hands_{hf.frames_key(frames, {k: CFG23['detector_settings']['mediapipe_hands'][k] for k in ('weights', 'num_hands', 'min_hand_detection_confidence', 'min_hand_presence_confidence', 'box_margin', 'remap')}, 'mediapipe_hands')}.npz"
            key_path.write_bytes(b"x")
            self.assertEqual(hf.ferr_cached_npz(frames, CFG23["detector_settings"]["mediapipe_hands"], d), key_path)
            del p


RUN22 = ROOT / "runs/22_f_response/20261021T042524Z"


@unittest.skipUnless(VID7158.exists() and VID7184.exists() and RUN17_7158.exists() and (RUN22 / "results.json").exists() and (RUN22 / "detection_cache").exists(),
                     "the input videos of IMG_7158 / IMG_7184, the 17 run and 22's official run (results and detection cache) are needed")
class FerrEndToEnd(unittest.TestCase):
    def test_22_reproduction_outputs_and_no_tags(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG23)
            cfg["workers"] = 2
            cfg["output"]["root"] = str(d / "runs")
            cfg["output"]["results_yaml"] = str(d / "r23.yaml")                     # never the tracked results file
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, jobs = run.ferr_main(["--config", str(cp), "--videos", "IMG_7158.MOV,IMG_7184.MOV", "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "diagnostic_pull.md", "results.json", "metadata.json", "log.txt", "per_frame.csv", "pull_per_frame.csv"):
                self.assertTrue((root / f).exists(), f)
            for f in ("factor_vs_error.png", "frame_preferred_f.png", "preferred_f_by_bins.png"):
                self.assertTrue((root / "figures" / f).exists(), f)
            self.assertFalse((root / "_tmp").exists())
            by = {j["video"]: j for j in jobs}
            for v in ("IMG_7158.MOV", "IMG_7184.MOV"):
                self.assertTrue(by[v]["same_as_22"], v)                              # the 22 settings reproduce 22's candidate f and verdict
                self.assertLess(by[v]["curve_mean_check"]["max_rel_diff"], 1e-9)
                self.assertEqual(by[v]["subset_all"]["f"], by[v]["ft"]["f"])
                self.assertEqual(sorted(by[v]["subsets"]), sorted(s["name"] for s in CFG23["subsets"]))
            self.assertEqual(by["IMG_7184.MOV"]["detections"], "22 run cache (found by the frame hashes)")
            self.assertTrue(all(x is None for x in by["IMG_7184.MOV"]["factors"]["detector_diff_pct"].values()))      # no wilor_yolo / rtmpose detection exists for it
            self.assertTrue(all(x is not None for x in by["IMG_7158.MOV"]["factors"]["detector_diff_pct"].values()))
            y = run.ferr_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r23.yaml")
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            ref_f = re.compile(r"(?<![\d.])(?:1680|880)(?:\.0+)?(?!\d)(?!\.\d)")           # the reference values as standalone numbers (not part of another number)
            for name in ("report.md", "per_frame.csv"):
                self.assertIsNone(ref_f.search((root / name).read_text()), name)       # reference f values: summary.md and the pull diagnostic only
            ytext = Path(y).read_text()
            self.assertIsNone(ref_f.search(ytext))
            summ = (root / "summary.md").read_text()
            self.assertIn("원인 확정", summ.split("## 1.")[0])                         # the exploratory disclaimer is at the top
            self.assertIn("1680", summ)
            self.assertIn("880", summ)
            self.assertIn("1680", (root / "diagnostic_pull.md").read_text())
            rho = summ.split("## 2.")[1].split("## 3.")[0]
            self.assertNotIn("7184", rho)                                            # the 0.5x video is not in the correlations


# ====================================================================================================
# long video check (25)
# ====================================================================================================

def toy_curves(f_true=(1500.0, 1800.0, 2100.0), noise=0.0):
    """Cost curves on the pipeline's f grid: quadratic in log f around each video's own minimum."""
    grid = hf.f_grid(PIPE["estimate"], 1920.0)
    return grid, np.array([(np.log(grid / f0)) ** 2 + 1.0 for f0 in f_true])


class LongPooled(unittest.TestCase):
    def test_pooled_curve_is_the_sum_of_the_video_curves(self):
        grid, cur = toy_curves()
        np.testing.assert_allclose(hf.long_pooled_curve(cur), cur[0] + cur[1] + cur[2])
        pooled = hf.long_pooled_estimate(cur, grid)
        self.assertAlmostEqual(pooled["f"], float(np.exp(np.mean(np.log([1500.0, 1800.0, 2100.0])))), delta=8.0)      # the minimum of the summed parabolas: the geometric mean
        self.assertFalse(pooled["at_grid_edge"])

    def test_one_video_gives_that_videos_ft(self):
        g = synth_wrist(seed=2, tilt_frames=80)
        a = analyze_synth(g, methods_full=("fixed_template",))
        sol = a["solutions"]["fixed_template"]
        one = hf.long_pooled_estimate([sol["profile_cost"]], a["grid"])
        self.assertEqual(one["f"], sol["f"])
        self.assertEqual(one["at_grid_edge"], sol["at_grid_edge"])

    def test_grid_edge_is_flagged(self):
        grid = hf.f_grid(PIPE["estimate"], 1920.0)
        r = hf.long_pooled_estimate([np.linspace(2.0, 1.0, len(grid))], grid)           # decreasing: the minimum is at the upper end
        self.assertTrue(r["at_grid_edge"])
        self.assertEqual(r["side"], "상한")

    def test_bootstrap_is_reproducible_with_a_fixed_seed_and_differs_between_seeds(self):
        grid, cur = toy_curves((1400.0, 1600.0, 1900.0, 2300.0, 1700.0))
        a = hf.long_pooled_bootstrap(cur, grid, 200, 42)
        b = hf.long_pooled_bootstrap(cur, grid, 200, 42)
        c = hf.long_pooled_bootstrap(cur, grid, 200, 7)
        self.assertEqual(a["f"], b["f"])
        self.assertNotEqual(a["f"], c["f"])
        lo, hi = hf.long_log_ci(a["f"], 1700.0)
        self.assertLess(lo, hi)
        self.assertEqual(hf.long_log_ci(hf.long_pooled_bootstrap(cur, grid, 200, 42)["f"], 1700.0), (lo, hi))


class LongThetaFilter(unittest.TestCase):
    def test_keeps_only_tilt_frames_at_or_above_the_limit(self):
        theta = np.array([5.0, 19.9, 20.0, 35.0, np.nan, 61.0, 10.0])
        tilt = [1, 2, 3, 4, 5, 6]
        self.assertEqual(hf.long_theta_keep(theta, tilt, 20.0), [2, 3, 5])            # frame 0 is not a tilt frame; nan is dropped; the limit itself stays
        self.assertEqual(hf.long_theta_keep(theta, tilt, 0.0), [1, 2, 3, 5, 6])


class LongHypotheses(unittest.TestCase):
    H = CFG25["hypotheses"]

    def test_h1(self):
        self.assertEqual(hf.long_h1([0.08, 0.09], [0.10, 0.11, 0.12], 0.85)["verdict"], "지지")           # 0.085 <= 0.85 * 0.11
        self.assertEqual(hf.long_h1([0.10, 0.11], [0.10, 0.11, 0.12], 0.85)["verdict"], "근거 없음")
        self.assertEqual(hf.long_h1([], [0.1], 0.85)["verdict"], "판단 불가")

    def test_h2_interval_rule_and_minimum_number_of_videos(self):
        self.assertEqual(hf.long_h2((0.02, 0.2), 4, 4)["verdict"], "공통 편향 있음")
        self.assertEqual(hf.long_h2((-0.2, -0.01), 5, 4)["sign"], "−")
        self.assertEqual(hf.long_h2((-0.05, 0.2), 4, 4)["verdict"], "확인 안 됨")
        self.assertEqual(hf.long_h2((0.02, 0.2), 3, 4)["verdict"], "판단 불가")                     # recorded but not judged

    def test_h3(self):
        b = float(np.log(1.1))
        f = [1848.0, 1870.0, 1900.0, 1700.0]               # +10%, +11%, +13%, +1% against 1680; the correction divides by 1.1
        r = hf.long_h3(f, 1680.0, b, 4, 4, self.H["fraction"])
        self.assertEqual(r["verdict"], "보정 효과 있음" if r["median_abs_after"] < r["median_abs_before"] and r["fraction_reduced"] >= 2 / 3 else "보정 효과 없음")
        self.assertAlmostEqual(r["fraction_reduced"], 0.75)
        self.assertEqual(hf.long_h3([1650.0, 1700.0, 1660.0, 1690.0], 1680.0, b, 4, 4, self.H["fraction"])["verdict"], "보정 효과 없음")      # already unbiased: the correction hurts
        self.assertEqual(hf.long_h3(f[:3], 1680.0, b, 3, 4, self.H["fraction"])["verdict"], "판단 불가")

    def test_h4(self):
        r = hf.long_h4([1900.0, 1850.0, 1700.0], [1700.0, 1720.0, 1690.0], 1680.0, 0.02, 2 / 3)
        self.assertEqual(r["verdict"], "개선")
        self.assertEqual(hf.long_h4([1800.0, 1850.0, 1700.0], [1790.0, 1850.0, 1700.0], 1680.0, 0.02, 2 / 3)["verdict"], "근거 없음")      # gain below 2 points
        self.assertEqual(hf.long_h4([1800.0, 1650.0, 1700.0], [1700.0, 1900.0, 1750.0], 1680.0, 0.02, 2 / 3)["verdict"], "근거 없음")      # not enough videos improve
        self.assertEqual(hf.long_h4([], [], 1680.0, 0.02, 2 / 3)["verdict"], "판단 불가")

    def test_calibration_shares(self):
        c = hf.long_calibration([1680.0 * np.exp(0.05), 1680.0 * np.exp(0.5)], [0.1, 0.1], 1680.0)
        self.assertEqual((c["within_1sigma"], c["within_2sigma"]), (0.5, 0.5))
        c2 = hf.long_calibration([1680.0 * np.exp(0.05 + 0.3), 1680.0 * np.exp(0.18)], [0.1, 0.1], 1680.0, b=0.3)
        self.assertEqual(c2["within_1sigma"], 0.5)

    def test_log_std_and_yaml_thresholds(self):
        self.assertAlmostEqual(hf.long_boot_log_std([1000.0, 1000.0 * np.e]), float(np.std([np.log(1000.0), np.log(1000.0) + 1.0], ddof=1)))
        self.assertTrue(np.isnan(hf.long_boot_log_std([1000.0])))
        self.assertEqual((self.H["h1_ratio"], self.H["h4_gain"], self.H["min_new_videos_h2_h3"], self.H["h4_theta_min_deg"]), (0.85, 0.02, 4, 20.0))
        self.assertAlmostEqual(self.H["fraction"], 2 / 3, places=6)


@unittest.skipUnless(RUN17_7158.exists() and (ROOT / "runs/22_f_response/20261021T042524Z/results.json").exists() and (ROOT / "runs/23_error_factors/20261022T002204Z/results.json").exists(),
                     "the 17 run of IMG_7158 and the official runs 22 and 23 are needed")
class LongReproduction(unittest.TestCase):
    def test_22_settings_reproduce_22_and_23_for_a_baseline_video(self):
        cjob = {k: CFG25[k] for k in ("candidate", "exclusion", "hypotheses", "detector_settings", "guide", "anycalib_guard")}
        with tempfile.TemporaryDirectory() as d:
            r = hf.long_video(("IMG_7158.MOV", "baseline", str(RUN17_7158), None, None, CFG25["pipeline"], dict(cjob, compare_estimates=False), d, None))
        r22 = {j["video"]: j for j in json.loads((ROOT / "runs/22_f_response/20261021T042524Z/results.json").read_text())["jobs"]}["IMG_7158.MOV"]
        r23 = {j["video"]: j for j in json.loads((ROOT / "runs/23_error_factors/20261022T002204Z/results.json").read_text())["jobs"]}["IMG_7158.MOV"]
        self.assertEqual(r["ft"]["f"], r22["original"]["exclude"]["solutions"]["fixed_template"]["f"])
        self.assertEqual(r["ft"]["verdict"], r22["original"]["exclude"]["verdict"])
        self.assertAlmostEqual(r["ft"]["boot_rel_std"], r23["factors"]["bootstrap_rel_std"], places=12)
        self.assertEqual(r["excluded"], [])
        self.assertEqual(len(r["curve"]["cost"]), len(r["curve"]["grid"]))
        self.assertEqual(r["motion"]["n_tilt_frames"], r23["factors"]["n_tilt_frames"])
        self.assertTrue(abs(r["ft"]["boot_log_std"] - r["ft"]["boot_rel_std"]) < 0.01)              # the log SD and the relative SD are nearly the same quantity


@unittest.skipUnless(VID7158.exists() and RUN17_7158.exists() and GPU, "the input video of IMG_7158, the 17 run and a GPU are needed")
class LongEndToEnd(unittest.TestCase):
    def test_bias_step_new_video_run_and_no_tags(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG25)
            cfg["baseline_videos"] = ["IMG_7157.MOV", "IMG_7176.MOV"]                  # two baseline videos keep the test short
            cfg["new_videos"] = ["IMG_7158.MOV"]                                       # a "new" video that is not in the baseline list
            cfg["workers"] = 2
            cfg["gpu_ids"] = [1]
            cfg["n_workers"] = 2
            cfg["output"]["root"] = str(d / "runs")
            cfg["output"]["results_yaml"] = str(d / "r25.yaml")
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            self.assertIsNone(yaml.safe_load(cp.read_text())["baseline"]["bias_b"])
            with self.assertRaises(SystemExit):                                         # the run refuses to start before b is fixed
                run.long_main(["--config", str(cp), "--videos", "IMG_7158.MOV", "--out", str(d / "o0")])
            rec = run.long_main(["--config", str(cp), "bias"])
            self.assertEqual([j["video"] for j in rec["videos"]], ["IMG_7157.MOV", "IMG_7176.MOV"])      # b uses the baseline videos only
            self.assertNotIn("IMG_7158.MOV", json.dumps(rec))
            c2 = yaml.safe_load(cp.read_text())
            self.assertAlmostEqual(c2["baseline"]["bias_b"], rec["b"], places=12)
            self.assertTrue(Path(c2["baseline"]["json"]).exists() or (ROOT.parent / c2["baseline"]["json"]).exists() or hf.resolve(c2["baseline"]["json"]).exists())
            root, new_all, h = run.long_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt", "per_frame.csv"):
                self.assertTrue((root / f).exists(), f)
            for f in ("f_with_bootstrap.png", "pooled_cost_curves.png", "theta_distribution.png"):
                self.assertTrue((root / "figures" / f).exists(), f)
            self.assertFalse((root / "_frames").exists())
            j = new_all[0]
            self.assertEqual(j["ft"]["f"], json.loads((ROOT / "runs/22_f_response/20261021T042524Z/results.json").read_text())["jobs"][2]["original"]["exclude"]["solutions"]["fixed_template"]["f"])   # IMG_7158 = 22's value
            self.assertEqual((h["H2"]["verdict"], h["H3"]["verdict"]), ("판단 불가", "판단 불가"))      # one new video: fewer than 4
            self.assertIn("anycalib_pinhole", j["anycalib"])
            self.assertIn("guide", j)
            y = run.long_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r25.yaml")
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            ref_f = re.compile(r"(?<![\d.])(?:1680)(?:\.0+)?(?!\d)(?!\.\d)")
            self.assertIsNone(ref_f.search((root / "report.md").read_text()))
            self.assertIsNone(ref_f.search(Path(y).read_text()))
            self.assertIn("1680", (root / "summary.md").read_text())
            yy = yaml.safe_load(Path(y).read_text())
            self.assertEqual(sorted(yy["hypotheses"]), ["H1", "H2", "H3", "H4"])


class LongYamlPatch(unittest.TestCase):
    def test_baseline_values_are_written_without_losing_comments_or_the_space_before_them(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("baseline:\n  bias_b: null                       # b = log\n  json: null                         # note\n  pooled_f_px: null\n")
            run.long_patch_yaml(p, {"bias_b": "0.5", "json": "x/y.json", "pooled_f_px": "1.5"})
            c = yaml.safe_load(p.read_text())["baseline"]
            self.assertEqual((c["bias_b"], c["json"], c["pooled_f_px"]), (0.5, "x/y.json", 1.5))
            self.assertIn("# b = log", p.read_text())
            with self.assertRaises(SystemExit):
                run.long_patch_yaml(p, {"missing_key": "1"})


# ====================================================================================================
# generalisation (26)
# ====================================================================================================

class GenRatingAndVerification(unittest.TestCase):
    V2, VER = CFG26["rating_v2"], CFG26["verification_v2"]

    def test_rating_rules(self):
        r = lambda tilt, th, sg: hf.gen_rating(tilt, th, sg, self.V2)      # noqa: E731
        self.assertEqual(r(20.0, 60.0, 0.08), "사용 가능")
        self.assertEqual(r(18.0, 55.0, 0.10), "사용 가능")                      # the limits belong to the better class
        self.assertEqual(r(20.0, 60.0, 0.12), "참고")                          # both shooting conditions hold, the log SD is looser
        self.assertEqual(r(20.0, 60.0, 0.15), "참고")
        self.assertEqual(r(20.0, 60.0, 0.16), "사용 불가")
        self.assertEqual(r(17.9, 60.0, 0.05), "사용 불가")                      # one shooting condition fails
        self.assertEqual(r(20.0, 54.9, 0.05), "사용 불가")
        self.assertEqual(r(20.0, 60.0, None), "사용 불가")
        self.assertEqual(r(None, 60.0, 0.05), "사용 불가")

    def test_thresholds_of_the_yaml_are_the_specified_ones(self):
        self.assertEqual((self.V2["tilt_min_s"], self.V2["theta_max_min_deg"], self.V2["usable_log_std_max"], self.V2["reference_log_std_max"]), (18.0, 55.0, 0.10, 0.15))
        self.assertEqual((self.VER["within_sigma"], self.VER["within_share_min"], self.VER["median_abs_err_max"], self.VER["min_usable"]), (2.0, 0.9, 0.10, 5))
        g = CFG26["generalization"]
        self.assertEqual((g["pooled_tol"], g["video_tol"], g["min_videos"], g["bootstrap_n"]), (0.10, 0.15, 3, 1000))
        self.assertAlmostEqual(g["video_fraction"], 2 / 3, places=6)
        self.assertEqual(CFG26["guide"]["conditions"]["box_short_ratio_max"], 0.65)
        self.assertEqual(CFG25["guide"]["conditions"]["box_short_ratio_max"], 0.6)                       # 25's value is untouched
        t = CFG26["trajectory"]
        self.assertEqual((t["hand_length_mm"], t["epsilons_pct"], t["tolerance"]), (180.0, [-20, -10, -5, 5, 10, 20], {"motion_pct": None, "direction_deg": None}))
        self.assertEqual(CFG26["calib_end_default_s"], 25.0)

    def rows(self, n_usable, errs, sigma=0.06):
        return [{"video": f"v{i}", "rating": "사용 가능", "f": 1000.0 * np.exp(e), "ref_f": 1000.0, "sigma": sigma} for i, e in enumerate(errs[:n_usable])] + [{"video": "x", "rating": "참고", "f": 1.0, "ref_f": 1000.0, "sigma": 0.1}]

    def test_verification(self):
        ok = hf.gen_verify(self.rows(5, [0.02, -0.05, 0.08, 0.0, -0.03]), self.VER)
        self.assertEqual(ok["verdict"], "판정 v2 검증됨")
        self.assertEqual(ok["n_usable"], 5)
        far = hf.gen_verify(self.rows(5, [0.02, -0.05, 0.30, 0.0, -0.03]), self.VER)                      # one video beyond 2 sigma (0.12): share 0.8 < 0.9
        self.assertEqual(far["verdict"], "검증 안 됨")
        self.assertAlmostEqual(far["within_share"], 0.8)
        big = hf.gen_verify(self.rows(5, [0.11, 0.12, 0.11, 0.1, 0.11], sigma=0.2), self.VER)           # all within 2 sigma but the median error is above 10%
        self.assertEqual(big["verdict"], "검증 안 됨")
        few = hf.gen_verify(self.rows(4, [0.01, 0.02, 0.0, 0.01]), self.VER)
        self.assertEqual(few["verdict"], "판단 불가")
        self.assertIn("4개 < 5개", few["reason"])
        self.assertEqual(hf.gen_verify([], self.VER)["verdict"], "판단 불가")


class GenGroups(unittest.TestCase):
    G = CFG26["generalization"]

    def curves(self, fs):
        grid, cur = toy_curves(tuple(fs))
        return grid, cur

    def test_group_pooled_f_is_the_sum_of_the_video_curves(self):
        grid, cur = self.curves((900.0, 880.0, 860.0, 910.0))
        d = hf.gen_group_verdict(cur, grid, [900.0, 880.0, 860.0, 910.0], 880.0, self.G)
        self.assertEqual(d["pooled_f"], hf.long_pooled_estimate(cur, grid)["f"])
        np.testing.assert_allclose(hf.long_pooled_curve(cur), cur.sum(0))
        self.assertEqual(d["n_videos"], 4)
        self.assertEqual(d["verdict"], "통과")

    def test_pass_fail_and_not_judged(self):
        grid, cur = self.curves((900.0, 880.0, 860.0))
        self.assertEqual(hf.gen_group_verdict(cur, grid, [900.0, 880.0, 860.0], 880.0, self.G)["verdict"], "통과")
        self.assertEqual(hf.gen_group_verdict(cur, grid, [900.0, 880.0, 860.0], 700.0, self.G)["verdict"], "실패")        # the pooled f is not within +-10% of ref_f
        grid2, cur2 = self.curves((880.0, 880.0, 880.0))
        d = hf.gen_group_verdict(cur2, grid2, [880.0, 1200.0, 1300.0], 880.0, self.G)                                      # pooled f fine, only 1 of 3 videos within +-15%
        self.assertTrue(d["pooled_within_tol"])
        self.assertAlmostEqual(d["video_share_within_tol"], 1 / 3)
        self.assertEqual(d["verdict"], "실패")
        d2 = hf.gen_group_verdict(cur2, grid2, [880.0, 880.0, 1300.0], 880.0, self.G)                                      # 2 of 3 within +-15%: exactly the 2/3 threshold
        self.assertEqual(d2["verdict"], "통과")
        self.assertEqual(hf.gen_group_verdict(cur[:2], grid, [900.0, 880.0], 880.0, self.G)["verdict"], "판단 불가")
        self.assertEqual(hf.gen_group_verdict([], np.array([]), [], 880.0, self.G)["verdict"], "영상 없음")
        ex = hf.gen_group_verdict(cur, grid, [900.0, 880.0, 860.0], 880.0, self.G, n_excluded=2, n_total=5)
        self.assertEqual((ex["n_excluded"], ex["n_total"]), (2, 5))                                                       # the excluded share is carried (G4)

    def test_video_entries_are_checked(self):
        good = {"file": "IMG_1.MOV", "group": "G3", "person_id": "P1", "phone": "main", "lens": "0.5x", "glove": False, "ref_f": 880}
        self.assertEqual(hf.gen_check_videos([good]), [good])
        with self.assertRaises(SystemExit):
            hf.gen_check_videos([])
        with self.assertRaises(SystemExit):
            hf.gen_check_videos([{k: v for k, v in good.items() if k != "ref_f"}])
        with self.assertRaises(SystemExit):
            hf.gen_check_videos([dict(good, group="G9")])
        with self.assertRaises(SystemExit):
            hf.gen_check_videos([dict(good, ref_f=0)])
        for g in hf.GEN_GROUPS:                                                                                      # every group is accepted
            self.assertEqual(len(hf.gen_check_videos([dict(good, group=g)])), 1)


class GenTrajectory(unittest.TestCase):
    TC = CFG26["trajectory"]

    def series(self):
        t = np.arange(0, 6.0, 1 / 15)
        u = 960 + 120 * np.sin(1.2 * t)
        v = 540 + 80 * np.cos(0.9 * t)
        ell = 400.0 * (1.0 + 0.15 * np.sin(0.7 * t))                 # the hand moves along the optical axis too
        return t, u, v, ell

    def test_changing_f_scales_only_z_and_the_depth_movement(self):
        t, u, v, ell = self.series()
        ref = hf.gen_trajectory(u, v, ell, 1680.0, 180.0, 1920, 1080)
        for e in self.TC["epsilons_pct"]:
            cmp_ = hf.gen_trajectory(u, v, ell, 1680.0 * (1 + e / 100.0), 180.0, 1920, 1080)
            np.testing.assert_allclose(cmp_[:, 0], ref[:, 0], rtol=1e-12)
            np.testing.assert_allclose(cmp_[:, 1], ref[:, 1], rtol=1e-12)
            np.testing.assert_allclose(cmp_[:, 2], ref[:, 2] * (1 + e / 100.0), rtol=1e-12)
            _, d0 = hf.gen_step_vectors(t, ref, 0.5, 0.1)
            _, d1 = hf.gen_step_vectors(t, cmp_, 0.5, 0.1)
            np.testing.assert_allclose(d1[:, :2], d0[:, :2], atol=1e-9)
            np.testing.assert_allclose(d1[:, 2], d0[:, 2] * (1 + e / 100.0), rtol=1e-9)                     # only the depth movement changes, by (1 + eps)
        np.testing.assert_allclose(ref[:, 0], (u - 960) * 180.0 / ell)                                       # X = (u - cx) L / l
        np.testing.assert_allclose(ref[:, 2], 1680.0 * 180.0 / ell)                                          # Z = f L / l

    def test_step_metrics(self):
        t, u, v, ell = self.series()
        ref = hf.gen_smooth_traj(t, hf.gen_trajectory(u, v, ell, 1680.0, 180.0, 1920, 1080), 0.3)
        same = hf.gen_step_metrics(t, ref, ref, self.TC)
        self.assertEqual((same["magnitude_err_median_pct"], same["direction_err_median_deg"], same["depth_err_median_mm"]), (0.0, 0.0, 0.0))
        m10 = hf.gen_step_metrics(t, ref, hf.gen_smooth_traj(t, hf.gen_trajectory(u, v, ell, 1680.0 * 1.1, 180.0, 1920, 1080), 0.3), self.TC)
        m20 = hf.gen_step_metrics(t, ref, hf.gen_smooth_traj(t, hf.gen_trajectory(u, v, ell, 1680.0 * 1.2, 180.0, 1920, 1080), 0.3), self.TC)
        self.assertGreater(m10["depth_err_median_mm"], 0.0)
        self.assertGreater(m20["depth_err_median_mm"], m10["depth_err_median_mm"])                          # a larger f error: a larger depth error
        self.assertGreater(m20["direction_err_median_deg"], m10["direction_err_median_deg"])
        self.assertGreater(m10["n_steps"], 5)
        still = np.tile(ref[:1], (len(t), 1))                                                              # no movement: no step is used
        self.assertEqual(hf.gen_step_metrics(t, still, still, self.TC)["n_steps"], 0)
        self.assertIsNone(hf.gen_step_metrics(t, still, still, self.TC)["magnitude_err_median_pct"])

    def test_variants_of_a_video_include_the_estimate_and_the_pooled_f(self):
        t, u, v, ell = self.series()
        rows = [{"time_s": float(a), "u": float(b), "v": float(c), "size_ratio": float(d / 400.0)} for a, b, c, d in zip(t, u, v, ell)]
        res = {"ref_f": 880.0, "size": [1920, 1080], "traj": {"rows": rows, "ell_front_px": 400.0}, "ft": {"f": 950.0, "at_grid_edge": False}}
        tv = run.gen_traj_variants(res, CFG26, 900.0)
        self.assertEqual(sorted(tv["variants"]), sorted(["ref", "est", "pooled"] + [f"eps_{int(e):+d}" for e in CFG26["trajectory"]["epsilons_pct"]]))
        self.assertEqual(tv["variants"]["est"]["f"], 950.0)
        self.assertEqual(tv["variants"]["eps_+10"]["f"], 880.0 * 1.1)
        res["ft"]["at_grid_edge"] = True
        self.assertNotIn("est", run.gen_traj_variants(res, CFG26, None)["variants"])                         # an estimate on the grid edge has no trajectory


class GenCalibEnd(unittest.TestCase):
    def test_frames_after_calib_end_are_not_used_for_f(self):
        g = synth_wrist(seed=4, tilt_frames=60)
        t = g["times"]
        calib_end = float(t[len(t) - 20])                                                                        # the last 20 tilt frames are the "work" segment
        pres = hf.gen_present_until(t, g["present"], calib_end)
        self.assertEqual(int(pres.sum()), int((t <= calib_end).sum()))
        self.assertTrue(np.array_equal(hf.gen_present_until(t, g["present"], None), g["present"]))
        a = hf.analyze(g["kp"], g["mask"], pres, g["box"], g["cls"], t, g["size"], PIPE, front_idx=g["front_idx"], tilt_start=g["tilt_start"], methods_full=())
        kp2 = g["kp"].copy()
        kp2[t > calib_end] = kp2[t > calib_end] + np.random.default_rng(0).normal(0, 60.0, kp2[t > calib_end].shape)      # wreck the work segment
        b = hf.analyze(kp2, g["mask"], pres, g["box"], g["cls"], t, g["size"], PIPE, front_idx=g["front_idx"], tilt_start=g["tilt_start"], methods_full=())
        self.assertEqual(a["solutions"]["fixed_template"]["f"], b["solutions"]["fixed_template"]["f"])
        self.assertEqual(max(i for i, r in enumerate(a["role"]) if r == "tilt"), int(np.flatnonzero(t <= calib_end).max()))
        full = hf.analyze(g["kp"], g["mask"], g["present"], g["box"], g["cls"], t, g["size"], PIPE, front_idx=g["front_idx"], tilt_start=g["tilt_start"], methods_full=())
        self.assertNotEqual(a["solutions"]["fixed_template"]["f"], full["solutions"]["fixed_template"]["f"])      # the frames after calib_end do matter when they are allowed


def gen_fake(video, group, **kw):
    """A minimal gen_video result for the report functions."""
    rows = [{"index": i, "time_s": 3.0 + i / 15.0, "u": 900.0 + 5 * i, "v": 500.0, "size_ratio": 1.0 + 0.01 * i, "theta2d_deg": 20.0} for i in range(40)]
    r = {"video": video, "group": group, "person_id": kw.get("person", "P1"), "phone": "main", "lens": "1x", "glove": group == "G4", "ref_f": 1680.0, "calib_end_s": None, "n_frames": 300, "duration_s": 20.0,
         "size": [1920, 1080], "detect_rate": kw.get("detect_rate", 1.0), "edge_rate": 0.0, "front_window": [1.0, 2.0], "excluded": kw.get("excluded", [])}
    if not r["excluded"]:
        r.update({"ft": {"f": 1700.0, "at_grid_edge": False, "side": None, "verdict": "REJECT", "sharpness": 1.004, "boot_log_std": 0.08, "boot_rel_std": 0.08, "n_boot": 30}, "curve": {"grid": [1.0], "cost": [1.0]},
                  "motion": {"tilt_start_s": 3.0, "n_tilt_frames": 300, "tilt_duration_s": 19.0, "theta_median_deg": 33.0, "theta_p90_deg": 50.0, "theta_max_deg": 60.0, "theta_over_45_frac": 0.2},
                  "hand_box_short_ratio_front": 0.63, "guide": {"t_transition": 4.0, "guide": True, "reasons": [], "items": {}}, "traj": {"rows": rows, "ell_front_px": 400.0, "s1_front": 1.0, "segment": "tilt phase"}, "rating": "사용 가능"})
    return r


class GenReports(unittest.TestCase):
    def test_every_group_path_reports_and_excluded_g4_shows_the_failure_rate(self):
        res = [gen_fake("IMG_A.MOV", "G1", person="P1"), gen_fake("IMG_B.MOV", "G1", person="P2"), gen_fake("IMG_C.MOV", "G4", excluded=["손 검출 실패 프레임 35.0% > 20%"], detect_rate=0.65),
               gen_fake("IMG_D.MOV", "G2"), gen_fake("IMG_E.MOV", "G5")]
        groups = {"G1": {"verdict": "판단 불가", "reason": "영상 2개 < 3개", "n_videos": 2, "n_excluded": 0, "by_person": {"P1": {"n_videos": 1, "pooled_f": 1700.0, "at_grid_edge": False}}},
                  "G2": {"verdict": "영상 없음", "n_videos": 0, "n_excluded": 0}, "G3": {"verdict": "영상 없음"}, "G4": {"verdict": "판단 불가", "n_videos": 0, "n_excluded": 1, "detect_fail_rates": [0.35]}, "G5": {"verdict": "판정 없음 (궤적 분석용)"}}
        ver = {"verdict": "판단 불가", "reason": "사용 가능 영상 3개 < 5개", "n_usable": 3, "n_videos": 4}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.gen_report(d / "report.md", res, [], {}, CFG26)
            run.gen_summary(d / "summary.md", res, [], groups, ver, {}, CFG26)
            rep, summ = (d / "report.md").read_text(), (d / "summary.md").read_text()
        self.assertIn("손 검출 실패 프레임 35.0%", rep)
        self.assertIn("35.0%", summ)                                                  # the failure rate of the excluded G4 video
        for g in ("G1", "G2", "G3", "G4", "G5"):
            self.assertIn(g, summ)
        self.assertIn("판정 v2 검증: 판단 불가", summ.split("## 1.")[0])             # the verdicts are at the top
        self.assertIn("영상 없음", summ.split("## 1.")[0])
        self.assertNotIn("1680", rep.split("## 4.")[0])                            # no ref_f comparison before the trajectory section of report.md
        for name in ("P1", "P2"):
            self.assertIn(name, rep)                                               # people appear as person_id only
        self.assertNotIn("얼굴", rep + summ)


    def test_orientation_column_and_the_lens_comparison_with_7184(self):
        res = [gen_fake("IMG_L.MOV", "G3"), dict(gen_fake("IMG_P.MOV", "G3"), size=[1080, 1920])]
        for r in res:
            r["lens"], r["ref_f"] = "0.5x", 880.0
        ex = [{"video": "IMG_7184.MOV", "source": "22 (0.5x)", "ref_f": 880.0, "f": 987.8, "at_grid_edge": False, "sigma": 0.08, "tilt_s": 10.0, "theta_max": 62.0, "sigma_is_rel_std": True, "size": [1920, 1080],
               "n_frames": 199, "duration_s": 199 / 15.0, "rating": "사용 불가"}]
        groups = {g: {"verdict": "영상 없음"} for g in ("G1", "G2", "G4", "G5")}
        groups["G3"] = {"verdict": "판단 불가", "reason": "영상 2개 < 3개", "n_videos": 2, "n_excluded": 0}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.gen_report(d / "report.md", res, ex, {}, CFG26)
            run.gen_summary(d / "summary.md", res, ex, groups, {"verdict": "판단 불가", "reason": "x", "n_usable": 2, "n_videos": 2}, {}, CFG26)
            rep, summ = (d / "report.md").read_text(), (d / "summary.md").read_text()
        self.assertEqual(run.gen_orient([1920, 1080]), "가로")
        self.assertEqual(run.gen_orient([1080, 1920]), "세로")
        for text in (rep, summ):
            rows = [l for l in text.splitlines() if "IMG_P.MOV" in l]
            self.assertTrue(rows and all("세로" in l for l in rows), text[:200])           # the portrait video says 세로 in every per-video table
            rows = [l for l in text.splitlines() if "IMG_L.MOV" in l]
            self.assertTrue(rows and all("가로" in l for l in rows))
        side = summ.split("### 0.5배")[1].split("## 4.")[0]
        self.assertIn("IMG_7184.MOV", side)
        self.assertIn("IMG_L.MOV", side)
        self.assertIn("IMG_P.MOV", side)
        self.assertIn("13.3", side)                                                         # the stored length of 7184 (199 frames at 15 fps)


@unittest.skipUnless(VID7158.exists() and (ROOT / "runs/25_long_video_check/20261022T042710Z/results.json").exists() and (Path(CFG26["input_dir"]) / "IMG_7191.MOV").exists() and (Path(CFG26["input_dir"]) / "IMG_7192.MOV").exists(),
                     "the input videos IMG_7191 / IMG_7192 and 25's official run are needed")
class GenEndToEnd(unittest.TestCase):
    def test_25_reproduction_g5_split_outputs_and_no_tags(self):
        r25 = {j["video"]: j for j in json.loads((ROOT / "runs/25_long_video_check/20261022T042710Z/results.json").read_text())["new"]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG26)
            cfg["videos"] = [{"file": "IMG_7191.MOV", "group": "G2", "person_id": "P2", "phone": "teammate", "lens": "1x", "glove": False, "ref_f": 1680},
                             {"file": "IMG_7192.MOV", "group": "G5", "person_id": "P1", "phone": "main", "lens": "1x", "glove": False, "ref_f": 1680, "calib_end_s": 14.0}]
            cfg["workers"], cfg["n_workers"] = 2, 2
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r26.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, results, groups, ver = run.gen_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt", "per_frame.csv", "trajectory.csv"):
                self.assertTrue((root / f).exists(), f)
            for f in ("f_by_group.png", "error_by_rating.png", "trajectory_eps_error.png", "trajectory_3d.png"):
                self.assertTrue((root / "figures" / f).exists(), f)
            self.assertFalse((root / "_frames").exists())
            by = {r["video"]: r for r in results}
            a = by["IMG_7191.MOV"]
            self.assertEqual(a["ft"]["f"], r25["IMG_7191.MOV"]["ft"]["f"])                                  # 25's candidate value for the same video (the whole video is used)
            self.assertAlmostEqual(a["ft"]["boot_log_std"], r25["IMG_7191.MOV"]["ft"]["boot_log_std"], places=12)
            self.assertEqual(a["rating"], hf.gen_rating(a["motion"]["tilt_duration_s"], a["motion"]["theta_max_deg"], a["ft"]["boot_log_std"], CFG26["rating_v2"]))
            g5 = by["IMG_7192.MOV"]
            self.assertEqual(g5["calib_end_s"], 14.0)
            self.assertLessEqual(g5["motion"]["tilt_duration_s"], 14.0 - g5["motion"]["tilt_start_s"] + 1e-6)         # f uses the first 14 s only
            self.assertGreater(min(q["time_s"] for q in g5["traj"]["rows"]), 14.0)                                  # the trajectory is the work segment after it
            self.assertNotEqual(g5["ft"]["f"], r25["IMG_7192.MOV"]["ft"]["f"])
            self.assertEqual(groups["G5"]["verdict"], "판정 없음 (궤적 분석용)")
            self.assertEqual(groups["G3"]["verdict"], "영상 없음")
            self.assertEqual(groups["G2"]["verdict"], "판단 불가")                                                   # one video < 3
            self.assertEqual(ver["verdict"], "판단 불가")
            y = run.gen_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r26.yaml")
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            ref_f = re.compile(r"(?<![\d.])(?:1680|880)(?:\.0+)?(?!\d)(?!\.\d)")
            self.assertIsNone(ref_f.search(Path(y).read_text()))                                                  # no ref_f comparison in the results yaml
            self.assertIn("1680", (root / "summary.md").read_text())


# ====================================================================================================
# 27: main-camera extension (G6 distance, G7 left hand)
# ====================================================================================================

GRID27 = np.exp(np.linspace(np.log(0.35 * 1920), np.log(1.6 * 1920), 241))


def curve27(f0, scale=1.0):
    """A cost curve with its minimum at f0 (parabola in log f)."""
    return scale * (np.log(GRID27) - np.log(f0)) ** 2


def by_distance27(spec):
    """spec = {distance: [f0 of the included videos]} -> the by_distance input of ext_distance_verdict (n_total = the included videos unless given as (f0s, n_total))."""
    out = {}
    for d, v in spec.items():
        f0s, n_total = (v, len(v)) if isinstance(v, list) else v
        out[d] = {"curves": [curve27(f) for f in f0s], "n_total": n_total}
    return out


class ExtDistanceVerdict(unittest.TestCase):
    D = CFG27["distance"]

    def test_pass_fail_and_not_judged_on_synthetic_input(self):
        ok = hf.ext_distance_verdict(by_distance27({50: [1700, 1720], 60: [1680, 1690], 75: [1650, 1660]}), GRID27, 1680.0, self.D)
        self.assertEqual(ok["verdict"], "통과")
        self.assertLessEqual(ok["max_over_min"], 1.15)
        far = hf.ext_distance_verdict(by_distance27({50: [1700, 1720], 60: [1680, 1690], 75: [1450, 1450]}), GRID27, 1680.0, self.D)           # 75 cm pooled f 13.7% below ref_f
        self.assertEqual(far["verdict"], "실패")
        self.assertFalse(far["distances"][75.0 if 75.0 in far["distances"] else 75]["pooled_within_tol"])
        spread = hf.ext_distance_verdict(by_distance27({50: [1530, 1530], 60: [1680, 1680], 75: [1830, 1830]}), GRID27, 1680.0, self.D)           # each within +-10%, max / min 1.196 > 1.15
        self.assertEqual(spread["verdict"], "실패")
        self.assertTrue(all(r["pooled_within_tol"] for r in spread["distances"].values()))
        self.assertGreater(spread["max_over_min"], 1.15)
        few = hf.ext_distance_verdict(by_distance27({50: [1700, 1720], 60: [1680], 75: ([1650, 1660], 3)}), GRID27, 1680.0, self.D)              # 60 cm has one included video
        self.assertEqual(few["verdict"], "판단 불가")
        self.assertIn("60", few["reason"])
        gone = hf.ext_distance_verdict(by_distance27({50: [1700, 1720], 60: ([], 3), 75: [1650, 1660]}), GRID27, 1680.0, self.D)                # every 60 cm video excluded
        self.assertEqual(gone["verdict"], "판단 불가")
        self.assertEqual(gone["distances"][60]["n_excluded"], 3)
        self.assertEqual(hf.ext_distance_verdict({}, GRID27, 1680.0, self.D)["verdict"], "영상 없음")

    def test_thresholds_come_from_the_yaml(self):
        spec = by_distance27({50: [1530, 1530], 60: [1680, 1680], 75: [1830, 1830]})
        loose = hf.ext_distance_verdict(spec, GRID27, 1680.0, dict(self.D, ratio_max=1.25))
        self.assertEqual(loose["verdict"], "통과")
        tight = hf.ext_distance_verdict(by_distance27({50: [1700, 1700], 60: [1680, 1680], 75: [1650, 1650]}), GRID27, 1680.0, dict(self.D, pooled_tol=0.01))
        self.assertEqual(tight["verdict"], "실패")
        self.assertEqual(hf.ext_distance_verdict(by_distance27({50: [1700, 1700], 60: [1680, 1690, 1670], 75: [1650, 1650]}), GRID27, 1680.0, dict(self.D, min_videos_per_distance=3))["verdict"], "판단 불가")

    def test_pooled_estimate_of_a_distance_is_the_sum_of_its_cost_curves(self):
        curves = [curve27(1600), curve27(1700, 2.0), curve27(1750, 0.5)]
        r = hf.ext_distance_verdict({50: {"curves": curves, "n_total": 3}, 60: {"curves": curves[:2], "n_total": 2}, 75: {"curves": curves[1:], "n_total": 2}}, GRID27, 1680.0, self.D)
        ref = hf.long_pooled_estimate(np.sum(curves, axis=0)[None, :], GRID27)
        self.assertEqual(r["distances"][50]["pooled_f"], ref["f"])
        self.assertEqual(r["distances"][60]["pooled_f"], hf.long_pooled_estimate(np.sum(curves[:2], axis=0)[None, :], GRID27)["f"])
        b = hf.long_pooled_bootstrap(np.asarray(curves), GRID27, int(self.D["bootstrap_n"]), int(self.D["bootstrap_seed"]))
        self.assertAlmostEqual(r["distances"][50]["interval_95"][0], float(np.quantile(b["f"], 0.025)), places=9)

    def test_records_slope_rank_correlation_and_rounds(self):
        def vid(name, d, f0, rnd):
            return {"video": name, "group": "G6", "distance_cm": float(d), "round": rnd, "ft": {"f": f0, "at_grid_edge": False}, "curve": {"cost": curve27(f0).tolist()}}
        res = [vid("IMG_1", 50, 1500, 1), vid("IMG_2", 60, 1600, 1), vid("IMG_3", 75, 1800, 1), vid("IMG_4", 50, 1520, 2), vid("IMG_5", 60, 1620, 2), vid("IMG_6", 75, 1830, 2)]
        entries = [{"file": r["video"], "group": "G6", "distance_cm": r["distance_cm"]} for r in res]
        self.assertEqual(hf.ext_rounds(entries), {"IMG_1": 1, "IMG_2": 1, "IMG_3": 1, "IMG_4": 2, "IMG_5": 2, "IMG_6": 2})
        rec = hf.ext_distance_records(res, entries, GRID27, self.D)
        self.assertGreater(rec["slope_video"], 0)                                                          # f grows with distance in this input
        rx, ry = np.array([0.5, 2.5, 4.5, 0.5, 2.5, 4.5]), np.array([0.0, 2.0, 4.0, 1.0, 3.0, 5.0])                    # mean ranks of the tied distances, ranks of f
        self.assertAlmostEqual(rec["spearman_video"], float(np.corrcoef(rx, ry)[0, 1]))
        self.assertEqual(rec["rounds"][1]["videos"], ["IMG_1", "IMG_2", "IMG_3"])
        self.assertAlmostEqual(rec["rounds"][2]["pooled_f"], hf.long_pooled_estimate(np.array([curve27(1520), curve27(1620), curve27(1830)]), GRID27)["f"])
        self.assertAlmostEqual(hf.ext_spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(hf.ext_spearman([1, 2], [1, 2]))
        self.assertIsNone(hf.ext_loglog_slope([50, 50], [1, 2]))
        self.assertAlmostEqual(hf.ext_loglog_slope([50, 100], [100, 200]), 1.0)

    def test_entry_check_and_hand_compare(self):
        good = [{"file": "a.MOV", "group": "G6", "distance_cm": 50, "hand": "right", "ref_f": 1680}]
        self.assertEqual(hf.ext_check_videos(good), good)
        for bad in ({"hand": "both"}, {"group": "G3"}, {"distance_cm": 0}, {"ref_f": None}):
            with self.assertRaises(SystemExit):
                hf.ext_check_videos([dict(good[0], **bad)])
        with self.assertRaises(SystemExit):
            hf.ext_check_videos([])
        res = [{"video": "R1", "group": "G6", "distance_cm": 60.0, "ft": {"f": 1700.0, "at_grid_edge": False}, "curve": {"cost": curve27(1700).tolist()}},
               {"video": "R2", "group": "G6", "distance_cm": 50.0, "ft": {"f": 1500.0, "at_grid_edge": False}, "curve": {"cost": curve27(1500).tolist()}},
               {"video": "L1", "group": "G7", "distance_cm": 60.0, "ft": {"f": 1800.0, "at_grid_edge": False}, "curve": {"cost": curve27(1800).tolist()}}]
        h = hf.ext_hand_compare(res, GRID27, CFG27["generalization"], 60.0)
        self.assertEqual(h["right"]["videos"], ["R1"])                                                      # only the 60 cm right-hand videos
        self.assertAlmostEqual(h["ratio_left_over_right"], h["left"]["pooled_f"] / h["right"]["pooled_f"])


def mirror27(g, W):
    """The same synthetic hand seen as a left hand: image x -> W - x, handedness 0."""
    m = copy.deepcopy(g)
    m["kp"][..., 0] = W - g["kp"][..., 0]
    m["box"][:, [0, 2]] = W - g["box"][:, [2, 0]]
    m["cls"] = np.zeros_like(g["cls"])
    return m


class ExtLeftHand(unittest.TestCase):
    """The pipeline treats a left hand like a right hand: the 21 joints keep MediaPipe's order for both hands and the template is built from the video's own front window,
    so the mirrored image of a hand gives the same f, the same wrist-excluded estimate and the same motion indicators."""

    def test_mirrored_hand_gives_the_same_f_and_the_wrist_is_still_excluded(self):
        ce = hf.pseg_wrist_cfg(PIPE)
        W = SYN["image"][0]
        for seed in (0, 1):
            g = synth_wrist(seed=seed, tilt_frames=60)
            m = mirror27(g, W)
            ra = analyze_synth(g, methods_full=("fixed_template",), cfg=ce)
            rb = analyze_synth(m, methods_full=("fixed_template",), cfg=ce)
            fa, fb = ra["solutions"]["fixed_template"]["f"], rb["solutions"]["fixed_template"]["f"]
            self.assertLess(abs(fb / fa - 1.0), 1e-6, (seed, fa, fb))
            self.assertEqual(ra["solutions"]["fixed_template"]["f"] > 0, True)
            self.assertAlmostEqual(ra["per_method"]["fixed_template"]["bootstrap"]["rel_std"], rb["per_method"]["fixed_template"]["bootstrap"]["rel_std"], places=6)
        garbage = {}
        for name, gx in (("right", synth_wrist((50.0, 60.0))), ("left", mirror27(synth_wrist((50.0, 60.0)), W))):
            other = synth_wrist((1700.0, 900.0))
            other = other if name == "right" else mirror27(other, W)
            garbage[name] = (analyze_synth(gx, methods_full=("fixed_template",), cfg=ce), analyze_synth(other, methods_full=("fixed_template",), cfg=ce))
        for name, (x, y) in garbage.items():
            self.assertEqual(x["solutions"]["fixed_template"]["f"], y["solutions"]["fixed_template"]["f"], name)                                  # the wrist point never enters, left or right
            self.assertEqual(x["template"], y["template"], name)

    def test_motion_indicators_are_the_same_for_the_mirrored_hand(self):
        ce = hf.pseg_wrist_cfg(PIPE)
        g = synth_wrist(seed=2, tilt_frames=60)
        m = mirror27(g, SYN["image"][0])
        out = []
        for x in (g, m):
            a = analyze_synth(x, methods_full=(), cfg=ce)
            out.append(hf.ext_motion(x["kp"], x["mask"], x["present"], x["box"], x["times"], a, ce, x["size"]))
        self.assertGreater(out[0]["n_tilt_frames"], 30)
        self.assertEqual(out[0]["n_tilt_frames"], out[1]["n_tilt_frames"])
        self.assertEqual(out[0]["axis_bins"], out[1]["axis_bins"])
        for k in ("speed", "bone_ratio_pct", "spread_change_deg"):
            for q in ("median", "p90"):
                self.assertAlmostEqual(out[0][k][q], out[1][k][q], places=5, msg=(k, q))


def ext_fake(video, group, distance, hand, rnd, f=1700.0, sigma=0.08):
    r = gen_fake(video, group)
    r.update({"distance_cm": float(distance), "hand": hand, "round": rnd, "person_id": None, "ref_f": 1680.0})
    r["ft"]["f"], r["ft"]["boot_log_std"] = f, sigma
    r["dynamics"] = {"n_tilt_frames": 300, "axis_bins": 5, "axis_counts": [3, 5, 9, 12, 4, 2], "axis_dominant": None, "speed": {"median": 0.8, "p90": 1.9, "n": 300},
                     "bone_ratio_pct": {"median": 3.1, "p90": 6.0, "n": 300}, "spread_change_deg": {"median": 4.2, "p90": 9.0, "n": 300}}
    r.pop("traj", None)
    return r


class ExtReports(unittest.TestCase):
    def _run(self, d, results, groups, ver, hand, rec, cfg=CFG27):
        run.ext_report(d / "report.md", results, groups, rec, hand, cfg)
        run.ext_summary(d / "summary.md", results, groups, ver, rec, hand, [], cfg)
        return (d / "report.md").read_text(), (d / "summary.md").read_text()

    def test_verdicts_on_top_dynamics_table_and_the_photographer_note(self):
        res = [ext_fake("IMG_7217.MOV", "G6", 60, "right", 1), ext_fake("IMG_7220.MOV", "G6", 60, "right", 2), ext_fake("IMG_7223.MOV", "G6", 60, "right", 3),
               ext_fake("IMG_7225.MOV", "G7", 60, "left", None, f=1750.0), ext_fake("IMG_7226.MOV", "G7", 60, "left", None, f=1760.0), ext_fake("IMG_7227.MOV", "G7", 60, "left", None, f=1740.0)]
        g6 = hf.ext_distance_verdict(by_distance27({60: [1700, 1700, 1700]}), GRID27, 1680.0, CFG27["distance"])
        g7 = hf.gen_group_verdict([curve27(1750), curve27(1760), curve27(1740)], GRID27, [1750.0, 1760.0, 1740.0], 1680.0, CFG27["generalization"], 0, 3)
        ver = {"verdict": "판단 불가", "reason": "사용 가능 영상 6개 ... x", "n_usable": 6, "n_videos": 6}
        hand = hf.ext_hand_compare([dict(r, curve={"cost": curve27(r["ft"]["f"]).tolist()}) for r in res], GRID27, CFG27["generalization"], 60.0)
        with tempfile.TemporaryDirectory() as d:
            rep, summ = self._run(Path(d), res, {"G6": g6, "G7": g7}, ver, hand, {})
        top = summ.split("## 1.")[0]
        for key in ("G6 거리", "G7 왼손", "판정 v2 검증"):
            self.assertIn(key, top)
        self.assertIn("촬영자 메모: 왼손 촬영이 오른손보다 어색했다고 함", summ.split("## 4.")[0].split("## 3.")[1])      # in the G7 section
        for text in (rep, summ):
            self.assertIn("기울기 축 덮임", text)
            self.assertIn("마디 길이 비 변화", text)
            self.assertIn("손가락 벌어짐 변화", text)
            for vid in ("IMG_7217.MOV", "IMG_7225.MOV"):
                self.assertTrue(any(vid in l and "|" in l and "0.80" in l for l in text.splitlines()), vid)         # the speed median of the video is in its row
        self.assertNotIn("1680", rep)                                                                          # no ref_f comparison in report.md
        self.assertNotIn("촬영자 메모", rep)

    def test_every_status_path_reports(self):
        res = [ext_fake("IMG_A.MOV", "G6", 50, "right", 1), dict(gen_fake("IMG_B.MOV", "G6", excluded=["손 검출 실패 프레임 35.0% > 20%"], detect_rate=0.65), distance_cm=60.0, hand="right", round=1, person_id=None),
               ext_fake("IMG_C.MOV", "G7", 60, "left", None)]
        g6 = hf.ext_distance_verdict(by_distance27({50: [1700], 60: ([], 1)}), GRID27, 1680.0, CFG27["distance"])
        groups = {"G6": g6, "G7": {"verdict": "판단 불가", "reason": "영상 1개 < 3개", "n_videos": 1, "n_excluded": 0}}
        with tempfile.TemporaryDirectory() as d:
            rep, summ = self._run(Path(d), res, groups, {"verdict": "판단 불가", "reason": "x", "n_usable": 2, "n_videos": 2}, {}, {})
        self.assertEqual(g6["verdict"], "판단 불가")
        self.assertIn("손 검출 실패 프레임 35.0%", rep)
        self.assertIn("판단 불가", summ.split("## 1.")[0])


R26 = ROOT / "runs/26_generalization/20261026T031941Z/results.json"


@unittest.skipUnless(R26.exists() and all((Path(CFG27["input_dir"]) / f"IMG_{n}.MOV").exists() for n in (7213, 7214, 7215)),
                     "26's official run and the input videos IMG_7213 / 7214 / 7215 are needed")
class ExtEndToEnd(unittest.TestCase):
    def test_26_reproduction_outputs_and_no_tags(self):
        r26 = {j["video"]: j for j in json.loads(R26.read_text())["videos"]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG27)
            cfg["videos"] = [{"file": "IMG_7213.MOV", "group": "G6", "distance_cm": 60, "hand": "right", "ref_f": 880},
                             {"file": "IMG_7214.MOV", "group": "G6", "distance_cm": 60, "hand": "right", "ref_f": 880},
                             {"file": "IMG_7215.MOV", "group": "G7", "distance_cm": 60, "hand": "left", "ref_f": 880}]
            cfg["workers"], cfg["n_workers"] = 3, 3
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r27.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, results, groups, ver = run.ext_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt", "per_frame.csv"):
                self.assertTrue((root / f).exists(), f)
            for f in ("f_by_distance.png", "f_by_group.png"):
                self.assertTrue((root / "figures" / f).exists(), f)
            self.assertFalse((root / "_frames").exists())
            for r in results:
                o = r26[r["video"]]                                                                          # 26's official values for the same video: same candidate, same numbers
                self.assertEqual(r["ft"]["f"], o["ft"]["f"], r["video"])
                self.assertAlmostEqual(r["ft"]["boot_log_std"], o["ft"]["boot_log_std"], places=12)
                self.assertEqual(r["rating"], o["rating"])
                self.assertEqual(r["motion"]["n_tilt_frames"], o["motion"]["n_tilt_frames"])
                self.assertEqual(r["dynamics"]["n_tilt_frames"], o["motion"]["n_tilt_frames"])
                self.assertEqual(r["guide"]["guide"], o["guide"]["guide"])
            g6 = [r for r in results if r["group"] == "G6"]
            pooled = hf.long_pooled_estimate([r["curve"]["cost"] for r in g6], np.array(g6[0]["curve"]["grid"]))
            self.assertEqual(groups["G6"]["distances"][60.0]["pooled_f"], pooled["f"])                         # the distance's pooled f = the sum of its two videos' cost curves
            self.assertEqual(groups["G6"]["verdict"], "통과" if abs(pooled["f"] / 880.0 - 1.0) <= CFG27["distance"]["pooled_tol"] else "실패")   # one distance: max / min = 1
            self.assertEqual(groups["G7"]["verdict"], "판단 불가")                                           # one video < 3
            self.assertEqual(ver["verdict"], "판단 불가")
            y = run.ext_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r27.yaml")
            texts = "".join(p.read_text(errors="ignore") for p in list(root.rglob("*")) + [Path(y)] if p.is_file() and p.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            self.assertIsNone(re.compile(r"(?<![\d.])(?:1680|880)(?:\.0+)?(?!\d)(?!\.\d)").search(Path(y).read_text()))        # no ref_f comparison in the results yaml
            self.assertIn("880", (root / "summary.md").read_text())


# ====================================================================================================
# 29: 0.5x lens at about 60 cm, hypothesis "the method always reads about 62.5 cm"
# ====================================================================================================

def pooled29(f0s, n_excluded=0, cfg=None):
    """The pooled estimate (ext_pooled) of videos whose cost curves have their minima at f0s, as wide_hypothesis_verdict takes it."""
    p = hf.ext_pooled([curve27(f) for f in f0s], GRID27, (cfg or CFG29)["hypothesis"])
    p["n_excluded"] = n_excluded
    return p


class WideHypothesis(unittest.TestCase):
    H = CFG29["hypothesis"]

    def test_support_refute_open_and_not_judged_on_synthetic_input(self):
        for f0s, want in (([917, 920, 915], "가설 지지"), ([1000, 1000], "가설 지지"), ([1200, 1210, 1190], "가설 반박"), ([700, 710], "가설 반박"), ([1060, 1070, 1050], "판단 보류"), ([790, 800], "판단 보류")):
            v = hf.wide_hypothesis_verdict(pooled29(f0s), self.H)
            self.assertEqual(v["verdict"], want, (f0s, v.get("pooled_f")))
        one = hf.wide_hypothesis_verdict(pooled29([917], n_excluded=2), self.H)
        self.assertEqual(one["verdict"], "판단 불가")
        self.assertIn("1개", one["reason"])
        none = hf.wide_hypothesis_verdict({"n_videos": 0, "n_excluded": 3}, self.H)
        self.assertEqual(none["verdict"], "판단 불가")

    def test_ranges_are_the_ones_fixed_in_the_yaml(self):
        v = hf.wide_hypothesis_verdict(pooled29([917, 917]), self.H)
        self.assertAlmostEqual(v["support_range"][0], 825.3, places=1)
        self.assertAlmostEqual(v["support_range"][1], 1008.7, places=1)
        self.assertAlmostEqual(v["refute_above"], 1146.25, places=2)
        self.assertAlmostEqual(v["refute_below"], 733.6, places=1)
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([826]), dict(self.H, min_videos=1))["verdict"], "가설 지지")
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([824]), dict(self.H, min_videos=1))["verdict"], "판단 보류")                       # just outside the support range, above the refuting 0.8
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([1009.5]), dict(self.H, min_videos=1))["verdict"], "판단 보류")
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([1146.5]), dict(self.H, min_videos=1))["verdict"], "가설 반박")
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([733.0]), dict(self.H, min_videos=1))["verdict"], "가설 반박")
        shifted = dict(self.H, predicted_f=1833.0)                                                                                     # another predicted f moves every range with it
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([1840, 1830], cfg=CFG29), shifted)["verdict"], "가설 지지")
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([917, 917]), shifted)["verdict"], "가설 반박")
        self.assertEqual(hf.wide_hypothesis_verdict(pooled29([917, 917]), dict(self.H, min_videos=3))["verdict"], "판단 불가")

    def test_a_grid_edge_value_refutes_and_is_reported_as_such(self):
        v = hf.wide_hypothesis_verdict(pooled29([5000, 5000]), self.H)                                                             # minimum beyond the grid: clipped to the upper edge
        self.assertEqual(v["verdict"], "가설 반박")
        self.assertTrue(v["at_grid_edge"])
        self.assertEqual(v["side"], "상한")
        lowv = hf.wide_hypothesis_verdict(pooled29([300, 300]), self.H)
        self.assertEqual((lowv["verdict"], lowv["side"]), ("가설 반박", "하한"))

    def test_implied_distance_and_entry_check(self):
        self.assertAlmostEqual(hf.wide_implied_distance(917.0, 60, 880.0), 62.5, places=1)
        self.assertAlmostEqual(hf.wide_implied_distance(1054.7, 30, 880.0), 35.9557, places=3)
        self.assertAlmostEqual(hf.wide_implied_distance(2092.4, 50, 1680.0), 62.274, places=2)
        self.assertIsNone(hf.wide_implied_distance(None, 60, 880.0))
        good = [{"file": "a.MOV", "group": "G8", "lens": "0.5x", "distance_cm": 60, "hand": "right", "ref_f": 880}]
        self.assertEqual(hf.wide_check_videos(good), good)
        for bad in ({"lens": None}, {"hand": "both"}, {"distance_cm": 0}, {"ref_f": -1}):
            with self.assertRaises(SystemExit):
                hf.wide_check_videos([dict(good[0], **bad)])
        with self.assertRaises(SystemExit):
            hf.wide_check_videos([])
        self.assertEqual([e["file"] for e in CFG29["videos"]], ["IMG_7231.MOV", "IMG_7232.MOV", "IMG_7233.MOV"])
        self.assertTrue(all(e["lens"] == "0.5x" and e["distance_cm"] == 60 and e["hand"] == "right" and e["ref_f"] == 880 for e in CFG29["videos"]))


class WidePortrait(unittest.TestCase):
    def test_principal_point_of_a_portrait_video_is_the_screen_centre(self):
        self.assertEqual(hf.principal_point(PIPE["estimate"], 1080, 1920), (540.0, 960.0))
        g = synth_wrist(seed=1, tilt_frames=40)
        W, H = 1080, 1920
        g["size"] = (W, H)
        g["kp"] = g["kp"] - np.array([SYN["image"][0] / 2, SYN["image"][1] / 2]) + np.array([W / 2, H / 2])                         # the same hand around the portrait centre
        g["box"] = g["box"] - np.array([SYN["image"][0] / 2, SYN["image"][1] / 2] * 2) + np.array([W / 2, H / 2] * 2)
        a = analyze_synth(g, methods_full=(), cfg=hf.pseg_wrist_cfg(PIPE))
        self.assertEqual(a["principal_point"], [540.0, 960.0])


def wide_fake(video, f=900.0, sigma=0.06, **kw):
    r = gen_fake(video, "G8")
    r.update({"lens": "0.5x", "distance_cm": 60.0, "hand": "right", "ref_f": 880.0, "person_id": None, "size": [1080, 1920], "front_jitter_pct": 0.42, "principal_point": [540.0, 960.0],
              "rating": "사용 가능", "implied_distance_cm": hf.wide_implied_distance(f, 60, 880.0), "anycalib": {"anycalib_pinhole": {"median": 1763.5, "n": 300}, "anycalib_gen": {"median": 1692.5, "n": 300}}})
    r["ft"]["f"], r["ft"]["boot_log_std"] = f, sigma
    r.update(kw)
    r.pop("traj", None)
    return r


class WideReports(unittest.TestCase):
    def test_verdict_on_top_values_only_report_and_the_comparison_table(self):
        res = [wide_fake("IMG_A.MOV", 900.0), wide_fake("IMG_B.MOV", 930.0), wide_fake("IMG_C.MOV", 905.0)]
        v = hf.wide_hypothesis_verdict(pooled29([900, 930, 905]), CFG29["hypothesis"])
        v["ref_f"], v["distance_cm"] = 880.0, 60.0
        v["implied_distance_cm"] = hf.wide_implied_distance(v["pooled_f"], 60.0, 880.0)
        v["implied_distance_interval_cm"] = [hf.wide_implied_distance(x, 60.0, 880.0) for x in v["interval_95"]]
        table = [{"condition": "1배 50 cm (실험 27)", "distance_cm": 50.0, "pooled_f": 2092.4, "ref_f": 1680.0, "implied_distance_cm": 62.27},
                 {"condition": "0.5배 30 cm (실험 26)", "distance_cm": 30.0, "pooled_f": 1054.7, "ref_f": 880.0, "implied_distance_cm": 35.96},
                 {"condition": "0.5배 60 cm (이번)", "distance_cm": 60.0, "pooled_f": v["pooled_f"], "ref_f": 880.0, "implied_distance_cm": v["implied_distance_cm"]}]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.wide_report(d / "report.md", res, v, CFG29)
            run.wide_summary(d / "summary.md", res, v, table, CFG29)
            run.wide_figures(d, res, v, table, CFG29)
            rep, summ = (d / "report.md").read_text(), (d / "summary.md").read_text()
            self.assertTrue((d / "f_by_video.png").exists() and (d / "implied_distance.png").exists())
        self.assertIn("가설 판정: 가설 지지", summ.split("## 1.")[0])
        for key in ("0.5배 30 cm (실험 26)", "1배 50 cm (실험 27)", "AnyCalib", "1763.5", "암시 거리"):
            self.assertIn(key, summ)
        self.assertNotIn("880", rep)                                                                                   # no ref_f in report.md
        self.assertNotIn("암시 거리(cm)", rep)                                                                          # no implied-distance column or value (900 px -> 61.4 cm) in report.md
        self.assertNotIn("61.4", rep)
        self.assertNotIn("1763.5", rep)                                                                                # no AnyCalib value in report.md
        self.assertIn("키포인트 떨림", rep)
        self.assertIn("0.42", rep)

    def test_not_judged_and_excluded_paths_report(self):
        res = [wide_fake("IMG_A.MOV"), dict(gen_fake("IMG_B.MOV", "G8", excluded=["가장자리 접촉 프레임 6.3% > 5%"]), lens="0.5x", distance_cm=60.0, hand="right", ref_f=880.0, person_id=None, size=[1080, 1920])]
        v = hf.wide_hypothesis_verdict(pooled29([900], n_excluded=1), CFG29["hypothesis"])
        v["ref_f"], v["distance_cm"] = 880.0, 60.0
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.wide_report(d / "report.md", res, v, CFG29)
            run.wide_summary(d / "summary.md", res, v, [], CFG29)
            rep, summ = (d / "report.md").read_text(), (d / "summary.md").read_text()
        self.assertIn("가설 판정: 판단 불가", summ)
        self.assertIn("가장자리 접촉 프레임 6.3%", rep)
        self.assertIn("세로", rep)


R27 = ROOT / "runs/27_main_camera_extension/20261026T044322Z/results.json"
R26B = ROOT / "runs/26_generalization/20261026T031941Z/results.json"


@unittest.skipUnless(R27.exists() and R26B.exists() and all((Path(CFG29["input_dir"]) / f"IMG_{n}.MOV").exists() for n in (7215, 7217, 7218)),
                     "27's and 26's official runs and the input videos IMG_7215 / 7217 / 7218 are needed")
class WideEndToEnd(unittest.TestCase):
    def test_27_reproduction_portrait_principal_point_anycalib_and_no_tags(self):
        r27 = {j["video"]: j for j in json.loads(R27.read_text())["videos"]}
        r26 = {j["video"]: j for j in json.loads(R26B.read_text())["videos"]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG29)
            cfg["videos"] = [{"file": "IMG_7217.MOV", "group": "G8", "lens": "1x", "distance_cm": 60, "hand": "right", "ref_f": 1680},
                             {"file": "IMG_7218.MOV", "group": "G8", "lens": "1x", "distance_cm": 75, "hand": "right", "ref_f": 1680},
                             {"file": "IMG_7215.MOV", "group": "G8", "lens": "0.5x", "distance_cm": 60, "hand": "right", "ref_f": 880}]
            cfg["workers"], cfg["n_workers"] = 3, 3
            cfg["anycalib_compare"]["enabled"] = GPU
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r29.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, results, verdict = run.wide_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            for f in ("f_by_video.png", "implied_distance.png"):
                self.assertTrue((root / "figures" / f).exists(), f)
            self.assertFalse((root / "_frames").exists())
            by = {r["video"]: r for r in results}
            for vid in ("IMG_7217.MOV", "IMG_7218.MOV"):                                                                 # 27's official values for the same videos
                self.assertEqual(by[vid]["ft"]["f"], r27[vid]["ft"]["f"], vid)
                self.assertAlmostEqual(by[vid]["ft"]["boot_log_std"], r27[vid]["ft"]["boot_log_std"], places=12)
                self.assertEqual(by[vid]["rating"], r27[vid]["rating"])
            p = by["IMG_7215.MOV"]                                                                                       # a portrait video of 26: the same f, principal point at the screen centre
            self.assertEqual(p["ft"]["f"], r26["IMG_7215.MOV"]["ft"]["f"])
            self.assertEqual(p["size"], [1080, 1920])
            self.assertEqual(p["principal_point"], [540.0, 960.0])
            self.assertGreater(p["front_jitter_pct"], 0.0)
            if GPU:
                self.assertTrue(all(set(r["anycalib"]) == {"anycalib_pinhole", "anycalib_gen"} for r in results))
            self.assertIn(verdict["verdict"], ("가설 지지", "가설 반박", "판단 보류", "판단 불가"))
            y = run.wide_results_yaml(root, cfg)
            self.assertEqual(Path(y), d / "r29.yaml")
            texts = "".join(q.read_text(errors="ignore") for q in list(root.rglob("*")) + [Path(y)] if q.is_file() and q.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            self.assertIsNone(re.compile(r"(?<![\d.])(?:1680|880)(?:\.0+)?(?!\d)(?!\.\d)").search(Path(y).read_text()))        # no ref_f comparison in the results yaml
            self.assertNotIn("implied", Path(y).read_text())


# ====================================================================================================
# 30: uncertainty joint sets that fit the wrist-excluded candidate
# ====================================================================================================

class JsetSets(unittest.TestCase):
    def test_new_sets_have_no_wrist_enough_points_and_flat17_is_kept(self):
        jc = CFG30["jointset"]
        cfg = hf.jset_config(CFG30["pipeline"], jc, "exclude")
        est = cfg["joints"]["sets"][cfg["joints"]["estimate_set"]]
        self.assertEqual(est, list(range(5, 21)))                                                      # the candidate's estimate set (wrist excluded) is unchanged
        names = jc["uncertainty_sets"]
        self.assertEqual(cfg["joints"]["uncertainty_sets"], names)
        self.assertIn("flat17", names)
        for k in names:
            ids = cfg["joints"]["sets"][k]
            self.assertNotIn(0, ids, k)
            self.assertTrue(set(ids) <= set(est), k)
            self.assertGreaterEqual(len(ids), 8, k)                                                    # the old palm5 had 4 collinear points left after the wrist was removed
        old = hf.pseg_wrist_cfg(CFG30["pipeline"])
        self.assertEqual(len(old["joints"]["sets"]["palm5"]), 4)
        self.assertEqual(CFG30["pipeline"]["joints"]["uncertainty_sets"], ["palm5", "palm_proximal9", "flat17"])    # the old setting is untouched in the yaml
        keep = hf.jset_config(CFG30["pipeline"], jc, "keep")
        self.assertIn(0, keep["joints"]["sets"]["flat17"])                                              # keep mode keeps the wrist

    def test_range_and_edge_helpers(self):
        grid = np.geomspace(672.0, 3072.0, 60)
        fs = {"a": 1000.0, "b": 1100.0, "c": 672.0}
        self.assertAlmostEqual(hf.jset_range(fs, 1000.0), (1100.0 - 672.0) / 1000.0)
        self.assertEqual(hf.jset_edge_sets(fs, grid), ["c"])
        self.assertEqual(hf.jset_edge_sets({"x": 3072.0, "y": 2000.0}, grid), ["x"])
        self.assertTrue(np.isnan(hf.jset_range({"a": 1.0}, 1.0)))

    def test_thresholds_and_decision_rule_are_in_the_yaml(self):
        self.assertEqual(CFG30["decision"], {"min_within_share": 0.5, "max_edge_share": 0.10})
        self.assertEqual(CFG30["pipeline"]["verdict"]["rough"]["jointset_range_max"], 0.12)
        self.assertEqual(len(CFG30["videos"]), 33)
        self.assertTrue(all(e["ref_f"] == (1680 if e["lens"] == "1x" else 880) for e in CFG30["videos"]))
        self.assertEqual(sum(1 for e in CFG30["videos"] if e.get("source17")), 8)


class JsetSynthetic(unittest.TestCase):
    def test_new_sets_agree_on_exact_planar_hands_and_f_is_the_candidates(self):
        g = synth_wrist(seed=1, tilt_frames=120)
        jc = CFG30["jointset"]
        pc = CFG30["pipeline"]
        cfg_new, cfg_old = hf.jset_config(pc, jc, "exclude"), hf.pseg_wrist_cfg(pc)
        a_new = analyze_synth(g, methods_full=("fixed_template",), cfg=cfg_new, n_boot=2)
        a_old = analyze_synth(g, methods_full=("fixed_template",), cfg=cfg_old, n_boot=2)
        f = a_new["solutions"]["fixed_template"]["f"]
        self.assertEqual(f, a_old["solutions"]["fixed_template"]["f"])                                 # the estimate does not depend on the uncertainty sets
        self.assertLess(abs(f / g["f_true"] - 1.0), 0.02)
        pm = a_new["per_method"]["fixed_template"]
        self.assertLess(pm["jointset_range"], float(pc["verdict"]["rough"]["jointset_range_max"]))      # on an ideal planar hand the new sets agree within the ROUGH limit
        self.assertEqual(set(pm["jointsets"]), set(jc["uncertainty_sets"]))
        self.assertEqual(set(a_old["per_method"]["fixed_template"]["jointsets"]), {"palm5", "palm_proximal9", "flat17"})


R27J = ROOT / "runs/27_main_camera_extension/20261026T044322Z/results.json"
R26J = ROOT / "runs/26_generalization/20261026T031941Z/results.json"


@unittest.skipUnless(R27J.exists() and R26J.exists() and all((Path(CFG30["input_dir"]) / f"IMG_{n}.MOV").exists() for n in (7213, 7220)),
                     "26's and 27's official runs and the input videos IMG_7213 / IMG_7220 are needed")
class JsetEndToEnd(unittest.TestCase):
    def test_reproduction_of_f_old_sets_verdict_outputs_and_no_tags(self):
        r27 = {j["video"]: j for j in json.loads(R27J.read_text())["videos"]}
        r26 = {j["video"]: j for j in json.loads(R26J.read_text())["videos"]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG30)
            cfg["videos"] = [e for e in CFG30["videos"] if e["file"] in ("IMG_7213.MOV", "IMG_7220.MOV")]
            cfg["workers"], cfg["n_workers"] = 2, 2
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r30.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, results, summ = run.jset_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            self.assertTrue((root / "figures/jointset_range_old_vs_new.png").exists())
            self.assertFalse((root / "_frames").exists())
            by = {r["video"]: r for r in results}
            self.assertEqual(by["IMG_7220.MOV"]["f"], r27["IMG_7220.MOV"]["ft"]["f"])                  # the candidate's f is 27's / 26's
            self.assertEqual(by["IMG_7213.MOV"]["f"], r26["IMG_7213.MOV"]["ft"]["f"])
            for r in results:
                self.assertTrue(r["f_same_as_old_sets"])
                self.assertEqual(r["verdict_old_sets"], "REJECT")                                      # what 25-29 stored: every video REJECT
                self.assertEqual(r["failed_rough_old_sets"], ["jointset_range"])
                self.assertEqual(r["detection_source"].split(" ")[0], "cache")
            self.assertAlmostEqual(by["IMG_7220.MOV"]["jointset_range_old"], 0.9235, places=3)
            y = run.jset_results_yaml(root, cfg)
            texts = "".join(q.read_text(errors="ignore") for q in list(root.rglob("*")) + [Path(y)] if q.is_file() and q.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)


# ====================================================================================================
# 31: do the verdict's diagnostics predict the real f error?
# ====================================================================================================

def cal_fake_rows(n, predictor_fn, seed=0, lens="1x"):
    """Rows like cal_rows': error from a hidden rule, every predictor of the yaml filled (random unless predictor_fn sets it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = {"video": f"V{i}", "lens": lens, "ref_f": 1680.0, "excluded": False, "at_grid_edge": False, "tilt_duration_s": 20.0, "theta_max": 60.0, "boot_log_std": 0.05, "verdict_new": "REJECT"}
        for k in CFG31["predictors"]:
            r[k] = float(rng.uniform(0.0, 1.0))
        r.update(predictor_fn(i, rng, r))
        r["signed_error"] = r["error"]
        r["f"] = 1680.0 * (1 + r["error"])
        r["accurate"] = bool(r["error"] <= CFG31["accurate_tol"])
        rows.append(r)
    return rows


class CalStats(unittest.TestCase):
    def test_spearman_interval_balanced_accuracy_and_thresholds(self):
        x = np.arange(20.0)
        sp = hf.cal_spearman_ci(x, x * 2 + 1, 500, 42)
        self.assertAlmostEqual(sp["rho"], 1.0)
        self.assertGreater(sp["ci"][0], 0.9)
        self.assertEqual(hf.cal_spearman_ci(x, np.zeros(20), 100, 1)["rho"], None)                      # a constant input has no correlation
        ba = hf.cal_balanced_accuracy([True, True, False, False], [True, False, False, False])
        self.assertAlmostEqual(ba["sensitivity"], 1.0)
        self.assertAlmostEqual(ba["specificity"], 2 / 3)
        self.assertAlmostEqual(ba["balanced_accuracy"], 0.5 * (1.0 + 2 / 3))
        self.assertIsNone(hf.cal_balanced_accuracy([True, True], [True, True])["balanced_accuracy"])     # one class only
        truth = np.array([True] * 10 + [False] * 10)
        xs = np.r_[np.linspace(0.0, 0.4, 10), np.linspace(0.6, 1.0, 10)]
        b = hf.cal_best_threshold(xs, truth, good_below=True)
        self.assertEqual(b["balanced_accuracy"], 1.0)
        self.assertTrue(0.4 < b["threshold"] < 0.6)
        self.assertEqual(hf.cal_loo(xs, truth, True)["balanced_accuracy"], 1.0)
        self.assertEqual(hf.cal_best_threshold(-xs, truth, good_below=False)["balanced_accuracy"], 1.0)  # the other direction
        bas = [hf.cal_loo(np.random.default_rng(s).uniform(size=20), truth, True)["balanced_accuracy"] for s in range(30)]
        self.assertLess(float(np.mean(bas)), 0.65)                                                     # unrelated predictors stay near chance by leave-one-out (single draws can reach 0.7 by luck)


class CalAnalyse(unittest.TestCase):
    def test_a_predictor_that_tracks_the_error_makes_the_verdict_calibratable(self):
        def fn(i, rng, r):
            x = float(rng.uniform(0.0, 1.0))
            return {"boot_log_std": x, "error": 0.02 + 0.30 * x}
        rows = cal_fake_rows(30, fn)
        res = hf.cal_analyse(rows, CFG31)
        self.assertEqual(res["decision"], "보정 가능")
        self.assertIn("boot_log_std", res["useful_predictors"])
        self.assertTrue(res["predictors"]["boot_log_std"]["informative"])
        th = res["proposal"]["boot_log_std"]
        self.assertEqual(th["accurate_if"], "below")
        self.assertTrue(abs(th["threshold"] - (CFG31["accurate_tol"] - 0.02) / 0.30) < 0.15)         # near the true cut (error 0.15 at x = 0.43)
        self.assertEqual(res["n_calibration"], 30)

    def test_unrelated_predictors_give_not_calibratable(self):
        def fn(i, rng, r):
            return {"error": float(rng.uniform(0.0, 0.4))}
        res = hf.cal_analyse(cal_fake_rows(25, fn, seed=5), CFG31)
        self.assertEqual(res["decision"], "보정 불가")
        self.assertEqual(res["useful_predictors"], [])
        self.assertEqual(res["proposal"], {})

    def test_excluded_edge_and_other_lens_rows_do_not_enter_the_calibration(self):
        def fn(i, rng, r):
            x = float(rng.uniform(0.0, 1.0))
            return {"boot_log_std": x, "error": 0.02 + 0.30 * x}
        rows = cal_fake_rows(24, fn)
        rows += [dict(r, video=r["video"] + "x", lens="0.5x") for r in cal_fake_rows(5, fn, seed=9)]
        rows[0]["excluded"] = True
        rows[1]["at_grid_edge"] = True
        res = hf.cal_analyse(rows, CFG31)
        self.assertEqual(res["n_calibration"], 22)
        self.assertEqual(res["n_held_out"], 5)
        self.assertIn("boot_log_std", res["held_out_check"])                                           # the held-out lens is only checked, never used to choose the threshold

    def test_rule_values_are_the_ones_fixed_in_the_yaml(self):
        self.assertEqual((CFG31["accurate_tol"], CFG31["rho_min"], CFG31["loo_ba_min"], CFG31["bootstrap_n"], CFG31["bootstrap_seed"], CFG31["calibration_lens"]), (0.15, 0.4, 0.70, 1000, 42, "1x"))
        self.assertEqual(CFG31["predictors"]["sharpness"]["expected_sign"], -1)
        self.assertEqual(CFG31["predictors"]["boot_log_std"]["expected_sign"], 1)


R30C = ROOT / CFG31["source"]["results_30"]


@unittest.skipUnless(R30C.exists(), "experiment 30's official run is needed")
class CalEndToEnd(unittest.TestCase):
    def test_outputs_numbers_and_no_tags(self):
        src = json.loads(R30C.read_text())
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG31)
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r31.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, rows, res = run.cal_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            self.assertTrue((root / "figures/predictors_vs_error.png").exists())
            n1x = [v for v in src["videos"] if "f" in v and v["lens"] == "1x"]
            self.assertEqual(len(rows), len([v for v in src["videos"] if "f" in v]))
            self.assertLessEqual(res["n_calibration"], len(n1x))
            by = {r["video"]: r for r in rows}
            v = next(x for x in src["videos"] if x["video"] == "IMG_7220.MOV")
            self.assertAlmostEqual(by["IMG_7220.MOV"]["error"], abs(v["f"] / 1680.0 - 1.0))                # error = |f / ref_f - 1| with the stored ref_f
            self.assertAlmostEqual(by["IMG_7220.MOV"]["signed_error"], v["f"] / 1680.0 - 1.0)
            self.assertTrue(by["IMG_7216.MOV"]["excluded"])                                               # 27's edge-contact exclusion is reproduced
            self.assertIn(res["decision"], ("보정 가능", "보정 불가"))
            y = run.cal_results_yaml(root, cfg)
            ytext = Path(y).read_text()
            self.assertIn(res["decision"], ytext)
            self.assertNotIn("1680", ytext)                                                              # no reference-f comparison in the results yaml
            texts = "".join(q.read_text(errors="ignore") for q in list(root.rglob("*")) + [Path(y)] if q.is_file() and q.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)


# ====================================================================================================
# 34: printed hand, is the f-versus-distance slope still there when the hand cannot change shape?
# ====================================================================================================

REF34 = {float(d): float(f) for d, f in CFG34["ref_f_by_distance"].items()}


def curves34(spec):
    """{distance: [cost curves with their minimum at the listed f]} on the shared test grid."""
    return {float(d): [curve27(f) for f in fs] for d, fs in spec.items()}


class PrVerdict(unittest.TestCase):
    V = CFG34["verdict"]

    def test_none_yes_hold_and_not_judged_on_synthetic_curves(self):
        flat = hf.pr_slope(curves34({50: [1662, 1658], 60: [1655, 1651], 75: [1646, 1643]}), GRID27, 300, 42)
        self.assertLess(abs(flat["slope"]), 0.05)                                                       # the checkerboard values themselves have slope -0.02
        v = hf.pr_verdict(flat, REF34, self.V, {50.0: 2, 60.0: 2, 75.0: 2})
        self.assertEqual(v["verdict"], "반비례 없음")
        self.assertTrue(all(v["within_f_tol"].values()))
        inv = hf.pr_slope(curves34({50: [2100, 2090], 60: [1750, 1745], 75: [1400, 1395]}), GRID27, 300, 42)
        self.assertLess(inv["slope"], -0.95)
        self.assertEqual(hf.pr_verdict(inv, REF34, self.V, {50.0: 2, 60.0: 2, 75.0: 2})["verdict"], "반비례 있음")
        hold = hf.pr_slope(curves34({50: [1900, 1890], 60: [1700, 1695], 75: [1500, 1490]}), GRID27, 300, 42)
        self.assertTrue(-0.7 < hold["slope"] < -0.3, hold["slope"])
        self.assertEqual(hf.pr_verdict(hold, REF34, self.V, {50.0: 2, 60.0: 2, 75.0: 2})["verdict"], "판단 보류")
        far = hf.pr_slope(curves34({50: [1990, 1980], 60: [1985, 1975], 75: [1980, 1975]}), GRID27, 300, 42)       # flat but 20 % above the reference f: the f_tol condition fails
        self.assertEqual(hf.pr_verdict(far, REF34, self.V, {50.0: 2, 60.0: 2, 75.0: 2})["verdict"], "판단 보류")
        self.assertEqual(hf.pr_verdict(flat, REF34, self.V, {50.0: 2, 60.0: 0, 75.0: 2})["verdict"], "판단 불가")     # a distance without an included video

    def test_thresholds_come_from_the_yaml_and_the_slope_matches_pooling(self):
        flat = hf.pr_slope(curves34({50: [1662, 1658], 60: [1655, 1651], 75: [1646, 1643]}), GRID27, 200, 42)
        tight = dict(self.V, f_tol=0.0001)
        self.assertEqual(hf.pr_verdict(flat, REF34, tight, None)["verdict"], "판단 보류")
        self.assertEqual(hf.pr_verdict(flat, REF34, dict(self.V, slope_none_min=0.5), None)["verdict"], "판단 보류")
        c = curves34({50: [1700, 1720], 60: [1650, 1660], 75: [1600, 1610]})
        r = hf.pr_slope(c, GRID27, 50, 7)
        for d in (50.0, 60.0, 75.0):
            self.assertEqual(r["pooled_f"][d], hf.long_pooled_estimate(np.sum(c[d], axis=0)[None, :], GRID27)["f"])      # pooled f = the sum of the distance's cost curves
        self.assertAlmostEqual(r["slope"], hf.ext_loglog_slope([50.0, 60.0, 75.0], [r["pooled_f"][d] for d in (50.0, 60.0, 75.0)]))
        self.assertEqual(hf.pr_slope(c, GRID27, 50, 7)["interval_95"], r["interval_95"])                            # fixed seed: reproducible
        self.assertEqual(CFG34["verdict"], {"slope_none_min": -0.3, "slope_yes_max": -0.7, "f_tol": 0.1})
        self.assertEqual(CFG34["ref_f_by_distance"], {50: 1660.6, 60: 1653.7, 75: 1644.9})
        self.assertEqual((CFG34["bootstrap"]["n"], CFG34["bootstrap"]["seed"]), (1000, 42))
        self.assertEqual([e["file"] for e in CFG34["videos"]], ["IMG_7249.MOV", "IMG_7251.MOV", "IMG_7253.MOV", "IMG_7254.MOV", "IMG_7255.MOV", "IMG_7256.MOV"])


class PrSynthetic(unittest.TestCase):
    def test_exact_planar_hand_gives_a_slope_near_zero_and_f_near_the_truth(self):
        ce = hf.pseg_wrist_cfg(PIPE)
        curves = {}
        grid = None
        for dist in (500.0, 600.0, 750.0):
            cs = []
            for seed in (0, 1):
                g = synth({"noise": {"px": 0.3}, "tilt_range_deg": [30, 30]}, seed=seed, scfg=dict(SYN, f_px=1660.0, distance_mm=dist, tilt_frames=120))
                a = analyze_synth(g, methods_full=(), cfg=ce)
                self.assertLess(abs(a["solutions"]["fixed_template"]["f"] / 1660.0 - 1.0), 0.03, (dist, seed))
                cs.append(a["solutions"]["fixed_template"]["profile_cost"])
                grid = np.array(a["grid"])
            curves[dist / 10.0] = cs
        r = hf.pr_slope(curves, grid, 200, 42)
        self.assertLessEqual(abs(r["slope"]), 0.1)                                                     # exact joints: no distance dependence
        self.assertLessEqual(max(abs(f / 1660.0 - 1.0) for f in r["pooled_f"].values()), 0.03)
        self.assertGreater(r["interval_95"][0], -0.3)


class PrTwoHands(unittest.TestCase):
    def _det(self, n, W, H, other_conf, other_scale, share):
        rng = np.random.default_rng(0)
        times = np.arange(n) / 15.0
        kp = hand_track(n, times, W, H, 1660.0, 600.0, rng)
        boxes, kxy, kconf, cls, offs = [], [], [], [], [0]
        for i in range(n):
            ks = [kp[i]]
            confs = [0.95]
            if i < int(share * n):                                                                      # a second hand: another hand-sized picture elsewhere
                c0 = np.nanmean(kp[i], axis=0)
                ks.append((kp[i] - c0) * other_scale + c0 + np.array([650.0, -120.0]))
                confs.append(other_conf)
            for k, cf in zip(ks, confs):
                lo, hi = k.min(0), k.max(0)
                boxes.append([lo[0], lo[1], hi[0], hi[1]])
                kxy.append(k)
                kconf.append(np.full(21, cf))
                cls.append(1)
            offs.append(len(boxes))
        det = {"offsets": np.array(offs), "boxes": np.array(boxes), "box_conf": np.array([float(c[0]) for c in kconf]), "cls": np.array(cls), "kxy": np.array(kxy), "kconf": np.array(kconf)}
        return det, times, kp

    def test_the_tracked_hand_is_the_big_confident_one_and_the_two_hand_share_is_reported(self):
        W, H = 1920, 1080
        for conf, scale in ((0.9, 0.4), (0.3, 1.4)):                                                      # a smaller hand; a bigger but unsure one (area x confidence stays lower)
            det, times, kp = self._det(120, W, H, conf, scale, 0.4)
            tr = hf.track(det, times, (W, H), PIPE["tracking"], front_start_s=float(PIPE["phases"]["front_skip_s"]))
            self.assertTrue(tr["present"].all())
            self.assertTrue(np.allclose(tr["kp"][tr["present"]], kp[tr["present"]]))                      # the first hand is followed in every frame
            sh = hf.pr_multi_hand_share(tr["n_candidates"])
            self.assertEqual(sh["frames_with_hand"], 120)
            self.assertEqual(sh["frames_with_two_or_more"], 48)
            self.assertAlmostEqual(sh["share"], 0.4)
        self.assertEqual(hf.pr_multi_hand_share(np.zeros(5, int))["share"], 0.0)


R27P = ROOT / "runs/27_main_camera_extension/20261026T044322Z/results.json"


@unittest.skipUnless(R27P.exists() and (Path(CFG34["input_dir"]) / "IMG_7249.MOV").exists() and (Path(CFG34["input_dir"]) / "IMG_7220.MOV").exists(),
                     "27's official run and the input videos IMG_7249 / IMG_7220 are needed")
class PrEndToEnd(unittest.TestCase):
    def test_27_reproduction_outputs_overlay_and_no_tags(self):
        r27 = {j["video"]: j for j in json.loads(R27P.read_text())["videos"]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = copy.deepcopy(CFG34)
            cfg["videos"] = [e for e in CFG34["videos"] if e["file"] == "IMG_7249.MOV"]
            cfg["compare_videos"] = [e for e in CFG34["compare_videos"] if e["file"] == "IMG_7220.MOV"]
            cfg["workers"], cfg["n_workers"] = 2, 2
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r34.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            root, results, ana = run.pr_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            self.assertTrue((root / "figures/f_vs_distance_loglog.png").exists())
            self.assertGreater((root / "overlays/overlay_IMG_7249.mp4").stat().st_size, 10000)            # one overlay video per printed-hand video, only in the run folder
            self.assertFalse((root / "overlays/overlay_IMG_7220.mp4").exists())
            self.assertFalse((root / "_frames").exists())
            by = {r["video"]: r for r in results}
            self.assertEqual(by["IMG_7220.MOV"]["ft"]["f"], r27["IMG_7220.MOV"]["ft"]["f"])              # 27's value for the same video: the same candidate, the same detections
            self.assertEqual(by["IMG_7249.MOV"]["kind"], "printed")
            self.assertTrue(0.0 <= by["IMG_7249.MOV"]["multi_hand"]["share"] <= 1.0)
            self.assertEqual(ana["verdict"]["verdict"], "판단 불가")                                       # one printed video at one distance: the other distances have no video
            y = run.pr_results_yaml(root, cfg)
            ytext = Path(y).read_text()
            for val in ("1660.6", "1653.7", "1644.9"):
                self.assertNotIn(val, ytext)                                                              # no reference-f values in the results yaml
            self.assertIsNone(re.compile(r"(?<![\d.])1680(?:\.0+)?(?!\d)(?!\.\d)").search(ytext))
            for line in (root / "summary.md").read_text().splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|") - 1, (line.strip().count("|") - 1), line)             # every table row is a plain row
            texts = "".join(q.read_text(errors="ignore") for q in list(root.rglob("*")) + [Path(y)] if q.is_file() and q.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)


# ====================================================================================================
# 35: hand markers, does the slope survive when the points are measured by image processing?
# ====================================================================================================

MK = CFG35["markers"]
MK_USED_IDS = [int(x) for x in MK["used_ids"]]


def mk_layout(seed=0, size=700.0):
    """21 marker positions of a hand-like layout (px): the planar hand points of the synthetic study for joints 5-20, stand-ins for 0-4."""
    P = hf.synth_hand_points(0.0)
    rng = np.random.default_rng(seed)
    xy = np.zeros((21, 2))
    for j in range(5, 21):
        xy[j] = P[j, :2]
    for j in range(5):
        xy[j] = np.array([-40.0 + 25.0 * j, 70.0 + 10.0 * j]) + rng.normal(0, 3, 2)
    xy = (xy - xy.mean(0)) / np.ptp(xy[:, 1]) * size
    return xy + np.array([960.0, 540.0])


def mk_homography_at(t, n):
    """A smooth rotation / scale / translation / weak perspective of the layout about its centre, as a 3 x 3 homography in image pixels."""
    th, s = np.radians(25.0) * np.sin(2 * np.pi * t / n), 1.0 + 0.05 * np.sin(2 * np.pi * t / n * 1.7)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * s
    c = np.array([960.0, 540.0])
    T = np.eye(3)
    T[:2, :2], T[:2, 2] = R, c - R @ c + np.array([12.0 * np.sin(t / 7.0), 8.0 * np.cos(t / 9.0)])
    T[2, :2] = [2e-5 * np.sin(t / 5.0), 1e-5 * np.cos(t / 6.0)]
    return T


def mk_apply(H, xy):
    return cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def mk_sequence(n, seed=0, extras=0, hidden=None, missing=(), drift=0.0, noise=0.3):
    """Detections of a synthetic marker sequence: truth (n, 21, 2), per-frame shuffled detections with `extras` clutter blobs far from every marker, markers in `hidden` = {marker: (first, last)} never detected
    in those frames, frames in `missing` without any detection, a translation drift (px per frame) added to the whole hand."""
    rng = np.random.default_rng(seed)
    ref = mk_layout(seed)
    truth = np.zeros((n, 21, 2))
    dets = []
    for i in range(n):
        H = mk_homography_at(i, n)
        pts = mk_apply(H, ref) + np.array([drift * i, 0.5 * drift * i])
        truth[i] = pts + rng.normal(0, noise, pts.shape)
        if i in missing:
            dets.append(None)
            continue
        keep = [j for j in range(21) if not (hidden and j in hidden and hidden[j][0] <= i <= hidden[j][1])]
        d = [truth[i][j] for j in keep]
        while len(d) < len(keep) + extras:
            q = np.array([rng.uniform(300, 1600), rng.uniform(100, 1000)])
            if np.min(np.linalg.norm(truth[i] - q, axis=1)) > 0.1 * 800:
                d.append(q)
        d = np.array(d)
        dets.append(d[rng.permutation(len(d))])
    return ref, truth, dets


def mk_match_errors(trk, truth, idx):
    """Per frame and marker the distance of the tracked position to the truth (nan where the marker is not tracked)."""
    return np.linalg.norm(trk["xy"][:, idx, :] - truth[:, idx, :], axis=-1)


class MkBlobs(unittest.TestCase):
    def test_centres_are_found_to_a_third_of_a_pixel_on_a_synthetic_frame(self):
        rng = np.random.default_rng(1)
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:] = (150, 170, 205)                                                                           # skin / wood like background (BGR)
        img = np.clip(img.astype(float) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        centres = [(900.0 + 60.0 * (k % 5) + rng.uniform(0, 1), 400.0 + 70.0 * (k // 5) + rng.uniform(0, 1)) for k in range(16)]        # continuous coordinates (pixel centre = index + 0.5)
        for cx, cy in centres:
            cv2.circle(img, (int(round((cx - 0.5) * 16)), int(round((cy - 0.5) * 16))), int(7 * 16), (20, 20, 215), -1, lineType=cv2.LINE_AA, shift=4)
        b = hf.mk_blobs(img, (800, 300, 1300, 700), MK)
        self.assertEqual(len(b["xy"]), 16)
        err = [np.min(np.linalg.norm(b["xy"] - np.array(c), axis=1)) for c in centres]
        self.assertLessEqual(max(err), 0.3, err)
        small = hf.mk_blobs(img, (800, 300, 1300, 700), MK, area_min_px=5000.0)                           # the area limit is an argument: a limit above every blob finds none
        self.assertEqual(len(small["xy"]), 0)

    def test_area_limit_scales_with_the_box_and_equals_40_px_at_50_cm(self):
        self.assertAlmostEqual(MK["area_min_frac"] * 331091.6, 40.0, delta=0.05)
        self.assertEqual(MK["used_ids"], [5, 6, 9, 10, 13, 14, 17, 18])
        self.assertEqual((MK["min_markers_ratio"], MK["circularity_min"], MK["s_min"], MK["v_min"], MK["ransac_frac"], MK["tracker"]), (0.75, 0.6, 130, 90, 0.03, "reacquire"))
        self.assertEqual(CFG35["precheck"]["min_tilt_valid_ratio"], 0.70)


class MkTracking(unittest.TestCase):
    USED = MK_USED_IDS

    def test_swapped_labels_are_dropped_by_the_check_on_the_used_markers(self):
        ref, truth, _ = mk_sequence(20, seed=2)
        cur = truth[7]
        thr = MK["ransac_frac"] * 800.0
        good = hf.mk_homography_check(ref[self.USED], cur[self.USED], thr, MK["min_inlier_ratio"], 6)
        self.assertTrue(good["ok"])
        sw = cur.copy()
        a, b = self.USED.index(5), self.USED.index(18)
        sw[[5, 18]] = sw[[18, 5]]                                                                          # two of the 8 used markers carry each other's labels
        bad = hf.mk_homography_check(ref[self.USED], sw[self.USED], thr, MK["min_inlier_ratio"], 6)
        self.assertFalse(bad["ok"])
        self.assertLess(bad["inliers"].sum(), 8 * MK["min_inlier_ratio"])
        self.assertGreaterEqual(np.linalg.norm(ref[5] - ref[18]), 0.1 * 800.0)

    def test_numbers_stay_through_motion_and_clutter_and_extras_are_ignored(self):
        ref, truth, dets = mk_sequence(80, seed=3, extras=3)
        ref_index = 0
        trk = hf.mk_track_reacquire(dets, ref_index, truth[ref_index], np.full(80, 800.0), MK, self.USED)
        self.assertTrue(trk["ok"].all(), trk["reason"][~trk["ok"]])
        self.assertLess(np.nanmax(mk_match_errors(trk, truth, list(range(21)))), 1.5)                  # every marker keeps its own number
        self.assertTrue(np.all(trk["ignored"][1:] == 3))                                                 # the clutter blobs are ignored and counted
        _, truth6, dets6 = mk_sequence(80, seed=4, extras=6)                                            # (a hand moving at most about 2 % of the box per frame, as the real videos do)
        wide = hf.mk_track_reacquire(dets6, 0, truth6[0], np.full(80, 800.0), MK, self.USED)
        self.assertTrue(wide["ok"].all())
        self.assertTrue(np.all(wide["ignored"][1:] == 6))                                                # more clutter: the 8 used markers are still the right ones
        self.assertLess(np.nanmax(mk_match_errors(wide, truth6, MK_USED_IDS)), 1.5)
        self.assertEqual(MK["tracker"], "reacquire")

    def test_a_marker_that_drops_out_and_frames_that_are_missed_are_re_acquired(self):
        ref, truth, dets = mk_sequence(70, seed=5, hidden={10: (10, 49)}, missing=(20, 21, 22, 23), drift=4.0)
        trk = hf.mk_track_reacquire(dets, 0, truth[0], np.full(70, 800.0), MK, self.USED)
        ok = trk["ok"]
        self.assertTrue(ok[[i for i in range(70) if i not in (20, 21, 22, 23)]].all(), trk["reason"][~ok])
        self.assertTrue(not ok[[20, 21, 22, 23]].any())                                                  # missed frames are skipped, the next frame is matched again
        err = mk_match_errors(trk, truth, [10])[:, 0]
        self.assertTrue(np.isnan(err[10:50]).all())                                                      # hidden: no position
        self.assertLess(np.nanmax(err[50:]), 1.5)                                                        # reappears after the hand moved by far more than the last-position limit: re-acquired with its own number
        self.assertLess(np.nanmax(mk_match_errors(trk, truth, [j for j in range(21) if j != 10])), 1.5)
        old = hf.mk_track(dets, 0, truth[0], np.full(70, 800.0), {**MK, "min_markers_ratio": 0.75}, self.USED)
        self.assertTrue(np.isnan(mk_match_errors(old, truth, [10])[50:, 0]).any() or not old["ok"][50:].all())   # the first tracker does not get it back

    def test_a_frame_with_too_few_used_markers_is_dropped(self):
        ref, truth, dets = mk_sequence(30, seed=6, hidden={5: (10, 12), 6: (10, 12), 9: (10, 12)})
        trk = hf.mk_track_reacquire(dets, 0, truth[0], np.full(30, 800.0), MK, self.USED)
        self.assertTrue(not trk["ok"][10:13].any())                                                      # 5 of 8 used markers: below 6
        self.assertTrue(trk["ok"][13:].all())


class MkPairing(unittest.TestCase):
    USED = MK_USED_IDS

    def test_distances_to_the_prediction_and_the_near_blob_share(self):
        ref, truth, dets = mk_sequence(60, seed=7, extras=2)
        dets = [d.copy() if d is not None else None for d in dets]
        for i in range(20, 30):                                                                           # a blob 1 % of the box away from used marker 5, closer ones win the match
            dets[i] = np.vstack([dets[i], truth[i][5] + np.array([8.0, 0.0])])
        trk = hf.mk_track_reacquire(dets, 0, truth[0], np.full(60, 800.0), MK, self.USED)
        self.assertTrue(trk["ok"].all())
        # (a blob that close can be paired instead of the true marker when the hand has moved by more than half the gap: that is the risk the near-blob share measures, so it is not asserted away)
        proc = {"track": trk, "used_idx": self.USED, "box_long": np.full(60, 800.0)}
        st = hf.mk_pairing_stats(proc)
        self.assertEqual(st["with_ignored"]["frames"], 59)                                               # 2 clutter blobs in every frame after the reference
        self.assertEqual(st["near_2pct_frames"], 10)
        self.assertAlmostEqual(st["near_2pct_share"], 10 / 59)
        self.assertEqual(st["without_ignored"]["frames"], 0)
        self.assertTrue(st["with_ignored"]["median_px"] < st["with_ignored"]["p95_px"] < 40.0)
        self.assertEqual(trk["pred"].shape, trk["xy"].shape)
        self.assertEqual(len(trk["ignored_xy"][25]), 3)                                                  # 2 clutter + the near blob

    def test_panels_are_drawn_from_frames_with_ignored_blobs_with_a_fixed_seed(self):
        ref, truth, dets = mk_sequence(40, seed=8, extras=1)
        trk = hf.mk_track_reacquire(dets, 0, truth[0], np.full(40, 800.0), MK, self.USED)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            frames = []
            for i in range(40):
                q = d / f"f{i}.png"
                cv2.imwrite(str(q), np.full((1080, 1920, 3), 90, np.uint8))
                frames.append({"png": str(q), "time_s": i / 15.0})
            proc = {"track": trk, "used_idx": self.USED, "regions": [(300, 100, 1600, 1000)] * 40, "names": list(range(21))}
            n1 = hf.mk_draw_pairing({"frames": frames}, proc, d / "a.png")
            n2 = hf.mk_draw_pairing({"frames": frames}, proc, d / "b.png")
            self.assertEqual((n1, n2), (10, 10))
            self.assertEqual((d / "a.png").read_bytes(), (d / "b.png").read_bytes())                        # fixed seed: the same ten frames, the same picture


class MkPerspective(unittest.TestCase):
    def test_halved_perspective_gives_k_about_one_half(self):
        rng = np.random.default_rng(0)
        src = mk_layout(0)[5:]
        scale = 800.0
        T_s = hf.mk_normalise(src, scale)
        pm_mk, pm_mp, pm_same = [], [], []
        for _ in range(60):
            p = rng.uniform(-0.3, 0.3, 2)
            ang = rng.uniform(-0.4, 0.4)
            A = np.array([[np.cos(ang), -np.sin(ang), rng.uniform(-0.1, 0.1)], [np.sin(ang), np.cos(ang), rng.uniform(-0.1, 0.1)], [p[0], p[1], 1.0]])
            for q, store in ((p, pm_mk), (0.5 * p, pm_mp)):
                Hn = A.copy()
                Hn[2, :2] = q
                H = np.linalg.inv(T_s) @ Hn @ T_s                                                        # the same similarity part, only the perspective row differs
                dst = mk_apply(H, src)
                store.append(hf.mk_perspective(src, dst, scale))
            pm_same.append(pm_mk[-1])
        k = hf.mk_perspective_k(np.array(pm_mp), np.array(pm_mk))
        self.assertAlmostEqual(k["k"], 0.5, delta=0.05)
        self.assertGreater(k["corr"], 0.99)
        self.assertAlmostEqual(hf.mk_perspective_k(np.array(pm_same), np.array(pm_mk))["k"], 1.0, places=6)
        self.assertAlmostEqual(hf.mk_perspective_k(np.zeros((60, 2)), np.array(pm_mk))["k"], 0.0, places=6)

    def test_planar_gap_is_zero_for_a_planar_hand_and_grows_with_relief(self):
        src = mk_layout(1)
        H = mk_homography_at(11, 40)
        cur = mk_apply(H, src)
        self.assertLess(hf.mk_planar_gap(src, cur, src[5:], cur[5:], 800.0), 1e-4)
        bumped = cur[5:] + np.array([6.0, -4.0])
        self.assertAlmostEqual(hf.mk_planar_gap(src, cur, src[5:], bumped, 800.0), 100.0 * np.hypot(6.0, 4.0) / 800.0, places=4)


class MkSynthetic(unittest.TestCase):
    def test_exact_planar_markers_give_a_slope_near_zero_through_the_unchanged_analysis(self):
        cfg = copy.deepcopy(PIPE)
        cfg["joints"]["sets"]["markers"] = MK_USED_IDS
        cfg["joints"]["estimate_set"] = "markers"
        cfg["joints"]["uncertainty_sets"] = ["markers"]
        curves, grid = {}, None
        for dist in (500.0, 600.0, 750.0):
            cs = []
            for seed in (0, 1):
                g = synth({"noise": {"px": 0.3}, "tilt_range_deg": [30, 30]}, seed=seed, scfg=dict(SYN, f_px=1660.0, distance_mm=dist, tilt_frames=120))
                g["mask"] = g["mask"].copy()
                a = analyze_synth(g, methods_full=(), cfg=cfg)
                self.assertLess(abs(a["solutions"]["fixed_template"]["f"] / 1660.0 - 1.0), 0.03, (dist, seed))
                cs.append(a["solutions"]["fixed_template"]["profile_cost"])
                grid = np.array(a["grid"])
            curves[dist / 10.0] = cs
        r = hf.pr_slope(curves, grid, 200, 42)
        self.assertLessEqual(abs(r["slope"]), 0.1)                                                         # exact markers: no distance dependence
        self.assertEqual(hf.pr_verdict(r, REF34, CFG35["verdict"], {50.0: 2, 60.0: 2, 75.0: 2})["verdict"], "반비례 없음")


R27M = ROOT / "runs/27_main_camera_extension/20261026T044322Z/results.json"


@unittest.skipUnless(R27M.exists() and all((Path(CFG35["input_dir"]) / f"IMG_{n}.MOV").exists() for n in (7220, 7267)), "27's official run and the input videos IMG_7220 / IMG_7267 are needed")
class MkEndToEnd(unittest.TestCase):
    def test_27_reproduction_official_run_outputs_overlay_and_no_tags(self):
        r27 = {j["video"]: j for j in json.loads(R27M.read_text())["videos"]}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # 27's setting reproduces: the MediaPipe 16-point part of a marker run is pr_video (27's gen_video unchanged)
            prep = hf.fresp_prepare(("IMG_7220.MOV", CFG35["input_dir"], CFG35["pipeline"], {}, str(d)))
            from_cache = hf.jset_find_detection(prep["frames"], CFG35["detector_settings"]["mediapipe_hands"], ["runs/*/*/detection_cache"], None, "mediapipe_hands")["npz"]
            base = {k: CFG35[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "session")}
            res = hf.pr_video(({"file": "IMG_7220.MOV", "distance_cm": 60, "kind": "real"}, prep, from_cache, CFG35["pipeline"], base))
            self.assertEqual(res["ft"]["f"], r27["IMG_7220.MOV"]["ft"]["f"])
            shutil.rmtree(d / "IMG_7220", ignore_errors=True)
            cfg = copy.deepcopy(CFG35)
            cfg["videos"] = [e for e in CFG35["videos"] if e["file"] == "IMG_7267.MOV"]
            cfg["workers"], cfg["n_workers"] = 2, 2
            cfg["output"] = {"root": str(d / "runs"), "results_yaml": str(d / "r35.yaml")}
            cp = d / "c.yaml"
            cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            pre_root, pre_res, proceed = run.mk_main(["--config", str(cp), "--out", str(d / "pre"), "precheck"])
            self.assertTrue((pre_root / "precheck.md").exists())
            self.assertTrue((pre_root / "figures/reference_IMG_7267.png").exists())
            self.assertFalse(proceed)                                                                       # only a 50 cm video: the other distances have none
            root, results, ana = run.mk_main(["--config", str(cp), "--out", str(d / "o")])
            for f in ("report.md", "summary.md", "results.json", "metadata.json", "log.txt"):
                self.assertTrue((root / f).exists(), f)
            self.assertGreater((root / "overlays/overlay_IMG_7267.mp4").stat().st_size, 10000)
            self.assertFalse((root / "_frames").exists())
            r = results[0]
            self.assertIn("f", r["marker"])
            self.assertIn("f", r["mediapipe8"])
            self.assertIn("ft", r["mediapipe"])
            self.assertEqual(ana["markers"]["verdict"]["verdict"], "판단 불가")                              # one distance only
            self.assertEqual(ana["interpretation"], "해석 보류")
            y = run.mk_results_yaml(root, cfg)
            ytext = Path(y).read_text()
            for val in ("1660.6", "1653.7", "1644.9"):
                self.assertNotIn(val, ytext)
            self.assertIsNone(re.compile(r"(?<![\d.])1680(?:\.0+)?(?!\d)(?!\.\d)").search(ytext))
            texts = "".join(q.read_text(errors="ignore") for q in list(root.rglob("*")) + [Path(y)] if q.is_file() and q.suffix in (".json", ".yaml", ".md", ".csv", ".txt"))
            for val in FAKE_TAGS.values():
                self.assertNotIn(val, texts)
            for key in ("ISO6709", "com.apple.quicktime", '"make"', '"model"'):
                self.assertNotIn(key, texts)
            for line in (root / "summary.md").read_text().splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)


# ---- experiment 39: markers only (no hand detector) ----------------------------------------------------
CFG39 = yaml.safe_load((ROOT / "configs/39_markers_only.yaml").read_text())
MO_IDS, MO_T = hf.mo_load_layout(ROOT / CFG39["layout"])


def mo_render(img_wh, pts_px, radius=4.0, extra=(), bg=(185, 160, 200)):
    """Synthetic frame: pinkish background, red dots at sub-pixel positions, optional extra (x, y, radius) red discs."""
    W, H = img_wh
    img = np.full((H, W, 3), bg, np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for x, y in pts_px:
        cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), int(round(radius * 16)), (40, 40, 215), -1, cv2.LINE_AA, 4)
    for x, y, r in extra:
        cv2.circle(img, (int(x), int(y)), int(r), (40, 40, 215), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


class MoBlobs(unittest.TestCase):
    def test_dots_found_with_subpixel_centres_and_big_red_object_ignored(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform([100, 80], [700, 400], (12, 2))
        img = mo_render((800, 480), pts, extra=[(760, 440, 40)])
        r = hf.mo_blobs(img, CFG39["markers"])
        self.assertEqual(len(r["xy"]), 12)                                                              # the 40 px disc (area ~5000) is over area_max
        d = np.linalg.norm(r["xy"][:, None, :] - (pts + 0.5)[None, :, :], axis=-1)
        self.assertLess(d.min(1).max(), 0.6)                                                            # continuous coordinates: pixel centre = index + 0.5

    def test_tiny_dots_of_75_cm_size_are_found(self):
        pts = np.array([[200.0, 200.0], [300.5, 240.2], [400.3, 180.7]])
        r = hf.mo_blobs(mo_render((640, 400), pts, radius=3.0), CFG39["markers"])                         # radius 3 px: area about 27 px (the 75 cm dots measured 40-300 px at contrast 30)
        self.assertEqual(len(r["xy"]), 3)

    def test_no_dots_in_a_plain_frame(self):
        self.assertEqual(len(hf.mo_blobs(mo_render((640, 400), []), CFG39["markers"])["xy"]), 0)

    def test_box_rule_is_the_worker_rule(self):
        b = hf.mo_box_of(np.array([[10.0, 20.0], [110.0, 220.0]]), 0.1)
        self.assertTrue(np.allclose(b, [0.0, 0.0, 120.0, 240.0]))


class MoNaming(unittest.TestCase):
    def placed(self, s=150.0, rot_deg=10.0, t=(900.0, 500.0)):
        z = (MO_T[:, 0] + 1j * MO_T[:, 1]) * s * np.exp(1j * np.deg2rad(rot_deg)) + (t[0] + 1j * t[1])
        return np.stack([z.real, z.imag], 1)

    def test_names_are_recovered_with_rotation_missing_points_and_distractors(self):
        rng = np.random.default_rng(1)
        P = self.placed()
        keep = np.ones(len(P), bool)
        keep[[3, 11]] = False
        det = np.vstack([P[keep] + rng.normal(0, 0.3, (keep.sum(), 2)), [[300.0, 900.0], [1500.0, 100.0], [1700.0, 950.0]]])
        order = rng.permutation(len(det))
        det = det[order]
        res = hf.mo_name(det, MO_IDS, MO_T, CFG39["naming"])
        self.assertIsNotNone(res)
        truth = np.concatenate([MO_IDS[keep], [-1, -1, -1]])[order]
        self.assertTrue(np.array_equal(res["names"], truth))
        self.assertAlmostEqual(res["scale_px"], 150.0, delta=2.0)
        self.assertLess(res["rms_px"], 1.0)

    def test_scale_outside_the_range_or_too_few_detections_give_none(self):
        self.assertIsNone(hf.mo_name(self.placed(s=600.0), MO_IDS, MO_T, CFG39["naming"]))
        self.assertIsNone(hf.mo_name(self.placed()[:6], MO_IDS, MO_T, CFG39["naming"]))

    def test_rotation_beyond_the_limit_is_not_named(self):
        self.assertIsNone(hf.mo_name(self.placed(rot_deg=70.0), MO_IDS, MO_T, CFG39["naming"]))


def mo_synth_video(d, f_px=850.0, dist_mm=500.0, size=(960, 540), seconds=10.0, fps=15.0, tilt_deg=40.0, pen=True, seed=0):
    """Frames of the planar marker layout (physical RMS radius 50 mm) seen by a pinhole camera: fronto-parallel until 3 s, then a tilt about the horizontal axis ramping to tilt_deg; a red pen and 3 distractor dots."""
    rng = np.random.default_rng(seed)
    W, H = size
    P = np.concatenate([MO_T * 50.0, np.zeros((len(MO_T), 1))], 1)
    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        a = np.deg2rad(0.0 if t < 3.0 else tilt_deg * min((t - 3.0) / 5.0, 1.0))
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        X = P @ R.T + np.array([0.0, 0.0, dist_mm])
        uv = f_px * X[:, :2] / X[:, 2:3] + np.array([W / 2.0, H / 2.0])
        extra = [(60, 480, 22)] if pen else []
        distract = np.array([[80.0, 60.0], [880.0, 70.0], [900.0, 500.0]]) if pen else np.zeros((0, 2))
        img = mo_render(size, np.vstack([uv + rng.normal(0, 0.05, uv.shape), distract]), radius=3.4, extra=extra)
        path = Path(d) / f"f{i:04d}.png"
        cv2.imwrite(str(path), img)
        frames.append({"png": str(path), "time_s": t})
    return {"frames": frames, "size": size}


class MoProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.prep = mo_synth_video(cls.tmp)
        cls.c = {k: CFG39[k] for k in ("exclusion", "markers", "marker_exclusion", "precheck", "naming", "layout")}
        cls.proc = hf.mo_process(cls.prep, CFG39["pipeline"], cls.c)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reference_frame_names_every_used_marker_and_the_distractors_are_ignored(self):
        p = self.proc
        self.assertNotIn("failed", p)
        self.assertTrue(0.5 <= p["ref_time_s"] < 1.5)
        self.assertEqual(sorted(int(x) for x in p["names"]), [int(x) for x in MO_IDS])                      # 20 layout markers, no distractor named
        self.assertGreaterEqual(p["tilt_valid_ratio"], 0.95)
        self.assertGreaterEqual(p["ignored_median"], 3.0)                                                   # pen end and the three corner dots are detections without a marker

    def test_kp_mask_and_box_follow_the_markers(self):
        p = self.proc
        i = int(np.flatnonzero(p["present"])[-1])
        self.assertTrue(p["mask"][i][[5, 6, 9, 10, 13, 14, 17, 18]].all())
        self.assertFalse(p["mask"][i][3])                                                                   # marker 3 is not in the layout
        b = p["box"][i]
        pts = p["kp"][i][p["mask"][i]]
        self.assertTrue(b[0] < pts[:, 0].min() and b[1] < pts[:, 1].min() and b[2] > pts[:, 0].max() and b[3] > pts[:, 1].max())

    def test_precheck_passes_and_draws_the_reference_picture(self):
        with tempfile.TemporaryDirectory() as d:
            r = hf.mo_precheck_video(({"file": "synth.MOV", "distance_cm": 50}, self.prep, CFG39["pipeline"], {**self.c, "out_dir": d}))
            self.assertTrue(r["passes"])
            self.assertTrue((Path(d) / "reference_synth.png").exists())

    def test_f_is_recovered_by_the_unchanged_analysis(self):
        res = hf.mo_video(({"file": "synth.MOV", "distance_cm": 50}, self.prep, CFG39["pipeline"], self.c))
        self.assertIn("f", res["marker"], res["marker"])
        self.assertLess(abs(res["marker"]["f"] / 850.0 - 1.0), 0.05)

    def test_a_video_without_markers_fails_with_a_reason_and_no_crash(self):
        with tempfile.TemporaryDirectory() as d:
            frames = []
            for i in range(30):
                pth = Path(d) / f"g{i}.png"
                cv2.imwrite(str(pth), mo_render((480, 270), []))
                frames.append({"png": str(pth), "time_s": i / 15.0})
            res = hf.mo_video(({"file": "empty.MOV", "distance_cm": 50}, {"frames": frames, "size": (480, 270)}, CFG39["pipeline"], self.c))
            self.assertIn("failed", res["marker"])
            self.assertFalse(res["marker"].get("f"))

    def test_a_dropped_marker_run_is_excluded_by_the_valid_share(self):
        c = copy.deepcopy(self.c)
        c["marker_exclusion"]["min_tilt_valid_ratio"] = 1.01
        res = hf.mo_video(({"file": "synth.MOV", "distance_cm": 50}, self.prep, CFG39["pipeline"], c))
        self.assertTrue(res["marker"]["excluded"])


class MoReports(unittest.TestCase):
    def fake_results(self):
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        cost = ((np.log(grid) - np.log(1700.0)) ** 2 + 0.01).tolist()
        out = []
        for f, d in (("IMG_7294.MOV", 50), ("IMG_7293.MOV", 50), ("IMG_7292.MOV", 60), ("IMG_7291.MOV", 60), ("IMG_7290.MOV", 75), ("IMG_7289.MOV", 75)):
            out.append({"video": f, "distance_cm": float(d), "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0,
                        "marker_stats": {"valid_ratio": 0.99, "tilt_valid_ratio": 0.98, "window_success": 15, "window_frames_50_150": 15, "per_marker_rate": {5: 0.99}},
                        "marker": {"f": 1700.0, "at_grid_edge": False, "sharpness": 1.5, "boot_log_std": 0.02, "curve": {"grid": grid.tolist(), "cost": cost}, "n_used_frames": 200, "theta_median": 30.0, "theta_max": 45.0,
                                   "theta_p90": 44.0, "tilt_duration_s": 20.0, "phase_rms_pct": {"front": 0.02, "tilt": 0.06}, "box_long_median_px": 500.0}})
        return out

    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG39["ref_f_by_distance"].items()}
        results = self.fake_results()
        cur = {d: [r["marker"]["curve"]["cost"] for r in results if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        grid = np.array(results[0]["marker"]["curve"]["grid"])
        sl = hf.pr_slope(cur, grid, 50, 42)
        per = {d: {**hf.ext_pooled(cur[d], grid, {"bootstrap_n": 50, "bootstrap_seed": 42}), "n_total": 2} for d in cur}
        ana = {"markers": {"slope": sl, "per_distance": per, "verdict": hf.pr_verdict(sl, ref, CFG39["verdict"], {50.0: 2, 60.0: 2, 75.0: 2})}, "checkerboard_slope": hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)])}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.mo_figures(d / "figures", results, ana, ref)
            run.mo_report(d / "report.md", results, CFG39)
            run.mo_summary(d / "summary.md", results, ana, ref, CFG39)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG39)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.mo_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists() and (d / "figures/cost_curves.png").exists())
            self.assertEqual(ana["markers"]["verdict"]["verdict"], "반비례 없음")
            summ = (d / "summary.md").read_text()
            self.assertIn("판정 (점 전용): 반비례 없음", summ)
            self.assertIn("1660.6", summ)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                for val in ("1660.6", "1653.7", "1644.9"):
                    self.assertNotIn(val, text, name)
                self.assertNotIn("1680", text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)


# ---- experiment 40: the supervisor's free template on the marker tracks ---------------------------------
CFG40 = yaml.safe_load((ROOT / "configs/40_free_template.yaml").read_text())


def ft_planar_problem(f=1000.0, n_frames=14, seed=0):
    """Planar 8-point template seen by a pinhole camera at 1920 x 1080 in a tilt sequence; returns (Problem, K)."""
    rng = np.random.default_rng(seed)
    T = np.array([[-60, -40], [-20, -45], [20, -42], [60, -38], [-58, 10], [-18, 12], [22, 9], [62, 14]], float)
    K = hf.K_of(f, 960.0, 540.0)
    frames = []
    for i in range(n_frames):
        a = np.deg2rad(5 + 40 * i / max(n_frames - 1, 1))
        R = cv2.Rodrigues(np.array([a, 0.2 * a, 0.0]))[0]
        X = np.c_[T, np.zeros(len(T))] @ R.T + np.array([10.0, 5.0, 600.0])
        uv = (K @ X.T).T
        uv = uv[:, :2] / uv[:, 2:3] + rng.normal(0, 0.1, (len(T), 2))
        frames.append((uv, np.ones(len(T), bool)))
    return hf.Problem(np.zeros((len(T), 2)), frames, 960.0, 540.0, np.full(n_frames, 4.0)), K, T


class FtParts(unittest.TestCase):
    def test_backproject_returns_the_template_coordinates(self):
        pb, K, T = ft_planar_problem()
        o, v = pb.frames[5]
        _, R, t = hf.frame_fit(T, o, v, K)
        back = hf.backproject(o, R, t, K)
        self.assertLess(np.abs(back - T).max(), 1.0)

    def test_rodrigues_matches_opencv(self):
        rv = np.array([[0.3, -0.2, 0.5], [0.0, 0.0, 0.0], [1.2, 0.1, -0.4]])
        R = hf._rodrigues(rv)
        for r, m in zip(rv, R):
            self.assertTrue(np.allclose(m, cv2.Rodrigues(r)[0], atol=1e-7))

    def test_restrict_drops_rare_joints_and_subsamples_frames(self):
        pb, _, _ = ft_planar_problem(n_frames=14)
        o, v = pb.frames[0]
        pb.frames = [(o2, v2.copy()) for o2, v2 in pb.frames]
        for i in range(14):
            pb.frames[i][1][7] = i < 2                                                                       # joint 7 is seen in 2 frames only
        pb.keep = list(range(14))
        q, ids = hf._restrict(pb, min_frames=3, max_frames=8)
        self.assertNotIn(7, ids.tolist())
        self.assertEqual(len(q.T), 7)
        self.assertLessEqual(len(q.keep), 8)

    def test_free_problem_selects_the_ids_and_the_present_frames(self):
        n = 6
        kp = np.random.default_rng(0).uniform(100, 800, (n, 21, 2))
        mask = np.ones((n, 21), bool)
        present = np.array([1, 1, 0, 1, 1, 1], bool)
        box = np.tile([100.0, 100.0, 700.0, 500.0], (n, 1))
        pb, idx = hf.free_problem({"kp": kp, "mask": mask, "present": present, "box": box}, (1920, 1080), CFG40["pipeline"]["estimate"], 0.02, ids=[5, 6, 9, 10])
        self.assertEqual(idx.tolist(), [0, 1, 3, 4, 5])
        self.assertEqual(len(pb.frames), 5)
        self.assertEqual(int(pb.frames[0][1].sum()), 4)
        self.assertTrue(np.allclose(pb.huber, 0.02 * 600.0))

    def test_estimate_run_ft_routes_and_rejects_unknown_methods(self):
        with self.assertRaises(ValueError):
            hf.ft_estimate_run(None, "nonsense", None, CFG40["pipeline"]["estimate"], 1.0)

    def test_sharpness_on_a_parabola_cost(self):
        grid = np.geomspace(500, 3000, 60)
        c = (np.log(grid) - np.log(1500.0)) ** 2 + 0.05
        s = hf.ft_sharpness(grid, c, 1500.0)
        self.assertGreater(s, 1.0)
        self.assertTrue(np.isnan(hf.ft_sharpness(grid, c, 520.0)))                                           # a neighbour outside the grid


class FtTemplate(unittest.TestCase):
    def test_free_template_runs_on_a_planar_problem_and_returns_the_documented_fields(self):
        pb, _, _ = ft_planar_problem(f=1000.0, n_frames=14)
        grid = np.geomspace(500.0, 2000.0, 7)
        r = hf.ft_estimate_run(pb, "free_template", grid, CFG40["pipeline"]["estimate"], 1.0)
        for k in ("f", "f_grid", "profile_f", "profile_cost", "at_grid_edge", "template", "joint_ids", "ba", "cost"):
            self.assertIn(k, r)
        self.assertEqual(len(r["profile_cost"]), 7)
        self.assertTrue(np.isfinite(r["f"]) and grid[0] <= r["f"] <= grid[-1])
        self.assertTrue(r["ba"]["success"] or r["ba"]["nfev"] > 0)
        self.assertEqual(r["template"].shape, (8, 2))

    def test_refine_false_gives_the_grid_value(self):
        pb, _, _ = ft_planar_problem()
        grid = np.geomspace(500.0, 2000.0, 7)
        r = hf.free_template(pb, grid, alt_iters=1, refine=False, max_frames=10)
        self.assertEqual(r["f"], r["f_grid"])
        self.assertNotIn("ba", r)

    def test_no_usable_frames_raises(self):
        pb, _, _ = ft_planar_problem()
        pb.keep = []
        with self.assertRaises(ValueError):
            hf.free_template(pb, np.geomspace(500.0, 2000.0, 5))


class FtVideo(unittest.TestCase):
    def test_video_through_the_marker_tracks_returns_f_and_curves_and_a_failure_is_a_result(self):
        with tempfile.TemporaryDirectory() as d:
            prep = mo_synth_video(d, seconds=6.0, fps=6.0)
            pcfg = copy.deepcopy(CFG40["pipeline"])
            pcfg["estimate"]["f_grid_n"] = 6
            c = {k: CFG40[k] for k in run.FT_KEYS}
            c["free_template"] = {"alt_iters": 1, "refine": True, "max_frames": 8}
            res = hf.ft_video(({"file": "synth.MOV", "distance_cm": 50}, prep, pcfg, c))
            x = res["marker"]
            self.assertIn("f", x, x)
            self.assertEqual(len(x["curve"]["grid"]), 6)
            self.assertEqual(sorted(x["joint_ids"]), [5, 6, 9, 10, 13, 14, 17, 18])
            c2 = copy.deepcopy(c)
            c2["free_template"] = {"alt_iters": 1, "refine": True, "max_frames": 8}
            c2["markers"]["used_ids"] = [5, 6, 9, 10, 13, 14, 17, 99]                                          # a joint the layout does not have
            res2 = hf.ft_video(({"file": "synth.MOV", "distance_cm": 50}, prep, pcfg, c2))
            self.assertIn("failed", res2["marker"])                                                          # no reference frame names marker 99: a result, not a crash
            self.assertNotIn("f", res2["marker"])


class FtConfig(unittest.TestCase):
    def test_the_config_has_every_section_the_run_command_reads(self):
        for k in run.FT_KEYS + ("videos", "input_dir", "ref_f_by_distance", "pipeline", "verdict", "bootstrap", "rating_v2", "workers", "output", "code_40", "compare"):
            self.assertIn(k, CFG40, k)
        self.assertEqual(CFG40["free_template"]["alt_iters"], 3)


class FtReports(unittest.TestCase):
    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG40["ref_f_by_distance"].items()}
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        cost = ((np.log(grid) - np.log(1700.0)) ** 2 + 0.01).tolist()
        results = []
        for f, d in (("IMG_7294.MOV", 50), ("IMG_7293.MOV", 50), ("IMG_7292.MOV", 60), ("IMG_7291.MOV", 60), ("IMG_7290.MOV", 75), ("IMG_7289.MOV", 75)):
            results.append({"video": f, "distance_cm": float(d), "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0, "marker_stats": {"tilt_valid_ratio": 0.98},
                            "marker": {"f": 1710.0, "f_grid": 1700.0, "at_grid_edge": False, "ba_at_bound": False, "curve": {"grid": grid.tolist(), "cost": cost}, "sharpness": 1.4, "cost_at_f": 0.01, "ba_success": True,
                                       "ba_nfev": 20, "ba_rms_px": 0.2, "n_frames_in_problem": 120, "n_points": 900, "joint_ids": [5], "elapsed_s": 600.0}})
        cur = {d: [r["marker"]["curve"]["cost"] for r in results if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        sl = hf.pr_slope(cur, grid, 50, 42)
        per = {d: {**hf.ext_pooled(cur[d], grid, {"bootstrap_n": 50, "bootstrap_seed": 42}), "n_total": 2} for d in cur}
        ana = {"markers": {"slope": sl, "per_distance": per, "verdict": hf.pr_verdict(sl, ref, CFG40["verdict"], {50.0: 2, 60.0: 2, 75.0: 2})}}
        prior = {"video": {r["video"]: 1690.0 for r in results}, "pooled": {50.0: 1693.2, 60.0: 1663.6, 75.0: 1662.9}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.ft_figures(d / "figures", results, ana, ref, prior)
            run.ft_report(d / "report.md", results, CFG40)
            run.ft_summary(d / "summary.md", results, ana, ref, CFG40, prior)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG40)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.ft_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists() and (d / "figures/cost_curves.png").exists())
            summ = (d / "summary.md").read_text()
            self.assertIn("판정 (free_template", summ)
            self.assertIn("1660.6", summ)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                for val in ("1660.6", "1653.7", "1644.9", "1680"):
                    self.assertNotIn(val, text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)



# ---- experiment 41: the supervisor's free_shape3d on the marker tracks ------------------------------------
CFG41 = yaml.safe_load((ROOT / "configs/41_free_shape3d.yaml").read_text())


def s3_relief_problem(f=1000.0, n_frames=20, relief=8.0, seed=0):
    """The planar 8-point template of ft_planar_problem with a small out-of-plane relief on the points; returns (Problem, K)."""
    rng = np.random.default_rng(seed)
    T = np.array([[-60, -40], [-20, -45], [20, -42], [60, -38], [-58, 10], [-18, 12], [22, 9], [62, 14]], float)
    K = hf.K_of(f, 960.0, 540.0)
    z = relief * np.array([0, 0, 1, -1, 1, 0, -1, 1.0])
    frames = []
    for i in range(n_frames):
        a = np.deg2rad(5 + 40 * i / max(n_frames - 1, 1))
        R = cv2.Rodrigues(np.array([a, 0.2 * a, 0.0]))[0]
        X = np.c_[T, z] @ R.T + np.array([10.0, 5.0, 600.0])
        uv = (K @ X.T).T
        uv = uv[:, :2] / uv[:, 2:3] + rng.normal(0, 0.05, (len(T), 2))
        frames.append((uv, np.ones(len(T), bool)))
    return hf.Problem(np.zeros((len(T), 2)), frames, 960.0, 540.0, np.full(n_frames, 4.0))


class S3Parts(unittest.TestCase):
    def test_free_shape3d_returns_the_documented_fields_and_f_stays_in_range(self):
        pb = s3_relief_problem()
        grid = np.geomspace(500.0, 2000.0, 7)
        r = hf.s3_estimate_run(pb, "free_shape3d", grid, {"free_template": {"z_rel": 0.2, "alt_iters": 1, "max_frames": 12}}, 1.0)
        for k in ("f", "f_planar", "shape", "joint_ids", "relief_rel", "relief_per_joint", "cost", "cost_planar", "profile_f", "profile_cost", "at_grid_edge", "at_bound", "ba"):
            self.assertIn(k, r)
        self.assertEqual(r["shape"].shape, (8, 3))
        self.assertTrue(grid[0] <= r["f"] <= grid[-1] and np.isfinite(r["relief_rel"]))
        self.assertEqual(len(r["relief_per_joint"]), 8)
        self.assertEqual(len(r["ba"]["anchors"]), 3)

    def test_the_three_anchors_are_distinct_and_the_planar_result_can_be_passed_in(self):
        pb = s3_relief_problem()
        grid = np.geomspace(500.0, 2000.0, 7)
        A = hf.free_template(pb, grid, alt_iters=1, max_frames=12)
        r = hf.free_shape3d(pb, grid, alt_iters=1, max_frames=12, planar=A)
        self.assertEqual(r["f_planar"], A["f"])
        self.assertEqual(len(set(r["ba"]["anchors"])), 3)
        with self.assertRaises(ValueError):
            hf.free_shape3d(pb, grid, max_frames=12, planar={**A, "joint_ids": [0, 1, 2]})

    def test_dispatch_routes_and_rejects_unknown_methods(self):
        with self.assertRaises(ValueError):
            hf.s3_estimate_run(None, "nonsense", None, {}, 1.0)
        pb = s3_relief_problem()
        grid = np.geomspace(500.0, 2000.0, 7)
        self.assertIn("f_grid", hf.s3_estimate_run(pb, "free_template", grid, {}, 1.0))
        self.assertIn("profile_cost", hf.s3_estimate_run(pb, "fixed_template", grid, {}, 1.0))

    def test_planar_points_give_a_relief_near_zero(self):
        pb = s3_relief_problem(relief=0.0)
        grid = np.geomspace(500.0, 2000.0, 7)
        r = hf.free_shape3d(pb, grid, alt_iters=1, max_frames=12)
        self.assertLess(r["relief_rel"], 0.05)

    def test_slope_of_the_per_video_f(self):
        fb = {50.0: [1700.0, 1690.0], 60.0: [1680.0, 1670.0], 75.0: [1660.0, 1650.0]}
        sl = hf.s3_slope(fb, {d: [False, False] for d in fb}, 200, 42)
        self.assertAlmostEqual(sl["pooled_f"][50.0], float(np.sqrt(1700.0 * 1690.0)), places=6)
        self.assertLess(sl["slope"], 0.0)
        self.assertGreater(sl["slope"], -0.2)
        self.assertEqual(len(sl["interval_95"]), 2)
        self.assertEqual(sl, hf.s3_slope(fb, {d: [False, False] for d in fb}, 200, 42))                      # fixed seed
        edge = hf.s3_slope(fb, {50.0: [False, True], 60.0: [False, False], 75.0: [False, False]}, 200, 42)
        self.assertIsNone(edge["slope"])
        self.assertIsNone(edge["interval_95"])
        v = hf.pr_verdict(sl, {50.0: 1660.6, 60.0: 1653.7, 75.0: 1644.9}, CFG41["verdict"], {50.0: 2, 60.0: 2, 75.0: 2})
        self.assertEqual(v["verdict"], "반비례 없음")
        self.assertEqual(hf.pr_verdict(edge, {50.0: 1660.6, 60.0: 1653.7, 75.0: 1644.9}, CFG41["verdict"], {50.0: 2, 60.0: 2, 75.0: 2})["verdict"], "판단 불가")


class S3Video(unittest.TestCase):
    def test_video_through_the_marker_tracks_returns_both_f_and_a_failure_is_a_result(self):
        with tempfile.TemporaryDirectory() as d:
            prep = mo_synth_video(d, seconds=6.0, fps=6.0)
            pcfg = copy.deepcopy(CFG41["pipeline"])
            pcfg["estimate"]["f_grid_n"] = 6
            c = {k: CFG41[k] for k in run.S3_KEYS}
            c["free_shape3d"] = {"z_rel": 0.2, "alt_iters": 1, "max_frames": 8}
            x = hf.s3_video(({"file": "synth.MOV", "distance_cm": 50}, prep, pcfg, c))["marker"]
            self.assertIn("f", x, x)
            for k in ("f_planar", "relief_rel", "at_bound", "curve", "anchors", "ba3_nfev"):
                self.assertIn(k, x)
            self.assertEqual(len(x["curve"]["grid"]), 6)
            c2 = copy.deepcopy(c)
            c2["markers"]["used_ids"] = [5, 6, 9, 10, 13, 14, 17, 99]
            res2 = hf.s3_video(({"file": "synth.MOV", "distance_cm": 50}, prep, pcfg, c2))
            self.assertIn("failed", res2["marker"])
            self.assertNotIn("f", res2["marker"])


class S3Config(unittest.TestCase):
    def test_the_config_has_every_section_the_run_command_reads(self):
        for k in run.S3_KEYS + ("videos", "input_dir", "ref_f_by_distance", "pipeline", "verdict", "bootstrap", "workers", "output", "code_41", "compare"):
            self.assertIn(k, CFG41, k)
        self.assertEqual((CFG41["free_shape3d"]["z_rel"], CFG41["free_shape3d"]["alt_iters"], CFG41["free_shape3d"]["max_frames"]), (0.2, 3, 120))
        self.assertEqual(CFG41["markers"], CFG40["markers"])
        self.assertEqual(CFG41["naming"], CFG40["naming"])
        self.assertEqual(CFG41["pipeline"]["estimate"], CFG40["pipeline"]["estimate"])
        self.assertEqual(CFG41["verdict"], CFG40["verdict"])


class S3Reports(unittest.TestCase):
    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG41["ref_f_by_distance"].items()}
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        cost = ((np.log(grid) - np.log(1700.0)) ** 2 + 0.01).tolist()
        results = []
        for f, d in (("IMG_7294.MOV", 50), ("IMG_7293.MOV", 50), ("IMG_7292.MOV", 60), ("IMG_7291.MOV", 60), ("IMG_7290.MOV", 75), ("IMG_7289.MOV", 75)):
            results.append({"video": f, "distance_cm": float(d), "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0, "marker_stats": {"tilt_valid_ratio": 0.98},
                            "marker": {"f": 1705.0, "f_planar": 1710.0, "f_grid": 1700.0, "at_grid_edge": False, "at_bound": False, "relief_rel": 0.01, "relief_per_joint": [0.01] * 8, "cost": 0.01, "cost_planar": 0.011,
                                       "curve": {"grid": grid.tolist(), "cost": cost}, "sharpness": 1.4, "ba3_success": True, "ba3_nfev": 20, "ba3_rms_px": 0.2, "anchors": [0, 1, 2], "n_frames_in_problem": 120,
                                       "n_points": 900, "joint_ids": [5, 6, 9, 10, 13, 14, 17, 18], "elapsed_s": 60.0}})
        fb = {d: [r["marker"]["f"] for r in results if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        sl = hf.s3_slope(fb, {d: [False, False] for d in fb}, 50, 42)
        cur = {d: [r["marker"]["curve"]["cost"] for r in results if r["distance_cm"] == d] for d in fb}
        psl = hf.pr_slope(cur, grid, 50, 42)
        nv = {50.0: 2, 60.0: 2, 75.0: 2}
        ana = {"shape3d": {"slope": sl, "verdict": hf.pr_verdict(sl, ref, CFG41["verdict"], nv)}, "planar": {"slope": psl, "verdict": hf.pr_verdict(psl, ref, CFG41["verdict"], nv)}}
        prior = {"39": {"video": {r["video"]: 1690.0 for r in results}, "pooled": {50.0: 1693.2, 60.0: 1663.6, 75.0: 1662.9}},
                 "40": {"video": {r["video"]: 1710.0 for r in results}, "pooled": {50.0: 1725.4, 60.0: 1638.2, 75.0: 1686.7}}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.s3_figures(d / "figures", results, ana, ref, prior)
            run.s3_report(d / "report.md", results, CFG41)
            run.s3_summary(d / "summary.md", results, ana, ref, CFG41, prior)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG41)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.s3_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists() and (d / "figures/relief_per_joint.png").exists())
            summ = (d / "summary.md").read_text()
            self.assertIn("판정 (free_shape3d", summ)
            self.assertIn("1660.6", summ)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                for val in ("1660.6", "1653.7", "1644.9", "1680"):
                    self.assertNotIn(val, text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)



# ---- experiment 42: free_shape3d on the real-hand marker videos of 35c -------------------------------------
CFG42 = yaml.safe_load((ROOT / "configs/42_free_shape3d_hand.yaml").read_text())


def h3_fake_proc(n=24, tilt_ratio=0.95, f=1700.0):
    """What mk_process returns, built from a planar 8-marker problem (joints 5, 6, 9, 10, 13, 14, 17, 18) at 1920 x 1080."""
    pb, _, _ = ft_planar_problem(f=f, n_frames=n)
    ids = [5, 6, 9, 10, 13, 14, 17, 18]
    kp = np.full((n, 21, 2), np.nan)
    for i, (o, v) in enumerate(pb.frames):
        kp[i, ids] = o
    box = np.tile([760.0, 340.0, 1160.0, 740.0], (n, 1))
    return {"size": [1920, 1080], "n_frames": n, "times": np.linspace(0, 20, n), "tilt_valid_ratio": tilt_ratio, "valid_ratio": 1.0, "present": np.ones(n, bool), "tr": {"box": box}, "kp": kp,
            "mask": np.isfinite(kp[..., 0]), "box_long": np.full(n, 400.0), "names": ids}


class H3Video(unittest.TestCase):
    def run_h3(self, proc, c):
        from unittest import mock
        pcfg = copy.deepcopy(CFG42["pipeline"])
        pcfg["estimate"]["f_grid_n"] = 6
        with mock.patch.object(hf._report, "mk_process", return_value=proc):
            return hf.h3_video(({"file": "x.MOV", "distance_cm": 50}, None, None, pcfg, c))

    def test_tracks_go_through_the_supervisors_call_and_the_35c_exclusion_rules_apply(self):
        c = {k: CFG42[k] for k in run.H3_KEYS}
        c["free_shape3d"] = {"z_rel": 0.2, "alt_iters": 1, "max_frames": 12}
        x = self.run_h3(h3_fake_proc(), c)["marker"]
        self.assertIn("f", x, x)
        self.assertEqual(sorted(x["joint_ids"]), [5, 6, 9, 10, 13, 14, 17, 18])
        for k in ("f_planar", "relief_rel", "at_bound", "curve", "anchors"):
            self.assertIn(k, x)
        low = self.run_h3(h3_fake_proc(tilt_ratio=0.5), c)["marker"]                                         # 35c's tilt-phase rule: excluded, not computed
        self.assertTrue(low["excluded"] and "f" not in low)
        edge = h3_fake_proc()
        edge["tr"]["box"] = np.tile([0.0, 340.0, 400.0, 740.0], (edge["n_frames"], 1))                       # the hand box touches the frame edge
        self.assertTrue(self.run_h3(edge, c)["marker"]["excluded"])
        bad = h3_fake_proc()
        bad["mask"][:] = False                                                                               # nothing visible: a result (failed), not a crash
        self.assertIn("failed", self.run_h3(bad, c)["marker"])
        self.assertIn("failed", self.run_h3({"size": [1920, 1080], "n_frames": 3, "times": np.arange(3.0), "failed": "정면 창에 기준 프레임이 없음"}, c)["marker"])


class H3Config(unittest.TestCase):
    def test_the_config_has_every_section_the_run_command_reads_and_35c_settings_are_unchanged(self):
        for k in run.H3_KEYS + ("videos", "input_dir", "ref_f_by_distance", "pipeline", "verdict", "bootstrap", "workers", "n_workers", "output", "code_42", "compare"):
            self.assertIn(k, CFG42, k)
        for k in ("markers", "marker_exclusion", "exclusion", "candidate", "detector_settings", "verdict", "bootstrap", "ref_f_by_distance", "videos"):
            self.assertEqual(CFG42[k], CFG35[k], k)
        self.assertEqual(CFG42["pipeline"]["estimate"], CFG35["pipeline"]["estimate"])
        self.assertEqual(CFG42["free_shape3d"], CFG41["free_shape3d"])


class H3Reports(unittest.TestCase):
    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG42["ref_f_by_distance"].items()}
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        cost = ((np.log(grid) - np.log(1700.0)) ** 2 + 0.01).tolist()
        results = []
        for f, d in (("IMG_7267.MOV", 50), ("IMG_7266.MOV", 50), ("IMG_7265.MOV", 60), ("IMG_7264.MOV", 60), ("IMG_7263.MOV", 75), ("IMG_7262.MOV", 75)):
            results.append({"video": f, "distance_cm": float(d), "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0, "marker_stats": {"tilt_valid_ratio": 0.98},
                            "marker": {"f": 1705.0, "f_planar": 1710.0, "f_grid": 1700.0, "at_grid_edge": False, "at_bound": False, "relief_rel": 0.01, "relief_per_joint": [0.01] * 8, "cost": 0.01, "cost_planar": 0.011,
                                       "curve": {"grid": grid.tolist(), "cost": cost}, "sharpness": 1.4, "ba3_success": True, "ba3_nfev": 20, "ba3_rms_px": 0.2, "anchors": [0, 1, 2], "n_frames_in_problem": 120,
                                       "n_points": 900, "joint_ids": [5, 6, 9, 10, 13, 14, 17, 18], "elapsed_s": 60.0}})
        fb = {d: [r["marker"]["f"] for r in results if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        sl = hf.s3_slope(fb, {d: [False, False] for d in fb}, 50, 42)
        cur = {d: [r["marker"]["curve"]["cost"] for r in results if r["distance_cm"] == d] for d in fb}
        psl = hf.pr_slope(cur, grid, 50, 42)
        nv = {50.0: 2, 60.0: 2, 75.0: 2}
        ana = {"shape3d": {"slope": sl, "verdict": hf.pr_verdict(sl, ref, CFG42["verdict"], nv)}, "planar": {"slope": psl, "verdict": hf.pr_verdict(psl, ref, CFG42["verdict"], nv)}}
        prior = {"markers": {"video": {r["video"]: 3072.0 for r in results}, "pooled": {50.0: 3072.0, 60.0: 2647.8, 75.0: 2658.1}}, "mediapipe8": {"video": {}, "pooled": {}}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.h3_figures(d / "figures", results, ana, ref, prior)
            run.h3_report(d / "report.md", results, CFG42)
            run.h3_summary(d / "summary.md", results, ana, ref, CFG42, prior)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG42)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.h3_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists() and (d / "figures/relief_per_joint.png").exists())
            summ = (d / "summary.md").read_text()
            self.assertIn("판정 (free_shape3d", summ)
            self.assertIn("1660.6", summ)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                for val in ("1660.6", "1653.7", "1644.9", "1680"):
                    self.assertNotIn(val, text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)
            self.assertEqual(run.h3_prior({"compare": {"35c": str(d / "none.json")}}), {"markers": {"video": {}, "pooled": {}}, "mediapipe8": {"video": {}, "pooled": {}}})



# ---- experiment 45: free_shape3d on the real-hand videos of 27 -------------------------------------------
CFG45 = yaml.safe_load((ROOT / "configs/45_free_shape3d_27.yaml").read_text())


def e3_fake_track(n=200, edge=False, seconds=20.0, f=1700.0):
    """What track() returns for a planar 16-joint hand (joints 5-20) seen in a tilt sequence at 1920 x 1080."""
    rng = np.random.default_rng(0)
    base = np.array([[-60, -40], [-20, -45], [20, -42], [60, -38], [-58, 10], [-18, 12], [22, 9], [62, 14]], float)
    T = np.r_[base, base + np.array([5.0, 40.0])]
    K = hf.K_of(f, 960.0, 540.0)
    kp = np.full((n, 21, 2), np.nan)
    for i in range(n):
        a = np.deg2rad(40 * max(0.0, i / (n - 1) - 0.15))
        R = cv2.Rodrigues(np.array([a, 0.2 * a, 0.0]))[0]
        X = np.c_[T, np.zeros(len(T))] @ R.T + np.array([10.0, 5.0, 600.0])
        uv = (K @ X.T).T
        kp[i, 5:21] = uv[:, :2] / uv[:, 2:3] + rng.normal(0, 0.1, (16, 2))
    box = np.tile([760.0, 340.0, 1160.0, 740.0], (n, 1))
    if edge:
        box = np.tile([0.0, 340.0, 400.0, 740.0], (n, 1))
    mask = np.isfinite(kp[..., 0])
    return {"present": np.ones(n, bool), "box": box, "cls": np.zeros(n, int), "box_conf": np.ones(n), "kp": kp, "kconf": np.ones((n, 21)), "mask": mask, "n_candidates": np.ones(n, int), "init_frame": 0}, np.linspace(0, seconds, n)


class E3Video(unittest.TestCase):
    def run_e3(self, tr, times, c=None, entry=None):
        from unittest import mock
        pcfg = copy.deepcopy(CFG45["pipeline"])
        pcfg["estimate"]["f_grid_n"] = 6
        c = c or {k: CFG45[k] for k in run.E3_KEYS}
        c["free_shape3d"] = {"z_rel": 0.2, "alt_iters": 1, "max_frames": 10}
        prep = {"frames": [{"time_s": float(t)} for t in times], "size": (1920, 1080)}
        with mock.patch.object(hf._report, "load_detections", return_value=None), mock.patch.object(hf._report, "check_alignment", return_value=None), mock.patch.object(hf._report, "track", return_value=tr):
            return hf.e3_video((entry or {"file": "x.MOV", "group": "G6", "distance_cm": 50, "hand": "right"}, prep, "none.npz", pcfg, c))

    def test_tracks_go_through_the_supervisors_call_and_27s_exclusion_rules_apply(self):
        tr, times = e3_fake_track()
        r = self.run_e3(tr, times)
        self.assertEqual(r["excluded"], [], r)
        x = r["est"]
        self.assertEqual(sorted(x["joint_ids"]), list(range(5, 21)))
        for k in ("f", "f_planar", "relief_rel", "at_bound", "curve", "anchors"):
            self.assertIn(k, x)
        self.assertEqual((r["group"], r["hand"], r["distance_cm"]), ("G6", "right", 50.0))
        tr2, t2 = e3_fake_track(edge=True)
        r2 = self.run_e3(tr2, t2)
        self.assertTrue(r2["excluded"] and "est" not in r2)                                                  # hand box on the frame edge
        tr3, t3 = e3_fake_track()
        tr3["present"][:] = False
        r3 = self.run_e3(tr3, t3)
        self.assertTrue(r3["excluded"] and "est" not in r3)                                                  # no hand detected
        tr4, t4 = e3_fake_track()
        tr4["mask"][:] = False
        self.assertIn("failed", self.run_e3(tr4, t4))                                                        # nothing visible: a result, not a crash


class E3Config(unittest.TestCase):
    def test_the_config_has_every_section_the_run_command_reads_and_27_settings_are_unchanged(self):
        for k in run.E3_KEYS + ("videos", "input_dir", "ref_f_by_distance", "pipeline", "verdict", "bootstrap", "workers", "n_workers", "output", "code_45", "compare"):
            self.assertIn(k, CFG45, k)
        C27 = yaml.safe_load((ROOT / "configs/27_main_camera_extension.yaml").read_text())
        for k in ("videos", "candidate", "exclusion", "detector_settings", "input_dir"):
            self.assertEqual(CFG45[k], C27[k], k)
        self.assertEqual(CFG45["pipeline"]["estimate"], C27["pipeline"]["estimate"])
        self.assertEqual(CFG45["free_shape3d"], CFG41["free_shape3d"])
        self.assertEqual(CFG45["verdict"], CFG41["verdict"])
        files = [e["file"] for e in CFG45["videos"]]
        self.assertFalse(set(files) & {f"IMG_{n}.MOV" for n in list(range(7249, 7259)) + list(range(7289, 7295))})   # printed hand (34) and tablet picture are not used


class E3Reports(unittest.TestCase):
    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG45["ref_f_by_distance"].items()}
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 8)
        cost = ((np.log(grid) - np.log(1700.0)) ** 2 + 0.01).tolist()
        est = {"f": 1705.0, "f_planar": 1710.0, "f_grid": 1700.0, "at_grid_edge": False, "at_bound": False, "relief_rel": 0.01, "relief_per_joint": [0.01] * 16, "cost": 0.01, "cost_planar": 0.011,
               "curve": {"grid": grid.tolist(), "cost": cost}, "sharpness": 1.4, "ba3_success": True, "ba3_nfev": 20, "ba3_rms_px": 0.2, "anchors": [0, 1, 2], "n_frames_in_problem": 120, "n_points": 900, "joint_ids": list(range(5, 21)), "elapsed_s": 60.0}
        results = []
        for e in CFG45["videos"]:
            r = {"video": e["file"], "group": e["group"], "distance_cm": float(e["distance_cm"]), "hand": e["hand"], "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0, "detect_rate": 1.0, "edge_rate": 0.0, "excluded": [], "est": est}
            results.append(r)
        results[0].pop("est")
        results[0]["excluded"] = ["가장자리 접촉 프레임 6.3% > 5%"]
        inc = [r for r in results if "est" in r and r["group"] == "G6"]
        fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        sl = hf.s3_slope(fb, {d: [False] * len(fb[d]) for d in fb}, 50, 42)
        cur = {d: [r["est"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in fb}
        psl = hf.pr_slope(cur, grid, 50, 42)
        nv = {d: len(fb[d]) for d in fb}
        ana = {"shape3d": {"slope": sl, "verdict": hf.pr_verdict(sl, ref, CFG45["verdict"], nv)}, "planar": {"slope": psl, "verdict": hf.pr_verdict(psl, ref, CFG45["verdict"], nv)}}
        prior = {r["video"]: 1800.0 for r in results}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.e3_figures(d / "figures", results, ana, ref, prior)
            run.e3_report(d / "report.md", results, CFG45)
            run.e3_summary(d / "summary.md", results, ana, ref, CFG45, prior)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG45)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.e3_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists())
            summ = (d / "summary.md").read_text()
            self.assertIn("판정 (G6", summ)
            self.assertIn("1660.6", summ)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                for val in ("1660.6", "1653.7", "1644.9", "1680"):
                    self.assertNotIn(val, text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)
            self.assertEqual(run.e3_prior({"compare": {"27": str(d / "none.json")}}), {})


# ---- experiment 46: the call of 45 on newly shot videos ----------------------------------------------------
CFG46 = yaml.safe_load((ROOT / "configs/46_free_shape3d_new.yaml").read_text())


class E4Config(unittest.TestCase):
    def test_the_config_has_every_section_the_run_command_reads_and_45_settings_are_unchanged(self):
        for k in run.E4_KEYS + ("videos", "input_dir", "ref_f_by_distance", "pipeline", "verdict", "bootstrap", "workers", "n_workers", "output", "code_46", "compare"):
            self.assertIn(k, CFG46, k)
        for k in ("candidate", "exclusion", "detector_settings", "free_shape3d", "verdict", "bootstrap", "input_dir"):
            self.assertEqual(CFG46[k], CFG45[k], k)
        self.assertEqual(CFG46["pipeline"]["estimate"], CFG45["pipeline"]["estimate"])
        self.assertEqual(CFG46["pipeline"]["phases"], CFG45["pipeline"]["phases"])
        self.assertEqual(CFG46["ref_f_by_distance"], {50: 1553, 60: 1553, 75: 1553})
        got = {(e["file"], e["distance_cm"]) for e in CFG46["videos"]}
        want = {(f"IMG_{n}.MOV", d) for n, d in ((7298, 50), (7299, 50), (7309, 50), (7310, 50), (7300, 60), (7301, 60), (7302, 60), (7303, 60), (7304, 75), (7305, 75), (7307, 75), (7308, 75))}
        self.assertEqual(got, want)


class E4Reports(unittest.TestCase):
    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG46["ref_f_by_distance"].items()}
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 8)
        cost = ((np.log(grid) - np.log(1600.0)) ** 2 + 0.01).tolist()
        est = {"f": 1605.0, "f_planar": 1610.0, "f_grid": 1600.0, "at_grid_edge": False, "at_bound": False, "relief_rel": 0.01, "relief_per_joint": [0.01] * 16, "cost": 0.01, "cost_planar": 0.011,
               "curve": {"grid": grid.tolist(), "cost": cost}, "sharpness": 1.4, "ba3_success": True, "ba3_nfev": 20, "ba3_rms_px": 0.2, "anchors": [0, 1, 2], "n_frames_in_problem": 120, "n_points": 900, "joint_ids": list(range(5, 21)), "elapsed_s": 60.0}
        results = [{"video": e["file"], "group": "G8", "distance_cm": float(e["distance_cm"]), "hand": "unrecorded", "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0, "detect_rate": 1.0, "edge_rate": 0.0, "excluded": [], "est": est}
                   for e in CFG46["videos"]]
        results[0].pop("est")
        results[0]["excluded"] = ["가장자리 접촉 프레임 6.3% > 5%"]
        inc = [r for r in results if "est" in r]
        fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        sl = hf.s3_slope(fb, {d: [False] * len(fb[d]) for d in fb}, 50, 42)
        cur = {d: [r["est"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in fb}
        psl = hf.pr_slope(cur, grid, 50, 42)
        nv = {d: len(fb[d]) for d in fb}
        ana = {"shape3d": {"slope": sl, "verdict": hf.pr_verdict(sl, ref, CFG46["verdict"], nv)}, "planar": {"slope": psl, "verdict": hf.pr_verdict(psl, ref, CFG46["verdict"], nv)}}
        prior = {50.0: 1675.3, 60.0: 1476.1, 75.0: 1167.1}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.e4_figures(d / "figures", results, ana, ref, prior)
            run.e4_report(d / "report.md", results, CFG46)
            run.e4_summary(d / "summary.md", results, ana, ref, CFG46, prior)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG46)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.e4_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists())
            summ = (d / "summary.md").read_text()
            self.assertIn("1553", summ)
            self.assertIn("판정 (free_shape3d", summ)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                self.assertNotIn("1553", text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)
            self.assertEqual(run.e4_prior({"compare": {"45": str(d / "none.json")}}), {})
            (d / "p45.json").write_text(json.dumps({"analysis": {"shape3d": {"slope": {"pooled_f": {"50.0": 1675.3, "60.0": 1476.1}}}}}))
            self.assertEqual(run.e4_prior({"compare": {"45": str(d / "p45.json")}}), {50.0: 1675.3, 60.0: 1476.1})


# ---- experiment 47: MMPose hand models as detectors ----------------------------------------------------------
CFG47 = yaml.safe_load((ROOT / "configs/47_mmpose_hand_detectors.yaml").read_text())
MM_PY = Path(CFG47["detector_settings"]["mmhand_rtmpose_m"]["python"])
MM_READY = MM_PY.exists() and all(Path(CFG47["detector_settings"][n]["checkpoint"]).exists() for n in hf.MM_DETECTORS)
VID47 = Path(CFG47["input_dir"]) / "IMG_7220.MOV"
LOCATION_TAGS = ("gps", "location", "latitude", "longitude", "com.apple.quicktime", "iphone", "make", "model_name", "creation_time")


def mm_test_frames(d: Path, n=3, video=VID47):
    """n frames of an input video (seconds 4, 6, 8, ...) as PNGs with the worker's frame list; a hand-sized box in the middle of every frame except the second (no box)."""
    cap = cv2.VideoCapture(str(video))
    rows = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, 4000 + 2000 * k)
        ok, img = cap.read()
        p = d / f"f{k}.png"
        cv2.imwrite(str(p), img)
        rows.append({"index": k, "png": str(p), "sha256": hf.sha256_file(p)})
    cap.release()
    H, W = img.shape[:2]
    box = [W * 0.35, H * 0.2, W * 0.65, H * 0.8]
    offsets = [0] + [sum(1 for j in range(i + 1) if j != 1) for i in range(n)]
    nb = offsets[-1]
    np.savez(d / "boxes.npz", offsets=np.array(offsets), boxes=np.tile(box, (nb, 1)), box_conf=np.ones(nb), cls=np.ones(nb, int))
    hf.write_list(rows, d / "frames.tsv")
    return rows


@unittest.skipUnless(MM_READY and VID47.exists() and GPU, "the det_mmpose_hand venv, the five checkpoints, IMG_7220 and a GPU are needed")
class MMWorker(unittest.TestCase):
    def test_each_wrapper_matches_the_official_inference_and_keeps_the_cache_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = mm_test_frames(d)
            for n in hf.MM_DETECTORS:
                r = hf.mm_official_check(rows, run.mm_det_settings(CFG47, n, 0), d / "boxes.npz", d / f"w_{n}", 3)
                self.assertEqual(r["n_boxes"], 2, n)                                                   # the frame without a box is not processed
                self.assertLessEqual(r["max_abs_px"], 0.1, n)
                self.assertEqual(WC.mm_order(r["names"]), list(range(21)), n)                         # the model's points are already in the cache order

    def test_a_frame_without_a_box_is_saved_as_a_detection_failure_and_the_cache_loads(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = mm_test_frames(d)
            s = run.mm_det_settings(CFG47, "mmhand_rtmpose_m", 0)
            p, extra = hf.swap_mmpose_hand(rows, s, d, print, d / "boxes.npz", "mmhand_rtmpose_m")
            z = hf.load_detections(p)
            hf.check_alignment(rows, z)
            self.assertEqual([int(x) for x in z["offsets"]], [0, 1, 1, 2])
            self.assertEqual(z["kxy"].shape, (2, 21, 2))
            self.assertEqual(z["info"]["detector"], "mmhand_rtmpose_m")
            self.assertEqual(z["info"]["checkpoint_sha256"], CFG47["detector_settings"]["mmhand_rtmpose_m"]["checkpoint_sha256"])
            self.assertEqual(extra["box_source"], "mediapipe")
            info = json.dumps(z["info"]).lower()
            for tag in LOCATION_TAGS:
                self.assertNotIn(tag, info, tag)


class MMOrder(unittest.TestCase):
    def test_the_order_map_follows_the_names_and_fails_on_a_missing_name(self):
        self.assertEqual(WC.mm_order(WC.MM_HAND_NAMES), list(range(21)))
        perm = list(reversed(WC.MM_HAND_NAMES))
        o = WC.mm_order(perm)
        self.assertEqual([perm[k] for k in o], WC.MM_HAND_NAMES)
        with self.assertRaises(RuntimeError):
            WC.mm_order(WC.MM_HAND_NAMES[:-1] + ["toe"])

    def test_hungarian_order_check_finds_a_swapped_pair_on_a_synthetic_hand(self):
        tr, _ = e3_fake_track(n=30)
        kp = np.nan_to_num(tr["kp"].copy())
        kp[:, :5] = kp[:, 5:10] + 30.0                                                                # fill the unused points with distinct positions
        sw = kp.copy()
        sw[:, [8, 12]] = sw[:, [12, 8]]
        m = np.ones((30, 21), bool)
        self.assertTrue(hf.order_assignment(kp, kp, m)["identity"])
        r = hf.order_assignment(kp, sw, m)
        self.assertFalse(r["identity"])
        self.assertEqual((r["assignment"][8], r["assignment"][12]), (12, 8))


class MMPipeline(unittest.TestCase):
    def fake_cache(self, d: Path, drop=()):
        tr, times = e3_fake_track(n=200)
        frames = [{"index": i, "png": str(d / f"{i}.png"), "sha256": f"{i:064x}", "time_s": float(t)} for i, t in enumerate(times)]
        per = [[] if i in drop else [{"box": tr["box"][i], "box_conf": 0.9, "cls": 1, "kxy": np.nan_to_num(tr["kp"][i], nan=1.0), "kconf": np.r_[np.zeros(5), np.ones(16)] * 0.9}]
               for i in range(len(times))]
        p = d / "mmhand_rtmpose_m_test.npz"
        WC.save(p, [(r["index"], r["png"], r["sha256"]) for r in frames], per, {"detector": "mmhand_rtmpose_m"})
        return frames, p

    def test_a_new_cache_runs_through_the_existing_ft_code(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            frames, p = self.fake_cache(d)
            pcfg = copy.deepcopy(CFG47["pipeline"])
            c = {k: CFG47[k] for k in run.MM_KEYS}
            r = hf.mm_video(({"file": "x.MOV", "source": "27", "distance_cm": 60}, {"frames": frames, "size": (1920, 1080)}, {"mmhand_rtmpose_m": p}, pcfg, c))
            x = r["detectors"]["mmhand_rtmpose_m"]
            self.assertEqual(x.get("excluded"), [], x)
            self.assertIn("ft", x)
            self.assertGreater(x["ft"]["f"], 0)
            self.assertNotIn("ref_f", json.dumps(hf.jsonable(r)))
            frames2, p2 = self.fake_cache(d, drop=set(range(0, 200, 3)))                                   # a third of the frames without a box: detection failure -> excluded (> 20 %)
            r2 = hf.mm_video(({"file": "x.MOV", "source": "27", "distance_cm": 60}, {"frames": frames2, "size": (1920, 1080)}, {"mmhand_rtmpose_m": p2}, pcfg, c))
            self.assertTrue(any("검출 실패" in e for e in r2["detectors"]["mmhand_rtmpose_m"]["excluded"]))

    def test_swap_run_all_mm_sends_old_names_to_swap_run_all_and_new_names_to_the_worker(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            frames, p = self.fake_cache(d)
            cfg = copy.deepcopy(CFG47)
            cfg["detectors"] = ["mediapipe_hands", "mmhand_rtmpose_m"]
            with mock.patch.object(hf._report, "swap_run_all", return_value={"mediapipe_hands": {"npz": p}}) as old, \
                 mock.patch.object(hf._report, "swap_mmpose_hand", return_value=(p, {"box_source": "mediapipe"})) as new:
                out = hf.swap_run_all_mm(frames, cfg, d / "c", print)
            self.assertEqual(old.call_args[0][1]["detectors"], ["mediapipe_hands"])
            self.assertEqual(new.call_args[0][4], p)                                                       # the boxes of the box source's run
            self.assertEqual(new.call_args[0][5], "mmhand_rtmpose_m")
            self.assertIn("npz", out["mmhand_rtmpose_m"])

    def test_box_cache_is_only_read(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {**CFG47, "box_caches": [d]}
            with self.assertRaises(hf.SkipDetector):
                hf.mm_box_cache([{"index": 0, "sha256": "0" * 64}], cfg)
            self.assertEqual(os.listdir(d), [])

    def test_thermal_guard_waits_from_70_until_below_65(self):
        temps = iter([72.0, 69.0, 66.0, 64.0])
        logs = []
        w = hf.mm_thermal_wait(0, CFG47["thermal"], logs.append, poll_s=0.0, temp_fn=lambda g: next(temps))
        self.assertGreaterEqual(w, 0.0)
        self.assertEqual(next(temps, None), None)                                                       # read until 64
        self.assertEqual(hf.mm_thermal_wait(0, CFG47["thermal"], logs.append, poll_s=0.0, temp_fn=lambda g: 50.0), 0.0)

    def test_agreement_uses_the_same_boxes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            frames, p = self.fake_cache(d)
            a = hf.load_detections(p)
            b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
            b["kxy"] = b["kxy"] + np.array([4.0, 3.0])                                                       # 5 px everywhere
            r = hf.mm_agreement(a, b, 0.3)
            self.assertAlmostEqual(r["median_pct"], 5.0 / 400.0 * 100.0, places=6)
            self.assertTrue(r["order"]["identity"])
            self.assertAlmostEqual(r["low_conf_frac"], 5 / 21)


@unittest.skipUnless(VID47.exists() and Path(CFG47["existing_results"]["27"]).exists(), "IMG_7220, experiment 27's run and its detection cache are needed")
class MMExistingDetector(unittest.TestCase):
    def test_the_existing_detector_setting_reproduces_experiment_27(self):
        prep = hf.fresp_prepare(("IMG_7220.MOV", CFG47["input_dir"], CFG47["pipeline"], {}, tempfile.mkdtemp(prefix="t47_")))
        try:
            box = hf.mm_box_cache(prep["frames"], CFG47)
            c = {k: CFG47[k] for k in run.MM_KEYS}
            e = next(x for x in CFG47["stage2_videos"] if x["file"] == "IMG_7220.MOV")
            r = hf.mm_video((e, prep, {"mediapipe_hands": box}, CFG47["pipeline"], c))
        finally:
            shutil.rmtree(Path(prep["frames"][0]["png"]).parents[2], ignore_errors=True)
        old = run.mm_existing_f(CFG47)["IMG_7220.MOV"]
        x = r["detectors"]["mediapipe_hands"]["ft"]
        self.assertAlmostEqual(x["f"], old["f"], delta=1e-6 * old["f"])
        self.assertAlmostEqual(x["sharpness"], old["sharpness"], places=9)


class MMReports(unittest.TestCase):
    def test_report_summary_and_yaml_keep_the_reference_in_summary_only_and_have_no_location_tags(self):
        names = list(hf.MM_DETECTORS)
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        mk = lambda f: {"ft": {"f": f, "at_grid_edge": False, "sharpness": 1.01, "verdict": "REJECT"}, "excluded": [], "curve": {"grid": grid.tolist(), "cost": ((np.log(grid) - np.log(f)) ** 2).tolist()}}   # noqa: E731
        stage2 = []
        for e in CFG47["stage2_videos"]:
            stage2.append({"video": e["file"], "source": e["source"], "distance_cm": e.get("distance_cm"), "lens": e["lens"], "detectors": {n: mk(1500.0 + 10 * k) for k, n in enumerate(["mediapipe_hands"] + names)}})
        stage2[0]["detectors"]["mmhand_scnet"] = {"excluded": ["가장자리 접촉 프레임 6.3% > 5%"]}
        existing = {s["video"]: {"source": s["source"], "f": 1490.0, "sharpness": 1.0, "at_grid_edge": False, "curve": ((np.log(grid) - np.log(1490.0)) ** 2).tolist()} for s in stage2}
        slopes = run.mm_stage2_slopes(stage2, ["mediapipe_hands"] + names, CFG47, existing)
        self.assertIsNotNone(slopes["mmhand_hrnetv2"]["slope"])
        env = {"info": {n: {"config_sha256": "c" * 64, "test_pipeline": ["LoadImage", "GetBBoxCenterScale", "TopdownAffine", "PackPoseInputs"], "input_size": [256, 256], "test_cfg": {"flip_test": True},
                            "remap": None, "names_ok": True} for n in names}, "versions": {"mmpose": "1.3.2"}}
        checks = {n: {"IMG_7220.MOV": {"frames": [0, 1], "n_boxes": 2, "max_abs_px": 0.0, "max_abs_conf": 0.0}} for n in names}
        agree = {n: {"IMG_7220.MOV": {"n_boxes": 2, "median_pct": 3.0, "p90_pct": 6.0, "per_joint_median_pct": [3.0] * 21, "low_conf_frac": 0.01}} for n in names}
        order = {n: {"identity": True, "assignment": list(range(21)), "n_boxes": 2} for n in names}
        fail = {"IMG_7220.MOV": {"n_frames": 300, "no_box_frac": 0.0, "low_conf": {n: 0.01 for n in names}}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.mm_report(d / "report.md", CFG47, env, checks, agree, order, fail, ["IMG_7220_frame00010.png"], {"official": True, "order": True, "stage2": "실행함"})
            run.mm_summary(d / "summary.md", CFG47, stage2, names, existing, slopes)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"verification": {"checks": checks, "order": order, "agreement": agree, "failure": fail, "verified": {"official": True, "order": True}, "env": env},
                                                                    "stage2": stage2, "slopes": slopes}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG47)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.mm_results_yaml(d, cfg)).read_text()
            summ, rep = (d / "summary.md").read_text(), (d / "report.md").read_text()
            self.assertIn("1680", summ)
            self.assertIn("880", summ)
            self.assertIn("1660.6", summ)
            for name, text in (("report", rep), ("yaml", y)):
                for val in ("1680", "1660.6", "1653.7", "1644.9", "기준 f 대비"):
                    self.assertNotIn(val, text, name)
            for name, text in (("report", rep), ("summary", summ), ("yaml", y)):
                low = text.lower()
                for tag in LOCATION_TAGS:
                    self.assertNotIn(tag, low, (name, tag))
            for text in (rep, summ):
                for line in text.splitlines():
                    if line.startswith("|") and "---" not in line:
                        cells = line.strip().strip("|").split("|")
                        self.assertTrue(all("|" not in c for c in cells))

    def test_the_config_has_every_section_and_27s_settings(self):
        C27 = yaml.safe_load((ROOT / "configs/27_main_camera_extension.yaml").read_text())
        for k in run.MM_KEYS + ("detectors", "hand_box_source", "box_caches", "gpu_ids", "verify", "run_stage2", "stage2_videos", "existing_results", "ref_f", "code_47", "pipeline", "thermal"):
            self.assertIn(k, CFG47, k)
        self.assertEqual(CFG47["detectors"], list(hf.MM_DETECTORS))
        for k in ("candidate", "exclusion", "guide", "input_dir"):
            self.assertEqual(CFG47[k], C27[k], k)
        self.assertEqual(CFG47["pipeline"]["estimate"], C27["pipeline"]["estimate"])
        self.assertEqual(CFG47["detector_settings"]["mediapipe_hands"], C27["detector_settings"]["mediapipe_hands"])
        files = [e["file"] for e in CFG47["stage2_videos"]]
        self.assertEqual(len(files), 18)
        self.assertTrue({"IMG_7220.MOV", "IMG_7192.MOV"} <= set(files))


# ---- experiment 48: experiment 47's models on the videos of 46 -------------------------------------------------
CFG48 = yaml.safe_load((ROOT / "configs/48_mmpose_hand_46_videos.yaml").read_text())


class MM2Reports(unittest.TestCase):
    def test_the_config_keeps_47s_detector_settings_and_lists_46s_videos(self):
        for k in ("detectors", "detector_settings", "candidate", "exclusion", "guide", "hand_box_source", "bootstrap", "thermal"):
            self.assertEqual(CFG48[k], CFG47[k], k)
        C46 = yaml.safe_load((ROOT / "configs/46_free_shape3d_new.yaml").read_text())
        self.assertEqual([(e["file"], e["distance_cm"]) for e in CFG48["stage2_videos"]], [(e["file"], e["distance_cm"]) for e in C46["videos"]])
        self.assertTrue(set(CFG48["verify"]["videos"]) <= {e["file"] for e in C46["videos"]})
        self.assertEqual(CFG48["ref_f"], 1553)
        self.assertIn("code_48", CFG48)

    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        names = list(hf.MM_DETECTORS)
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        mk = lambda f: {"ft": {"f": f, "at_grid_edge": False, "sharpness": 1.01, "verdict": "REJECT"}, "excluded": [], "curve": {"grid": grid.tolist(), "cost": ((np.log(grid) - np.log(f)) ** 2).tolist()}}   # noqa: E731
        stage2 = [{"video": e["file"], "source": "46", "distance_cm": e["distance_cm"], "lens": "1x", "detectors": {n: mk(1400.0 + 10 * k - e["distance_cm"]) for k, n in enumerate(["mediapipe_hands"] + names)}}
                  for e in CFG48["stage2_videos"]]
        stage2[2]["detectors"]["mmhand_scnet"] = {"excluded": ["가장자리 접촉 프레임 9.1% > 5%"]}
        existing = {"videos": {s["video"]: 1100.0 for s in stage2}, "slope": {"distances": [50.0, 60.0, 75.0], "pooled_f": {"50.0": 1106.5, "60.0": 982.4, "75.0": 1034.3}, "slope": None, "interval_95": None}}
        slopes = run.mm2_stage2_slopes(stage2, ["mediapipe_hands"] + names, CFG48)
        self.assertEqual(slopes["mmhand_scnet"]["n_videos"][50.0], 3)
        env = {"info": {n: {"config_sha256": "c" * 64, "test_pipeline": ["LoadImage"], "input_size": [256, 256], "test_cfg": {"flip_test": True}, "remap": None, "names_ok": True} for n in names}, "versions": {}}
        checks = {n: {"IMG_7298.MOV": {"frames": [0], "n_boxes": 1, "max_abs_px": 0.0, "max_abs_conf": 0.0}} for n in names}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.mm2_report(d / "report.md", CFG48, env, checks, {}, {n: {"identity": True, "n_boxes": 1} for n in names}, {}, [], {"official": True, "order": True, "stage2": "실행함"})
            run.mm2_summary(d / "summary.md", CFG48, stage2, names, existing, slopes)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"verification": {"checks": checks, "order": {}, "agreement": {}, "failure": {}, "verified": {"official": True, "order": True}, "env": env},
                                                                    "stage2": stage2, "slopes": slopes}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG48)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.mm2_results_yaml(d, cfg)).read_text()
            summ, rep = (d / "summary.md").read_text(), (d / "report.md").read_text()
            self.assertTrue(rep.startswith("# 실험 48:"))
            self.assertIn("1553", summ)
            self.assertIn("1106.5", summ)
            for name, text in (("report", rep), ("yaml", y)):
                self.assertNotIn("1553", text, name)
            for name, text in (("report", rep), ("summary", summ), ("yaml", y)):
                for tag in LOCATION_TAGS:
                    self.assertNotIn(tag, text.lower(), (name, tag))
            self.assertIn("distance", yaml.safe_load(y))


# ---- experiment 49: experiment 46 on the videos IMG_7312-7323 -------------------------------------------------
CFG49 = yaml.safe_load((ROOT / "configs/49_free_shape3d_7312.yaml").read_text())


class E5Config(unittest.TestCase):
    def test_the_config_keeps_46s_settings_and_lists_the_new_videos(self):
        for k in CFG46:
            if k not in ("videos", "experiment", "pipeline", "output", "code_46"):
                self.assertEqual(CFG49[k], CFG46[k], k)
        self.assertEqual(CFG49["pipeline"]["estimate"], CFG46["pipeline"]["estimate"])
        want = {(f"IMG_{n}.MOV", d) for d, ns in ((75, range(7312, 7316)), (60, range(7316, 7320)), (50, range(7320, 7324))) for n in ns}
        self.assertEqual({(e["file"], e["distance_cm"]) for e in CFG49["videos"]}, want)
        self.assertIn("code_49", CFG49)


class E5Reports(unittest.TestCase):
    def test_summary_report_and_yaml_keep_the_reference_in_summary_only(self):
        ref = {float(d): float(f) for d, f in CFG49["ref_f_by_distance"].items()}
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 8)
        cost = ((np.log(grid) - np.log(1600.0)) ** 2 + 0.01).tolist()
        est = {"f": 1605.0, "f_planar": 1610.0, "f_grid": 1600.0, "at_grid_edge": False, "at_bound": False, "relief_rel": 0.01, "relief_per_joint": [0.01] * 16, "cost": 0.01, "cost_planar": 0.011,
               "curve": {"grid": grid.tolist(), "cost": cost}, "sharpness": 1.4, "ba3_success": True, "ba3_nfev": 20, "ba3_rms_px": 0.2, "anchors": [0, 1, 2], "n_frames_in_problem": 120, "n_points": 900, "joint_ids": list(range(5, 21)), "elapsed_s": 60.0}
        results = [{"video": e["file"], "group": "G8", "distance_cm": float(e["distance_cm"]), "hand": "unrecorded", "size": [1920, 1080], "n_frames": 300, "duration_s": 20.0, "detect_rate": 1.0, "edge_rate": 0.0, "excluded": [], "est": est}
                   for e in CFG49["videos"]]
        results[0].pop("est")
        results[0]["excluded"] = ["가장자리 접촉 프레임 6.3% > 5%"]
        inc = [r for r in results if "est" in r]
        fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in (50.0, 60.0, 75.0)}
        sl = hf.s3_slope(fb, {d: [False] * len(fb[d]) for d in fb}, 50, 42)
        cur = {d: [r["est"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in fb}
        psl = hf.pr_slope(cur, grid, 50, 42)
        nv = {d: len(fb[d]) for d in fb}
        ana = {"shape3d": {"slope": sl, "verdict": hf.pr_verdict(sl, ref, CFG49["verdict"], nv)}, "planar": {"slope": psl, "verdict": hf.pr_verdict(psl, ref, CFG49["verdict"], nv)}}
        prior = {50.0: 1675.3, 60.0: 1476.1, 75.0: 1167.1}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "figures").mkdir()
            run.e5_figures(d / "figures", results, ana, ref, prior)
            run.e5_report(d / "report.md", results, CFG49)
            run.e5_summary(d / "summary.md", results, ana, ref, CFG49, prior)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG49)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.e5_results_yaml(d, cfg)).read_text()
            self.assertTrue((d / "figures/f_vs_distance_loglog.png").exists())
            summ = (d / "summary.md").read_text()
            self.assertIn("1553", summ)
            self.assertIn("판정 (free_shape3d", summ)
            self.assertTrue(summ.startswith("# 실험 49 summary") and (d / "report.md").read_text().startswith("# 실험 49:"))
            self.assertNotIn("7298", summ + (d / "report.md").read_text() + y)
            for name, text in (("report", (d / "report.md").read_text()), ("yaml", y)):
                self.assertNotIn("1553", text, name)
            for line in summ.splitlines():
                if line.startswith("|") and "---" not in line:
                    self.assertEqual(line.count("|"), line.strip().count("|"), line)
            self.assertEqual(run.e5_prior({"compare": {"45": str(d / "none.json")}}), {})
            (d / "p45.json").write_text(json.dumps({"analysis": {"shape3d": {"slope": {"pooled_f": {"50.0": 1675.3, "60.0": 1476.1}}}}}))
            self.assertEqual(run.e5_prior({"compare": {"45": str(d / "p45.json")}}), {50.0: 1675.3, 60.0: 1476.1})


# ---- experiment 50: experiment 48 on the videos of 49 ---------------------------------------------------------
CFG50 = yaml.safe_load((ROOT / "configs/50_mmpose_hand_7312.yaml").read_text())


class MM3Reports(unittest.TestCase):
    def test_the_config_keeps_48s_settings_and_names_the_source_experiment_in_every_output(self):
        for k in ("detectors", "detector_settings", "candidate", "exclusion", "guide", "hand_box_source", "bootstrap", "thermal", "ref_f"):
            self.assertEqual(CFG50[k], CFG48[k], k)
        self.assertEqual([(e["file"], e["distance_cm"]) for e in CFG50["stage2_videos"]], [(e["file"], e["distance_cm"]) for e in CFG49["videos"]])
        names = list(hf.MM_DETECTORS)
        grid = np.geomspace(0.35 * 1920, 1.6 * 1920, 60)
        mk = lambda f: {"ft": {"f": f, "at_grid_edge": False, "sharpness": 1.01}, "excluded": [], "curve": {"grid": grid.tolist(), "cost": ((np.log(grid) - np.log(f)) ** 2).tolist()}}   # noqa: E731
        stage2 = [{"video": e["file"], "source": "49", "distance_cm": e["distance_cm"], "lens": "1x", "detectors": {n: mk(1400.0 + 10 * k) for k, n in enumerate(["mediapipe_hands"] + names)}} for e in CFG50["stage2_videos"]]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "r49.json").write_text(json.dumps({"videos": [{"video": s["video"], "est": {"f": 1200.0}} for s in stage2], "analysis": {"shape3d": {"slope": None}}}))
            cfg = copy.deepcopy(CFG50)
            cfg["existing_results"] = {"49": str(d / "r49.json")}
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            existing = run.mm3_existing(cfg)
            self.assertEqual(len(existing["videos"]), 12)
            slopes = run.mm2_stage2_slopes(stage2, ["mediapipe_hands"] + names, cfg)
            env = {"info": {n: {"remap": None, "names_ok": True} for n in names}, "versions": {}}
            run.mm3_report(d / "report.md", cfg, env, {}, {}, {}, {}, [], {"official": True, "order": True, "stage2": "실행함"})
            run.mm3_summary(d / "summary.md", cfg, stage2, names, existing, slopes)
            (d / "results.json").write_text(json.dumps(hf.jsonable({"verification": {"checks": {}, "order": {}, "agreement": {}, "failure": {}, "verified": {}, "env": env}, "stage2": stage2, "slopes": slopes}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            y = Path(run.mm3_results_yaml(d, cfg)).read_text()
            summ, rep = (d / "summary.md").read_text(), (d / "report.md").read_text()
            self.assertTrue(summ.startswith("# 실험 50 summary") and rep.startswith("# 실험 50:"))
            for text in (summ, rep, y):
                self.assertNotIn("실험 46", text)
            self.assertIn("실험 49 free_shape3d", summ)
            self.assertIn("1200.0", summ)
            self.assertNotIn("1553", rep + y)


# ---- experiment 51: free_shape3d on the joints of every detector of 48 / 50 ---------------------------------------
CFG51 = yaml.safe_load((ROOT / "configs/51_free_shape3d_mmpose.yaml").read_text())


class FSReports(unittest.TestCase):
    def test_config_uses_46s_settings_and_47s_detectors(self):
        for k in ("candidate", "exclusion", "free_shape3d", "verdict", "bootstrap"):
            self.assertEqual(CFG51[k], CFG46[k], k)
        self.assertEqual(CFG51["detectors"], ["mediapipe_hands"] + CFG47["detectors"])
        self.assertEqual(CFG51["detector_settings"]["mediapipe_hands"], CFG46["detector_settings"]["mediapipe_hands"])
        for n in CFG47["detectors"]:
            self.assertEqual(CFG51["detector_settings"][n], CFG47["detector_settings"][n])
        self.assertEqual([s["config"] for s in CFG51["sets"]], ["configs/48_mmpose_hand_46_videos.yaml", "configs/50_mmpose_hand_7312.yaml"])

    def test_summary_has_f_with_error_and_report_and_yaml_have_no_reference(self):
        est = lambda f: {"f": f, "f_planar": f, "relief_rel": 0.02, "at_bound": False, "at_grid_edge": False}   # noqa: E731
        results = []
        for st in CFG51["sets"]:
            for det in CFG51["detectors"]:
                for k, d in enumerate((50.0, 50.0, 60.0, 60.0, 75.0, 75.0)):
                    results.append({"set": st["name"], "detector": det, "video": f"V{k}.MOV", "distance_cm": d, "excluded": [], "est": est(1500.0 + 10 * k)})
        results[0] = {**results[0], "excluded": ["가장자리 접촉 프레임 9.1% > 5%"]}
        results[0].pop("est")
        ana = run.fs_analyze(results, CFG51)
        self.assertIn(ana["실험 46 영상"]["mmhand_scnet"]["verdict"]["verdict"], ("반비례 없음", "반비례 있음", "판단 보류", "판단 불가"))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run.fs_report(d / "report.md", results, CFG51, {})
            run.fs_summary(d / "summary.md", results, ana, CFG51, {"실험 46 영상": {"V1.MOV": 1510.0}})
            (d / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False))
            (d / "metadata.json").write_text(json.dumps({"git": {"commit": "abcdef1234", "git_dirty": False}}))
            cfg = copy.deepcopy(CFG51)
            cfg["output"]["results_yaml"] = str(d / "r.yaml")
            y = Path(run.fs_results_yaml(d, cfg)).read_text()
            summ, rep = (d / "summary.md").read_text(), (d / "report.md").read_text()
            self.assertIn("1510.0 (-2.8%)", summ)
            self.assertIn("최대 차이 0.0000 px", summ)
            self.assertNotIn("1553", rep + y)
            self.assertIn("1553", summ)


if __name__ == "__main__":
    unittest.main()
