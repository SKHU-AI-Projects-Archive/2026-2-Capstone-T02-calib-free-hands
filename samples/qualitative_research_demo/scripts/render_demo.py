"""Render the qualitative comparison from the cached hand predictions.

The hand model is NOT re-run. For a given focal f the pipeline's own formula
(rgb_predictor._cam_crop_to_full) gives

    tz = 2 f / (s * box_size)        ->  tz(f) = tz_legacy * f / f_legacy
    tx, ty                          ->  contain no focal at all

so every condition here uses bit-identical detections, crops, MANO results and
root-relative joints, and differs only in the camera translation that the focal
produces. The audit table written beside the videos shows that explicitly.

Nothing in these clips is ground truth: there is no dataset-provided camera
calibration for them, so the conditions are labelled by what they are —
a pipeline focal convention and a model focal estimate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (CACHE, CALIB, DEMO, OUT_ANYCALIB, OUT_COMPARISON, OUT_E2,  # noqa: E402
                    OUT_FRAMES, OUT_LEGACY, REPO, read_json, write_csv,
                    write_json)

sys.path.insert(0, str(REPO / "demo"))
from hand_topology import (HAND_BASE, HAND_BASE_COLORS, HAND_CONNECTIONS,  # noqa: E402
                           HAND_LABELS, JOINTS_PER_HAND, LEFT, RIGHT,
                           hex_to_bgr, joint_color, raw_to_view)

WRIST = 0
PANEL_W, PANEL_H = 960, 480          # RGB panel
PANEL_3D_H = 620                     # 3D panel (taller: the depth axis is the point)
HEAD_H = 60
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- conditions
def scale_cam_t(cam_t_legacy, focal_legacy, focal_new):
    """tz scales with the focal; tx and ty do not depend on it."""
    out = cam_t_legacy.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(focal_legacy > 0, focal_new / focal_legacy, 0.0)
    out[..., 2] = cam_t_legacy[..., 2] * ratio
    return out


def absolute_joints(kp3d_rel, cam_t):
    """(F,42,3) root-relative + per-hand translation -> absolute camera space."""
    out = kp3d_rel.copy()
    for side in (LEFT, RIGHT):
        b = HAND_BASE[side]
        out[:, b:b + JOINTS_PER_HAND, :] += cam_t[:, side, None, :]
    return out


# ------------------------------------------------------------------- drawing
def draw_overlay(frame_bgr, kp2d, bbox, scores, detected, height):
    """The existing demo's 2D overlay, rescaled to the panel height."""
    img = frame_bgr.copy()
    for side in (LEFT, RIGHT):
        if not detected[side]:
            continue
        b = HAND_BASE[side]
        kp = kp2d[b:b + JOINTS_PER_HAND]
        base = hex_to_bgr(HAND_BASE_COLORS[side])
        x1, y1, x2, y2 = bbox[side].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), base, 3)
        cv2.putText(img, f"{HAND_LABELS[side]} {scores[side]:.2f}",
                    (x1, max(y1 - 8, 16)), FONT, 1.0, base, 2)
        for s, e in HAND_CONNECTIONS:
            cv2.line(img, tuple(kp[s].astype(int)), tuple(kp[e].astype(int)),
                     hex_to_bgr(joint_color(e)), 3, cv2.LINE_AA)
        for i, (u, v) in enumerate(kp):
            cv2.circle(img, (int(u), int(v)), 7 if i == WRIST else 4,
                       hex_to_bgr(joint_color(i)), -1, cv2.LINE_AA)
    return fit_panel(img, PANEL_W, height)


def fit_panel(img, w, h):
    """Letterbox to exactly (w, h) on black, preserving aspect."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(int(iw * s), 1), max(int(ih * s), 1)),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def render_3d(kp_abs, detected, lims, title, subtitle, w=PANEL_W, h=PANEL_3D_H):
    """The existing demo's 3D panel (hand_topology.raw_to_view, Z-up display)."""
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    for side in (LEFT, RIGHT):
        if not detected[side]:
            continue
        b = HAND_BASE[side]
        pts = raw_to_view(kp_abs[b:b + JOINTS_PER_HAND])
        for s, e in HAND_CONNECTIONS:
            ax.plot(*zip(pts[s], pts[e]), color=joint_color(e), linewidth=2)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[joint_color(i) for i in range(JOINTS_PER_HAND)],
                   s=16, depthshade=False)
        ax.text(*pts[WRIST], f" {HAND_LABELS[side]}",
                color=HAND_BASE_COLORS[side], fontsize=10)
        # a ray from the camera to the wrist makes the absolute depth legible
        ax.plot(*zip((0.0, 0.0, 0.0), tuple(pts[WRIST])),
                color=HAND_BASE_COLORS[side], linewidth=0.8, linestyle=":",
                alpha=0.7)
    ax.scatter([0], [0], [0], c="white", s=26, marker="o")
    ax.text(0, 0, 0, " camera", color="white", fontsize=8)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel("X [m]", color="white", fontsize=9)
    ax.set_ylabel("depth [m]", color="white", fontsize=9)
    ax.set_zlabel("up [m]", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("gray")
    ax.view_init(elev=18, azim=-60)
    ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
    plt.close(fig)
    return fit_panel(buf, w, h)


MIN_SPAN_M = 0.35


def view_limits(*abs_sets, detected, pad=0.08):
    """One fixed box covering EVERY condition, so the panels are comparable.

    Robust 3-97 percentile over all conditions jointly, per axis, with a
    minimum span so a nearly still hand does not blow the scale up. Per-axis
    rather than a cube because the depth difference between the two conditions
    is metres while the hand itself is centimetres; a cube would shrink the
    skeleton to a dot. Both panels get the SAME limits and the same view angle,
    which is what makes them comparable. No ground truth is consulted - there is
    none for these clips.
    """
    pts = []
    for kp in abs_sets:
        for f in range(kp.shape[0]):
            for side in (LEFT, RIGHT):
                if detected[f, side]:
                    b = HAND_BASE[side]
                    pts.append(raw_to_view(kp[f, b:b + JOINTS_PER_HAND]))
    if not pts:
        return [(-0.3, 0.3), (-0.8, -0.2), (-0.3, 0.3)]
    pts = np.concatenate(pts)
    lo = np.percentile(pts, 3, axis=0) - pad
    hi = np.percentile(pts, 97, axis=0) + pad
    out = []
    for i in range(3):
        mid, span = (lo[i] + hi[i]) / 2, max(hi[i] - lo[i], MIN_SPAN_M)
        out.append((mid - span / 2, mid + span / 2))
    return out


def banner(w, h, lines, color=(255, 255, 255)):
    img = np.zeros((h, w, 3), np.uint8)
    for i, (txt, scale, thick, col) in enumerate(lines):
        cv2.putText(img, txt, (14, 26 + i * 26), FONT, scale, col, thick,
                    cv2.LINE_AA)
    return img


def root_z(kp_abs, detected, side):
    if not detected[side]:
        return float("nan")
    return float(kp_abs[HAND_BASE[side] + WRIST, 2])


# -------------------------------------------------------------------- writer
class H264Writer:
    """H.264 / yuv420p through the bundled ffmpeg, with an mp4v fallback."""

    def __init__(self, path: Path, fps: float, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.proc = None
        self.vw = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                 "-r", f"{fps:.6f}", "-i", "-", "-an",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.codec = "h264 (libx264, yuv420p)"
        except Exception:
            self.vw = cv2.VideoWriter(str(self.path),
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            self.codec = "mp4v fallback"

    def write(self, frame):
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.vw.write(frame)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.vw is not None:
            self.vw.release()


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="sample filename, e.g. 2_screw.mp4")
    ap.add_argument("--tag", default="", help="cache suffix")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--current", default="auto", choices=["auto", "e2", "anycalib"])
    args = ap.parse_args()

    stem = Path(args.video).stem
    cache = np.load(CACHE / f"{stem}{args.tag}_hands.npz", allow_pickle=False)
    est = read_json(CALIB / f"{stem}_video_estimate.json")

    kp3d_rel = cache["kp3d_rel"]
    cam_t_legacy = cache["cam_t_legacy"]
    focal_legacy_per_hand = cache["focal_used_legacy"]
    detected = cache["detected"]
    kp2d = cache["kp2d"]
    bbox = cache["bbox"]
    scores = cache["scores"]
    frame_indices = cache["frame_indices"]
    fps = float(cache["fps"])
    src = Path(str(cache["source_path"]))
    f_legacy = float(np.max(focal_legacy_per_hand))

    f_any = est["anycalib"]["video_focal_px"]
    f_geo = est["geocalib"]["video_focal_px"]
    f_e2 = est["e2_frozen"]["video_focal_px"]
    if args.current == "auto":
        cur_name, f_cur = (("FROZEN E2 - exploratory ensemble", f_e2) if f_e2
                           else ("AnyCalib video estimate", f_any))
    elif args.current == "e2":
        cur_name, f_cur = "FROZEN E2 - exploratory ensemble", f_e2
    else:
        cur_name, f_cur = "AnyCalib video estimate", f_any
    if f_cur is None:
        raise SystemExit("no calibration estimate available for this clip")

    n = len(frame_indices) if not args.max_frames else min(args.max_frames,
                                                           len(frame_indices))
    conditions = {"legacy": f_legacy, "anycalib": f_any}
    if f_e2:
        conditions["e2"] = f_e2
    abs_by_cond = {}
    for name, f in conditions.items():
        ct = scale_cam_t(cam_t_legacy, focal_legacy_per_hand, f)
        abs_by_cond[name] = absolute_joints(kp3d_rel, ct)

    cur_key = "e2" if (args.current in ("auto", "e2") and f_e2) else "anycalib"
    lims = view_limits(abs_by_cond["legacy"][:n], abs_by_cond[cur_key][:n],
                       detected=detected[:n])

    # ---- sanity audit on the first frame that has a hand --------------------
    audit_rows = []
    fi = next((i for i in range(n) if detected[i].any()), None)
    if fi is not None:
        side = LEFT if detected[fi, LEFT] else RIGHT
        b = HAND_BASE[side]
        la = abs_by_cond["legacy"][fi, b + WRIST]
        ca = abs_by_cond[cur_key][fi, b + WRIST]
        same = lambda a, b_: "SAME" if np.allclose(a, b_, atol=0, rtol=0) else "DIFFERENT"
        audit_rows = [
            {"field": "frame", "legacy": int(frame_indices[fi]),
             "current": int(frame_indices[fi]), "verdict": "SAME"},
            {"field": "bbox", "legacy": np.array2string(bbox[fi, side], precision=2),
             "current": np.array2string(bbox[fi, side], precision=2),
             "verdict": "SAME (single inference, reused)"},
            {"field": "detector score", "legacy": round(float(scores[fi, side]), 6),
             "current": round(float(scores[fi, side]), 6), "verdict": "SAME"},
            {"field": "root-relative joints (sum |.|)",
             "legacy": round(float(np.abs(kp3d_rel[fi, b:b + 21]).sum()), 9),
             "current": round(float(np.abs(kp3d_rel[fi, b:b + 21]).sum()), 9),
             "verdict": "SAME (bit-identical array)"},
            {"field": "kp2d (sum |.|)",
             "legacy": round(float(np.abs(kp2d[fi, b:b + 21]).sum()), 6),
             "current": round(float(np.abs(kp2d[fi, b:b + 21]).sum()), 6),
             "verdict": "SAME"},
            {"field": "focal [px]", "legacy": round(f_legacy, 2),
             "current": round(f_cur, 2), "verdict": "DIFFERENT"},
            {"field": "cam_t tx [m]",
             "legacy": round(float(cam_t_legacy[fi, side, 0]), 6),
             "current": round(float(cam_t_legacy[fi, side, 0]), 6),
             "verdict": "SAME (tx carries no focal)"},
            {"field": "cam_t tz [m]",
             "legacy": round(float(cam_t_legacy[fi, side, 2]), 6),
             "current": round(float(cam_t_legacy[fi, side, 2] * f_cur / f_legacy), 6),
             "verdict": "DIFFERENT (tz proportional to focal)"},
            {"field": "wrist absolute Z [m]", "legacy": round(float(la[2]), 6),
             "current": round(float(ca[2]), 6), "verdict": same(la[2], ca[2])},
        ]
        write_csv(DEMO / "metadata" / f"{stem}_focal_isolation_audit.csv", audit_rows)

    # ---- render ------------------------------------------------------------
    cap = cv2.VideoCapture(str(src))
    writers = {k: H264Writer(
        {"legacy": OUT_LEGACY, "anycalib": OUT_ANYCALIB, "e2": OUT_E2}[k]
        / f"{stem}_{k}.mp4", fps, (PANEL_W, HEAD_H + PANEL_H + PANEL_3D_H))
        for k in conditions}
    comp = H264Writer(OUT_COMPARISON / f"{stem}_comparison.mp4", fps,
                      (PANEL_W * 2, HEAD_H + PANEL_H + PANEL_3D_H))
    depth_rows = []
    snap_at = {int(round(x)) for x in np.linspace(0, max(n - 1, 0), 3)}

    for k in range(n):
        idx = int(frame_indices[k])
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        det = detected[k]
        over = draw_overlay(frame, kp2d[k], bbox[k], scores[k], det, PANEL_H)
        t = idx / max(fps, 1e-9)

        panels = {}
        for name, f in conditions.items():
            kp = abs_by_cond[name][k]
            zs = [root_z(kp, det, LEFT), root_z(kp, det, RIGHT)]
            sub = "  ".join(f"{HAND_LABELS[s]} root Z = {zs[s]:.3f} m"
                            for s in (LEFT, RIGHT) if det[s]) or "no hand detected"
            label = {"legacy": "LEGACY - pipeline focal convention",
                     "anycalib": "AnyCalib video estimate",
                     "e2": "FROZEN E2 - exploratory ensemble"}[name]
            panels[name] = render_3d(kp, det, lims,
                                     f"CAMERA SPACE - {label}   f = {f:,.0f} px",
                                     sub)
            head = banner(PANEL_W, HEAD_H, [
                (f"{src.name}   frame {idx}   t = {t:6.2f} s", 0.6, 1,
                 (255, 255, 255)),
                (f"{label}   f = {f:,.0f} px", 0.6, 1, (0, 255, 255)),
            ])
            writers[name].write(np.vstack([head, over, panels[name]]))

        # comparison: legacy | current
        headL = banner(PANEL_W, HEAD_H, [
            (f"{src.name}   frame {idx}   t = {t:6.2f} s", 0.6, 1, (255, 255, 255)),
            (f"LEGACY - pipeline focal convention   f = {f_legacy:,.0f} px", 0.6, 1,
             (120, 200, 255)),
        ])
        headR = banner(PANEL_W, HEAD_H, [
            ("2D overlay is identical on both sides - the focal cancels in the "
             "projection", 0.5, 1, (170, 170, 170)),
            (f"CURRENT CALIBRATION ESTIMATE - {cur_name}   f = {f_cur:,.0f} px",
             0.6, 1, (0, 255, 255)),
        ])
        row1 = np.hstack([np.vstack([headL, over]), np.vstack([headR, over])])
        row2 = np.hstack([panels["legacy"], panels[cur_key]])
        canvas = np.vstack([row1, row2])
        comp.write(canvas)
        if k in snap_at:
            OUT_FRAMES.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT_FRAMES / f"{stem}_t{idx:05d}.png"), canvas)

        for side in (LEFT, RIGHT):
            if not det[side]:
                continue
            depth_rows.append({
                "frame": idx, "time_sec": round(t, 3),
                "hand": HAND_LABELS[side],
                "legacy_root_z_m": round(root_z(abs_by_cond["legacy"][k], det, side), 6),
                "anycalib_root_z_m": round(root_z(abs_by_cond["anycalib"][k], det, side), 6),
                "e2_root_z_m": (round(root_z(abs_by_cond["e2"][k], det, side), 6)
                                if "e2" in abs_by_cond else ""),
            })
        if (k + 1) % 60 == 0:
            print(f"  {stem}: {k + 1}/{n} frames", flush=True)

    cap.release()
    for wtr in writers.values():
        wtr.close()
    comp.close()
    write_csv(DEMO / f"{stem}_root_depth.csv", depth_rows)

    # ---- root-depth comparison plot ---------------------------------------
    if depth_rows:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for side_name, style in (("Left", "-"), ("Right", "--")):
            rows = [r for r in depth_rows if r["hand"] == side_name]
            if not rows:
                continue
            t = [r["time_sec"] for r in rows]
            ax.plot(t, [r["legacy_root_z_m"] for r in rows], style, color="#1f77b4",
                    label=f"{side_name} LEGACY (f={f_legacy:,.0f})")
            ax.plot(t, [r[f"{cur_key}_root_z_m"] for r in rows], style,
                    color="#d62728",
                    label=f"{side_name} {cur_name} (f={f_cur:,.0f})")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("wrist root Z in camera space [m]")
        ax.set_title(f"{src.name} - absolute wrist depth under two focal "
                     f"conditions\nsame hand predictions; only the focal differs. "
                     f"No ground truth exists for this clip.")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(DEMO / f"{stem}_root_depth_comparison.png", dpi=130)
        plt.close(fig)

    write_json(DEMO / "metadata" / f"{stem}_render.json", {
        "video": src.name, "frames_rendered": n, "fps": fps,
        "codec": comp.codec,
        "pipeline_focal_px": f_legacy,
        "anycalib_video_focal_px": f_any,
        "geocalib_video_focal_px": f_geo,
        "e2_video_focal_px": f_e2,
        "current_demo_method": cur_name,
        "axis_limits_display_space": [[float(a), float(b)] for a, b in lims],
        "axis_limit_rule": "robust 3-97 percentile over BOTH conditions jointly, "
                           "padded, cubed; identical limits and view angle on "
                           "both sides",
        "hand_prediction_reuse": "single inference pass; both conditions use the "
                                 "same detections, crops, MANO results and "
                                 "root-relative joints",
        "what_differs": "only cam_t[2] (tz), which the pipeline defines as "
                        "2*f/(s*box_size)",
        "not_ground_truth": "these clips have no dataset-provided calibration; "
                            "neither condition is known to be metrically correct",
    })
    print(f"[{stem}] rendered {n} frames, codec={comp.codec}")
    print(f"  legacy f={f_legacy:,.1f}  anycalib f={f_any}  geocalib f={f_geo}  "
          f"e2 f={f_e2}  -> current = {cur_name}")


if __name__ == "__main__":
    main()
