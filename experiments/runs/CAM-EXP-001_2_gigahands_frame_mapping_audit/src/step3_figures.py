"""Step 3 - bimanual four-colour overlays, official-repro comparison, and the
single main explanation figure.

Every figure draws BOTH hands at once, because the CAM-EXP-001.1 failure mode
(annotation on the wrong hand) is invisible when one hand is drawn alone.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from audit_common import (COLOR_LEGEND, COLORS, HANDS, RUN_DIR, Sequence,
                          centroid_dist, is_zero_pattern, median_err, read_csv,
                          rel, repro_tile, sequences, video_frame, write_csv)
from experiments.src.visualization.overlay import EDGES_21

log = logging.getLogger("cam-exp-001.2")
FIG = RUN_DIR / "figures"
CONF = 0.5


def _plot_hand(ax, pts, color, mask=None, ms=2.6, lw=0.9):
    if pts is None:
        return
    P = np.asarray(pts, float)[:, :2]
    m = np.ones(len(P), bool) if mask is None else np.asarray(mask, bool)
    m = m & np.isfinite(P).all(1)
    if len(P) >= 21:
        for a, b in EDGES_21:
            if m[a] and m[b]:
                ax.plot(P[[a, b], 0], P[[a, b], 1], "-", color=color, lw=lw, alpha=0.9)
    ax.plot(P[m, 0], P[m, 1], "o", color=color, ms=ms, mew=0)


def bimanual_axis(ax, s: Sequence, cam_name: str, frame: int, title: str = "",
                  zoom: bool = True, show_2d=True, show_3d=True):
    """Draw both hands, 2D annotation and 3D reprojection, in four colours."""
    cam = s.cameras.get(cam_name)
    vid = s.video_path(cam_name)
    im = video_frame(vid, frame) if vid else None
    if im is None:
        ax.axis("off")
        ax.set_title((title + "\n(annotated video not in download)").strip(), fontsize=6.5)
        return {}
    ax.imshow(im)
    info, pts_all = {}, []
    for h in HANDS:
        X = s.joints3d(h, frame)
        g = s.joints2d(h, cam_name, frame)
        chosen = frame in s.chosen[h]
        zero = bool(g is not None and is_zero_pattern(g))
        if show_2d and g is not None and not zero:
            m = g[:, 2] >= CONF
            _plot_hand(ax, g, COLORS[(h, "2d")], m)
            pts_all.append(g[m, :2])
        if show_3d and X is not None and chosen:
            uv, _ = cam.project(X)
            _plot_hand(ax, uv, COLORS[(h, "3d")])
            pts_all.append(uv[np.isfinite(uv).all(1)])
        err, _ = (median_err(cam, X, g) if (g is not None and not zero and chosen)
                  else (float("nan"), 0))
        info[h] = {"chosen": int(chosen), "zero": int(zero), "err": err,
                   "valid2d": int(g is not None and not zero)}
    if zoom and pts_all:
        P = np.vstack([p for p in pts_all if len(p)])
        if len(P):
            lo, hi = P.min(0), P.max(0)
            pad = max(60, 0.25 * float(np.max(hi - lo)))
            ax.set_xlim(max(0, lo[0] - pad), min(im.shape[1], hi[0] + pad))
            ax.set_ylim(min(im.shape[0], hi[1] + pad), max(0, lo[1] - pad))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.5)
    return info


def legend_handles():
    return [Line2D([], [], color=COLORS[(h, k)], marker="o", ls="-",
                   label=f"{h.upper()} {'2D annotation' if k == '2d' else '3D reprojection'}")
            for h in HANDS for k in ("2d", "3d")]


def caption(s: Sequence, cam, frame, info) -> str:
    def f(h):
        i = info.get(h, {})
        if not i:
            return "n/a"
        if i.get("zero"):
            return "2D=(0,0)"
        if not i.get("chosen"):
            return "not chosen"
        e = i.get("err", float("nan"))
        return f"{e:.1f}px" if np.isfinite(e) else "n/a"
    return (f"{s.name} {cam.replace('brics-odroid-', '')} f{frame}\n"
            f"L {f('left')} | R {f('right')}  chosenL/R="
            f"{int(frame in s.chosen['left'])}/{int(frame in s.chosen['right'])}")


def grid(items, out, title, ncols=3, zoom=True):
    """items: list of (Sequence, camera, frame)"""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (s, cam, fr) in zip(axes.ravel(), items):
        ax.axis("on")
        info = bimanual_axis(ax, s, cam, fr, zoom=zoom)
        ax.set_title(caption(s, cam, fr, info), fontsize=6.5)
    fig.suptitle(f"{title}\n{COLOR_LEGEND}", fontsize=9)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=7,
               frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("  wrote", rel(out))


# ---------------------------------------------------------------------------
def resolve_tile_index(s: Sequence, cam_name: str, frame: int, cam_list) -> tuple:
    """Find which montage tile belongs to a camera, by image matching.

    The tile order of repro_2d_vid is not documented, so rather than assuming
    it follows the sorted camera list we match each candidate tile against the
    actual RGB frame and take the best correlation. Returns (index, score).
    """
    import cv2
    vid = s.video_path(cam_name)
    im = video_frame(vid, frame) if vid else None
    if im is None:
        return -1, 0.0
    best, best_s = -1, -1.0
    ref = cv2.resize(im, (64, 36)).astype(np.float32)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    for i in range(42):
        t = repro_tile(s, "2d", frame, i)
        if t is None:
            continue
        c = cv2.resize(np.ascontiguousarray(t), (64, 36)).astype(np.float32)
        c = (c - c.mean()) / (c.std() + 1e-6)
        score = float((ref * c).mean())
        if score > best_s:
            best_s, best = score, i
    return best, best_s


def official_comparison(cases, cam_list_by_seq) -> list:
    """5-panel: RGB | official repro_2d | official repro_3d | our 2D | our 3D."""
    out_rows = []
    for tag, s, cam_name, frame in cases:
        cams = cam_list_by_seq[s.name]
        idx, score = resolve_tile_index(s, cam_name, frame, cams)
        nominal = cams.index(cam_name) if cam_name in cams else -1
        t2 = repro_tile(s, "2d", frame, idx) if idx >= 0 else None
        t3 = repro_tile(s, "3d", frame, idx) if idx >= 0 else None
        vid = s.video_path(cam_name)
        im = video_frame(vid, frame) if vid else None
        fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        if im is not None:
            axes[0].imshow(im)
        axes[0].set_title("A. original RGB", fontsize=8)
        for ax, t, name in ((axes[1], t2, "B. official repro_2d_vid tile"),
                            (axes[2], t3, "C. official repro_3d_vid tile")):
            if t is not None:
                ax.imshow(t)
            else:
                ax.text(.5, .5, "tile unavailable", ha="center", va="center", fontsize=8)
            ax.set_title(name, fontsize=8)
        info2 = bimanual_axis(axes[3], s, cam_name, frame, zoom=False, show_3d=False)
        axes[3].set_title("D. our 2D annotation overlay", fontsize=8)
        info3 = bimanual_axis(axes[4], s, cam_name, frame, zoom=False, show_2d=False)
        axes[4].set_title("E. our 3D reprojection overlay", fontsize=8)
        ui = s.union_index(frame)
        fig.suptitle(
            f"[{tag}] {s.name} {cam_name} frame {frame} (union position {ui}) - "
            f"{caption(s, cam_name, frame, info2).splitlines()[-1]}\n{COLOR_LEGEND}",
            fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        out = FIG / "official_repro_comparison" / f"{tag}_{s.name}_{cam_name}_f{frame}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=95)
        plt.close(fig)
        le = info2.get("left", {}).get("err", float("nan"))
        re_ = info2.get("right", {}).get("err", float("nan"))
        out_rows.append({
            "tag": tag, "sequence": s.name, "camera": cam_name, "frame": frame,
            "union_position": ui,
            "tile_index_matched": idx, "tile_index_nominal_sorted_order": nominal,
            "tile_match_score": round(score, 4),
            "tile_order_equals_sorted_camera_order": int(idx == nominal),
            "left_err_px": round(le, 2) if np.isfinite(le) else "",
            "right_err_px": round(re_, 2) if np.isfinite(re_) else "",
            "figure": rel(out),
        })
        print("  wrote", rel(out))
    return out_rows


def frame_mapping_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    boxes = [
        (0.06, 0.78, "RGB video frame i\nrgb_vid/<cam>/<cam>_<ts>.mp4", "#dbeafe"),
        (0.06, 0.60, "timestamp sidecar line i\nframe_<us>_<i>  (0-based, contiguous)", "#dbeafe"),
        (0.06, 0.42, "keypoints_2d/<hand>/<take>/<cam>_<ts>.jsonl  row i\n"
                     "21 x [u, v, conf]", "#dcfce7"),
        (0.06, 0.24, "keypoints_3d/<take>/<hand>.jsonl  row i\n"
                     "21 x [x, y, z, conf]   (rows 0..max(chosen))", "#fee2e2"),
        (0.06, 0.06, "chosen_frames_<hand>.json\nset of frame ids with a valid 3D pose",
         "#fef9c3"),
    ]
    for x, y, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.52, 0.12, facecolor=c,
                                   edgecolor="#334155", lw=1.2))
        ax.text(x + 0.26, y + 0.06, txt, ha="center", va="center", fontsize=8.5)
    for y0, y1, lbl in ((0.78, 0.72, "same index i"), (0.60, 0.54, "same index i"),
                        (0.42, 0.36, "same index i"), (0.24, 0.18, "membership test")):
        ax.annotate("", xy=(0.32, y1), xytext=(0.32, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4))
        ax.text(0.34, (y0 + y1) / 2, lbl, fontsize=8, va="center", color="#334155")

    ax.add_patch(plt.Rectangle((0.66, 0.24), 0.30, 0.30, facecolor="#ede9fe",
                               edgecolor="#334155", lw=1.2))
    ax.text(0.81, 0.39,
            "params/<take>.json\nrepro_2d_vid, repro_3d_vid, mano_vid\n\n"
            "indexed by POSITION in\nsorted(chosen_left | chosen_right)\n"
            "NOT by frame id",
            ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(0.66, 0.34), xytext=(0.58, 0.12),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.6))
    ax.text(0.585, 0.20, "union_position(frame)", fontsize=8, color="#7c3aed", rotation=28)
    ax.text(0.03, 0.97, "GigaHands frame index relationships (verified in CAM-EXP-001.2)",
            fontsize=12, fontweight="bold")
    ax.text(0.03, 0.93,
            "Verified on all 5 demo sequences: 2D rows = timestamps = RGB frames; "
            "3D rows = max(chosen)+1; params/repro = |union|.", fontsize=8.5)
    ax.text(0.03, 0.005,
            "Caveat: in p52-instrument-0034, 25 of 40 cameras ship an rgb_vid mp4 whose "
            "filename timestamp differs from the annotated take (+13.4 s / +154 s), i.e. a "
            "different segment. Match video by filename timestamp, never by directory alone.",
            fontsize=7.5, color="#b91c1c")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "frame_mapping_diagram.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "frame_mapping_diagram.png"))


def main() -> dict:
    seqs = {d.name: Sequence(d) for d in sequences()}
    census = read_csv(RUN_DIR / "results" / "raw" / "gigahands_demo_qc_census.csv.gz")
    cam_list_by_seq = {n: s.cameras_2d("left") for n, s in seqs.items()}

    # index the census by (sequence, camera, frame) so both hands can be judged
    per_frame = defaultdict(dict)
    for r in census:
        per_frame[(r["sequence"], r["camera"], r["frame"])][r["hand"]] = r

    swap_cams = {(r["sequence"], r["camera"])
                 for r in read_csv(RUN_DIR / "results" / "summary" /
                                   "systematic_swap_cameras.csv")
                 if r["verdict"] == "systematic_swap"}

    def pick(status, n, both_hands=False, prefer_swap_cams=False):
        """Choose representative frames, spread over cameras and sequences.

        Selection walks camera-by-camera so a grid never shows one camera
        repeatedly, and can require that BOTH hands share the status (used for
        the clean 'good' examples).
        """
        cands = []
        for (seq, cam, fr), hands in per_frame.items():
            if any(h.get("video_available") != "1" for h in hands.values()):
                continue
            hit = ([h["status"] for h in hands.values()].count(status)
                   == (2 if both_hands else 1))
            if both_hands:
                hit = all(h["status"] == status for h in hands.values()) and len(hands) == 2
            else:
                hit = any(h["status"] == status for h in hands.values())
            if not hit:
                continue
            pri = 0 if (prefer_swap_cams and (seq, cam) in swap_cams) else 1
            cands.append((pri, seq, cam, int(fr)))
        cands.sort()
        out, used = [], defaultdict(int)
        for rounds in range(1, 4):
            for pri, seq, cam, fr in cands:
                if len(out) >= n:
                    break
                if used[(seq, cam)] >= rounds:
                    continue
                if (seqs[seq], cam, fr) in out:
                    continue
                used[(seq, cam)] += 1
                out.append((seqs[seq], cam, fr))
            if len(out) >= n:
                break
        return out[:n]

    good = pick("good", 9, both_hands=True)
    swap = pick("likely_hand_identity_error", 12, prefer_swap_cams=True)
    unexp = pick("bad_unexplained", 9)
    zero = pick("invalid_2d_zero", 6)

    grid(good, FIG / "both_hands" / "good_bimanual_grid.png",
         "Good views - both hands, annotation and reprojection coincide")
    grid(swap + unexp, FIG / "both_hands" / "bad_bimanual_grid.png",
         "Previously 'bad' views - both hands drawn, mapping verified correct")
    grid(swap, FIG / "both_hands" / "suspected_swap_grid.png",
         "Suspected hand-identity error: LEFT annotation sits on the RIGHT projection "
         "and vice versa")
    grid(zero, FIG / "both_hands" / "sentinel_grid.png",
         "Invalid 2D: all 21 joints exactly (0,0) - that hand is simply not annotated")
    # No re-indexing repaired anything, so this figure documents that explicitly.
    grid(unexp, FIG / "both_hands" / "mapping_corrected_grid.png",
         "Remaining unexplained views AFTER the audited mapping "
         "(no re-indexing changed any case - mapping was already correct)")

    # same frame across several cameras, to show the swap is camera-dependent
    swapcams = read_csv(RUN_DIR / "results" / "summary" / "systematic_swap_cameras.csv")
    sysm = [r for r in swapcams if r["verdict"] == "systematic_swap"]
    multi = []
    if sysm:
        sname = sysm[0]["sequence"]
        s = seqs[sname]
        fr = None
        for r in census:
            if (r["sequence"] == sname and r["camera"] == sysm[0]["camera"]
                    and r["status"] == "likely_hand_identity_error"
                    and r["video_available"] == "1"):
                fr = int(r["frame"]); break
        if fr is not None:
            goodcams = [r["camera"] for r in census
                        if r["sequence"] == sname and r["frame"] == str(fr)
                        and r["status"] == "good" and r["video_available"] == "1"]
            cams = [sysm[0]["camera"]] + list(dict.fromkeys(goodcams))[:5]
            multi = [(s, c, fr) for c in cams]
            grid(multi, FIG / "mapping_examples" / "same_frame_multiview.png",
                 f"Same sequence and frame ({sname} f{fr}) seen from several cameras: "
                 "the swap follows the camera, not the frame", ncols=3)

    cases = ([("swap", s, c, f) for (s, c, f) in swap[:4]]
             + [("zero", s, c, f) for (s, c, f) in zero[:2]]
             + [("good", s, c, f) for (s, c, f) in good[:4]])
    rows = official_comparison(cases, cam_list_by_seq)
    write_csv(RUN_DIR / "results" / "summary" / "official_repro_verdict.csv", rows)

    frame_mapping_diagram()
    main_figure(seqs, good, swap, zero, unexp)
    return {"official_rows": rows}


def main_figure(seqs, good, swap, zero, unexp) -> None:
    """One picture that carries the whole result of CAM-EXP-001.2."""
    rows = [
        ("ROW 1  NORMAL bimanual case - 2D annotation and 3D reprojection agree on both hands",
         good[:4]),
        ("ROW 2  Previously 'bad' - mapping re-verified as correct; the annotation sits on "
         "the OTHER hand (green pairs with orange, blue pairs with red)", swap[:4]),
        ("ROW 3  Invalid 2D - all 21 joints exactly (0,0); that hand is simply not annotated",
         zero[:4]),
        ("ROW 4  Still unresolved - large error with no clean hand-swap explanation",
         unexp[:4]),
    ]
    # One extra grid row per band is reserved for a horizontal header, so the
    # labels never collide with the images the way rotated side labels do.
    fig = plt.figure(figsize=(17.5, 16.0))
    gs = fig.add_gridspec(8, 4, height_ratios=[0.12, 1] * 4, hspace=0.30, wspace=0.06)
    for r, (label, items) in enumerate(rows):
        hax = fig.add_subplot(gs[2 * r, :])
        hax.axis("off")
        hax.text(0.0, 0.35, label, fontsize=10.5, fontweight="bold", va="center",
                 color="#1e293b")
        for c in range(4):
            ax = fig.add_subplot(gs[2 * r + 1, c])
            ax.axis("off")
            if c < len(items):
                ax.axis("on")
                sq, cam, fr = items[c]
                info = bimanual_axis(ax, sq, cam, fr)
                ax.set_title(caption(sq, cam, fr, info), fontsize=6.5)
    fig.suptitle(
        "CAM-EXP-001.2 - GigaHands frame/annotation mapping audit\n"
        "Frame mapping verified: 2D row = 3D row = RGB frame index; chosen_frames gates "
        "3D validity. No case was repaired by re-indexing.\n" + COLOR_LEGEND,
        fontsize=12)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.subplots_adjust(top=0.90, bottom=0.045, left=0.015, right=0.985)
    out = FIG / "CAM_EXP_001_2_MAIN_EXPLANATION.png"
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print("  wrote", rel(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
