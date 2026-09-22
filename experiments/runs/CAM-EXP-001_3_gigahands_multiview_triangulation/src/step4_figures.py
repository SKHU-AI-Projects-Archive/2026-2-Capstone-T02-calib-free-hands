"""Step 4 - human-facing figures.

The red/orange skeletons in every figure of this run are NOT the dataset's 3D
reprojected. They are our own reconstruction from the OTHER cameras only, with
the pictured camera held out. Each figure states this, because the whole point
of CAM-EXP-001.3 is that the comparison is independent.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import COLOR_LEGEND, COLORS, MANIFESTS, RUN_DIR, TAKES, read_csv, rel
import loco
import verdicts
from experiments.src.datasets.gigahands import is_zero_2d
from experiments.src.visualization.overlay import EDGES_21, load_frame

log = logging.getLogger("cam-exp-001.3")
FIG = RUN_DIR / "figures"
BANNER = ("red / orange = 3D reconstructed from the OTHER cameras only, then projected "
          "into this held-out camera")


def _draw(ax, pts, color, mask=None, ms=2.6, lw=0.9):
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


def legend_handles():
    lbl = {("left", "2d"): "held-out LEFT 2D annotation",
           ("left", "3d"): "LEFT reconstructed from other cameras",
           ("right", "2d"): "held-out RIGHT 2D annotation",
           ("right", "3d"): "RIGHT reconstructed from other cameras"}
    return [Line2D([], [], color=COLORS[k], marker="o", ls="-", label=v)
            for k, v in lbl.items()]


def panel(ax, take, cam_name, frame, thr, caption_extra="") -> bool:
    """One held-out camera view with both annotations and both reconstructions."""
    vid = take.video_path(cam_name)
    im = load_frame(vid, frame) if vid else None
    if im is None:
        ax.axis("off")
        ax.set_title(f"{take.name} {cam_name} f{frame}\n(annotated video not downloaded)",
                     fontsize=6)
        return False
    ax.imshow(im[:, :, ::-1])
    cam = take.cameras.get(cam_name)
    pts_all = []
    for h in ("left", "right"):
        hyp = loco.reconstruct(take, h, frame, cam_name,
                               thr["triangulation_inlier_px"],
                               thr["min_inliers_for_reconstruction"])
        uv = loco.project_hypothesis(cam, hyp)
        g = take.joints2d(h, cam_name, frame)
        if g is not None and not is_zero_2d(g).all():
            m = g[:, 2] >= 0.5
            _draw(ax, g, COLORS[(h, "2d")], m)
            pts_all.append(g[m, :2])
        if np.isfinite(uv).any():
            _draw(ax, uv, COLORS[(h, "3d")])
            pts_all.append(uv[np.isfinite(uv).all(1)])
    if pts_all:
        P = np.vstack([p for p in pts_all if len(p)])
        lo, hi = P.min(0), P.max(0)
        pad = max(60, 0.25 * float(np.max(hi - lo)))
        ax.set_xlim(max(0, lo[0] - pad), min(im.shape[1], hi[0] + pad))
        ax.set_ylim(min(im.shape[0], hi[1] + pad), max(0, lo[1] - pad))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{take.name} {cam_name.replace('brics-odroid-', '')} f{frame}\n"
                 f"{caption_extra}", fontsize=6)
    return True


def grid(items, out, title, thr, ncols=3):
    """items: list of (take, camera, frame, caption)"""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (take, cam, fr, cap) in zip(axes.ravel(), items):
        ax.axis("on")
        panel(ax, take, cam, fr, thr, cap)
    fig.suptitle(f"{title}\n{BANNER}", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=7,
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("  wrote", rel(out))


def pick(rows, predicate, n, need_video=True, take_by_name=None):
    """Spread picks over sequences and cameras."""
    cand = [r for r in rows if predicate(r)]
    if need_video:
        cand = [r for r in cand
                if take_by_name[r["sequence"]].video_path(r["camera"]) is not None]
    used = Counter()
    out = []
    for rounds in (1, 2, 3):
        for r in cand:
            if len(out) >= n:
                break
            key = (r["sequence"], r["camera"])
            if used[key] >= rounds:
                continue
            used[key] += 1
            out.append(r)
        if len(out) >= n:
            break
    return out[:n]


def explainer_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.95, "Leave-one-camera-out adjudication", fontsize=14,
            fontweight="bold")
    ax.text(0.02, 0.89,
            "The camera under test never contributes to the 3D it is judged against.",
            fontsize=10)
    boxes = [(0.03, 0.55, 0.26, 0.22, "1. Other cameras' 2D\nkeypoints_2d from every\n"
              "camera EXCEPT the one\nunder test", "#dbeafe"),
             (0.36, 0.55, 0.26, 0.22, "2. Robust triangulation\nRANSAC over camera pairs\n"
              "+ calibration only\n(no dataset 3D used)", "#dcfce7"),
             (0.69, 0.55, 0.28, 0.22, "3. Project into the\nheld-out camera\n"
              "gives predicted LEFT\nand RIGHT positions", "#fef9c3")]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#334155", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
    for x0, x1 in ((0.29, 0.36), (0.62, 0.69)):
        ax.annotate("", xy=(x1, 0.66), xytext=(x0, 0.66),
                    arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.6))
    ax.add_patch(plt.Rectangle((0.03, 0.17, ), 0.94, 0.30, facecolor="#f8fafc",
                               edgecolor="#334155", lw=1.2))
    ax.text(0.05, 0.42, "4. Compare the held-out camera's own annotation with both predictions",
            fontsize=10, fontweight="bold", va="top")
    ax.text(0.05, 0.36,
            "E_LL = LEFT annotation  vs  LEFT prediction        "
            "E_LR = LEFT annotation  vs  RIGHT prediction\n"
            "E_RR = RIGHT annotation vs  RIGHT prediction       "
            "E_RL = RIGHT annotation vs  LEFT prediction",
            fontsize=9, family="monospace", va="top")
    ax.text(0.05, 0.245,
            "normal   : E_LL and E_RR small        -> the annotation agrees\n"
            "swapped  : E_LR and E_RL small        -> the annotation names the wrong hand\n"
            "bad 2D   : none of them small         -> the annotation matches nothing",
            fontsize=9, family="monospace", va="top")
    ax.text(0.03, 0.07,
            "The dataset's own 3D is read only afterwards, to check the reconstruction - "
            "never to build or filter it.",
            fontsize=9, style="italic", color="#b91c1c")
    fig.tight_layout()
    fig.savefig(FIG / "diagrams" / "leave_one_camera_out_explainer.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "diagrams" / "leave_one_camera_out_explainer.png"))


def consistency_plot(rows) -> None:
    same = np.array([float(r["provided3d_mpjpe_mm"]) for r in rows
                     if r["provided3d_mpjpe_mm"] not in ("", None)])
    other = np.array([float(r["provided3d_other_hand_mpjpe_mm"]) for r in rows
                      if r["provided3d_other_hand_mpjpe_mm"] not in ("", None)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.logspace(-2, 3.2, 70)
    axes[0].hist(np.clip(same, 1e-2, None), bins=bins, color="#22c55e", alpha=0.85,
                 label=f"same hand (median {np.median(same):.2f} mm)")
    axes[0].hist(np.clip(other, 1e-2, None), bins=bins, color="#ef4444", alpha=0.6,
                 label=f"other hand (median {np.median(other):.1f} mm)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MPJPE between our reconstruction and the dataset 3D [mm]", fontsize=9)
    axes[0].set_ylabel("observations", fontsize=9)
    axes[0].set_title("Independent reconstruction vs dataset-provided 3D", fontsize=10)
    axes[0].legend(fontsize=8)

    ok = np.isfinite(same)
    axes[1].hist(np.clip(same[ok], 1e-2, 50), bins=np.linspace(0, 50, 60), color="#3b82f6")
    axes[1].axvline(float(np.median(same)), color="#b91c1c", ls="--",
                    label=f"median {np.median(same):.2f} mm")
    axes[1].set_xlabel("same-hand MPJPE [mm] (linear, clipped at 50)", fontsize=9)
    axes[1].set_ylabel("observations", fontsize=9)
    axes[1].set_title("Agreement where both are defined", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.suptitle("CAM-EXP-001.3 - 3D consistency", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "3d_consistency_plot.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "3d_consistency_plot.png"))


def identity_heatmap(rows) -> None:
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["sequence"], r["camera"])
        per[key][1] += 1
        if r["case"].startswith(verdicts.CASE_SWAP):
            per[key][0] += 1
    seqs = sorted({k[0] for k in per})
    cams = sorted({k[1] for k in per})
    M = np.full((len(seqs), len(cams)), np.nan)
    for (s, c), (sw, tot) in per.items():
        M[seqs.index(s), cams.index(c)] = sw / tot if tot else np.nan
    fig, ax = plt.subplots(figsize=(max(12, len(cams) * 0.32), 3.4))
    im = ax.imshow(M, aspect="auto", cmap="inferno", vmin=0, vmax=max(0.05, np.nanmax(M)))
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(seqs, fontsize=7)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([c.replace("brics-odroid-", "") for c in cams], rotation=90, fontsize=6)
    ax.set_title("Confirmed 2D hand-identity swap rate per sequence x camera", fontsize=10)
    fig.colorbar(im, ax=ax, label="share of observations")
    fig.tight_layout()
    fig.savefig(FIG / "per_camera_identity_error_heatmap.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "per_camera_identity_error_heatmap.png"))


def qc_status_bar(rows, ax) -> None:
    """Status counts come from the manifest, which is where PASS_SINGLE_HAND is
    assigned (a strict pass whose partner hand is not clean)."""
    man = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
    src = read_csv(man) if man.exists() else rows
    c = Counter(r["qc_status"] for r in src)
    order = [verdicts.PASS_STRICT, verdicts.PASS_SINGLE_HAND, verdicts.REVIEW,
             verdicts.EXCLUDE]
    vals = [c.get(k, 0) for k in order]
    colors = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444"]
    bars = ax.bar(range(len(order)), vals, color=colors)
    tot = max(sum(vals), 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v}\n{v / tot:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax.set_ylabel("observations", fontsize=9)
    ax.set_title(f"Final QC status over {tot} observations", fontsize=10)
    ax.margins(y=0.18)


def main(thr: dict) -> None:
    rows = read_csv(RUN_DIR / "results" / "raw" / "full_demo_qc.csv.gz")
    take_by_name = {t.name: t for t in TAKES}
    explainer_diagram()
    consistency_plot(rows)
    identity_heatmap(rows)

    def cap(r, tag):
        return (f"{tag}\nL {r['heldout_same_hand_error_px'] or '-'}px "
                f"3D {r['provided3d_mpjpe_mm'] or '-'}mm")

    good = pick(rows, lambda r: r["case"] == verdicts.CASE_GOOD, 9, True, take_by_name)
    swap = pick(rows, lambda r: r["case"] == verdicts.CASE_SWAP, 9, True, take_by_name)
    bad2d = pick(rows, lambda r: r["case"] == verdicts.CASE_BAD2D, 9, True, take_by_name)
    susp = pick(rows, lambda r: r["case"] == verdicts.CASE_PROVIDED3D, 6, True, take_by_name)
    insuf = pick(rows, lambda r: r["case"] == verdicts.CASE_INSUFFICIENT, 6, True,
                 take_by_name)

    grid([(take_by_name[r["sequence"]], r["camera"], int(r["frame"]),
           f"GOOD {r['hand']} same={r['heldout_same_hand_error_px']}px "
           f"other={r['heldout_other_hand_error_px']}px") for r in good],
         FIG / "triangulation_examples" / "good_control_grid.png",
         "Multi-view confirmed: annotation agrees with the independent reconstruction "
         "(green pairs with red, blue with orange)", thr)

    grid([(take_by_name[r["sequence"]], r["camera"], int(r["frame"]),
           f"SWAP {r['hand']} same={r['heldout_same_hand_error_px']}px "
           f"other={r['heldout_other_hand_error_px']}px") for r in swap],
         FIG / "confirmed_swaps" / "confirmed_2d_swap_grid.png",
         "Confirmed per-camera 2D hand-identity swap: the held-out annotation matches the "
         "OTHER hand's independent reconstruction (green pairs with orange, blue with red)",
         thr)

    grid([(take_by_name[r["sequence"]], r["camera"], int(r["frame"]),
           f"BAD2D {r['hand']} same={r['heldout_same_hand_error_px']}px "
           f"other={r['heldout_other_hand_error_px']}px") for r in bad2d + insuf],
         FIG / "unresolved" / "unresolved_grid.png",
         "Unresolved: the held-out annotation matches neither reconstruction, or the "
         "geometry is insufficient to decide", thr)

    if susp:
        grid([(take_by_name[r["sequence"]], r["camera"], int(r["frame"]),
               f"3D-SUSPECT {r['hand']} same={r['provided3d_mpjpe_mm']}mm "
               f"other={r['provided3d_other_hand_mpjpe_mm']}mm") for r in susp],
             FIG / "unresolved" / "provided3d_suspect_grid.png",
             "Dataset-3D identity suspect: our reconstruction matches the OTHER hand's "
             "dataset 3D better than its own", thr)

    qc_ex = []
    for st, tag in ((verdicts.PASS_STRICT, "PASS_STRICT"),
                    (verdicts.REVIEW, "REVIEW"), (verdicts.EXCLUDE, "EXCLUDE")):
        qc_ex += [(take_by_name[r["sequence"]], r["camera"], int(r["frame"]),
                   f"{tag}\n{r['hand']} {r['reason'][:46]}")
                  for r in pick(rows, lambda r, s=st: r["qc_status"] == s, 3, True,
                                take_by_name)]
    grid(qc_ex, FIG / "qc_examples" / "qc_status_grid.png",
         "Representative PASS_STRICT / REVIEW / EXCLUDE observations", thr, ncols=3)

    main_figure(rows, take_by_name, thr, good, swap, bad2d, susp, insuf)


def main_figure(rows, take_by_name, thr, good, swap, bad2d, susp, insuf) -> None:
    fig = plt.figure(figsize=(17.5, 19.5))
    gs = fig.add_gridspec(9, 4, height_ratios=[0.13, 1] * 4 + [1.1], hspace=0.34,
                          wspace=0.06)
    third = susp if susp else []
    third_label = ("ROW 3  Dataset-3D identity suspect - our reconstruction matches the "
                   "OTHER hand's dataset 3D"
                   if susp else
                   "ROW 3  Dataset-3D identity suspect - NONE FOUND; the dataset 3D "
                   "matched our independent reconstruction everywhere it was testable")
    bands = [
        ("ROW 1  Normal control - held-out annotation agrees with the independent "
         "reconstruction (green with red, blue with orange)", good[:4]),
        ("ROW 2  Confirmed per-camera 2D hand-identity swap - annotation matches the "
         "OTHER hand's reconstruction (green with orange, blue with red)", swap[:4]),
        (third_label, third[:4]),
        ("ROW 4  Unresolved - annotation matches neither reconstruction, or geometry "
         "insufficient", (bad2d + insuf)[:4]),
    ]
    for r, (label, items) in enumerate(bands):
        hax = fig.add_subplot(gs[2 * r, :])
        hax.axis("off")
        hax.text(0.0, 0.3, label, fontsize=10.5, fontweight="bold", va="center",
                 color="#1e293b")
        for c in range(4):
            ax = fig.add_subplot(gs[2 * r + 1, c])
            ax.axis("off")
            if c < len(items):
                ax.axis("on")
                rr = items[c]
                panel(ax, take_by_name[rr["sequence"]], rr["camera"], int(rr["frame"]),
                      thr, f"{rr['hand']} same={rr['heldout_same_hand_error_px']}px "
                           f"other={rr['heldout_other_hand_error_px']}px")
            elif c == 0 and not items:
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "no cases found", ha="center", va="center",
                        fontsize=11, color="#64748b")
    ax = fig.add_subplot(gs[8, :2])
    qc_status_bar(rows, ax)
    ax.tick_params(axis="x", labelsize=7)
    ax2 = fig.add_subplot(gs[8, 2:])
    c = Counter(r["case"] for r in rows)
    keys = [k for k, _ in c.most_common()]
    vals = [c[k] for k in keys]
    ax2.barh(range(len(keys)), vals, color="#64748b")
    ax2.set_yticks(range(len(keys)))
    ax2.set_yticklabels([k.replace("_", " ")[:44] for k in keys], fontsize=7)
    ax2.invert_yaxis()
    tot = max(sum(vals), 1)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v} ({v / tot:.1%})", va="center", fontsize=7)
    ax2.set_title("Adjudicated case distribution", fontsize=10)
    ax2.margins(x=0.22)

    fig.suptitle(
        "CAM-EXP-001.3 - GigaHands independent multi-view triangulation and QC\n"
        "Every red/orange skeleton is reconstructed from the OTHER cameras only and "
        "projected into the held-out camera; the dataset's 3D is used only afterwards, "
        "to check the reconstruction.\n" + COLOR_LEGEND, fontsize=11)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.subplots_adjust(top=0.925, bottom=0.035, left=0.015, right=0.985)
    out = FIG / "CAM_EXP_001_3_MAIN_EXPLANATION.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("  wrote", rel(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t = json.load(open(RUN_DIR / "results" / "summary" / "_thresholds.json"))["thresholds"]
    main(t)
