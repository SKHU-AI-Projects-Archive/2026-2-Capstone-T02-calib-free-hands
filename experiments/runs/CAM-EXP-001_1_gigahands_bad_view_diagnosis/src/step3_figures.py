"""Step 3 - build every image a human needs to judge the bad-view diagnosis.

Case panels are laid out with matplotlib so captions and metadata travel with
the pixels; the underlying frames come from the GigaHands rgb_vid mp4s.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bad_view_common import RUN_DIR, VIEW_KEYS, rel, similarity_residual
from step2_hypotheses import base_mask, load_targets, read_rows
from experiments.src.visualization.overlay import EDGES_21, load_frame

log = logging.getLogger("cam-exp-001.1")
FIG = RUN_DIR / "figures"
GT_C, PR_C = "#22c55e", "#ef4444"      # green = annotation, red = projection


def _frame_rgb(rec, frame_idx):
    im = load_frame(rec["video"], int(frame_idx))
    if im is None:
        return None
    return im[:, :, ::-1]               # BGR -> RGB


def _draw_pts(ax, pts, mask, color, edges=True, ms=3.2):
    P = np.asarray(pts, float)
    if edges and len(P) >= 21:
        for a, b in EDGES_21:
            if mask[a] and mask[b] and np.isfinite(P[[a, b]]).all():
                ax.plot(P[[a, b], 0], P[[a, b], 1], "-", color=color, lw=1.0, alpha=0.85)
    ok = mask & np.isfinite(P[:, 0]) & np.isfinite(P[:, 1])
    ax.plot(P[ok, 0], P[ok, 1], "o", color=color, ms=ms, mew=0)


def _axis_img(ax, im, title):
    ax.imshow(im)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def case_figure(rec, row, out_path: Path, case_label: str) -> bool:
    """Six-panel view: raw, GT, projection, both, zoom, metadata."""
    im = _frame_rgb(rec, row["frame"])
    if im is None:
        return False
    H, W = im.shape[:2]
    m = base_mask(rec)
    uv, _ = rec["cam"].project(rec["X"])
    gt = rec["gt"]
    mm = m & np.isfinite(uv[:, 0])
    sentinel = bool(np.all(gt[m] == 0)) if m.sum() else False
    err = (np.linalg.norm(uv - gt, axis=1)[mm] if mm.sum() else np.array([]))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.0))
    _axis_img(axes[0][0], im, "1. original RGB")
    _axis_img(axes[0][1], im, "2. GT 2D annotation (green)")
    _draw_pts(axes[0][1], gt, m, GT_C)
    if sentinel:
        axes[0][1].text(0.5, 0.5, "GT 2D is (0,0) for every joint\n= hand not detected",
                        transform=axes[0][1].transAxes, ha="center", va="center",
                        fontsize=10, color="white",
                        bbox=dict(facecolor="#b91c1c", alpha=0.85, pad=6))
    _axis_img(axes[0][2], im, "3. projected GT 3D (red)")
    _draw_pts(axes[0][2], uv, mm, PR_C)
    _axis_img(axes[1][0], im, "4. both overlaid (+ residual links)")
    _draw_pts(axes[1][0], uv, mm, PR_C)
    _draw_pts(axes[1][0], gt, m, GT_C)
    for j in np.where(mm & m)[0]:
        if np.isfinite(gt[j]).all() and np.isfinite(uv[j]).all():
            axes[1][0].plot([gt[j, 0], uv[j, 0]], [gt[j, 1], uv[j, 1]],
                            "-", color="#facc15", lw=0.7, alpha=0.9)

    # Zoom on the projected hand, which always exists even when GT is a sentinel.
    ax = axes[1][1]
    _axis_img(ax, im, "5. zoom on projected hand")
    _draw_pts(ax, uv, mm, PR_C, ms=4.5)
    _draw_pts(ax, gt, m, GT_C, ms=4.5)
    if mm.sum():
        c = uv[mm].mean(0)
        pad = max(90.0, float(np.ptp(uv[mm], axis=0).max()) * 0.9)
        ax.set_xlim(max(0, c[0] - pad), min(W, c[0] + pad))
        ax.set_ylim(min(H, c[1] + pad), max(0, c[1] - pad))

    ax = axes[1][2]; ax.axis("off")
    resid_sim, resid_tr = (similarity_residual(uv[mm], gt[mm]) if mm.sum() >= 3
                           else (float("nan"), float("nan")))
    lines = [
        f"{case_label}",
        f"class      : {row['view_class']}",
        f"sequence   : {row['sequence']}",
        f"camera     : {row['camera']}",
        f"frame      : {row['frame']}    hand: {row['hand']}",
        "",
        f"view median err : {row['err_median_px'] or 'n/a'} px",
        f"view mean   err : {row['err_mean_px'] or 'n/a'} px",
        f"view p90    err : {row['err_p90_px'] or 'n/a'} px",
        f"frame median err: {np.median(err):.2f} px" if err.size else "frame median err: n/a",
        f"joints compared : {int(mm.sum())} / {len(uv)}",
        f"zero-sentinel   : {'YES (no detection)' if sentinel else 'no'}",
        "",
        f"resid after similarity : {resid_sim:.1f} px" if np.isfinite(resid_sim) else "",
        f"resid after translation: {resid_tr:.1f} px" if np.isfinite(resid_tr) else "",
        "",
        "green = dataset GT 2D",
        "red   = our projection of GT 3D",
        "yellow= per-joint residual",
    ]
    ax.text(0.0, 1.0, "\n".join(x for x in lines if x != ""), va="top", ha="left",
            family="monospace", fontsize=8.5, transform=ax.transAxes)
    ax.set_title("6. metadata", fontsize=8)

    fig.suptitle(f"CAM-EXP-001.1 {case_label} - {row['sequence']} {row['camera']} "
                 f"frame {row['frame']} ({row['hand']} hand)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=85)
    plt.close(fig)
    return True


def thumb(ax, rec, row, caption: str) -> None:
    im = _frame_rgb(rec, row["frame"])
    if im is None:
        ax.axis("off"); return
    m = base_mask(rec)
    uv, _ = rec["cam"].project(rec["X"])
    mm = m & np.isfinite(uv[:, 0])
    ax.imshow(im)
    _draw_pts(ax, uv, mm, PR_C, ms=2.4)
    _draw_pts(ax, gt := rec["gt"], m, GT_C, ms=2.4)
    if mm.sum():
        c = np.vstack([uv[mm], gt[m][np.isfinite(gt[m][:, 0])] if m.sum() else uv[mm]])
        lo, hi = c.min(0), c.max(0)
        pad = 60
        ax.set_xlim(max(0, lo[0] - pad), min(im.shape[1], hi[0] + pad))
        ax.set_ylim(min(im.shape[0], hi[1] + pad), max(0, lo[1] - pad))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(caption, fontsize=7)


def grid(recs_rows, captions, out: Path, title: str, ncols: int = 3) -> None:
    n = len(recs_rows)
    if not n:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (rec, row), cap in zip(axes.ravel(), recs_rows, captions):
        ax.axis("on")
        thumb(ax, rec, row, cap)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print("  wrote", rel(out))


def classify_failure_mode(rec, row) -> str:
    m = base_mask(rec)
    if m.sum() == 0:
        return "no_valid_joints"
    if np.all(rec["gt"][m] == 0):
        return "A_no_detection_sentinel"
    uv, _ = rec["cam"].project(rec["X"])
    mm = m & np.isfinite(uv[:, 0])
    if mm.sum() < 5:
        return "E_behind_camera"
    e = np.linalg.norm(uv - rec["gt"], axis=1)[mm]
    frac = float((e > 50).mean())
    if frac < 0.1:
        return "ok"
    resid_sim, resid_tr = similarity_residual(uv[mm], rec["gt"][mm])
    if frac < 0.9:
        return "D_partial_joint_corruption"
    if np.isfinite(resid_tr) and resid_tr < 20:
        return "C_whole_view_translation"
    if np.isfinite(resid_sim) and resid_sim < 20:
        return "B_shape_preserved_wrong_placement"
    return "F_structural_mismatch"


def main() -> dict:
    rows = read_rows(RUN_DIR / "results" / "raw" / "per_view_error_summary.csv")
    cases = read_rows(RUN_DIR / "results" / "raw" / "representative_cases.csv")
    by_key = {tuple(r[k] for k in VIEW_KEYS): r for r in rows}

    # --- pure-table figures -------------------------------------------------
    _error_rank_bar(rows)
    _lag_plot()
    _distortion_plot()

    # --- image figures ------------------------------------------------------
    need = [by_key[tuple(c[k] for k in VIEW_KEYS)] for c in cases
            if tuple(c[k] for k in VIEW_KEYS) in by_key]
    store = load_targets(need)
    log.info("loaded %d case views for figures", len(store))

    made = []
    counters = defaultdict(int)
    for c in cases:
        key = tuple(c[k] for k in VIEW_KEYS)
        rec, row = store.get(key), by_key.get(key)
        if rec is None or row is None:
            continue
        role = c["role"]
        prefix = {"good": "good_case", "bad": "bad_case", "pair_good": "pair_good",
                  "pair_bad": "pair_bad", "zero_sentinel": "sentinel_case"}.get(role, role)
        counters[prefix] += 1
        name = f"{prefix}_{counters[prefix]:02d}.png"
        if case_figure(rec, row, FIG / "cases" / name, f"{role} #{counters[prefix]}"):
            made.append({**c, "figure": rel(FIG / "cases" / name),
                         "failure_mode": classify_failure_mode(rec, row)})
    print(f"  wrote {len(made)} case panels -> {rel(FIG / 'cases')}")

    def pick(role):
        return [(store[tuple(c[k] for k in VIEW_KEYS)], by_key[tuple(c[k] for k in VIEW_KEYS)])
                for c in cases if c["role"] == role
                and tuple(c[k] for k in VIEW_KEYS) in store]

    def caps(role):
        out = []
        for c in cases:
            if c["role"] != role:
                continue
            k = tuple(c[k2] for k2 in VIEW_KEYS)
            if k not in store:
                continue
            out.append(f"{c['sequence'].split('/')[0]} {c['camera'].replace('brics-odroid-','')}\n"
                       f"f{c['frame']} {c['hand']} - median "
                       f"{float(c['err_median_px']):.1f} px" if c["err_median_px"]
                       else f"{c['camera']} f{c['frame']} {c['hand']} - sentinel")
        return out

    grid(pick("good"), caps("good"), FIG / "good_view_examples_grid.png",
         "Good views - projection (red) lands on the annotation (green)")
    grid(pick("bad"), caps("bad"), FIG / "bad_view_examples_grid.png",
         "Bad views - non-sentinel failures with the largest per-view median error")

    # paired good/bad, same sequence+frame+hand, good above bad
    pairs = defaultdict(dict)
    for c in cases:
        if c["role"] in ("pair_good", "pair_bad") and c["pair_id"]:
            pairs[c["pair_id"]][c["role"]] = c
    ordered, pcaps = [], []
    for pid in sorted(pairs):
        p = pairs[pid]
        if "pair_good" not in p or "pair_bad" not in p:
            continue
        for role in ("pair_good", "pair_bad"):
            c = p[role]
            k = tuple(c[k2] for k2 in VIEW_KEYS)
            if k not in store:
                continue
            ordered.append((store[k], by_key[k]))
            tag = "GOOD" if role == "pair_good" else "BAD "
            em = (f"{float(c['err_median_px']):.1f} px" if c["err_median_px"]
                  else "sentinel (0,0)")
            pcaps.append(f"{pid} {tag} {c['camera'].replace('brics-odroid-','')}\n"
                         f"{c['sequence'].split('/')[0]} f{c['frame']} {c['hand']} - {em}")
    grid(ordered, pcaps, FIG / "paired_good_bad_comparison_grid.png",
         "Same sequence / frame / hand, different camera: good (left of each pair) vs bad",
         ncols=2)

    _failure_gallery(rows, by_key)
    return {"cases": made}


def _error_rank_bar(rows) -> None:
    real = sorted([float(r["err_median_px"]) for r in rows
                   if not int(r["is_zero_sentinel_view"]) and r["err_median_px"] != ""])
    n_sent = sum(int(r["is_zero_sentinel_view"]) for r in rows)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(real))
    a = np.array(real)
    colors = np.where(a <= 10, "#22c55e", np.where(a >= 50, "#ef4444", "#f59e0b"))
    ax.bar(x, a, width=1.0, color=colors, linewidth=0)
    ax.set_yscale("log")
    ax.axhline(10, color="#16a34a", ls="--", lw=1, label="good threshold 10 px")
    ax.axhline(50, color="#b91c1c", ls="--", lw=1, label="bad threshold 50 px")
    ax.set_xlabel(f"GigaHands views sorted by median reprojection error "
                  f"(n={len(real)} annotated; {n_sent} further views are zero-sentinel "
                  f"and carry no annotation)", fontsize=8)
    ax.set_ylabel("per-view median error [px]", fontsize=9)
    ax.set_title("CAM-EXP-001.1 per-view reprojection error, ranked", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "error_rank_bar.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "error_rank_bar.png"))


def _lag_plot() -> None:
    p = RUN_DIR / "results" / "raw" / "lag_sweep_results.csv"
    if not p.exists():
        return
    byc = defaultdict(lambda: defaultdict(list))
    for r in read_rows(p):
        if r["median_err_px"]:
            byc[r["view_class"]][int(r["lag"])].append(float(r["median_err_px"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for cls, color in (("good", "#22c55e"), ("bad", "#ef4444")):
        lags = sorted(byc[cls])
        if not lags:
            continue
        med = [np.median(byc[cls][l]) for l in lags]
        ax.plot(lags, med, "o-", color=color, label=f"{cls} views")
    ax.set_xlabel("3D frame index offset applied to the 2D annotation (lag)", fontsize=9)
    ax.set_ylabel("median reprojection error [px]", fontsize=9)
    ax.set_yscale("log")
    ax.axvline(0, color="#64748b", ls=":", lw=1)
    ax.set_title("H2 lag sweep - no lag repairs the bad views", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "lag_sweep_plot.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "lag_sweep_plot.png"))


def _distortion_plot() -> None:
    p = RUN_DIR / "results" / "raw" / "distortion_compare.csv"
    if not p.exists():
        return
    rows = read_rows(p)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, cls in zip(axes, ("good", "bad")):
        sel = [r for r in rows if r["view_class"] == cls]
        on = np.array([float(r["median_err_distortion_on_px"]) for r in sel])
        off = np.array([float(r["median_err_distortion_off_px"]) for r in sel])
        if not on.size:
            continue
        ax.scatter(on, off, s=10, alpha=0.6,
                   color="#22c55e" if cls == "good" else "#ef4444")
        lim = [max(1e-2, min(on.min(), off.min())), max(on.max(), off.max()) * 1.2]
        ax.plot(lim, lim, "k--", lw=1, label="no change")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("distortion applied [px]", fontsize=9)
        ax.set_ylabel("distortion disabled [px]", fontsize=9)
        ax.set_title(f"{cls} views (n={len(sel)})\n"
                     f"median {np.median(on):.2f} -> {np.median(off):.2f} px when disabled",
                     fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("H4 distortion on/off - points above the line mean distortion helps",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "distortion_effect_plot.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "distortion_effect_plot.png"))


MODE_CAPTION = {
    "A_no_detection_sentinel": "A. no detection\nGT 2D = (0,0), conf 1.0",
    "B_shape_preserved_wrong_placement": "B. shape kept, placement wrong\n(similarity fits)",
    "C_whole_view_translation": "C. whole-view shift\n(pure translation fits)",
    "D_partial_joint_corruption": "D. partial joint corruption\n(only some joints off)",
    "E_behind_camera": "E. too few projectable joints",
    "F_structural_mismatch": "F. structural mismatch\n(different shape entirely)",
}


def _failure_gallery(rows, by_key) -> None:
    """One or two examples per observed failure mode."""
    cand = [r for r in rows if r["view_class"] in ("bad", "zero_sentinel")]
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(cand))[:400]
    store = load_targets([cand[i] for i in idx])
    buckets = defaultdict(list)
    for i in idx:
        r = cand[i]
        key = tuple(r[k] for k in VIEW_KEYS)
        rec = store.get(key)
        if rec is None:
            continue
        mode = classify_failure_mode(rec, r)
        if mode in ("ok", "no_valid_joints"):
            continue
        if len(buckets[mode]) < 2:
            buckets[mode].append((rec, r))
    items, caps_ = [], []
    counts = {}
    for mode in sorted(buckets):
        counts[mode] = len(buckets[mode])
        for rec, r in buckets[mode]:
            items.append((rec, r))
            em = (f"{float(r['err_median_px']):.0f} px" if r["err_median_px"] else "no annot.")
            caps_.append(f"{MODE_CAPTION.get(mode, mode)}\n{r['camera'].replace('brics-odroid-','')} "
                         f"f{r['frame']} {r['hand']} - {em}")
    grid(items, caps_, FIG / "failure_mode_gallery.png",
         "Failure-mode gallery - bad and sentinel views grouped by observed pattern",
         ncols=4)
    # record the mode census alongside the picture
    census = defaultdict(int)
    for i in idx:
        r = cand[i]
        rec = store.get(tuple(r[k] for k in VIEW_KEYS))
        if rec is not None:
            census[classify_failure_mode(rec, r)] += 1
    total = sum(census.values())
    out = [{"failure_mode": k, "n": v, "share": round(v / total, 4)}
           for k, v in sorted(census.items(), key=lambda kv: -kv[1])]
    from bad_view_common import write_csv
    write_csv(RUN_DIR / "results" / "summary" / "failure_mode_census.csv", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
