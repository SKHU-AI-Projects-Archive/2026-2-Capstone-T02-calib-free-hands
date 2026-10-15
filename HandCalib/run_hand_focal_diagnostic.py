"""Experiment 14 diagnostics (not pre-registered; results are diagnostics, never f values).

    python HandCalib/run_hand_focal_diagnostic.py --config HandCalib/configs/14_hand_template_focal.yaml \
        --official runs/14_hand_template_focal/<run>/IMG_7141 runs/14_hand_template_focal/<run>/IMG_7142

Reuses each official run's detection cache (detections.npz) and frame list; the videos are not read. Writes a new folder
runs/14_hand_template_focal/<UTC>_diagnostic/. The official results, verdicts and yaml thresholds are not touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hand_focal import diagnostic as D                     # noqa: E402
from hand_focal import estimate as E                       # noqa: E402
from hand_focal import provenance as PV                    # noqa: E402
from hand_focal import report as RP                        # noqa: E402
from hand_focal.paths import resolve                       # noqa: E402
from hand_focal.track import box_long                      # noqa: E402
from run_hand_focal import utc_stamp                       # noqa: E402

SETS = ("palm4", "palm5", "palm_proximal9", "flat17")
N_SEEDS = 5


def analyse_video(run_dir: Path, cfg: dict) -> dict:
    t0 = time.time()
    L0 = D.load_official(run_dir, cfg)
    a, tr, W, H = L0["a"], L0["tr"], L0["W"], L0["H"]
    L = float(max(W, H))
    J = cfg["joints"]
    est_ids = list(J["sets"][J["estimate_set"]])
    kp, mask = tr["kp"][:, est_ids], tr["mask"][:, est_ids]
    T = np.array(a["template"])
    theta = np.array([np.nan if v is None else v for v in a["theta2d_deg"]], float)
    role = np.array(a["role"])
    used = [i for i in range(len(role)) if role[i] in ("front", "tilt")]
    front = list(a["template_frames"])
    cx, cy = a["principal_point"]
    bl = box_long(tr["box"])
    huber = cfg["estimate"]["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    edge = D.edge_flags(tr["box"], tr["present"], W, H)
    g_off = E.f_grid(cfg["estimate"], L)
    hi = g_off[-1]
    grid = np.r_[g_off, np.geomspace(hi, 8.0 * L, 31)[1:]]          # official grid + extension up to 8 x long side (diagnostic only)
    off_sel = np.arange(len(g_off))

    # ---- cost matrices per joint set
    Cs, Rs = {}, {}
    for s in SETS:
        ids = list(J["sets"][s])
        pos = [est_ids.index(j) for j in ids]
        C, R = D.frame_cost_matrix(T, tr["kp"], tr["mask"], used, pos, ids, grid, cx, cy, huber)
        Cs[s], Rs[s] = C, R
    th_u, edge_u = theta[used], edge[used]

    # ---- D1 subsets
    subsets = {"all": np.ones(len(used), bool)}
    for t in D.THETA_SUBSETS:
        subsets[f"theta<={t:g}"] = np.isfinite(th_u) & (th_u <= t)
    subsets["no_edge"] = ~edge_u
    d1 = {}
    for sn, sel in subsets.items():
        d1[sn] = {}
        for s in SETS:
            C = Cs[s][sel]
            ok = np.isfinite(C).all(1)
            prof = np.nanmean(C[ok], 0) if ok.sum() else np.full(len(grid), np.nan)
            st = D.profile_stats(grid, prof, hi) if ok.sum() >= 3 else {"official": None, "extended": None}
            d1[sn][s] = {"n_frames": int(ok.sum()), "n_tilt": int((ok & (role[used] == "tilt")[sel]).sum()), "profile": prof.tolist(), **st}

    # ---- D2 per-frame slopes (flat17, official grid)
    C = Cs["flat17"][:, off_sel]
    lg = np.log(g_off)
    kref = int(np.argmin(np.abs(grid - L)))
    slope = np.full(len(used), np.nan)
    rel = np.full(len(used), np.nan)
    fmin_at = np.array([None] * len(used), dtype=object)
    for r in range(len(used)):
        if np.isfinite(C[r]).all():
            slope[r] = np.polyfit(lg, C[r], 1)[0]
            rel[r] = slope[r] / C[r].mean()
            k = int(np.argmin(C[r]))
            fmin_at[r] = "lower" if k == 0 else ("upper" if k == len(g_off) - 1 else "interior")
    rms_ref = Rs["flat17"][:, kref] / np.where(bl[used] > 0, bl[used], np.nan)
    ok = np.isfinite(rel)
    tilt_u = role[used] == "tilt"

    def corr(x, y, m):
        m = m & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            return None
        r_ = spearmanr(x[m], y[m])
        return {"rho": float(r_.statistic), "p": float(r_.pvalue), "n": int(m.sum())}

    bins = D.THETA_BINS
    by_bin = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = ok & (th_u >= b0) & (th_u < b1)
        by_bin.append({"theta": [b0, b1], "n": int(m.sum()), "neg_frac": float((rel[m] < 0).mean()) if m.sum() else None,
                       "rel_slope_median": float(np.median(rel[m])) if m.sum() else None, "slope_sum": float(np.nansum(slope[m]))})
    d2 = {"n": int(ok.sum()), "neg_frac_all": float((rel[ok] < 0).mean()), "rel_slope_median": float(np.median(rel[ok])),
          "frame_min_at": {k: int(sum(1 for x in fmin_at[ok] if x == k)) for k in ("lower", "interior", "upper")},
          "edge": {"n_edge": int((ok & edge_u).sum()), "neg_frac_edge": float((rel[ok & edge_u] < 0).mean()) if (ok & edge_u).any() else None,
                   "neg_frac_no_edge": float((rel[ok & ~edge_u] < 0).mean()) if (ok & ~edge_u).any() else None,
                   "rel_slope_median_edge": float(np.median(rel[ok & edge_u])) if (ok & edge_u).any() else None,
                   "rel_slope_median_no_edge": float(np.median(rel[ok & ~edge_u])) if (ok & ~edge_u).any() else None},
          "spearman_rel_slope_vs_theta": corr(rel, th_u, ok), "spearman_rel_slope_vs_rms": corr(rel, rms_ref, ok),
          "spearman_rms_vs_theta": corr(rms_ref, th_u, ok), "by_theta_bin": by_bin, "slope_sum_total": float(np.nansum(slope[ok])),
          "frames": [{"index": int(used[r]), "theta2d": None if not np.isfinite(th_u[r]) else float(th_u[r]), "edge": bool(edge_u[r]),
                      "slope": None if not np.isfinite(slope[r]) else float(slope[r]), "rel_slope": None if not np.isfinite(rel[r]) else float(rel[r]),
                      "rms_rel_at_ref": None if not np.isfinite(rms_ref[r]) else float(rms_ref[r]), "own_min": fmin_at[r]} for r in range(len(used))]}

    # ---- D3 same-pose replay (flat17)
    sigma_xy = a["front_jitter_px"] / np.sqrt(2.0)
    real_prof = np.array(d1["all"]["flat17"]["profile"])
    d3 = {"sigma_xy_px": sigma_xy, "seeds": N_SEEDS, "real": D.profile_stats(grid, real_prof, hi), "assumed": {}}
    for m_ in D.ASSUMED:
        fa = m_ * L
        runs = []
        for s in range(N_SEEDS):
            c = D.same_pose_replay(T, kp, mask, used, front, grid, cx, cy, huber, fa, sigma_xy, 1000 + s, int(J["template_align_iters"]))
            st = D.profile_stats(grid, c, hi)
            runs.append({"profile": c.tolist(), **st, "min_over_assumed": st["official"]["min_at_px"] / fa if st["official"] else None})
        d3["assumed"][f"{m_:g}"] = {"f_assumed_px": fa, "runs": runs,
                                    "n_upper_edge": sum(1 for r in runs if r["official"]["at"] == "upper"),
                                    "n_interior_within10": sum(1 for r in runs if r["official"]["interior"] and abs(r["min_over_assumed"] - 1) <= 0.10),
                                    "min_over_assumed_median": float(np.median([r["min_over_assumed"] for r in runs]))}
    real_edge = d3["real"]["official"]["at"] == "upper"
    all_upper = all(v["n_upper_edge"] == N_SEEDS for v in d3["assumed"].values())
    all_ok = all(v["n_interior_within10"] == N_SEEDS for v in d3["assumed"].values())
    if all_upper:
        cls = "shooting"
    elif all_ok and real_edge:
        cls = "detector_or_finger"
    else:
        cls = "undetermined"
    d3["classification"] = cls

    # ---- D4 per-joint residuals by theta bin (flat17 at f = 1.0 x L)
    K = E.K_of(L, cx, cy)
    acc = {b: [[] for _ in est_ids] for b in range(len(bins) - 1)}
    nb = np.zeros(len(bins) - 1, int)
    for r, i in enumerate(used):
        v = mask[i]
        if v.sum() < 4 or not np.isfinite(theta[i]):
            continue
        res, _, _ = E.frame_fit(T, kp[i], v, K)
        b = int(np.searchsorted(bins, theta[i], side="right") - 1)
        b = min(max(b, 0), len(bins) - 2)
        nb[b] += 1
        for pos, j in enumerate(np.flatnonzero(v)):
            acc[b][j].append(res[pos] / bl[i])
    table = np.full((len(est_ids), len(bins) - 1), np.nan)
    for b in acc:
        for j in range(len(est_ids)):
            if acc[b][j]:
                table[j, b] = float(np.sqrt(np.mean(np.square(acc[b][j]))))
    ratio = table / table[:, [0]]
    first = []
    for b in range(1, len(bins) - 1):
        col = ratio[:, b]
        if np.isfinite(col).any():
            top = np.argsort(-np.nan_to_num(col, nan=-1))[:3]
            first.append({"theta": [bins[b], bins[b + 1]], "top_joints": [int(est_ids[j]) for j in top], "ratio": [float(col[j]) for j in top]})
    d4 = {"f_ref_px": L, "bins": list(bins), "n_frames_per_bin": nb.tolist(), "joint_ids": est_ids, "rms_rel": table.tolist(), "ratio_to_first_bin": ratio.tolist(),
          "top_rising": first}
    return {"run_dir": str(run_dir), "video": L0["res"]["video"], "size": [W, H], "grid": grid.tolist(), "official_hi_px": float(hi),
            "official_f_check": a["solutions"]["fixed_template"]["f"], "n_used": len(used), "front_jitter_px": a["front_jitter_px"],
            "d1": d1, "d2": d2, "d3": d3, "d4": d4, "time_s": time.time() - t0, "_arrays": {"th": th_u, "edge": edge_u, "rel": rel, "rms": rms_ref}}


def figures(out: Path, name: str, v: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RP._font()
    made = []
    grid = np.array(v["grid"])
    hi = v["official_hi_px"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, s in zip(axs, SETS):
        for sn, d in v["d1"].items():
            p = np.array(d[s]["profile"], float)
            if np.isfinite(p).any():
                ax.plot(grid, p / np.nanmin(p), lw=1.3, label=f"{sn} ({d[s]['n_frames']})")
        ax.axvline(hi, color="#444", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(s)
        ax.set_xlabel("f 후보 (px)")
        ax.set_ylim(0.95, 3)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("비용 / 최소")
    axs[0].legend(fontsize=6)
    made.append(_save(fig, out / f"{name}_d1_profiles.png"))

    A_ = v["_arrays"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for e, col, lab in ((False, "#2a6fbb", "가장자리 아님"), (True, "#c47a00", "가장자리 접촉")):
        m = (A_["edge"] == e) & np.isfinite(A_["rel"])
        axs[0].scatter(A_["th"][m], A_["rel"][m], s=12, color=col, label=lab)
        axs[1].scatter(100 * A_["rms"][m], A_["rel"][m], s=12, color=col, label=lab)
    for ax in axs:
        ax.axhline(0, color="#444", lw=0.8)
        ax.grid(alpha=0.3)
        ax.set_ylabel("프레임 비용의 log f 기울기 / 평균 비용")
    axs[0].set_xlabel("θ2d (°)")
    axs[1].set_xlabel("프레임 재투영 RMS / 박스 (%) at f = 긴 변")
    axs[0].legend(fontsize=7)
    made.append(_save(fig, out / f"{name}_d2_frame_slopes.png"))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    real = np.array(v["d1"]["all"]["flat17"]["profile"])
    ax.plot(grid, real / np.nanmin(real), color="k", lw=2, label="실제 (flat17, 전체)")
    cols = {"0.7": "#2a6fbb", "1": "#3a9a4a", "1.3": "#c47a00"}
    for k, d in v["d3"]["assumed"].items():
        for j, r in enumerate(d["runs"]):
            p = np.array(r["profile"])
            ax.plot(grid, p / np.nanmin(p), color=cols[k], lw=1, alpha=0.6, label=f"합성 재현 가정 {k}×긴 변" if j == 0 else None)
        ax.axvline(d["f_assumed_px"], color=cols[k], ls=":", lw=1)
    ax.axvline(hi, color="#444", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0.95, 3)
    ax.set_xlabel("f 후보 (px)")
    ax.set_ylabel("비용 / 최소")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    made.append(_save(fig, out / f"{name}_d3_replay.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    t = np.array(v["d4"]["rms_rel"], float) * 100
    im = ax.imshow(t, aspect="auto", cmap="viridis")
    b = v["d4"]["bins"]
    ax.set_xticks(range(len(b) - 1))
    ax.set_xticklabels([f"{b[i]}–{b[i + 1]}\n(n={v['d4']['n_frames_per_bin'][i]})" for i in range(len(b) - 1)], fontsize=7)
    ax.set_yticks(range(len(v["d4"]["joint_ids"])))
    ax.set_yticklabels(v["d4"]["joint_ids"], fontsize=7)
    ax.set_xlabel("θ2d 구간 (°)")
    ax.set_ylabel("관절 번호")
    fig.colorbar(im, ax=ax, label="잔차 RMS / 박스 (%)")
    made.append(_save(fig, out / f"{name}_d4_joint_residuals.png"))
    return made


def _save(fig, p):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


CLS_KO = {"shooting": "촬영(자세 분포) 문제", "detector_or_finger": "검출기·손가락 평면성 문제", "undetermined": "판단 불가"}


def summary(vs: dict, cfg) -> list:
    L = ["# 실험 14 진단 (사전 등록 아님)", "", "**진단 결과이며 f 추정값이 아님.** 아래 '최솟값 위치'는 비용 프로파일의 모양을 보기 위한 수치이며 초점거리로 쓰지 않는다. "
         "공식 결과·판정·yaml 문턱값은 바꾸지 않았다.", "",
         "- 입력: 공식 실행의 검출 캐시(detections.npz)와 프레임 목록. 영상은 다시 읽지 않음. 추적·템플릿·θ2d는 공식 코드 그대로 재실행.",
         "- 공식 격자 0.35–1.6×긴 변(60점)에 진단용으로 8×긴 변까지 연장(30점). '내부 최솟값'은 공식 격자 기준.",
         "- 분류 규칙(실행 전 고정): D3 합성 재현이 모든 가정 f에서 상한 → 촬영(자세 분포) / 합성은 모든 가정에서 가정 f ±10% 안의 내부 최솟값인데 실제만 상한 → 검출기·손가락 평면성 / 그 밖 → 판단 불가.", ""]
    for name, v in vs.items():
        d1, d2, d3, d4 = v["d1"], v["d2"], v["d3"], v["d4"]
        L += [f"## {v['video']} ({v['size'][0]}×{v['size'][1]}, 사용 프레임 {v['n_used']}, 공식 격자 상한 {v['official_hi_px']:.0f} px)", "",
              f"**분류: {CLS_KO[d3['classification']]}**", "", "### 진단 1: 부분집합별 비용 프로파일", "",
              "| 프레임 부분집합 | 관절 집합 | 프레임(기울기) | 공식 격자 최솟값 | 위치(px) | 날카로움 | 연장 격자 최솟값 위치(px) |", "|---|---|---|---|---|---|---|"]
        for sn, d in d1.items():
            for s in SETS:
                x = d[s]
                if x["official"] is None:
                    L.append(f"| {sn} | {s} | {x['n_frames']}({x['n_tilt']}) | 프레임 부족 | - | - | - |")
                    continue
                o, e = x["official"], x["extended"]
                at = {"interior": "내부", "upper": "상한", "lower": "하한"}[o["at"]]
                L.append(f"| {sn} | {s} | {x['n_frames']}({x['n_tilt']}) | {at} | {o['min_at_px']:.0f} | {o['sharpness']:.2f} | "
                         f"{e['min_at_px']:.0f}{' (연장 상한)' if e['at'] == 'upper' else ''} |")
        L += ["", "### 진단 2: 프레임별 기여 (flat17, 공식 격자)", "",
              f"- 프레임 {d2['n']}장 중 비용이 f 증가에 따라 감소(기울기 < 0)하는 비율 {100 * d2['neg_frac_all']:.0f}%, 상대 기울기 중앙값 {d2['rel_slope_median']:+.3f}",
              f"- 프레임 단독 최솟값 위치: 하한 {d2['frame_min_at']['lower']}, 내부 {d2['frame_min_at']['interior']}, 상한 {d2['frame_min_at']['upper']}",
              f"- 가장자리 접촉 {d2['edge']['n_edge']}장: 감소 비율 {_pct(d2['edge']['neg_frac_edge'])} (비접촉 {_pct(d2['edge']['neg_frac_no_edge'])}), "
              f"상대 기울기 중앙값 {_f(d2['edge']['rel_slope_median_edge'])} (비접촉 {_f(d2['edge']['rel_slope_median_no_edge'])})",
              f"- Spearman: 기울기–θ2d {_rho(d2['spearman_rel_slope_vs_theta'])}, 기울기–프레임 RMS {_rho(d2['spearman_rel_slope_vs_rms'])}, RMS–θ2d {_rho(d2['spearman_rms_vs_theta'])}",
              "", "| θ2d 구간 | 프레임 | 감소 비율 | 상대 기울기 중앙값 | 기울기 합 |", "|---|---|---|---|---|"]
        for b in d2["by_theta_bin"]:
            L.append(f"| {b['theta'][0]}–{b['theta'][1]}° | {b['n']} | {_pct(b['neg_frac'])} | {_f(b['rel_slope_median'])} | {b['slope_sum']:.1f} |")
        L += ["", "### 진단 3: 같은 자세 합성 재현 (flat17, fixed_template, 잡음 σ = 정면 지터/√2 = "
              f"{d3['sigma_xy_px']:.2f} px/축, 시드 {d3['seeds']}개)", "",
              f"- 실제: 공식 격자 {('내부 ' + format(d3['real']['official']['min_at_px'], '.0f') + ' px') if d3['real']['official']['interior'] else d3['real']['official']['at']}, "
              f"연장 격자 최솟값 위치 {d3['real']['extended']['min_at_px']:.0f} px", "",
              "| 가정 f (×긴 변, px) | 상한에 붙은 시드 | 가정 ±10% 내부 최솟값 시드 | 최솟값 위치/가정 중앙값 | 연장 격자 최솟값 위치(시드별) |", "|---|---|---|---|---|"]
        for k, d in d3["assumed"].items():
            L.append(f"| {k} ({d['f_assumed_px']:.0f}) | {d['n_upper_edge']}/{d3['seeds']} | {d['n_interior_within10']}/{d3['seeds']} | {d['min_over_assumed_median']:.3f} | "
                     + ", ".join(f"{r['extended']['min_at_px']:.0f}" for r in d["runs"]) + " |")
        L += ["", f"### 진단 4: 관절별 잔차 (f 기준 = 긴 변 {d4['f_ref_px']:.0f} px, flat17, 잔차/박스 %)", "",
              "| 관절 | " + " | ".join(f"{d4['bins'][i]}–{d4['bins'][i + 1]}° (n={d4['n_frames_per_bin'][i]})" for i in range(len(d4['bins']) - 1)) + " |",
              "|---|" + "---|" * (len(d4["bins"]) - 1)]
        for j, row in zip(d4["joint_ids"], d4["rms_rel"]):
            L.append(f"| {j} | " + " | ".join("-" if x is None or not np.isfinite(x) else f"{100 * x:.2f}" for x in row) + " |")
        L.append("")
        for f_ in d4["top_rising"]:
            L.append(f"- θ2d {f_['theta'][0]}–{f_['theta'][1]}°: 첫 구간 대비 가장 많이 커진 관절 " +
                     ", ".join(f"{j} (×{r:.1f})" for j, r in zip(f_["top_joints"], f_["ratio"])))
        L += ["", f"그림: {name}_d1_profiles.png, {name}_d2_frame_slopes.png, {name}_d3_replay.png, {name}_d4_joint_residuals.png", ""]
    return L


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def _f(x):
    return "-" if x is None else f"{x:+.3f}"


def _rho(c):
    return "-" if c is None else f"ρ={c['rho']:+.2f} (p={c['p']:.2g}, n={c['n']})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--official", nargs="+", required=True)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    out = resolve(cfg["output"]["root"]) / f"{utc_stamp()}_diagnostic"
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    vs = {}
    for r in a.official:
        rd = resolve(r)
        name = rd.name
        print(f"== {name} ==", flush=True)
        v = analyse_video(rd, cfg)
        v["figures"] = figures(out / "figures", name, v)
        vs[name] = v
        print(f"  분류 {v['d3']['classification']}, {v['time_s']:.0f} s", flush=True)
    lines = summary(vs, cfg)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    RP.write_json(out / "diagnostic.json", {k: {kk: vv for kk, vv in v.items() if kk != "_arrays"} for k, v in vs.items()})
    RP.write_json(out / "metadata.json", PV.build(text, a.config, {"diagnostic": True, "official_runs": a.official}))
    print("\n".join(lines))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
