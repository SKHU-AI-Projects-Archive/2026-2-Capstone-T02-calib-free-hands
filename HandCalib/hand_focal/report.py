"""Writers: figures, per_frame.csv, results.json, result.yaml, report.md (Korean). Points only - no video frames or images of the hand."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

MANDATORY = "이 f는 핀홀·주점 중심·왜곡 없음 가정에서 손 2D 키포인트만으로 얻은 추정값이며 정답 데이터와 비교하지 않았다."
METHOD_KO = {"fixed_template": "fixed_template(정면=평면 가정)", "joint_delta": "joint_delta(정면 기울기 δ 동시 추정)"}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=1))


def fmt(x, d=1, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{100 * x:.{d}f}%" if pct else f"{x:.{d}f}"


def derived(f: float, W: int, H: int, target_sizes) -> dict:
    L = max(W, H)
    fov = lambda s: math.degrees(2 * math.atan(s / (2 * f)))      # noqa: E731
    out = {"f_over_long_side": f / L, "fov_h_deg": fov(W), "fov_v_deg": fov(H), "fov_d_deg": fov(math.hypot(W, H)), "at_sizes": []}
    for tw, th in target_sizes:
        same = (tw >= th) == (W >= H) and abs(tw / th - W / H) < 0.02
        out["at_sizes"].append({"size": [tw, th], "f_px": f * max(tw, th) / L,
                                "note": None if same else "가로세로 비율/방향이 달라 긴 변 기준으로만 환산(같은 화각 크롭 가정 아님)"})
    return out


# ---------------------------------------------------------------- figures
def figures(out_dir: Path, a: dict, times, chosen: str, anyc: dict | None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    made = []
    times = np.asarray(times)
    th = np.array([np.nan if v is None else v for v in a["theta2d_deg"]], float)
    ax_ = np.array([np.nan if v is None else v for v in a["axis2d_deg"]], float)
    role = np.array(a["role"])

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(times, th, ".-", lw=1, ms=3, color="#2a6fbb", label="θ2d")
    fw = a["front_window"]
    ax.axvspan(fw["t0"], fw["t1"], color="#999", alpha=0.25, label="정면 창")
    ax.axvline(a["tilt_start_s"], color="#444", ls="--", lw=1, label="기울기 구간 시작")
    if a.get("auto_transition_s") is not None:
        ax.axvline(a["auto_transition_s"], color="#c47a00", ls=":", lw=1, label="자동 전환")
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("θ2d (°)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    made.append(_save(fig, fig_dir / "theta2d_time.png"))

    fig, ax = plt.subplots(figsize=(4, 3))
    cov = a["axis_coverage"]
    ax.bar([f"{30 * k}-{30 * k + 30}" for k in range(6)], cov["counts"], color="#2a6fbb")
    ax.set_xlabel(f"압축 축 방향 (°), θ2d ≥ {cov['theta_min_deg']:g}°")
    ax.set_ylabel("프레임 수")
    ax.tick_params(axis="x", labelsize=7)
    made.append(_save(fig, fig_dir / "axis_distribution.png"))

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m, col in (("fixed_template", "#2a6fbb"), ("joint_delta", "#c47a00")):
        s = a["solutions"][m]
        if s.get("profile_cost"):
            c = np.array(s["profile_cost"], float)
            ax.plot(s["profile_f"], c / np.nanmin(c), "-", color=col, lw=1.5, label=m)
            ax.axvline(s["f"], color=col, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("f (px, 정규화 해상도)")
    ax.set_ylabel("비용 / 최소")
    ax.set_ylim(0.9, 4)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    made.append(_save(fig, fig_dir / "f_profile.png"))

    pm = a["per_method"][chosen]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    T = np.array(a["template"])
    fr = a["template_frames"][len(a["template_frames"]) // 2]
    from . import template as TP
    if "obs_front_example" in a:
        P = np.array(a["obs_front_example"])
        s, R, t = TP.similarity_align(T, P)
        Q = (s * (R @ T.T)).T + t
        ax.scatter(P[:, 0], P[:, 1], s=14, color="#2a6fbb", label=f"관측(프레임 {fr})")
        ax.scatter(Q[:, 0], Q[:, 1], s=30, facecolors="none", edgecolors="#c47a00", label="템플릿(닮음 정합)")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    made.append(_save(fig, fig_dir / "template_overlay.png"))

    fig, ax = plt.subplots(figsize=(6, 3))
    pj = pm["diagnostics"]["per_joint_rms_px"]
    ax.bar([str(k) for k in pj], [np.nan if v is None else v for v in pj.values()], color="#2a6fbb")
    ax.set_xlabel("관절 번호 (OpenPose)")
    ax.set_ylabel("재투영 RMS (px)")
    made.append(_save(fig, fig_dir / "per_joint_residual.png"))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    bf = pm["bootstrap"]["f"]
    if bf:
        ax.hist(bf, bins=min(15, max(5, len(bf) // 2)), color="#2a6fbb")
    ax.set_xlabel("부트스트랩 f (px)")
    ax.set_ylabel("횟수")
    made.append(_save(fig, fig_dir / "bootstrap_f.png"))

    if anyc:
        fig, ax = plt.subplots(figsize=(8, 3))
        for m, d in anyc["models"].items():
            ax.plot(anyc["times"], d["f"], ".-", ms=3, lw=1, label=f"AnyCalib {m}")
        ax.axvline(a["tilt_start_s"], color="#444", ls="--", lw=1)
        ax.set_xlabel("시간 (s)")
        ax.set_ylabel("f (px)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        made.append(_save(fig, fig_dir / "anycalib_f_time.png"))
    return made


def _save(fig, p: Path) -> str:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def _font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Malgun Gothic"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


# ---------------------------------------------------------------- tables
def per_frame_csv(path: Path, frames, tr, a, chosen, anyc) -> None:
    res_by_frame = {}
    sol = a["per_method"][chosen]
    for fi, r in zip(sol["diagnostics"]["frames_used"], sol["diagnostics"].get("frame_rms_px", [])):
        res_by_frame[fi] = r
    af = {}
    if anyc:
        m0 = next(iter(anyc["models"]))
        af = dict(zip(anyc["frames"], anyc["models"][m0]["f"]))
    bl = np.maximum(tr["box"][:, 2] - tr["box"][:, 0], tr["box"][:, 3] - tr["box"][:, 1])
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "video_frame", "time_s", "present", "cls", "box_conf", "box_long_px", "n_visible_points", "theta2d_deg", "axis2d_deg",
                    "role", "removed_reason", "reproj_rms_px", "anycalib_f_px"])
        for i, r in enumerate(frames):
            th, ax_ = a["theta2d_deg"][i], a["axis2d_deg"][i]
            w.writerow([r["index"], r["frame"], f"{r['time_s']:.4f}", int(tr["present"][i]), int(tr["cls"][i]), f"{tr['box_conf'][i]:.4f}",
                        "" if not np.isfinite(bl[i]) else f"{bl[i]:.2f}", int(tr["mask"][i].sum()),
                        "" if th is None or not np.isfinite(th) else f"{th:.3f}", "" if ax_ is None or not np.isfinite(ax_) else f"{ax_:.2f}",
                        a["role"][i], a["reason"][i], "" if i not in res_by_frame else f"{res_by_frame[i]:.3f}",
                        "" if i not in af else f"{af[i]:.2f}"])


def summary_row(name, a, chosen, anyc, duration) -> dict:
    if "per_method" not in a:
        return {"video": name, "duration_s": duration, "verdict": "REJECT", "f_hand": None, "unc": None, "fixed": None, "joint": None, "anycalib": None,
                "reason": a.get("failed")}
    pm = a["per_method"][chosen]
    return {"video": name, "duration_s": duration, "verdict": pm["verdict"]["verdict"], "f_hand": pm["f"], "unc": pm["f_rel_uncertainty"],
            "fixed": a["solutions"]["fixed_template"]["f"], "joint": a["solutions"]["joint_delta"]["f"],
            "anycalib": (anyc["models"][next(iter(anyc["models"]))]["summary"]["median"] if anyc else None)}


# ---------------------------------------------------------------- report.md
def report_md(path: Path, ctx: dict) -> None:
    a, chosen, v = ctx["analysis"], ctx["estimator"], ctx["verdict"]
    pm = a["per_method"][chosen]
    f = pm["f"]
    L = []
    L.append(f"# 손 템플릿 초점거리 추정: {ctx['video']}")
    L.append("")
    label = "신뢰 가능한 f" if v == "GOOD" else ("대략적인 f(ROUGH, 신뢰 가능한 f가 아님)" if v == "ROUGH" else "참고용 값(REJECT, 신뢰 가능한 f가 아님)")
    L.append(f"- 판정(손 방식에만 적용): **{v}**")
    L.append(f"- f_hand ({chosen}): **{f:.1f} px** — {label}, 정규화 해상도 {ctx['W']}×{ctx['H']} 기준")
    L.append(f"- 휴리스틱 상대 불확실도 f_rel_uncertainty: {fmt(pm['f_rel_uncertainty'], 1, True)} (보정된 신뢰구간이 아님)")
    L.append(f"- 두 방식 값: fixed_template {a['solutions']['fixed_template']['f']:.1f} px, joint_delta {a['solutions']['joint_delta']['f']:.1f} px "
             f"(δ {a['solutions']['joint_delta']['delta_deg']:.1f}°, φ {a['solutions']['joint_delta']['phi_deg']:.0f}°); 정면 가정 민감도 {fmt(a['method_diff_rel'], 1, True)}")
    if ctx.get("anycalib"):
        for m, d in ctx["anycalib"]["models"].items():
            s = d["summary"]
            L.append(f"- AnyCalib {m} (별도 값, 손 방식과 무관): 중앙값 {s['median']:.1f} px, IQR {s['iqr'][0]:.1f}–{s['iqr'][1]:.1f}, "
                     f"p10–p90 {s['p10_p90'][0]:.1f}–{s['p10_p90'][1]:.1f}, 정면 {fmt(s['front_median'])} / 기울기 {fmt(s['tilt_median'])}, CV {fmt(s['cv'], 1, True)}")
    else:
        L.append("- AnyCalib: 실행하지 않음")
    if ctx.get("expected"):
        e = ctx["expected"]
        L.append(f"- 합성 기준 예상 오차({e['condition']}, 정면 지터 {fmt(e['front_jitter_rel_condition'], 2, True)} vs 영상 {fmt(e['front_jitter_rel_video'], 2, True)}): "
                 f"중앙값 {fmt(e['median_abs'], 1, True)}, p90 {fmt(e['p90_abs'], 1, True)} — {e['note']}")
    L.append("")
    L.append(MANDATORY)
    L.append("")
    if pm["advice"] and v != "GOOD":
        L.append("## 재촬영 안내")
        L += [f"- {x}" for x in pm["advice"]]
        L.append("")
    L.append("## 판정 근거 (yaml 문턱값, 첫 실행 전에 고정)")
    met = pm["metrics"]
    L.append("| 항목 | 값 | GOOD 통과 | ROUGH 통과 |")
    L.append("|---|---|---|---|")
    keys = [("detect_rate", "검출률", lambda x: fmt(x, 3)), ("front_frames", "정면 프레임 수", str), ("finger_extension_ok", "손가락 펴짐", str),
            ("theta_p90", "θ2d p90 (°)", fmt), ("axis_bins", "축 구간(6) 커버", None), ("reproj_rms", "재투영 RMS / 박스", None),
            ("jointset_range", "관절 집합 범위", lambda x: fmt(x, 1, True)), ("bootstrap", "부트스트랩 상대 표준편차", None),
            ("consistency", "일관성 범위", None), ("delta", "|δ| (°)", None), ("sharpness", "프로파일 선명도", None)]
    vals = {"detect_rate": met["detect_rate"], "front_frames": met["front_frames"], "finger_extension_ok": met["finger_extension_ok"],
            "theta_p90": met["theta_p90"], "axis_bins": met["axis_bins6"], "reproj_rms": fmt(met["reproj_rms_rel"], 2, True),
            "jointset_range": met["jointset_range"], "bootstrap": fmt(met["bootstrap_rel_std"], 2, True), "consistency": fmt(met["consistency_range"], 1, True),
            "delta": fmt(abs(met["delta_deg"])), "sharpness": fmt(met["sharpness"], 2)}
    g, r = pm["verdict"]["good_checks"], pm["verdict"]["rough_checks"]
    for k, name, fn in keys:
        x = vals[k]
        s = fn(x) if fn else str(x)
        L.append(f"| {name} | {s} | {'O' if g.get(k) else 'X'} | {('O' if r[k] else 'X') if k in r else '-'} |")
    L.append(f"| 대상 프레임 수 | {met['n_target_frames']} | 최소 {ctx['min_target_frames']} | |")
    L.append("")
    L.append("## 진단")
    d = pm["diagnostics"]
    L.append(f"- 재투영 RMS {fmt(d['reproj_rms_px'], 2)} px ({fmt(d['reproj_rms_rel'], 2, True)} of 박스 긴 변), 비용 +15% {d['cost_plus15']:.3g}, −15% {d['cost_minus15']:.3g}, 최소 {d['cost']:.3g}")
    L.append(f"- θ2d (기울기 구간) 중앙값 {fmt(a['theta_tilt']['median'])}°, p90 {fmt(a['theta_tilt']['p90'])}°, 최대 {fmt(a['theta_tilt']['max'])}°; 축 6구간 {a['axis_coverage']['counts']}; "
             f"PnP 법선 방위 8구간 커버 {d['azimuth_bins8']}/8")
    L.append(f"- 정면 창 {fmt(a['front_window']['t0'], 2)}–{fmt(a['front_window']['t1'], 2)} s, 템플릿 프레임 {a['front_frames']}장, 지터 {fmt(a['front_jitter_px'], 2)} px "
             f"({fmt(a['front_jitter_rel'], 2, True)}), 손가락 펴짐 최소 {fmt(a['finger_extension']['min'], 3)}")
    L.append(f"- 기울기 구간 시작 {a['tilt_start_s']:.2f} s ({a['phase_mode']}), 자동 전환 {fmt(a.get('auto_transition_s'), 2)} s")
    L.append(f"- 사용 프레임: 정면 {a['n_front_used']}, 기울기 {a['n_tilt_used']}; 제외 사유 {ctx['removed']}")
    L.append(f"- 손 크기(박스 긴 변 중앙값) {fmt(a['hand_size_px']['box_long_median'])} px, 가장자리 프레임 비율 {fmt(a['edge_frame_rate'], 1, True)}, "
             f"좌우 클래스 일관성 {fmt(a['lr']['consistency'], 3) if a.get('lr') else 'nan'} (다수 클래스 {ctx['lr_name']})")
    L.append(f"- 관절 집합별 f: " + ", ".join(f"{k} {fmt(x)}" for k, x in pm["jointsets"].items()))
    b = pm["bootstrap"]
    L.append(f"- 블록 부트스트랩 {b['n_done']}/{b['n_requested']}회{' (시간 예산으로 축소)' if b['reduced'] else ''}, 상대 표준편차 {fmt(b['rel_std'], 2, True)}")
    c = pm["consistency"]
    L.append(f"- 일관성: 앞/뒤 절반 {fmt(c['first_half'])} / {fmt(c['second_half'])}, 수평축/수직축 {fmt(c['axis_horizontal'])} ({c['n_horizontal']}장) / "
             f"{fmt(c['axis_vertical'])} ({c['n_vertical']}장)")
    L.append(f"- 순서 점검(OpenPose): {ctx['order_check']}")
    L.append("")
    L.append("## 환산 (f_hand 기준)")
    dv = ctx["derived"]
    L.append(f"- f / 긴 변 {dv['f_over_long_side']:.4f}, 화각 가로 {dv['fov_h_deg']:.1f}°, 세로 {dv['fov_v_deg']:.1f}°, 대각 {dv['fov_d_deg']:.1f}°")
    for s in dv["at_sizes"]:
        L.append(f"- {s['size'][0]}×{s['size'][1]}: {s['f_px']:.1f} px" + (f" ({s['note']})" if s["note"] else ""))
    L.append("")
    if ctx["warnings"]:
        L.append("## 경고")
        L += [f"- {w}" for w in ctx["warnings"]]
        L.append("")
    L.append("## 촬영 가이드")
    from .verdict import SHOOTING_GUIDE
    L += [f"- {x}" for x in SHOOTING_GUIDE]
    L.append("")
    L.append("## 한계")
    L.append("- 핀홀, 주점 = 영상 중심, 정사각 화소, 왜곡 없음(기본값)을 가정한다. 손바닥과 펴진 손가락을 평면으로 본다.")
    L.append("- 키포인트는 WiLoR 파이프라인의 YOLO 검출 모듈(detector.pt)의 2D 출력만 사용했다. 이 검출기의 2D 라벨 생성 방식은 확인하지 못했다.")
    L.append("- 불확실도 결합식 max(2·부트스트랩, 집합범위/2, 일관성/2, 방식차/2)은 휴리스틱이며 정답 데이터로 보정되지 않았다.")
    L.append("- 정답 f가 없어 이 영상의 오차는 계산하지 않았다. 합성 기준 예상 오차는 합성 조건의 값이다.")
    path.write_text("\n".join(L) + "\n")
