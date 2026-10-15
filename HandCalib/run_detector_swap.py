"""Experiment 16: 14's hand-template f pipeline with the 2D hand keypoint detector swapped.

    python HandCalib/run_detector_swap.py --config HandCalib/configs/16_detector_swap.yaml [--video V] [--detectors a,b] [--no-anycalib]

Per video: one normalised frame list (14's rules; checked against 14's frame hashes when 14 processed the video), every detector on
the same PNGs (14 npz format), then 14's process_video with the mock (npz) adapter for each detector. f values are reported per
detector; no differences or ratios between f values are computed. Profile minima on the extended grid are diagnostics only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hand_focal as RUN14                                  # noqa: E402  (14 runner, unchanged)
from detector_swap import analysis16 as A16                     # noqa: E402
from detector_swap import detect as DT                          # noqa: E402
from hand_focal import frames as FR                             # noqa: E402
from hand_focal import normalize as NZ                          # noqa: E402
from hand_focal import report as RP                             # noqa: E402
from hand_focal.paths import PROJECT_ROOT, REPO_ROOT, resolve   # noqa: E402

MANDATORY = "이 f는 핀홀·주점 중심·왜곡 없음 가정에서 2D 키포인트만으로 얻은 추정값이며 정답 데이터와 비교하지 않았다."
VENVS = {"wilor_yolo": "/home/juson/project/MonoDETR/.venv/bin/python", "mediapipe_hands": "/home/juson/venvs/det_mediapipe/bin/python",
         "rtmpose_hand": "/home/juson/venvs/det_rtmpose/bin/python"}


# ---------------------------------------------------------------- display rules
def edge_side(sol, grid_hi):
    return "상한" if sol["f"] >= grid_hi * (1 - 1e-6) else "하한"


def f_text(sol, grid_hi, unc=None):
    """f with its uncertainty, or '값 없음 (상한/하한)' when the solution is a grid edge."""
    if sol is None:
        return "값 없음 (실패)"
    if sol["at_grid_edge"]:
        return f"값 없음 ({edge_side(sol, grid_hi)})"
    return f"{sol['f']:.1f} px" + ("" if unc is None or not np.isfinite(unc) else f" (±{100 * unc:.1f}%)")


def spread_text(x, edge):
    """Bootstrap / consistency spreads: '계산 불가' when the solution is pinned to a grid edge (never 0%)."""
    if edge:
        return "계산 불가"
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


# ---------------------------------------------------------------- provenance
def pip_freeze(py):
    try:
        return subprocess.check_output([py, "-m", "pip", "freeze"], text=True, env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"}).splitlines()
    except Exception as e:                                    # noqa: BLE001
        return [f"unavailable: {e}"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detector_provenance(cfg16):
    out = {}
    for name in cfg16["detectors"]:
        s = cfg16["detector_settings"][name]
        w = Path(s["weights"])
        out[name] = {k: s.get(k) for k in ("package", "weights", "weights_url", "device", "license", "training_labels", "conf_min")}
        out[name]["model_name"] = s.get("model")
        out[name]["weights_sha256"] = sha256_file(w) if w.exists() else None
        out[name]["python"] = s["python"]
        out[name]["pip_freeze"] = pip_freeze(s["python"])
    if "rtmpose_hand" in cfg16["detectors"]:
        z = Path(cfg16["detector_settings"]["rtmpose_hand"]["weights"]).parents[2].parent
        zips = list(z.glob("*.zip"))
        out["rtmpose_hand"]["zip_sha256"] = {p.name: sha256_file(p) for p in zips}
    return out


# ---------------------------------------------------------------- one video
def frames_14(cfg16, video_name):
    p = cfg16["reuse_14"]["official"].get(video_name)
    if not p:
        return None, None
    d = resolve(p)
    return d, json.loads((d / "results.json").read_text())


def process(video: Path, cfg16, cfg_text, cfg_path, run_root: Path, study, study_path, use_anycalib, log):
    pcfg = cfg16["pipeline"]
    name = video.stem
    vdir = run_root / name
    vdir.mkdir(parents=True)
    tmp = Path(tempfile.mkdtemp(prefix="hf16_"))
    T = {}
    res = {"video": video.name, "detectors": {}, "warnings": []}
    try:
        t = time.time()
        info = NZ.probe(video)
        if info["duration_s"] < float(pcfg["min_duration_s"]):
            res["warnings"].append(f"영상 길이 {info['duration_s']:.1f} s < {pcfg['min_duration_s']} s")
        norm = NZ.normalize(video, tmp / "normalized.mkv", pcfg)
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        frames, k = FR.extract(tmp / "normalized.mkv", tmp / "frames", norm["output"]["fps"], None, pcfg["frames"]["detect_fps"])
        T["frames_s"] = time.time() - t
        d14, r14 = frames_14(cfg16, video.name)
        same14 = None
        if r14:
            l14 = r14["frames"]["list"]
            same14 = [(x["index"], x["sha256"]) for x in l14] == [(x["index"], x["sha256"]) for x in frames]
            if not same14:
                res["warnings"].append("14의 프레임 목록(번호·해시)과 다릅니다")
        res["frames"] = {"n": len(frames), "step": k, "same_as_14": same14, "list": [{kk: r[kk] for kk in ("index", "frame", "time_s", "sha256")} for r in frames]}
        log(f"== {video.name} == 프레임 {len(frames)}장 (14 목록과 동일: {same14})")
        t = time.time()
        dets = DT.run_all(frames, cfg16, resolve(cfg16["output"]["root"]) / "detection_cache", log)
        T["detect_s"] = time.time() - t
        L0s = {}
        for det in cfg16["detectors"]:
            dd = dets.get(det, {"skipped": "not run"})
            if "skipped" in dd:
                res["detectors"][det] = {"skipped": dd["skipped"]}
                continue
            s = cfg16["detector_settings"][det]
            pc = copy.deepcopy(pcfg)
            pc["detector"] = {"name": "mock", "mock_npz": str(dd["npz"]), "weights": s["weights"], "classes": pcfg["detector"]["classes"], "cache": True}
            pc["tracking"]["conf_min"] = float(s["conf_min"])
            pc_text = yaml.safe_dump(pc, sort_keys=False)
            work = run_root / "_work" / det
            t = time.time()
            row = RUN14.process_video(video, pc, pc_text, f"{cfg_path}#pipeline[{det}]", work, pcfg["estimate"]["estimator"],
                                      "14 합성 연구 규칙(fixed_template), 16 yaml에 고정", study, study_path, False)
            ddir = vdir / det
            shutil.move(str(work / name), str(ddir))
            r = json.loads((ddir / "results.json").read_text())
            prof, L0 = A16.profiles(ddir, pc, cfg16)
            L0s[det] = L0
            (ddir / "profiles.json").write_text(json.dumps(RP.jsonable(prof), ensure_ascii=False))
            T[f"pipeline_{det}_s"] = time.time() - t
            res["detectors"][det] = summarize_detector(r, prof, L0, dd, s)
            figures_detector(ddir, prof, r, det)
            report_detector(ddir / "report.md", video.name, det, res["detectors"][det], s)
            log(console_line(det, res["detectors"][det]))
        res["diff2d"] = A16.diff2d(L0s, pcfg, cfg16)
        figure_diff(vdir, res["diff2d"], cfg16)
        # AnyCalib: reuse 14's values when 14 processed this video, else run on the same PNGs
        t = time.time()
        if use_anycalib and pcfg["anycalib"]["enabled"]:
            if r14 and cfg16["reuse_14"]["reuse_anycalib"] and r14.get("anycalib"):
                res["anycalib"] = {"reused_from_14": str(d14), **{k: r14["anycalib"][k] for k in ("models", "hub_check", "skipped_extra_models")}}
            else:
                from hand_focal import anycalib_ref as AC
                ac = AC.run(frames, pcfg["anycalib"], float(pcfg["phases"]["front_phase_s"]), log)
                res["anycalib"] = {"reused_from_14": None, **{k: ac[k] for k in ("models", "hub_check", "skipped_extra_models")}}
        T["anycalib_s"] = time.time() - t
        res["shooting"] = shooting(res, L0s, W, H)
        res["size"] = [W, H]
        res["timings_s"] = T
        for line in video_block(res):
            log(line)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_root / "_work", ignore_errors=True)


def summarize_detector(r, prof, L0, dd, s):
    a = r.get("analysis", {})
    out = {"verdict": r.get("verdict"), "npz": str(dd["npz"]), "detector_info": dd.get("info"), "conf_min": s["conf_min"]}
    if "solutions" not in a:
        out["failed"] = a.get("failed")
        return out
    grid_hi = a["grid"][-1]
    pm = a["per_method"]["fixed_template"]
    tr = L0["tr"]
    pres = tr["present"]
    out.update({
        "grid_px": [a["grid"][0], grid_hi],
        "fixed_template": {"f": a["solutions"]["fixed_template"]["f"], "at_grid_edge": a["solutions"]["fixed_template"]["at_grid_edge"],
                           "side": edge_side(a["solutions"]["fixed_template"], grid_hi) if a["solutions"]["fixed_template"]["at_grid_edge"] else None},
        "joint_delta": {"f": a["solutions"]["joint_delta"]["f"], "at_grid_edge": a["solutions"]["joint_delta"]["at_grid_edge"],
                        "side": edge_side(a["solutions"]["joint_delta"], grid_hi) if a["solutions"]["joint_delta"]["at_grid_edge"] else None,
                        "delta_deg": a["solutions"]["joint_delta"]["delta_deg"]},
        "f_rel_uncertainty": pm["f_rel_uncertainty"], "bootstrap_rel_std": pm["bootstrap"]["rel_std"], "consistency_range": pm["consistency"]["range"],
        "jointsets_f": pm["jointsets"], "failed_good": pm["verdict"]["failed_good"], "failed_rough": pm["verdict"]["failed_rough"], "advice": pm["advice"],
        "metrics": pm["metrics"], "sharpness": pm["diagnostics"]["sharpness"], "reproj_rms_rel": pm["diagnostics"]["reproj_rms_rel"],
        "detect_rate": a["detect_rate"], "masked_point_rate_present": float(1 - tr["mask"][pres].mean()) if pres.any() else None,
        "order_check": r["tracking"]["order_check"], "lr": a.get("lr"), "profiles": {k: {g: {kk: v[g][kk] for kk in ("interior", "at", "min_at_px", "sharpness", "n_frames")}
                                                                                        for g in ("official", "extended")} for k, v in prof.get("sets", {}).items()},
        "theta_tilt": a["theta_tilt"], "axis_coverage": a["axis_coverage"]["counts"], "hand_box_long_median_px": a["hand_size_px"]["box_long_median"],
        "edge_frame_rate": a["edge_frame_rate"], "front_window": [a["front_window"].get("t0"), a["front_window"].get("t1")], "front_frames": a["front_frames"],
        "auto_transition_s": a.get("auto_transition_s"), "warnings": r.get("warnings", [])})
    return out


def console_line(det, d):
    if "skipped" in d:
        return f"    {det:<16}: 건너뜀 ({d['skipped'][:80]})"
    if "failed" in d:
        return f"    {det:<16}: f = 값 없음 (분석 실패: {d['failed']})   판정 {d['verdict']}"
    ft = d["fixed_template"]
    unc = None if ft["at_grid_edge"] else d["f_rel_uncertainty"]
    s = f"값 없음 ({ft['side']})" if ft["at_grid_edge"] else f"{ft['f']:.0f} px" + (f" (±{100 * unc:.0f}%)" if unc is not None and np.isfinite(unc) else "")
    jd = d["joint_delta"]
    sj = f"값 없음 ({jd['side']})" if jd["at_grid_edge"] else f"{jd['f']:.0f} px"
    return f"    {det:<16}: f = {s:<22} 판정 {d['verdict']:<7} (joint_delta: {sj})"


def shooting(res, L0s, W, H):
    ref = next((d for d in ("wilor_yolo", "mediapipe_hands", "rtmpose_hand") if d in res["detectors"] and "metrics" in res["detectors"][d]), None)
    if ref is None:
        return {"reference": None, "warnings": []}
    d = res["detectors"][ref]
    short = min(W, H)
    fw = d["front_window"]
    w = [f"손 박스 긴 변이 화면 짧은 변의 {100 * d['hand_box_long_median_px'] / short:.0f}%",
         f"가장자리 접촉 프레임 {100 * d['edge_frame_rate']:.0f}%",
         f"정면 창 {fw[0]:.2f}–{fw[1]:.2f} s" if fw[0] is not None else "정면 창 없음",
         f"자동 전환 {d['auto_transition_s']:.2f} s" if d["auto_transition_s"] is not None else "자동 전환 없음",
         f"θ2d p90 {d['theta_tilt']['p90']:.0f}°, 최대 {d['theta_tilt']['max']:.0f}°", f"축 분포 {d['axis_coverage']}"]
    return {"reference": ref, "warnings": w}


def video_block(res):
    L = [f"  [{res['video']}]"]
    for det, d in res["detectors"].items():
        L.append(console_line(det, d))
    ac = res.get("anycalib")
    if ac:
        for m, v in ac["models"].items():
            s = v["summary"]
            L.append(f"    AnyCalib({m.replace('anycalib_', '')}): 중앙값 {s['median']:.1f} px (IQR {s['iqr'][0]:.1f}–{s['iqr'][1]:.1f})"
                     + (" — 14 결과 재사용" if ac.get("reused_from_14") else ""))
    L.append("    촬영 조건 경고(" + str(res["shooting"]["reference"]) + " 기준): " + ", ".join(res["shooting"]["warnings"]))
    return L


# ---------------------------------------------------------------- per-detector report and figures
def report_detector(path: Path, video, det, d, s):
    L = [f"# {video} — {det}", "", MANDATORY, ""]
    if "failed" in d:
        L += [f"- 판정 {d['verdict']}: 분석 실패({d['failed']}). f = 값 없음", ""]
        path.write_text("\n".join(L) + "\n")
        return
    ft, jd = d["fixed_template"], d["joint_delta"]
    edge = ft["at_grid_edge"]
    L += [f"- 판정(손 방식에만 적용): **{d['verdict']}**" + ("" if d["verdict"] == "GOOD" else " — 신뢰 가능한 f가 아님"),
          f"- f (fixed_template): **{'값 없음 (' + ft['side'] + ')' if edge else format(ft['f'], '.1f') + ' px'}** (탐색 범위 {d['grid_px'][0]:.0f}–{d['grid_px'][1]:.0f} px)",
          f"- f (joint_delta): {'값 없음 (' + jd['side'] + ')' if jd['at_grid_edge'] else format(jd['f'], '.1f') + ' px'} (δ {jd['delta_deg']:.1f}°)",
          f"- 휴리스틱 상대 불확실도: {'계산 불가' if edge else spread_text(d['f_rel_uncertainty'], False)}; 부트스트랩 상대 표준편차 "
          f"{spread_text(d['bootstrap_rel_std'], edge)}; 일관성 범위 {spread_text(d['consistency_range'], edge)}",
          f"- GOOD 실패 항목: {', '.join(d['failed_good']) or '없음'}; ROUGH 실패 항목: {', '.join(d['failed_rough']) or '없음'}",
          f"- 검출률 {d['detect_rate']:.3f}, 검출 프레임 중 마스크된 점 비율 {100 * d['masked_point_rate_present']:.1f}% (conf_min {s['conf_min']}), "
          f"순서 점검 {d['order_check'].get('ok')}, 재투영 RMS/박스 {100 * d['reproj_rms_rel']:.2f}%, 선명도 {d['sharpness']:.2f}", "",
          "## 관절 집합별 비용 프로파일 (진단, f 값 아님)", "", "| 관절 집합 | 공식 범위 최솟값 | 위치(px) | 날카로움 | 연장 격자(0.2–5×긴 변) 최솟값 | 위치(px) | 프레임 |",
          "|---|---|---|---|---|---|---|"]
    ko = {"interior": "내부", "upper": "상한", "lower": "하한", "none": "-"}
    for k, v in d["profiles"].items():
        o, e = v["official"], v["extended"]
        L.append(f"| {k} | {ko[o['at']]} | {_px(o['min_at_px'])} | {_n(o['sharpness'])} | {ko[e['at']]} | {_px(e['min_at_px'])} | {o['n_frames']} |")
    L += ["", "- 14 형식 원자료(results.json, result.yaml)는 그대로 두었다. 그 안의 f_hand_px는 f_hand_at_grid_edge가 true이면 격자 경계값이며 f가 아니다.", ""]
    if d.get("advice") and d["verdict"] != "GOOD":
        L += ["## 재촬영 안내"] + [f"- {x}" for x in d["advice"]] + [""]
    path.write_text("\n".join(L) + "\n")


def _px(x):
    return "-" if x is None else f"{x:.0f}"


def _n(x):
    return "-" if x is None else f"{x:.3f}"


def _plt():
    RP._font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RP._font()
    return plt


def figures_detector(ddir, prof, r, det):
    if "sets" not in prof:
        return
    plt = _plt()
    fd = ddir / "figures"
    fd.mkdir(exist_ok=True)
    off, ext = np.array(prof["official_grid"]), np.array(prof["extended_grid"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    p = np.array(prof["sets"]["flat17"]["extended"]["profile"], float)
    ax.plot(ext, p / np.nanmin(p), color="#2a6fbb", lw=1.8, label="flat17, 연장 격자")
    po = np.array(prof["sets"]["flat17"]["official"]["profile"], float)
    ax.plot(off, po / np.nanmin(p), "o", ms=2.5, color="#c47a00", label="flat17, 공식 격자")
    ax.axvspan(off[0], off[-1], color="#999", alpha=0.15, label="공식 탐색 범위")
    ax.set_xscale("log")
    ax.set_ylim(0.95, 4)
    ax.set_xlabel("f 후보 (px)")
    ax.set_ylabel("비용 / 연장 격자 최소")
    ax.set_title(f"{det} (진단)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fd / "profile_official_extended.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, v in prof["sets"].items():
        q = np.array(v["extended"]["profile"], float)
        if np.isfinite(q).any():
            ax.plot(ext, q / np.nanmin(q), lw=1.4, label=f"{k} ({v['extended']['n_frames']})")
    ax.axvspan(off[0], off[-1], color="#999", alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylim(0.95, 4)
    ax.set_xlabel("f 후보 (px)")
    ax.set_ylabel("비용 / 최소")
    ax.set_title(f"{det}: 관절 집합별 (진단)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fd / "profile_jointsets.png", dpi=110)
    plt.close(fig)


def figure_diff(vdir, diff, cfg16):
    if not diff:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = cfg16["diff2d"]["theta_bins_deg"]
    lab = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]
    for key, v in diff.items():
        y1 = [np.nan if b["median_rel"] is None else 100 * b["median_rel"] for b in v["by_theta"]]
        y2 = [np.nan if b["radial_median_rel"] is None else 100 * b["radial_median_rel"] for b in v["by_theta"]]
        axs[0].plot(lab, y1, "o-", label=f"{v['b']} − {v['a']}")
        axs[1].plot(lab, y2, "o-", label=f"{v['b']} − {v['a']}")
    axs[0].set_ylabel("관절 2D 차이 중앙값 / 박스 (%)")
    axs[1].set_ylabel("바깥쪽(+) 방향 성분 중앙값 / 박스 (%)")
    axs[1].axhline(0, color="#444", lw=0.8)
    for ax in axs:
        ax.set_xlabel("θ2d 구간 (°, 쌍의 첫 검출기 기준)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    (vdir / "figures").mkdir(exist_ok=True)
    fig.savefig(vdir / "figures" / "diff2d_theta.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- run report
def report_run(path: Path, results: list, cfg16, prov):
    L = ["# 실험 16: 검출기 교체 비교 (손 템플릿 f, 14 파이프라인)", "", MANDATORY, "",
         "f 값끼리 또는 외부 값과의 차이·비율은 계산하지 않았다. f가 탐색 범위 경계에 붙은 경우 '값 없음'으로 적고, 부트스트랩·일관성은 '계산 불가'로 적는다.", ""]
    for r in results:
        L += ["```"] + video_block(r) + ["```", ""]
    L += ["## 전체 요약", "", "| 영상 | 검출기 | f (fixed_template) | 불확실도 | 판정 | 공식 범위 내부 최솟값(flat17) | 프로파일 날카로움 | joint_delta f |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for det, d in r["detectors"].items():
            if "skipped" in d or "failed" in d:
                L.append(f"| {r['video']} | {det} | 값 없음 | - | {d.get('verdict', '-')} | - | - | - |")
                continue
            ft, jd = d["fixed_template"], d["joint_delta"]
            pf = d["profiles"].get("flat17", {}).get("official", {})
            L.append(f"| {r['video']} | {det} | {'값 없음 (' + ft['side'] + ')' if ft['at_grid_edge'] else format(ft['f'], '.1f') + ' px'} | "
                     f"{'계산 불가' if ft['at_grid_edge'] else spread_text(d['f_rel_uncertainty'], False)} | {d['verdict']} | "
                     f"{'있음' if pf.get('interior') else '없음'} | {_n(pf.get('sharpness'))} | "
                     f"{'값 없음 (' + jd['side'] + ')' if jd['at_grid_edge'] else format(jd['f'], '.1f') + ' px'} |")
        ac = r.get("anycalib")
        if ac:
            L.append(f"| {r['video']} | AnyCalib(별도) | " + ", ".join(f"{m}: 중앙값 {v['summary']['median']:.1f} px" for m, v in ac["models"].items()) + " | | | | | |")
    L.append("")
    L += ["## 관절 집합별 비용 프로파일 (진단, f 값 아님)", "", "| 영상 | 검출기 | 관절 집합 | 공식 범위 | 위치(px) | 날카로움 | 연장 격자 | 위치(px) |", "|---|---|---|---|---|---|---|---|"]
    ko = {"interior": "내부", "upper": "상한", "lower": "하한", "none": "-"}
    for r in results:
        for det, d in r["detectors"].items():
            for k, v in d.get("profiles", {}).items():
                o, e = v["official"], v["extended"]
                L.append(f"| {r['video']} | {det} | {k} | {ko[o['at']]} | {_px(o['min_at_px'])} | {_n(o['sharpness'])} | {ko[e['at']]} | {_px(e['min_at_px'])} |")
    L += ["", "## 검출기별 검출 품질", "", "| 영상 | 검출기 | 검출률 | 마스크된 점 비율 | conf_min | 순서 점검 | 좌우 일관성 | 재투영 RMS/박스 | θ2d p90 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for det, d in r["detectors"].items():
            if "metrics" not in d:
                continue
            L.append(f"| {r['video']} | {det} | {d['detect_rate']:.3f} | {100 * d['masked_point_rate_present']:.1f}% | {d['conf_min']} | {d['order_check'].get('ok')} | "
                     f"{_n((d.get('lr') or {}).get('consistency'))} | {100 * d['reproj_rms_rel']:.2f}% | {d['theta_tilt']['p90']:.0f}° |")
    L += ["", "## 검출기 사이 2D 차이 (같은 프레임·같은 손, f 비교 아님)", "",
          "| 영상 | 쌍 (B − A) | θ2d 구간 | 프레임 | 차이 중앙값 px | 차이/박스 중앙값 | p90 | 바깥쪽 성분/박스 중앙값 |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for key, v in r.get("diff2d", {}).items():
            for b in v["by_theta"]:
                if not b["frames"]:
                    continue
                L.append(f"| {r['video']} | {v['b']} − {v['a']} | {b['theta'][0]}–{b['theta'][1]}° | {b['frames']} | {b['median_px']:.1f} | "
                         f"{100 * b['median_rel']:.2f}% | {100 * b['p90_rel']:.2f}% | {100 * b['radial_median_rel']:+.2f}% |")
    L += ["", "점 순서 교차 확인(같은 프레임에서 검출기 쌍의 관절 거리 행렬, Hungarian 배정이 항등이면 순서 일치):", ""]
    for r in results:
        for key, v in r.get("diff2d", {}).items():
            oa = v.get("order_assignment")
            L.append(f"- {r['video']} {v['b']} vs {v['a']}: " + ("자료 없음" if oa is None else ("항등(일치)" if oa["identity"] else f"불일치 {oa['assignment']}")))
    L += ["", "- 바깥쪽 성분: 첫 검출기(A)의 손 중심 → A의 관절 방향으로 잰 (B − A) 성분. 양수면 B의 점이 손 바깥쪽에 있다.", "",
          "## 검출기 설치 정보, 라이선스, 학습 데이터 출처", ""]
    for det, p in prov.items():
        L.append(f"- **{det}**: {p.get('model_name')}; 패키지 {p.get('package') or '-'}; 가중치 {p.get('weights')} (sha256 {str(p.get('weights_sha256'))[:16]}…)"
                 f"{'; 출처 ' + p['weights_url'] if p.get('weights_url') else ''}; 실행 장치 {p.get('device')}; 라이선스 {p.get('license')}; "
                 f"학습 라벨: {p.get('training_labels')}")
    L += ["- sapiens: 기본 꺼짐, 이번 실행에서 구현·실행하지 않음(CC BY-NC 4.0).", "",
          "## 검출기별 입력 형식 처리", "",
          "- mediapipe_hands: IMAGE 모드(프레임 독립), 최대 2손. 정규화 좌표 × (W, H) → 픽셀. 점별 신뢰도가 없어 손의 handedness 점수를 21점 모두에 넣음. "
          "박스는 랜드마크 외접 상자 + 크기의 10%(검출기가 박스를 주지 않음). z·world landmarks 미사용. 좌우 라벨은 반환값 그대로(거울상 보정 없음).",
          "- rtmpose_hand: wilor_yolo의 박스만 입력(키포인트 미사용). 박스 1.25배 크롭(rtmlib 내부 패딩 1.25와 일치), RGB 입력, 점별 SimCC 점수. 박스·박스 점수·좌우는 박스 출처 값.",
          "- 세 검출기 모두 OpenPose 순서(재배열 없음). 순서 점검 결과는 위 표.", "",
          "## 한계", "",
          "- 박스 정의가 검출기마다 다르다(Huber 폭·가장자리 판정·정규화에 쓰임). 파이프라인 설정은 같다.",
          "- 학습 라벨 출처는 문서로 확인한 범위만 적었다. wilor_yolo는 확인 불가.",
          "- 정답 f가 없어 오차를 계산하지 않았다.", ""]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/16_detector_swap.yaml"))
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--detectors", default=None, help="comma list overriding the yaml detectors")
    ap.add_argument("--no-anycalib", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg16 = yaml.safe_load(text)
    if a.detectors:
        cfg16["detectors"] = a.detectors.split(",")
    if cfg16["detector_settings"].get("sapiens", {}).get("enabled") or "sapiens" in cfg16["detectors"]:
        raise SystemExit("sapiens는 이번 실험에서 구현하지 않았습니다(기본 꺼짐)")
    sp = resolve(cfg16["reuse_14"]["synthetic_study"])
    study = json.loads(sp.read_text())
    if a.video:
        videos = [Path(a.video).expanduser()]
    else:
        d = Path(a.input_dir or cfg16["input_dir"]).expanduser()
        videos = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in cfg16["video_extensions"]})
    run_root = Path(a.out).expanduser() if a.out else resolve(cfg16["output"]["root"]) / RUN14.utc_stamp()
    run_root.mkdir(parents=True)
    log = RUN14.Log(run_root / "log.txt")
    log(f"16 검출기 교체: {cfg16['detectors']} → {run_root}")
    t0 = time.time()
    results = []
    for v in videos:
        results.append(process(v, cfg16, text, a.config, run_root, study, str(sp), not a.no_anycalib, log))
    prov = detector_provenance(cfg16)
    report_run(run_root / "report.md", results, cfg16, prov)
    (run_root / "results.json").write_text(json.dumps(RP.jsonable({"videos": results, "detectors": cfg16["detectors"]}), ensure_ascii=False, indent=1))
    from hand_focal import provenance as PV
    meta = PV.build(text, a.config, {"code_14": cfg16["code_14"], "detectors": prov, "elapsed_s": time.time() - t0,
                                     "devices": {k: cfg16["detector_settings"][k].get("device") for k in cfg16["detectors"]}})
    (run_root / "metadata.json").write_text(json.dumps(RP.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results


if __name__ == "__main__":
    main()
