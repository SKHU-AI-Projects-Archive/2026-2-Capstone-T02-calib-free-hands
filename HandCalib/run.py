"""HandCalib hand-template focal length: orchestration, reports and the command line for experiments 17 and 18 (analysis code in handfocal.py).

    python HandCalib/run.py protocol       --config HandCalib/configs/17_protocol_rerun.yaml [--videos A,B] [--detectors a,b] [results-yaml --run DIR]
    python HandCalib/run.py frame-filter   --config HandCalib/configs/17_frame_filter_diag.yaml [--source VIDEO=RUN_DIR]
    python HandCalib/run.py same-session   --config HandCalib/configs/18_same_session_check.yaml [results-yaml --run DIR]
    python HandCalib/run.py overlay        --video-dir runs/17_protocol_rerun/<UTC>/IMG_XXXX

Consolidated from run_hand_focal.py (14, process_video only), run_detector_swap.py (16, the per-detector processing that 17 calls),
run_protocol_rerun.py and run_frame_filter_diag.py (17),
run_same_session.py (18) and run_keypoint_overlay.py; each is a section below. Input videos are only read; no location or device metadata is written.
"""
from __future__ import annotations

import argparse
import copy
import cv2
import datetime as dt
import hashlib
import json
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import handfocal as hf  # noqa: E402


# ====================================================================================================
# run14  (was HandCalib/run_hand_focal.py)
# Experiment 14's per-video processing (process_video), kept because 16-18 run it for every detector.
#
# Input videos are only read. Frames go to a temporary directory that is deleted at the end; no frame image is written to the results.
# No location or device metadata is read into results, reports or logs (only technical stream fields).
# ====================================================================================================

class Log:
    def __init__(self, path: Path | None = None):
        self.fh = path.open("a") if path else None

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.fh:
            self.fh.write(msg + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------- estimator resolution


# ---------------------------------------------------------------- one video
def process_video(video: Path, cfg: dict, cfg_text: str, cfg_path: str, out_root: Path, estimator: str, basis: str, study, study_path,
                  use_anycalib: bool, tmp_base: Path | None = None) -> dict:
    name = video.stem
    out = out_root / name
    if out.exists():
        raise SystemExit(f"결과 폴더가 이미 있습니다(덮어쓰지 않음): {out}")
    out.mkdir(parents=True)
    log = Log(out / "log.txt")
    T = {}
    t_all = time.time()
    log(f"== {video.name} ==")
    tmp = Path(tempfile.mkdtemp(prefix="hf14_", dir=str(tmp_base) if tmp_base else None))
    warnings = []
    try:
        # 1. normalise
        t = time.time()
        info = hf.probe(video)
        if info["duration_s"] < float(cfg["min_duration_s"]):
            warnings.append(f"영상 길이 {info['duration_s']:.1f} s 가 {cfg['min_duration_s']} s 보다 짧습니다")
            log(f"  경고: {warnings[-1]}")
        norm = hf.normalize(video, tmp / "normalized.mkv", cfg)
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        fps = norm["output"]["fps"]
        T["normalize_s"] = time.time() - t
        log(f"  정규화: {info['display_w']}x{info['display_h']} (회전 {info['rotation_deg']}°) -> {W}x{H}, {fps:.3f} fps, 톤매핑 {norm['path']['tonemap']}")

        # 2. frames
        t = time.time()
        frames, k = hf.extract_frames(tmp / "normalized.mkv", tmp / "frames", fps, None, cfg["frames"]["detect_fps"])
        times = np.array([r["time_s"] for r in frames])
        T["frames_s"] = time.time() - t
        log(f"  프레임: {len(frames)}장 (매 {k}번째, 실효 {fps / k:.2f} fps)")

        # 3. detection
        t = time.time()
        det = hf.detector_run(frames, cfg["detector"], hf.resolve(cfg["output"]["root"]) / "detection_cache", log)
        T["detect_s"] = time.time() - t
        shutil.copy2(det["cache_file"], out / "detections.npz")
        log(f"  검출: 후보 {len(det['boxes'])}개, {T['detect_s']:.1f} s")

        # 4. tracking
        tr = hf.track(det, times, (W, H), cfg["tracking"], front_start_s=float(cfg["phases"]["front_skip_s"]))
        order = hf.order_check(tr["kp"], tr["mask"])
        if order["ok"] is False:
            warnings.append("키포인트 순서가 OpenPose 순서와 맞지 않아 보입니다(" + ", ".join(k for k in ("middle_longest", "pinky_shortest", "thumb_base_nearer_index") if not order[k]) + ")")
        if (tr["n_candidates"] > 1).any():
            warnings.append(f"손 후보가 2개 이상인 프레임 {int((tr['n_candidates'] > 1).sum())}장 (IoU 추적으로 대상 손 유지)")
        log(f"  추적: 검출률 {tr['present'].mean():.3f}, 순서 점검 {order['ok']}")

        # 5. analysis
        t = time.time()
        a = hf.analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), cfg, methods_full=(estimator,),
                      seed=int(cfg["experiment"]["seed"]))
        T["analysis_s"] = time.time() - t
        warnings += a["warnings"]

        # 6. AnyCalib (separate; never used by the hand method)
        anyc = None
        if use_anycalib and cfg["anycalib"]["enabled"]:
            t = time.time()
            anyc = hf.anycalib_run(frames, cfg["anycalib"], a.get("tilt_start_s", float(cfg["phases"]["front_phase_s"])), log)
            T["anycalib_s"] = time.time() - t
        T["total_s"] = time.time() - t_all

        # 7. outputs
        lr_names = {int(k): v for k, v in cfg["detector"]["classes"].items()}
        res = {"video": video.name, "estimator": estimator, "estimator_basis": basis, "synthetic_study": study_path,
               "normalization": norm, "frames": {"n": len(frames), "step": k, "detect_fps_effective": fps / k,
                                                "list": [{kk: r[kk] for kk in ("index", "frame", "time_s", "sha256")} for r in frames]},
               "tracking": {"detect_rate": float(tr["present"].mean()), "init_frame": tr["init_frame"], "n_multi_candidate_frames": int((tr["n_candidates"] > 1).sum()),
                            "order_check": order}, "detector_info": det.get("info"), "warnings": warnings, "timings_s": T}
        if "per_method" not in a:
            res.update({"verdict": "REJECT", "analysis": a})
            log(f"  판정 REJECT: {a.get('failed')}")
        else:
            pm = a["per_method"][estimator]
            expected = None
            if study:
                expected = hf.expected_error(study, a["front_jitter_rel"], estimator)
            dv = hf.derived(pm["f"], W, H, cfg["target_sizes"])
            removed = {}
            for r_ in a["reason"]:
                if r_:
                    removed[r_] = removed.get(r_, 0) + 1
            res.update({"verdict": pm["verdict"]["verdict"], "f_hand_px": pm["f"], "f_hand_at_grid_edge": a["solutions"][estimator]["at_grid_edge"], "f_rel_uncertainty": pm["f_rel_uncertainty"],
                        "f_fixed_template_px": a["solutions"]["fixed_template"]["f"], "f_joint_delta_px": a["solutions"]["joint_delta"]["f"],
                        "frontal_assumption_sensitivity": a["method_diff_rel"], "derived": dv, "expected_error_synthetic": expected,
                        "removed_frames": removed, "analysis": a, "anycalib": anyc})
            lr_name = lr_names.get(a["lr"]["majority_cls"]) if a.get("lr") else None
            ctx = {"video": video.name, "analysis": a, "estimator": estimator, "verdict": pm["verdict"]["verdict"], "W": W, "H": H, "anycalib": anyc,
                   "expected": expected, "derived": dv, "warnings": warnings, "removed": removed, "order_check": order, "lr_name": lr_name,
                   "min_target_frames": cfg["verdict"]["min_target_frames"]}
            hf.report_md(out / "report.md", ctx)
            hf.per_frame_csv(out / "per_frame.csv", frames, tr, a, estimator, anyc)
            res["figures"] = hf.report_figures(out, a, times, estimator, anyc)
            log(f"  f_hand ({estimator}) {pm['f']:.1f} px, 판정 {pm['verdict']['verdict']}, 휴리스틱 상대 불확실도 {100 * pm['f_rel_uncertainty']:.1f}%")
            for m_ in hf.METHODS:
                if a["solutions"][m_]["at_grid_edge"]:
                    log(f"  경고: {m_} 값은 f 격자 경계({a['solutions'][m_]['f']:.0f} px)입니다(내부 최솟값 없음)")
            log(f"  두 방식: fixed_template {res['f_fixed_template_px']:.1f} px, joint_delta {res['f_joint_delta_px']:.1f} px")
            if pm["verdict"]["verdict"] != "GOOD":
                for x in pm["advice"]:
                    log(f"  안내: {x}")
        hf.write_json(out / "results.json", res)
        small = {k: res.get(k) for k in ("video", "estimator", "verdict", "f_hand_px", "f_hand_at_grid_edge", "f_rel_uncertainty", "f_fixed_template_px", "f_joint_delta_px",
                                         "frontal_assumption_sensitivity")}
        if anyc:
            small["anycalib_median_px"] = {m: d["summary"]["median"] for m, d in anyc["models"].items()}
        small["timings_s"] = T
        (out / "result.yaml").write_text(yaml.safe_dump(hf.jsonable(small), allow_unicode=True, sort_keys=False))
        extra = {"video": video.name, "detector": {"weights": cfg["detector"]["weights"], "weights_sha256": hf.sha256_file(cfg["detector"]["weights"])
                                                   if cfg["detector"]["name"] == "wilor_yolo" else None, "info": det.get("info")},
                 "ffmpeg": norm["tools"], "anycalib": ({m: {"weight": d["weight"], "sha256": d["weight_sha256"]} for m, d in anyc["models"].items()}
                                                       | {"hub_check": anyc["hub_check"]}) if anyc else None,
                 "estimator": estimator, "estimator_basis": basis, "synthetic_study": study_path}
        hf.write_json(out / "metadata.json", hf.provenance_build(cfg_text, cfg_path, extra))
        log("  단계 시간(s): " + ", ".join(f"{k} {v:.1f}" for k, v in T.items()))
        return hf.summary_row(video.name, a, estimator, anyc, info["duration_s"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        log.close()


# ====================================================================================================
# run16  (was HandCalib/run_detector_swap.py)
# Experiment 16: 14's hand-template f pipeline with the 2D hand keypoint detector swapped.
#
#     (16's command line was removed; 17 calls swap_process for every video and detector)
#
# Per video: one normalised frame list (14's rules; checked against 14's frame hashes when 14 processed the video), every detector on
# the same PNGs (14 npz format), then 14's process_video with the mock (npz) adapter for each detector. f values are reported per
# detector; no differences or ratios between f values are computed. Profile minima on the extended grid are diagnostics only.
# ====================================================================================================

MANDATORY = "이 f는 핀홀·주점 중심·왜곡 없음 가정에서 2D 키포인트만으로 얻은 추정값이며 정답 데이터와 비교하지 않았다."


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


def detector_provenance(cfg16):
    out = {}
    for name in cfg16["detectors"]:
        s = cfg16["detector_settings"][name]
        w = Path(s["weights"])
        out[name] = {k: s.get(k) for k in ("package", "weights", "weights_url", "device", "license", "training_labels", "conf_min")}
        out[name]["model_name"] = s.get("model")
        out[name]["weights_sha256"] = hf.sha256_file(w) if w.exists() else None
        out[name]["python"] = s["python"]
        out[name]["pip_freeze"] = pip_freeze(s["python"])
    if "rtmpose_hand" in cfg16["detectors"]:
        z = Path(cfg16["detector_settings"]["rtmpose_hand"]["weights"]).parents[2].parent
        zips = list(z.glob("*.zip"))
        out["rtmpose_hand"]["zip_sha256"] = {p.name: hf.sha256_file(p) for p in zips}
    return out


# ---------------------------------------------------------------- one video
def frames_14(cfg16, video_name):
    p = cfg16["reuse_14"]["official"].get(video_name)
    if not p:
        return None, None
    d = hf.resolve(p)
    return d, json.loads((d / "results.json").read_text())


def swap_process(video: Path, cfg16, cfg_text, cfg_path, run_root: Path, study, study_path, use_anycalib, log):
    pcfg = cfg16["pipeline"]
    name = video.stem
    vdir = run_root / name
    vdir.mkdir(parents=True)
    tmp = Path(tempfile.mkdtemp(prefix="hf16_"))
    T = {}
    res = {"video": video.name, "detectors": {}, "warnings": []}
    try:
        t = time.time()
        info = hf.probe(video)
        if info["duration_s"] < float(pcfg["min_duration_s"]):
            res["warnings"].append(f"영상 길이 {info['duration_s']:.1f} s < {pcfg['min_duration_s']} s")
        norm = hf.normalize(video, tmp / "normalized.mkv", pcfg)
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        frames, k = hf.extract_frames(tmp / "normalized.mkv", tmp / "frames", norm["output"]["fps"], None, pcfg["frames"]["detect_fps"])
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
        dets = hf.swap_run_all(frames, cfg16, hf.resolve(cfg16["output"]["root"]) / "detection_cache", log)
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
            row = process_video(video, pc, pc_text, f"{cfg_path}#pipeline[{det}]", work, pcfg["estimate"]["estimator"],
                                      "14 합성 연구 규칙(fixed_template), 16 yaml에 고정", study, study_path, False)
            ddir = vdir / det
            shutil.move(str(work / name), str(ddir))
            r = json.loads((ddir / "results.json").read_text())
            prof, L0 = hf.swap_profiles(ddir, pc, cfg16)
            L0s[det] = L0
            (ddir / "profiles.json").write_text(json.dumps(hf.jsonable(prof), ensure_ascii=False))
            T[f"pipeline_{det}_s"] = time.time() - t
            res["detectors"][det] = summarize_detector(r, prof, L0, dd, s)
            figures_detector(ddir, prof, r, det)
            report_detector(ddir / "report.md", video.name, det, res["detectors"][det], s)
            log(console_line(det, res["detectors"][det]))
        res["diff2d"] = hf.swap_diff2d(L0s, pcfg, cfg16)
        figure_diff(vdir, res["diff2d"], cfg16)
        # AnyCalib: reuse 14's values when 14 processed this video, else run on the same PNGs
        t = time.time()
        if use_anycalib and pcfg["anycalib"]["enabled"]:
            if r14 and cfg16["reuse_14"]["reuse_anycalib"] and r14.get("anycalib"):
                res["anycalib"] = {"reused_from_14": str(d14), **{k: r14["anycalib"][k] for k in ("models", "hub_check", "skipped_extra_models")}}
            else:
                ac = hf.anycalib_run(frames, pcfg["anycalib"], float(pcfg["phases"]["front_phase_s"]), log)
                res["anycalib"] = {"reused_from_14": None, **{k: ac[k] for k in ("models", "hub_check", "skipped_extra_models")}}
        T["anycalib_s"] = time.time() - t
        res["shooting"] = shooting(res, L0s, W, H)
        res["size"] = [W, H]
        res["timings_s"] = T
        for line in swap_video_block(res):
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


def swap_video_block(res):
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
    hf._font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hf._font()
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


# ---------------------------------------------------------------- main


# ====================================================================================================
# run17  (was HandCalib/run_protocol_rerun.py)
# Experiment 17: 16's detector-swap pipeline (imported unchanged) on protocol-compliant videos.
#
#     python HandCalib/run.py protocol --config HandCalib/configs/17_protocol_rerun.yaml            # yaml detectors / videos
#     python HandCalib/run.py protocol --config ... results-yaml --run runs/17_protocol_rerun/<UTC>  # after all official runs
#
# Per video: shooting-condition check (14's indicators, yaml thresholds) first, then each detector's f (values only, no differences or
# ratios), the internal quality ranking (not accuracy), and AnyCalib as a separate value.
# ====================================================================================================

RANK_NOTE = "내부 품질 순위이며 정답 f와의 정확도를 뜻하지 않는다."


def select_videos(cfg, override=None):
    names = override or cfg.get("videos") or []
    d = Path(cfg["input_dir"]).expanduser()
    if names:
        vs = [d / n for n in names]
        missing = [str(v) for v in vs if not v.exists()]
        if missing:
            raise SystemExit(f"영상이 없습니다: {missing}")
        return vs
    exts = {e.lower() for e in cfg["video_extensions"]}
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


def det_line(det, d, rk=None):
    if "skipped" in d:
        return f"    {det:<16}: 건너뜀 ({d['skipped'][:80]})"
    if "fixed_template" not in d:
        return f"    {det:<16}: f = 값 없음 (분석 실패)   판정 {d.get('verdict')}"
    ft = d["fixed_template"]
    if ft["at_grid_edge"]:
        s = f"값 없음 ({ft['side']})"
    else:
        u = d["f_rel_uncertainty"]
        s = f"{ft['f']:.0f} px" + (f" (±{100 * u:.0f}%)" if u is not None and np.isfinite(u) else "")
    sh = (d.get("profiles", {}).get("flat17", {}).get("official", {}) or {}).get("sharpness")
    jd = d["joint_delta"]
    sj = f"값 없음 ({jd['side']})" if jd["at_grid_edge"] else f"{jd['f']:.0f} px"
    r = f"   순위 {rk['rank']:g}" if rk else ""
    return f"    {det:<16}: f = {s:<16} 판정 {d['verdict']:<7} 날카로움 {_n(sh)}   (joint_delta: {sj}){r}"


def protocol_video_block(v):
    c = v["conditions"]
    head = "가이드 충족" if c["guide"] else "가이드 미충족"
    L = [f"  [{v['video']}]{' (이전 영상)' if v['previous'] else ''}  촬영 조건: {head} ({hf.conditions_text(c['metrics'])})"]
    if not c["guide"]:
        L.append("    미충족 이유: " + "; ".join(c["reasons"]))
    for det, d in v["detectors"].items():
        L.append(det_line(det, d, v["ranking"].get(det)))
    ac = v.get("anycalib")
    if ac:
        parts = [f"{m.replace('anycalib_', '')}: 중앙값 {x['summary']['median']:.1f} px (IQR {x['summary']['iqr'][0]:.1f}–{x['summary']['iqr'][1]:.1f})"
                 for m, x in ac["models"].items()]
        L.append("    AnyCalib " + ", ".join(parts) + (" — 14 결과 재사용" if ac.get("reused_from_14") else ""))
    return L


def figure_conditions(vdir: Path, conds: dict, ccfg):
    plt = _plt()
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for det, m in conds.items():
        if "series" not in m:
            continue
        s = m["series"]
        t = np.array(s["t"])
        axs[0].plot(t, s["theta"], ".-", ms=3, lw=1, label=det)
        ax_ = np.array(s.get("axis", [np.nan] * len(t)), float)
        th = np.array(s["theta"], float)
        ax_[~(th >= 10)] = np.nan
        axs[1].plot(t, ax_, ".", ms=4, label=det)
        br = np.array(s["box_short_ratio"], float)
        axs[2].plot(t, br, "-", lw=1.2, label=det)
        e = np.array(s["edge"], bool)
        axs[2].plot(t[e], br[e], "x", ms=5, color="#c00")
    axs[0].axhline(ccfg["theta_p95_max_deg"], color="#444", ls="--", lw=0.8)
    axs[0].axvline(ccfg["transition_min_s"], color="#888", ls=":", lw=0.8)
    axs[0].set_ylabel("θ2d (°)")
    axs[1].set_ylabel("기울기(압축) 축 방향 (°)\nθ2d ≥ 10°만")
    axs[1].set_ylim(0, 180)
    axs[2].axhline(ccfg["box_short_ratio_max"], color="#444", ls="--", lw=0.8)
    axs[2].set_ylabel("손 박스 긴 변 / 짧은 변\n(× = 가장자리 접촉)")
    axs[2].set_xlabel("시간 (s)")
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    (vdir / "figures").mkdir(exist_ok=True)
    fig.savefig(vdir / "figures" / "shooting_conditions.png", dpi=110)
    plt.close(fig)


def protocol_process(video, cfg, cfg_text, cfg_path, run_root, study, study_path, use_anycalib, log):
    v = swap_process(video, cfg, cfg_text, cfg_path, run_root, study, study_path, use_anycalib, log)
    vdir = run_root / video.stem
    pcfg = cfg["pipeline"]
    conds = {}
    for det, d in v["detectors"].items():
        if "skipped" in d:
            continue
        tc = dict(pcfg["tracking"], conf_min=float(cfg["detector_settings"][det]["conf_min"]))
        conds[det] = hf.metrics_from_run(vdir / det, tc, float(pcfg["phases"]["front_skip_s"]))
    ref = cfg["conditions"]["reference_detector"]
    if ref not in conds:
        ref = next(iter(conds), None)
    m = conds.get(ref, {"failed": "검출기 없음"})
    chk = hf.conditions_check(m, cfg["conditions"])
    v["conditions"] = {"reference": ref, "metrics": {k: x for k, x in m.items() if k != "series"}, **chk,
                       "by_detector": {d: {"metrics": {k: x for k, x in mm.items() if k != "series"}, **hf.conditions_check(mm, cfg["conditions"])}
                                       for d, mm in conds.items()}}
    v["previous"] = video.name in set(cfg.get("previous_videos") or [])
    v["ranking"] = hf.rank(v["detectors"], cfg["ranking"])
    figure_conditions(vdir, conds, cfg["conditions"])
    (vdir / "conditions_series.json").write_text(json.dumps(hf.jsonable({d: mm.get("series") for d, mm in conds.items()})))
    for line in protocol_video_block(v):
        log(line)
    return v


def protocol_report(path, vids, rsum, cfg, prov):
    c = cfg["conditions"]
    L = ["# 실험 17: 프로토콜 준수 영상으로 검출기 3종 재실행", "", MANDATORY, "",
         "f 값끼리 또는 외부 값과의 차이·비율은 계산하지 않았다. f가 탐색 범위 경계에 붙은 경우 '값 없음'으로 적고, 부트스트랩·일관성은 '계산 불가'로 적는다.", "",
         "## 1. 촬영 조건 점검 (검출기 결과보다 먼저, 문턱값은 실행 전 yaml에 고정)", "",
         f"가이드: 손 박스/짧은 변 중앙값 ≤ {c['box_short_ratio_max']}, 가장자리 접촉 ≤ {100 * c['edge_rate_max']:.0f}%, 자동 전환 ≥ {c['transition_min_s']} s, "
         f"θ2d p95 ≤ {c['theta_p95_max_deg']:.0f}°, 기울기 구간 θ2d 중앙값 {c['theta_tilt_median_range_deg'][0]:.0f}–{c['theta_tilt_median_range_deg'][1]:.0f}°, "
         f"축 덮임 ≥ {c['axis_bins_min']}/6, 손가락 펴짐 ≥ {c['finger_extension_min']}. 판정은 기준 검출기({c['reference_detector']}) 지표로 한다.", "",
         "| 영상 | 기준 검출기 판정 | 박스/짧은 변 | 가장자리 | 자동 전환 (s) | 정면 지터/박스 | θ2d p95 | 기울기 중앙값 | θ2d 최대 | 축 덮임 | 손가락 펴짐 | 미충족 이유 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in vids:
        m = v["conditions"]["metrics"]
        if "failed" in m:
            L.append(f"| {v['video']} | 분석 실패 | | | | | | | | | | {m['failed']} |")
            continue
        L.append(f"| {v['video']}{' (이전)' if v['previous'] else ''} | {'충족' if v['conditions']['guide'] else '미충족'} | {hf.cond_f(m['box_short_ratio'], 2)} | "
                 f"{hf.cond_p(m['edge_rate'])} | {'없음' if m['transition_s'] is None else format(m['transition_s'], '.2f')} | {hf.cond_p(m['front_jitter_rel'], 2)} | "
                 f"{hf.cond_f(m['theta_p95_deg'], 0)}° | {hf.cond_f(m['theta_tilt_median_deg'], 0)}° | {hf.cond_f(m['theta_max_deg'], 0)}° | {m['axis_bins']}/6 | "
                 f"{hf.cond_f(m['finger_extension_min'], 3)} | {'; '.join(v['conditions']['reasons']) or '-'} |")
    L += ["", "검출기별 같은 지표:", "", "| 영상 | 검출기 | 충족 | 지표 |", "|---|---|---|---|"]
    for v in vids:
        for d, x in v["conditions"]["by_detector"].items():
            L.append(f"| {v['video']} | {d} | {'충족' if x['guide'] else '미충족'} | {hf.conditions_text(x['metrics'])} |")
    L += ["", "## 2. 영상별 결과", ""]
    for v in vids:
        L += ["```"] + protocol_video_block(v) + ["```", ""]
    L += ["## 3. 요약 표 (영상별)", "", "| 영상 | 촬영 조건 | 검출기 | f (fixed_template) | 불확실도 | 판정 | 날카로움(flat17) | joint_delta f | 품질 순위 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in vids:
        g = "충족" if v["conditions"]["guide"] else "미충족"
        for det, d in v["detectors"].items():
            rk = v["ranking"].get(det, {})
            if "fixed_template" not in d:
                L.append(f"| {v['video']} | {g} | {det} | 값 없음 | - | {d.get('verdict', '-')} | - | - | {rk.get('rank', '-')} |")
                continue
            ft, jd = d["fixed_template"], d["joint_delta"]
            sh = d.get("profiles", {}).get("flat17", {}).get("official", {}).get("sharpness")
            L.append(f"| {v['video']} | {g} | {det} | {f_text(ft, d['grid_px'][1])} | "
                     f"{'계산 불가' if ft['at_grid_edge'] else spread_text(d['f_rel_uncertainty'], False)} | {d['verdict']} | {_n(sh)} | "
                     f"{f_text(jd, d['grid_px'][1])} | {rk.get('rank', '-'):g} |")
        ac = v.get("anycalib")
        if ac:
            L.append(f"| {v['video']} | {g} | AnyCalib(별도) | " + ", ".join(f"{m}: 중앙값 {x['summary']['median']:.1f} px" for m, x in ac["models"].items())
                     + " | | | | | |")
    L += ["", "## 4. 검출기 품질 순위 (내부 지표)", "", f"**{RANK_NOTE}**", "",
          "규칙(실행 전 yaml에 고정): ① 범주 GOOD > ROUGH > REJECT > 경계(f가 탐색 범위 경계) > 실패 → ② flat17 공식 범위 프로파일 날카로움(높을수록) → "
          "③ f_rel_uncertainty(낮을수록, 계산 불가는 최하) → ④ 재투영 RMS/손 박스(낮을수록) → ⑤ 검출률(높을수록). 완전 동점은 순위의 평균.", "",
          "| 검출기 | 가이드 충족 영상의 평균 순위 | 해당 영상 수 | GOOD | ROUGH | REJECT | 경계에 붙은 횟수 | 실패 |", "|---|---|---|---|---|---|---|---|"]
    for det, s in rsum.items():
        vc = s["verdict_counts"]
        L.append(f"| {det} | {'-' if s['mean_rank_guide'] is None else format(s['mean_rank_guide'], '.2f')} | {s['n_guide_videos']} | {vc['GOOD']} | {vc['ROUGH']} | "
                 f"{vc['REJECT']} | {s['n_edge']} | {s['n_failed']} |")
    if all(s["n_guide_videos"] == 0 for s in rsum.values()):
        L.append("\n가이드를 충족한 영상이 없어 평균 순위를 내지 않았다.")
    L += ["", "영상별 순위:", "", "| 영상 | 검출기 | 순위 | 범주 | 날카로움 | 불확실도 | 재투영 RMS/박스 | 검출률 |", "|---|---|---|---|---|---|---|---|"]
    for v in vids:
        for det, r in sorted(v["ranking"].items(), key=lambda kv: kv[1]["rank"]):
            k = r["keys"]
            L.append(f"| {v['video']} | {det} | {r['rank']:g} | {r['category']} | {_n(k['sharpness'])} | "
                     f"{'계산 불가' if k['f_rel_uncertainty'] is None else format(100 * k['f_rel_uncertainty'], '.1f') + '%'} | "
                     f"{'-' if k['reproj_rms_rel'] is None else format(100 * k['reproj_rms_rel'], '.2f') + '%'} | {_n(k['detect_rate'])} |")
    L += ["", "## 5. 관절 집합별 비용 프로파일 (진단, f 값 아님)", "", "| 영상 | 검출기 | 관절 집합 | 공식 범위 | 위치(px) | 날카로움 | 연장 격자(0.2–5×) | 위치(px) |",
          "|---|---|---|---|---|---|---|---|"]
    ko = {"interior": "내부", "upper": "상한", "lower": "하한", "none": "-"}
    for v in vids:
        for det, d in v["detectors"].items():
            for k, x in d.get("profiles", {}).items():
                o, e = x["official"], x["extended"]
                L.append(f"| {v['video']} | {det} | {k} | {ko[o['at']]} | {_px(o['min_at_px'])} | {_n(o['sharpness'])} | {ko[e['at']]} | {_px(e['min_at_px'])} |")
    L += ["", "## 6. AnyCalib (검출기와 무관한 별도 값)", "", "| 영상 | 모델 | 중앙값 | IQR | p10–p90 | 정면 구간 중앙값 | 기울기 구간 중앙값 | 변동계수 | 출처 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in vids:
        ac = v.get("anycalib")
        if not ac:
            continue
        for mname, x in ac["models"].items():
            s = x["summary"]
            L.append(f"| {v['video']} | {mname} | {s['median']:.1f} | {s['iqr'][0]:.1f}–{s['iqr'][1]:.1f} | {s['p10_p90'][0]:.1f}–{s['p10_p90'][1]:.1f} | "
                     f"{hf.fmt(s['front_median'])} | {hf.fmt(s['tilt_median'])} | {100 * s['cv']:.1f}% | {'14 재사용' if ac.get('reused_from_14') else '이번 실행'} |")
    L += ["", "## 7. 검출기 정보", ""]
    for det, p in prov.items():
        L.append(f"- **{det}**: {p.get('model_name')}; 가중치 sha256 {str(p.get('weights_sha256'))[:16]}…; 실행 장치 {p.get('device')}; 라이선스 {p.get('license')}; "
                 f"학습 라벨: {p.get('training_labels')}")
    L += ["", "## 8. 한계", "",
          "- 정답 f가 없어 오차를 계산하지 않았다. 품질 순위는 내부 지표 순위이며 정확도 순위가 아니다.",
          "- 촬영 조건은 기준 검출기의 2D 키포인트로 잰 값이다(θ2d는 2D 아핀 지표).",
          "- 박스 정의가 검출기마다 다르다(16과 같음). 파이프라인 설정과 판정 문턱값은 모든 검출기에 같다.", ""]
    path.write_text("\n".join(L) + "\n")


def protocol_results_yaml(run_dir: Path, cfg):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    out = {"experiment": "17_protocol_rerun", "run": str(run_dir.relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "note": MANDATORY + " f 값끼리·외부 값과의 비교 수치 없음. " + RANK_NOTE, "videos": {}}
    for v in r["videos"]:
        e = {"previous_video": v["previous"], "guide_satisfied": v["conditions"]["guide"], "guide_reasons": v["conditions"]["reasons"], "detectors": {}}
        for det, d in v["detectors"].items():
            if "fixed_template" not in d:
                e["detectors"][det] = {"f_px": None, "status": d.get("skipped") or d.get("failed"), "verdict": d.get("verdict")}
                continue
            ft = d["fixed_template"]
            sh = d.get("profiles", {}).get("flat17", {}).get("official", {}).get("sharpness")
            e["detectors"][det] = {"f_fixed_template_px": None if ft["at_grid_edge"] else round(ft["f"], 1),
                                   "f_status": f"값 없음 ({ft['side']})" if ft["at_grid_edge"] else "value",
                                   "f_joint_delta_px": None if d["joint_delta"]["at_grid_edge"] else round(d["joint_delta"]["f"], 1),
                                   "f_rel_uncertainty": None if ft["at_grid_edge"] else round(d["f_rel_uncertainty"], 4),
                                   "verdict": d["verdict"], "sharpness_flat17": None if sh is None else round(sh, 4),
                                   "quality_rank": v["ranking"][det]["rank"]}
        if v.get("anycalib"):
            e["anycalib_median_px"] = {k: round(x["summary"]["median"], 1) for k, x in v["anycalib"]["models"].items()}
        out["videos"][v["video"]] = e
    out["ranking_summary"] = r["ranking_summary"]
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def protocol_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/17_protocol_rerun.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list overriding yaml videos")
    ap.add_argument("--detectors", default=None, help="comma list overriding yaml detectors")
    ap.add_argument("--no-anycalib", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(protocol_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.detectors:
        cfg["detectors"] = a.detectors.split(",")
    videos = select_videos(cfg, a.videos.split(",") if a.videos else None)
    sp = hf.resolve(cfg["reuse_14"]["synthetic_study"])
    study = json.loads(sp.read_text())
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    log = Log(run_root / "log.txt")
    log(f"17 프로토콜 재실행: 검출기 {cfg['detectors']}, 영상 {[v.name for v in videos]} → {run_root}")
    t0 = time.time()
    vids = [protocol_process(v, cfg, text, a.config, run_root, study, str(sp), not a.no_anycalib, log) for v in videos]
    guide = {v["video"]: v["conditions"]["guide"] for v in vids}
    rsum = hf.rank_summary({v["video"]: v["ranking"] for v in vids}, guide)
    prov = detector_provenance(cfg)
    protocol_report(run_root / "report.md", vids, rsum, cfg, prov)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": vids, "ranking_summary": rsum, "detectors": cfg["detectors"]}),
                                                      ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_16": cfg["code_16"], "detectors": prov, "elapsed_s": time.time() - t0,
                                     "devices": {k: cfg["detector_settings"][k].get("device") for k in cfg["detectors"]}})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log("")
    log("검출기별 요약 (" + RANK_NOTE + ")")
    for det, s in rsum.items():
        log(f"  {det}: 가이드 충족 영상 평균 순위 {'-' if s['mean_rank_guide'] is None else format(s['mean_rank_guide'], '.2f')} "
            f"({s['n_guide_videos']}개), 판정 {s['verdict_counts']}, 경계 {s['n_edge']}회")
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, vids


# ====================================================================================================
# runff  (was HandCalib/run_frame_filter_diag.py)
# Diagnostic: the 17 pipeline with the tilt-phase frame filter (theta2d <= 45 deg, box off the border), on the 17 runs' detections.
#
#     python HandCalib/run.py frame-filter --config HandCalib/configs/17_frame_filter_diag.yaml [--source VIDEO=RUN_DIR]
#
# Not an official estimator. The comparison with the user's checkerboard f (reference_f_px) is made only in this diagnostic.
# ====================================================================================================

FF_HEAD = "진단 결과이며 공식 f 추정값이 아님. 체커보드 f와의 비교는 이 진단에서만 한다."
FF_KO = {"interior": "내부", "upper": "상한", "lower": "하한"}


def ff_fval(sol, hi):
    if sol is None:
        return "값 없음 (실패)"
    if sol["at_grid_edge"]:
        return "값 없음 (상한)" if sol["f"] >= hi * (1 - 1e-6) else "값 없음 (하한)"
    return f"{sol['f']:.1f} px"


def ff_rel(sol, ref, hi):
    if sol is None:
        return "-"
    if sol["at_grid_edge"]:
        return f"{'≥ +' if sol['f'] >= hi * (1 - 1e-6) else '≤ '}{100 * (sol['f'] / ref - 1):.0f}%"
    return f"{100 * (sol['f'] / ref - 1):+.1f}%"


def ff_official(run_dir: Path):
    r = json.loads((run_dir / "results.json").read_text())
    a = r.get("analysis", {})
    if "solutions" not in a:
        return None, None, r.get("verdict")
    return a["solutions"]["fixed_template"], a["solutions"]["joint_delta"], r["verdict"]


def ff_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", action="append", default=None, help="VIDEO=RUN_DIR; replaces the yaml sources (repeatable)")
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.source:
        cfg["sources"] = dict(x.split("=", 1) for x in a.source)
    c17 = yaml.safe_load(hf.resolve(cfg["pipeline_config"]).read_text())
    out = hf.resolve(cfg["output"]["root"]) / f"{utc_stamp()}_diag_framefilter"
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    ref = float(cfg["reference_f_px"])
    log = Log(out / "log.txt")
    log(f"프레임 거르기 진단 → {out}")
    res = {}
    t0 = time.time()
    for video, src in cfg["sources"].items():
        res[video] = {}
        for det in cfg["detectors"]:
            pc = copy.deepcopy(c17["pipeline"])
            pc["tracking"]["conf_min"] = float(c17["detector_settings"][det]["conf_min"])
            rd = hf.resolve(src) / det
            t = time.time()
            r = hf.ff_run_one(rd, pc, cfg["filter"])
            ft0, jd0, v0 = ff_official(rd)
            r["official"] = {"fixed_template": ft0, "joint_delta": jd0, "verdict": v0}
            res[video][det] = r
            if "failed" in r:
                log(f"  {video} {det}: 실패 ({r['failed']})")
                continue
            hi = r["grid_px"][1]
            pm = r["per_method"]["fixed_template"]
            log(f"  {video} {det}: 기울기 프레임 {r['n_tilt_kept']}/{r['n_tilt_before']} 유지, 템플릿 동일 {r['same_template']}, "
                f"f {ff_fval(r['solutions']['fixed_template'], hi)} (판정 {pm['verdict']}), 거르기 전 {ff_fval(ft0, hi)} — {time.time() - t:.0f} s")
    ff_figure(out, res, ref)
    lines = ff_summary(res, cfg, ref)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "diagnostic.json").write_text(json.dumps(hf.jsonable(res), ensure_ascii=False))
    (out / "metadata.json").write_text(json.dumps(hf.jsonable(hf.provenance_build(text, a.config, {"sources": cfg["sources"], "pipeline_config_sha256": hf.sha256_text(hf.resolve(cfg["pipeline_config"]).read_text()),
                                                                                         "elapsed_s": time.time() - t0})), ensure_ascii=False, indent=1))
    for ln in lines:
        log(ln)
    log.close()
    return out, res


def ff_summary(res, cfg, ref):
    f = cfg["filter"]
    L = ["# 17 진단: 기울기 구간 프레임 거르기", "", f"**{FF_HEAD}**", "",
         f"- 규칙(실행 전 yaml에 고정, 정답과 무관한 가이드 문턱값): 기울기 구간(3초 이후) 프레임 중 θ2d ≤ {f['theta_max_deg']:.0f}°이고 손 박스가 화면 가장자리"
         f"({f['edge_margin_px']:.0f} px)에 닿지 않는 프레임만 사용. 정면 창·템플릿·파이프라인 설정·판정 문턱값은 17과 같음. 판정의 검출률은 원래 추적 검출률.",
         f"- 입력: 17 공식 실행의 검출 결과(새 검출 없음). 비교 기준: 사용자 체커보드 f = {ref:.0f} px (1080p, 같은 렌즈·설정). 추정·판정에는 쓰지 않음.", "",
         "## 1. 거르기 전후 (fixed_template)", "",
         "| 영상 | 검출기 | 기울기 프레임 유지 | 거르기 전 f | 1680 대비 | 거른 뒤 f | 1680 대비 | 거른 뒤 판정 | 불확실도 | 날카로움 | 재투영 RMS/박스 | 거른 뒤 joint_delta f |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, dd in res.items():
        for det, r in dd.items():
            if "failed" in r:
                L.append(f"| {v} | {det} | {r.get('n_kept_tilt', '-')} | | | 실패 ({r['failed']}) | | | | | | |")
                continue
            hi = r["grid_px"][1]
            pm = r["per_method"]["fixed_template"]
            ft, ft0 = r["solutions"]["fixed_template"], r["official"]["fixed_template"]
            edge = ft["at_grid_edge"]
            L.append(f"| {v} | {det} | {r['n_tilt_kept']}/{r['n_tilt_before']} | {ff_fval(ft0, hi)} | {ff_rel(ft0, ref, hi)} | **{ff_fval(ft, hi)}** | {ff_rel(ft, ref, hi)} | "
                     f"{pm['verdict']} | {'계산 불가' if edge else format(100 * pm['f_rel_uncertainty'], '.0f') + '%'} | {pm['sharpness']:.3f} | "
                     f"{100 * pm['reproj_rms_rel']:.2f}% | {ff_fval(r['solutions']['joint_delta'], hi)} |")
    L += ["", "## 2. 거른 뒤 관절 집합별 비용 프로파일 최솟값 (연장 격자 0.2–5×긴 변, 진단)", "",
          "| 영상 | 검출기 | palm4 | palm5 | palm_proximal9 | flat17 |", "|---|---|---|---|---|---|"]
    for v, dd in res.items():
        for det, r in dd.items():
            if "profiles" not in r:
                continue
            cells = []
            for s in ("palm4", "palm5", "palm_proximal9", "flat17"):
                e = r["profiles"][s]["extended"]
                cells.append(f"{e['min_at_px']:.0f} ({100 * (e['min_at_px'] / ref - 1):+.0f}%){' 끝' if e['at'] != 'interior' else ''}")
            L.append(f"| {v} | {det} | " + " | ".join(cells) + " |")
    L += ["", "## 3. 판정 실패 항목 (거른 뒤)", ""]
    for v, dd in res.items():
        for det, r in dd.items():
            if "per_method" in r:
                pm = r["per_method"]["fixed_template"]
                L.append(f"- {v} {det}: {pm['verdict']} — GOOD 실패 {', '.join(pm['failed_good']) or '없음'}; ROUGH 실패 {', '.join(pm['failed_rough']) or '없음'}")
    L += ["", "## 한계", "", "- 진단이며 공식 추정 방식이 아니다. 영상 4개, 그중 가이드 충족 영상은 없다.",
          "- 체커보드 f는 사용자 제공 값이며, 이 진단의 비교에만 썼다.", ""]
    return L


def ff_figure(out, res, ref):
    plt = _plt()
    vids = list(res)
    dets = list(next(iter(res.values())))
    fig, axs = plt.subplots(len(vids), len(dets), figsize=(4.2 * len(dets), 2.8 * len(vids)), squeeze=False)
    for i, v in enumerate(vids):
        for j, d in enumerate(dets):
            ax = axs[i][j]
            r = res[v][d]
            if "profiles" in r:
                e = r["profiles"]["flat17"]["extended"]
                p = np.array(e["profile"], float)
                ax.plot(e["grid"], p / np.nanmin(p), color="#2a6fbb", lw=1.5, label="거른 뒤")
                o = r["profiles"]["flat17"]["official"]["grid"]
                ax.axvspan(o[0], o[-1], color="#999", alpha=0.12)
            ax.axvline(ref, color="#c00", ls="--", lw=1, label=f"체커보드 {ref:.0f}")
            ax.set_xscale("log")
            ax.set_ylim(0.95, 3)
            ax.set_title(f"{v} / {d}", fontsize=8)
            ax.grid(alpha=0.3, which="both")
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "figures" / "flat17_profiles_filtered.png", dpi=100)
    plt.close(fig)


# ====================================================================================================
# run18  (was HandCalib/run_same_session.py)
# Experiment 18: one recording with a hand segment and a checkerboard segment.
#
#     CUDA_VISIBLE_DEVICES=<free gpu> python HandCalib/run.py same-session --config HandCalib/configs/18_same_session_check.yaml
#     python HandCalib/run.py same-session --config ... results-yaml --run runs/18_same_session_check/<UTC>
#
# Hand segment: 17's process (imported unchanged) on the cut of the normalised video. Checkerboard segment: this recording's own
# reference f (models A-D median, plus the pinhole no-distortion model E). The checkerboard f is never an input to the hand method or
# AnyCalib; the comparisons are written to summary.md only.
# ====================================================================================================

def gpu_info():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:                                             # noqa: BLE001
        q = []
    return {"CUDA_VISIBLE_DEVICES": vis, "gpus_at_start": q}


# ---------------------------------------------------------------- segmentation
def session_scan(norm_video: Path, fps: float, cfg, tmp: Path, cache_dir: Path, log):
    sc = cfg["segmentation"]
    frames, k = hf.extract_frames(norm_video, tmp / "scan", fps, None, sc["scan_fps"])
    times = np.array([r["time_s"] for r in frames])
    board = np.array([hf.board_found(cv2.imread(r["png"], cv2.IMREAD_GRAYSCALE), cfg["checkerboard"]["inner_corners"], sc["board_detect_long_side"])
                      for r in frames])
    s = cfg["detector_settings"]["wilor_yolo"]
    dcfg = {"python": s["python"], "weights": s["weights"], "conf": s["conf"], "iou": s["iou"], "imgsz": s.get("imgsz"), "device": s["device"], "cache": True}
    det = hf.detector_wilor_yolo(frames, dcfg, cache_dir, log)
    hand = np.zeros(len(frames), bool)
    for i in range(len(frames)):
        a, b = int(det["offsets"][i]), int(det["offsets"][i + 1])
        hand[i] = bool(b > a and (det["box_conf"][a:b] >= sc["hand_min_box_conf"]).any())
    shutil.rmtree(tmp / "scan", ignore_errors=True)
    return times, hand, board


def figure_timeline(path: Path, times, hand, board, seg):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(times, hand * 1.0 + 0.02, "|", ms=10, color="#2a6fbb", label="손 검출(wilor_yolo)")
    ax.plot(times, board * 1.0 - 0.02, "|", ms=10, color="#c47a00", label="체커보드 검출")
    if seg.get("hand"):
        ax.axvspan(*seg["hand"], color="#2a6fbb", alpha=0.12, label="손 구간")
    if seg.get("board"):
        ax.axvspan(*seg["board"], color="#c47a00", alpha=0.12, label="체커보드 구간")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["없음", "있음"])
    ax.set_xlabel("시간 (s)")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figures_board(d: Path, b: dict):
    if "models" not in b:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    im = axs[0].imshow(np.array(b["coverage_cells"]) > 0, cmap="Greens", vmin=0, vmax=1)
    axs[0].set_title(f"격자 덮임 {100 * b['metrics']['coverage']:.0f}% (8×8)")
    axs[1].hist(b["tilts_deg"], bins=15, color="#2a6fbb")
    axs[1].set_xlabel("보드 기울기 (°)")
    axs[1].set_title(f"p90 {b['metrics']['tilt_p90_deg']:.0f}°, 최대 {b['metrics']['tilt_max_deg']:.0f}°")
    ms = list(b["models"])
    axs[2].bar(ms, [b["models"][m]["fx"] for m in ms], color=["#2a6fbb"] * (len(ms) - 1) + ["#c47a00"])
    axs[2].axhline(b["f_reference_px"], color="#444", ls="--", lw=1)
    lo = min(b["models"][m]["fx"] for m in ms)
    hi = max(b["models"][m]["fx"] for m in ms)
    axs[2].set_ylim(lo - 0.05 * lo, hi + 0.05 * hi)
    axs[2].set_title("모델별 fx (점선: A–D 중앙값)")
    fig.tight_layout()
    fig.savefig(d / "checkerboard_figures.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- one video
def session_board_frames(norm_video: Path, fps, seg, ccfg, tmp: Path):
    rows, k = hf.extract_frames(norm_video, tmp / "board", fps, None, ccfg["sample_fps"])
    out = []
    for r in rows:
        if seg[0] <= r["time_s"] <= seg[1]:
            out.append({"index": r["index"], "time_s": r["time_s"], "gray": cv2.imread(r["png"], cv2.IMREAD_GRAYSCALE)})
    shutil.rmtree(tmp / "board", ignore_errors=True)
    return out


def session_process(video: Path, cfg, cfg_text, cfg_path, run_root: Path, study, study_path, log):
    pcfg = cfg["pipeline"]
    tmp = Path(tempfile.mkdtemp(prefix="hf18_"))
    T = {}
    try:
        t = time.time()
        norm = hf.normalize(video, tmp / "normalized.mkv", pcfg)
        W, H, fps = norm["output"]["display_w"], norm["output"]["display_h"], norm["output"]["fps"]
        T["normalize_s"] = time.time() - t
        t = time.time()
        if cfg["segment_mode"] == "manual":
            m = cfg["manual_segments"][video.name]
            seg = {"hand": list(m["hand"]), "board": list(m["board"]), "warnings": [], "mode": "manual"}
            times = hand = board = None
        else:
            times, hand, board = session_scan(tmp / "normalized.mkv", fps, cfg, tmp, hf.resolve(cfg["output"]["root"]) / "detection_cache", log)
            seg = hf.segment_split(times, hand, board, cfg["segmentation"]["hand_gap_s"])
            seg["mode"] = "auto"
        T["segment_s"] = time.time() - t
        log(f"== {video.name} == 구간: 손 {seg['hand'][0]:.1f}–{seg['hand'][1]:.1f} s, 체커보드 "
            + (f"{seg['board'][0]:.1f}–{seg['board'][1]:.1f} s" if seg["board"] else "없음") + ("; 경고: " + "; ".join(seg["warnings"]) if seg["warnings"] else ""))
        # hand segment: exact cut of the normalised video, named like the original so 17's folders read <video>/<detector>
        t = time.time()
        hdir = tmp / "hand"
        hdir.mkdir()
        hand_cut = hdir / (video.stem + ".mkv")
        cut_info = hf.cut_segment(tmp / "normalized.mkv", hand_cut, seg["hand"][0], seg["hand"][1], fps)
        v17 = protocol_process(hand_cut, cfg, cfg_text, cfg_path, run_root, study, study_path, True, log)
        T["hand_s"] = time.time() - t
        vdir = run_root / video.stem
        if times is not None:
            (vdir / "figures").mkdir(exist_ok=True)
            figure_timeline(vdir / "figures" / "segments_timeline.png", times, hand, board, seg)
        # checkerboard segment
        t = time.time()
        bres = {"failed": "체커보드 구간 없음", "verdict": "REJECT"}
        if seg["board"]:
            fr = session_board_frames(tmp / "normalized.mkv", fps, seg["board"], cfg["checkerboard"], tmp)
            bres = hf.board_analyse(fr, cfg["checkerboard"], W, H, log)
        bdir = vdir / "checkerboard"
        bdir.mkdir(exist_ok=True)
        (bdir / "checkerboard.json").write_text(json.dumps(hf.jsonable(bres), ensure_ascii=False))
        figures_board(bdir, bres)
        T["board_s"] = time.time() - t
        res = {"video": video.name, "size": [W, H], "fps": fps, "normalization": {k: norm[k] for k in ("output", "path")}, "segments": seg,
               "scan": None if times is None else {"t": times.tolist(), "hand": hand.tolist(), "board": board.tolist()},
               "hand_cut": cut_info, "hand": v17, "checkerboard": bres, "timings_s": T}
        for ln in session_block(res):
            log(ln)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def board_line(b):
    if "f_reference_px" not in b:
        return f"    체커보드 기준 f   : 값 없음 ({b.get('failed')})"
    unc = max(b["metrics"]["model_range"] / 2, 2 * b["metrics"]["bootstrap_rel_std"], b["metrics"]["segment_range"] / 2)
    return (f"    체커보드 기준 f   : {b['f_reference_px']:.0f} px (모델 범위 {100 * b['metrics']['model_range']:.1f}%, 부트스트랩 {100 * b['metrics']['bootstrap_rel_std']:.2f}%, "
            f"구간 범위 {100 * b['metrics']['segment_range']:.1f}%)  판정 {b['verdict']},  모델 E(핀홀 무왜곡): {b['f_model_E_px']:.0f} px")


def session_block(res):
    s = res["segments"]
    L = [f"  [{res['video']}]  구간: 손 {s['hand'][0]:.1f}–{s['hand'][1]:.1f} s, 체커보드 "
         + (f"{s['board'][0]:.1f}–{s['board'][1]:.1f} s" if s["board"] else "없음"), board_line(res["checkerboard"])]
    v = res["hand"]
    for det, d in v["detectors"].items():
        L.append(det_line(det, d, v["ranking"].get(det)))
    ac = v.get("anycalib")
    if ac:
        L.append("    AnyCalib " + ", ".join(f"{m.replace('anycalib_', '')}: 중앙값 {x['summary']['median']:.1f} px (IQR {x['summary']['iqr'][0]:.1f}–{x['summary']['iqr'][1]:.1f})"
                                           for m, x in ac["models"].items()))
    c = v["conditions"]
    L.append(f"    손 구간 촬영 조건: {'가이드 충족' if c['guide'] else '가이드 미충족'} ({hf.conditions_text(c['metrics'])})")
    if c["reasons"]:
        L.append("      미충족 이유: " + "; ".join(c["reasons"]))
    return L


# ---------------------------------------------------------------- reports
def session_report(path, vids, cfg, prov, gpu):
    L = ["# 실험 18: 같은 녹화의 체커보드로 기준 f 확인", "", MANDATORY, "",
         "체커보드 기준 f는 비교용이며 손 방식·AnyCalib의 입력·보정·초기값으로 쓰지 않았다. 비교 수치는 summary.md에만 적었다.", "",
         "## 1. 영상별 결과", ""]
    for v in vids:
        L += ["```"] + session_block(v) + ["```", ""]
        if v["segments"]["warnings"]:
            L.append("구간 분할 경고: " + "; ".join(v["segments"]["warnings"]))
            L.append("")
    L += ["## 2. 요약 표", "", "| 영상 | 체커보드 기준 f (A–D 중앙값) | 모델 E f | 체커보드 판정 | wilor_yolo | mediapipe_hands | rtmpose_hand | AnyCalib pinhole | AnyCalib gen |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in vids:
        b = v["checkerboard"]
        cells = []
        for det in cfg["detectors"]:
            d = v["hand"]["detectors"].get(det, {})
            if "fixed_template" not in d:
                cells.append("값 없음")
            else:
                cells.append(f"{f_text(d['fixed_template'], d['grid_px'][1])} ({d['verdict']})")
        ac = v["hand"].get("anycalib") or {"models": {}}
        am = {m: f"{x['summary']['median']:.1f} px" for m, x in ac["models"].items()}
        L.append(f"| {v['video']} | {'-' if 'f_reference_px' not in b else format(b['f_reference_px'], '.1f') + ' px'} | "
                 f"{'-' if 'f_model_E_px' not in b or b['f_model_E_px'] is None else format(b['f_model_E_px'], '.1f') + ' px'} | {b['verdict']} | "
                 + " | ".join(cells) + f" | {am.get('anycalib_pinhole', '-')} | {am.get('anycalib_gen', '-')} |")
    L += ["", "## 3. 체커보드 보정 상세", ""]
    for v in vids:
        b = v["checkerboard"]
        if "models" not in b:
            L.append(f"- {v['video']}: {b.get('failed')}")
            continue
        m = b["metrics"]
        L += [f"### {v['video']}", "",
              f"- 표본: {b['n_sampled']}장 중 검출 {b['n_detected']}장, 선택 {b['n_selected']}장 (가장 먼 점 우선 표집), 패턴 {b['pattern_size']}",
              f"- 지표: 보드 기울기 p90 {m['tilt_p90_deg']:.1f}°, 최대 {m['tilt_max_deg']:.1f}°, 구별되는 자세 {m['distinct_poses']}개, 격자 덮임 {100 * m['coverage']:.0f}%, "
              f"재투영 RMS {m['rms_px']:.3f} px, 모델 범위 {100 * m['model_range']:.2f}%, 부트스트랩 상대 표준편차 {100 * m['bootstrap_rel_std']:.2f}%, 시간 구간 범위 {100 * m['segment_range']:.2f}%",
              f"- 판정 {b['verdict']} (GOOD 실패 항목: {', '.join(b['failed_good']) or '없음'}; ROUGH 실패 항목: {', '.join(b['failed_rough']) or '없음'})", "",
              "| 모델 | fx | fy | cx | cy | k1 | k2 | RMS (px) |", "|---|---|---|---|---|---|---|---|"]
        for k, x in b["models"].items():
            L.append(f"| {k} | {x['fx']:.1f} | {x['fy']:.1f} | {x['cx']:.1f} | {x['cy']:.1f} | {x['dist'][0]:+.4f} | {x['dist'][1]:+.4f} | {x['rms']:.3f} |")
        L.append(f"\n시간 구간별 f_reference: " + ", ".join(hf.fmt(x) for x in b["segment_f"]))
        L.append("")
    L += ["## 4. 방법 메모", "",
          "- 모든 f는 표준 규격(긴 변 1920)으로 변환한 같은 영상의 픽셀 기준이다(손 구간과 체커보드 구간 모두 같은 변환 프레임).",
          "- 체커보드: 초당 10장 표본 → findChessboardCornersSB(긴 변 960 축소) → 원해상도 cornerSubPix → 자세 기술자(법선·중심·크기)의 가장 먼 점 우선 표집 최대 80장 → "
          "cv2.calibrateCamera 모델 A–E. f_reference = A–D fx 중앙값. 부트스트랩은 시간순 5장 블록 30회, 시간 구간은 3등분.",
          "- 로컬 도구 video_calib의 코드는 서버에서 찾지 못해 명세 규칙으로 새로 구현했다. ROUGH 문턱값은 새로 정해 실행 전 yaml에 고정했다.",
          f"- 회귀 확인(IMG_7135, 1678.6 px): {'영상이 서버에 없어 하지 못함' if not cfg.get('regression_video') else '아래 summary.md 참고'}.",
          f"- 사용 GPU: CUDA_VISIBLE_DEVICES={gpu['CUDA_VISIBLE_DEVICES']}", ""]
    L += ["## 5. 검출기 정보", ""]
    for det, p in prov.items():
        L.append(f"- **{det}**: {p.get('model_name')}; 가중치 sha256 {str(p.get('weights_sha256'))[:16]}…; 장치 {p.get('device')}; 라이선스 {p.get('license')}")
    path.write_text("\n".join(L) + "\n")


def session_rel(x, ref):
    return f"{100 * (x / ref - 1):+.1f}%"


def session_summary_md(path, vids, cfg, regression):
    prev = float(cfg["previous_reference_f_px"])
    L = ["# 실험 18 비교 요약 (summary.md에만 기록)", "", MANDATORY, "",
         "아래 상대 차이는 이 영상의 체커보드 기준 f 대비 값이다. 체커보드 판정이 GOOD이 아니면 '기준 불확실'로 표시한다.", ""]
    for v in vids:
        b = v["checkerboard"]
        L.append(f"## {v['video']}")
        L.append("")
        if "f_reference_px" not in b:
            L += [f"- 체커보드 기준 f 없음: {b.get('failed')} — 비교하지 않음", ""]
            continue
        fr = b["f_reference_px"]
        tag = "" if b["verdict"] == "GOOD" else " (기준 불확실: 체커보드 판정 " + b["verdict"] + ")"
        L += [f"- 체커보드 기준 f = {fr:.1f} px{tag}; 모델 E = {b['f_model_E_px']:.1f} px ({session_rel(b['f_model_E_px'], fr)})",
              f"- 이전 기준 {prev:.0f} px 대비 이 영상 기준 f: {session_rel(fr, prev)}", "",
              "| 값 | f | 이 영상 기준 f 대비 | 모델 E 대비 |", "|---|---|---|---|"]
        for det in cfg["detectors"]:
            d = v["hand"]["detectors"].get(det, {})
            if "fixed_template" not in d:
                L.append(f"| {det} | 값 없음 | - | - |")
                continue
            for mname in ("fixed_template", "joint_delta"):
                s = d[mname]
                if s["at_grid_edge"]:
                    hi = d["grid_px"][1]
                    side = "상한" if s["f"] >= hi * (1 - 1e-6) else "하한"
                    L.append(f"| {det} {mname} | 값 없음 ({side}) | {'≥' if side == '상한' else '≤'} {session_rel(s['f'], fr)} | - |")
                else:
                    L.append(f"| {det} {mname} | {s['f']:.1f} px | {session_rel(s['f'], fr)} | {session_rel(s['f'], b['f_model_E_px'])} |")
        for m, x in (v["hand"].get("anycalib") or {"models": {}})["models"].items():
            L.append(f"| AnyCalib {m} | {x['summary']['median']:.1f} px | {session_rel(x['summary']['median'], fr)} | {session_rel(x['summary']['median'], b['f_model_E_px'])} |")
        L.append("")
    if regression:
        L += ["## 회귀 확인 (IMG_7135)", ""]
        if "f_reference_px" in regression:
            d = regression["f_reference_px"] / cfg["regression_expected_f_px"] - 1
            L.append(f"- f_reference {regression['f_reference_px']:.1f} px vs 1678.6 px: {100 * d:+.2f}% → {'±1% 이내' if abs(d) <= cfg['regression_tolerance'] else '±1% 밖'}")
        else:
            L.append(f"- 실패: {regression.get('failed')}")
    else:
        L += ["## 회귀 확인 (IMG_7135)", "", "- IMG_7135.MOV가 서버에 없어 하지 못했다."]
    path.write_text("\n".join(L) + "\n")


def regression_check(cfg, log):
    p = cfg.get("regression_video")
    if not p:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="hf18r_"))
    try:
        norm = hf.normalize(Path(p).expanduser(), tmp / "n.mkv", cfg["pipeline"])
        W, H, fps = norm["output"]["display_w"], norm["output"]["display_h"], norm["output"]["fps"]
        dur = norm["output"]["duration_s"] or 1e9
        fr = session_board_frames(tmp / "n.mkv", fps, [0.0, float(dur)], cfg["checkerboard"], tmp)
        return hf.board_analyse(fr, cfg["checkerboard"], W, H, log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def session_results_yaml(run_dir: Path, cfg):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    out = {"experiment": "18_same_session_check", "run": str(run_dir.relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "gpu": m.get("gpu"), "note": MANDATORY + " 비교 수치는 summary.md에만 있다.", "videos": {}}
    for v in r["videos"]:
        b = v["checkerboard"]
        e = {"segments": {"hand_s": v["segments"]["hand"], "board_s": v["segments"]["board"]},
             "checkerboard": {"f_reference_px": None if "f_reference_px" not in b else round(b["f_reference_px"], 1),
                              "f_model_E_px": None if not b.get("f_model_E_px") else round(b["f_model_E_px"], 1), "verdict": b["verdict"]},
             "hand_guide_satisfied": v["hand"]["conditions"]["guide"], "detectors": {}}
        for det, d in v["hand"]["detectors"].items():
            if "fixed_template" not in d:
                e["detectors"][det] = {"f_px": None, "verdict": d.get("verdict")}
                continue
            ft = d["fixed_template"]
            e["detectors"][det] = {"f_fixed_template_px": None if ft["at_grid_edge"] else round(ft["f"], 1),
                                   "f_status": f"값 없음 ({ft['side']})" if ft["at_grid_edge"] else "value", "verdict": d["verdict"]}
        if v["hand"].get("anycalib"):
            e["anycalib_median_px"] = {k: round(x["summary"]["median"], 1) for k, x in v["hand"]["anycalib"]["models"].items()}
        out["videos"][v["video"]] = e
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def session_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/18_same_session_check.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(session_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.videos:
        cfg["videos"] = a.videos.split(",")
    videos = select_videos(cfg)
    gpu = gpu_info()
    sp = hf.resolve(cfg["reuse_14"]["synthetic_study"])
    study = json.loads(sp.read_text())
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    log = Log(run_root / "log.txt")
    log(f"18 같은 녹화 확인: {[v.name for v in videos]} → {run_root} (CUDA_VISIBLE_DEVICES={gpu['CUDA_VISIBLE_DEVICES']})")
    t0 = time.time()
    vids = [session_process(v, cfg, text, a.config, run_root, study, str(sp), log) for v in videos]
    reg = regression_check(cfg, log)
    prov = detector_provenance(cfg)
    session_report(run_root / "report.md", vids, cfg, prov, gpu)
    session_summary_md(run_root / "summary.md", vids, cfg, reg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": vids, "regression": reg}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_17": cfg["code_17"], "detectors": prov, "gpu": gpu, "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, vids


# ====================================================================================================
# runov  (was HandCalib/run_keypoint_overlay.py)
# Overlay video of each detector's 2D hand keypoints (the tracked target hand the pipeline used) on the normalised frames.
#
#     python HandCalib/run.py overlay --video-dir runs/17_protocol_rerun/<UTC>/IMG_XXXX [--input /path/IMG_XXXX.MOV]
#
# Reads <video-dir>/<detector>/{results.json, detections.npz}, re-creates the same normalised PNG frames (hashes checked), re-tracks
# with 14's function and the detector's conf_min, and writes <video-dir>/overlay/<detector>.mp4 and side_by_side.mp4.
# Kept points: filled circles and skeleton; masked points (low confidence / outside): hollow grey circles. Temporary frames are deleted.
# The overlay videos stay in the (git-ignored) run folder.
# ====================================================================================================

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
FINGER_COLOR = {0: (255, 255, 255), 1: (0, 140, 255), 2: (0, 220, 0), 3: (255, 160, 0), 4: (255, 0, 200), 5: (0, 0, 255)}   # BGR: wrist, thumb..pinky
ROLE_KO = {"front": "front window", "tilt": "tilt", "none": "excluded"}


def finger_of(j):
    return 0 if j == 0 else (j - 1) // 4 + 1


def overlay_draw(img, kp, mask, box, label, lines):
    out = img.copy()
    s = max(1.0, max(img.shape[:2]) / 1920 * 1.6)
    if box is not None and np.isfinite(box).all():
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 200, 200), max(1, int(s)))
    if kp is not None:
        for a, b in BONES:
            if mask[a] and mask[b]:
                c = FINGER_COLOR[finger_of(b)]
                cv2.line(out, tuple(np.round(kp[a]).astype(int)), tuple(np.round(kp[b]).astype(int)), c, max(1, int(2 * s)), cv2.LINE_AA)
        for j in range(21):
            if not np.isfinite(kp[j]).all():
                continue
            p = tuple(np.round(kp[j]).astype(int))
            if mask[j]:
                cv2.circle(out, p, int(5 * s), FINGER_COLOR[finger_of(j)], -1, cv2.LINE_AA)
                cv2.circle(out, p, int(5 * s), (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, p, int(5 * s), (150, 150, 150), max(1, int(s)), cv2.LINE_AA)
    y = int(40 * s)
    for i, t in enumerate([label] + lines):
        org = (int(15 * s), y + i * int(45 * s))
        cv2.putText(out, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), max(3, int(4 * s)), cv2.LINE_AA)
        cv2.putText(out, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(2, int(1.5 * s)), cv2.LINE_AA)
    return out


def overlay_encode(frames_dir: Path, pattern: str, fps: float, out: Path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:.4f}", "-i", str(frames_dir / pattern), "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-map_metadata", "-1", str(out)], check=True)


def overlay_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="run folder of one video (contains one sub-folder per detector)")
    ap.add_argument("--input", default=None, help="original video (default: input_dir/<name>.MOV from the pipeline config)")
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/17_protocol_rerun.yaml"))
    a = ap.parse_args(argv)
    vdir = hf.resolve(a.video_dir)
    cfg = yaml.safe_load(Path(a.config).read_text())
    dets = [d for d in cfg["detectors"] if (vdir / d / "results.json").exists()]
    r0 = json.loads((vdir / dets[0] / "results.json").read_text())
    src = Path(a.input).expanduser() if a.input else Path(cfg["input_dir"]).expanduser() / r0["video"]
    tmp = Path(tempfile.mkdtemp(prefix="hfov_"))
    try:
        norm = hf.normalize(src, tmp / "n.mkv", cfg["pipeline"])
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        frames, k = hf.extract_frames(tmp / "n.mkv", tmp / "f", norm["output"]["fps"], None, cfg["pipeline"]["frames"]["detect_fps"])
        if [(x["index"], x["sha256"]) for x in frames] != [(x["index"], x["sha256"]) for x in r0["frames"]["list"]]:
            raise SystemExit("re-created frames differ from the run's frame list")
        times = np.array([x["time_s"] for x in frames])
        fps_out = norm["output"]["fps"] / k
        out_dir = vdir / "overlay"
        out_dir.mkdir(exist_ok=True)
        tracks = {}
        for d in dets:
            r = json.loads((vdir / d / "results.json").read_text())
            det = hf.load_detections(vdir / d / "detections.npz")
            hf.check_alignment(frames, det)
            tc = dict(cfg["pipeline"]["tracking"], conf_min=float(cfg["detector_settings"][d]["conf_min"]))
            tr = hf.track(det, times, (W, H), tc, front_start_s=float(cfg["pipeline"]["phases"]["front_skip_s"]))
            a_ = r.get("analysis", {})
            th = a_.get("theta2d_deg") or [None] * len(frames)
            role = a_.get("role") or ["none"] * len(frames)
            tracks[d] = (tr, th, role)
        for d in dets + ["side_by_side"]:
            (tmp / d).mkdir()
        for i, fr in enumerate(frames):
            img = cv2.imread(fr["png"], cv2.IMREAD_COLOR)
            panels = []
            for d in dets:
                tr, th, role = tracks[d]
                kp = tr["kp"][i] if tr["present"][i] else None
                lines = [f"t {fr['time_s']:.2f} s   {ROLE_KO.get(role[i], role[i])}",
                         "no hand" if kp is None else f"theta2d {th[i]:.0f} deg   kept {int(tr['mask'][i].sum())}/21" if th[i] is not None else f"kept {int(tr['mask'][i].sum())}/21"]
                o = overlay_draw(img, kp, tr["mask"][i], tr["box"][i] if tr["present"][i] else None, d, lines)
                cv2.imwrite(str(tmp / d / f"o_{i:05d}.png"), o)
                panels.append(o)
            sbs = np.concatenate(panels, axis=1)
            scale = min(1.0, 2880 / sbs.shape[1])
            if scale < 1:
                sbs = cv2.resize(sbs, (int(sbs.shape[1] * scale) // 2 * 2, int(sbs.shape[0] * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmp / "side_by_side" / f"o_{i:05d}.png"), sbs)
        made = []
        for d in dets + ["side_by_side"]:
            overlay_encode(tmp / d, "o_%05d.png", fps_out, out_dir / f"{d}.mp4")
            made.append(str(out_dir / f"{d}.mp4"))
        print("\n".join(made))
        return made
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ====================================================================================================
# command line
# ====================================================================================================

COMMANDS = {"protocol": protocol_main, "frame-filter": ff_main, "same-session": session_main, "overlay": overlay_main}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit("usage: run.py {" + ",".join(COMMANDS) + "} [options]   (see the module docstring)")
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
