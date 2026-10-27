"""HandCalib hand-template focal length: orchestration, reports and the command line for experiments 17 and 18 (analysis code in handfocal.py).

    python HandCalib/run.py protocol       --config HandCalib/configs/17_protocol_rerun.yaml [--videos A,B] [--detectors a,b] [results-yaml --run DIR]
    python HandCalib/run.py frame-filter   --config HandCalib/configs/17_frame_filter_diag.yaml [--source VIDEO=RUN_DIR]
    python HandCalib/run.py same-session   --config HandCalib/configs/18_same_session_check.yaml [results-yaml --run DIR]
    python HandCalib/run.py overlay        --video-dir runs/17_protocol_rerun/<UTC>/IMG_XXXX
    python HandCalib/run.py phase-segmentation --config HandCalib/configs/20_phase_segmentation.yaml [--videos A,B] [--detectors a,b] [results-yaml --run DIR]
    python HandCalib/run.py guide-revision --config HandCalib/configs/21_guide_revision.yaml [--videos A,B] [--detectors a,b] [results-yaml --run DIR | overlay --run DIR]
    python HandCalib/run.py f-response     --config HandCalib/configs/22_f_response.yaml [--videos A,B] [results-yaml --run DIR]
    python HandCalib/run.py error-factors  --config HandCalib/configs/23_error_factors.yaml [--videos A,B] [results-yaml --run DIR]
    python HandCalib/run.py long-video-check --config HandCalib/configs/25_long_video_check.yaml [bias | --videos A,B | results-yaml --run DIR]
    python HandCalib/run.py generalization --config HandCalib/configs/26_generalization.yaml [--videos A,B] [results-yaml --run DIR]
    python HandCalib/run.py main-camera-extension --config HandCalib/configs/27_main_camera_extension.yaml [--videos A,B] [results-yaml --run DIR]
    python HandCalib/run.py wide-60cm --config HandCalib/configs/29_wide_60cm.yaml [--videos A,B] [results-yaml --run DIR]
    python HandCalib/run.py jointset-sets --config HandCalib/configs/30_jointset_sets.yaml [--videos A,B] [results-yaml --run DIR]

Consolidated from run_hand_focal.py (14, process_video only), run_detector_swap.py (16, the per-detector processing that 17 calls),
run_protocol_rerun.py and run_frame_filter_diag.py (17),
run_same_session.py (18) and run_keypoint_overlay.py; each is a section below. Input videos are only read; no location or device metadata is written.
"""
from __future__ import annotations

import argparse
import copy
import csv
import cv2
import datetime as dt
import hashlib
import json
import math
import numpy as np
import os
import re
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
# pseg  (experiment 20: front -> tilt transition detection; new section)
# Re-detects the transition with the corrected rule (handfocal.pseg_*) on the 17 runs' detections (no video is read, nothing is re-detected),
# compares it with the legacy rule and with an optional manual time, and re-checks the shooting guide with each. The f estimate is 17's, unchanged;
# the diagnostic variants (auto phase, large-tilt exclusion, wrist exclusion) are reported in summary.md only.
#
#     python HandCalib/run.py phase-segmentation --config HandCalib/configs/20_phase_segmentation.yaml [--videos A,B] [--detectors a,b] [--transition-method M]
#     python HandCalib/run.py phase-segmentation --config ... results-yaml --run runs/20_phase_segmentation/<UTC>
# ====================================================================================================

PSEG_NOTE = "전환 시각 검출과 가이드 판정을 다시 낸 실험이다. f 추정 방식과 판정 문턱값은 바꾸지 않았다."


def pseg_fval(sol):
    """'값 없음 (상한/하한)' for a grid-edge solution, else 'f px'."""
    if sol is None:
        return "값 없음"
    return f"값 없음 ({sol['side']})" if sol["at_grid_edge"] else f"{sol['f']:.1f} px"


def pseg_t(x):
    return "없음" if x is None else f"{x:.2f}"


def pseg_manual_diff(t, manual):
    return "-" if manual is None or t is None else f"{t - manual:+.2f}"


def pseg_figure(path: Path, j: dict, manual):
    """theta2d, d (raw and smoothed) with its threshold, and the keypoint speed with its threshold; the transitions of both rules are marked."""
    plt = _plt()
    S = j["series"]
    t = np.array(S["t"])
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True)
    lines = (("legacy_theta", "#c00", "기존(θ2d)"), ("linear_d", "#2a6fbb", "linear_d"), ("linear_d_no_motion", "#3a9a4a", "linear_d(움직임 조건 끔, 진단)"))
    for ax in axs:
        if S["front"][0] is not None:
            ax.axvspan(S["front"][0], S["front"][1], color="#999", alpha=0.18)
        for key, col, lab in lines:
            tv = j[key]["t_transition"]
            if tv is not None:
                ax.axvline(tv, color=col, lw=1.4, ls="-" if key != "linear_d_no_motion" else "--", label=lab if ax is axs[0] else None)
        if manual is not None:
            ax.axvline(manual, color="k", lw=1.2, ls=":", label="수동 기준" if ax is axs[0] else None)
        ax.grid(alpha=0.3)
    axs[0].plot(t, S["theta"], ".-", ms=3, lw=1, color="#2a6fbb")
    axs[0].axhline(10.0, color="#c00", lw=0.8, ls="--")
    axs[0].set_ylabel("θ2d (°)\n(점선: 기존 문턱 10°)")
    axs[0].legend(fontsize=7, loc="upper right")
    l = j["linear_d"]
    axs[1].plot(t, S["d"], ".", ms=3, color="#9bb7d9", label="d")
    axs[1].plot(t, S["d_smooth"], "-", lw=1.4, color="#2a6fbb", label="d (0.3 s 중앙값)")
    if l.get("d_thr") is not None:
        axs[1].axhline(l["d_thr"], color="#c47a00", lw=1.2, label=f"d_thr {l['d_thr']:.3f}")
        axs[1].axhline(l["d0"], color="#888", lw=0.9, ls="--", label=f"d0 {l['d0']:.3f}")
    axs[1].set_ylabel("d = 1 − s2/s1")
    axs[1].legend(fontsize=7, loc="upper right")
    axs[2].plot(t, S["v"], "-", lw=1, color="#8a4fbf", label="v")
    if l.get("v_thr") is not None:
        axs[2].axhline(l["v_thr"], color="#c47a00", lw=1.2, label=f"v_thr {l['v_thr']:.3f}")
        axs[2].axhline(l["v0"], color="#888", lw=0.9, ls="--", label=f"v0 {l['v0']:.3f}")
    axs[2].set_ylabel("키포인트 속도 v (1/s)")
    axs[2].set_xlabel("시간 (s)  (회색: 정면 창)")
    axs[2].legend(fontsize=7, loc="upper right")
    axs[0].set_title(f"{j['video']} / {j['detector']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pseg_report(path: Path, jobs: list, cfg: dict):
    c20 = cfg["phase_segmentation"]
    man = cfg.get("manual_transition_s") or {}
    tol = float(cfg["manual_tolerance_s"])
    L = ["# 실험 20: 정면 → 기울기 전환 시각 검출 교정", "", PSEG_NOTE, "",
         "## 1. 방법과 설정", "",
         f"- 기존(legacy_theta): θ2d = arccos(s2/s1) ≥ {cfg['pipeline']['phases']['auto_theta_deg']:.0f}°가 {cfg['pipeline']['phases']['auto_sustain_s']} s 지속되는 첫 시각. θ2d는 0° 근처에서 예민해서 키포인트 잡음만으로도 10°가 나온다.",
         f"- 교정(linear_d): d = 1 − s2/s1 (= 1 − cos θ2d). 영상마다 정면 창 d의 {c20['noise_percentile']}번째 백분위수 d0를 잡음 바닥으로 하고, "
         f"d_thr = max({c20['k_d']} × d0, 1 − cos {c20['theta_min_deg']:.0f}°). d에 {c20['median_window_s']} s 중앙값 필터(분할에만 사용)를 적용하고, 정면 창이 끝난 뒤 "
         f"d_thr 초과가 {c20['sustain_s']} s 지속되는 첫 시각을 전환으로 한다.",
         f"- 움직임 조건({'켬' if c20['use_motion'] else '끔'}): 키포인트 속도 v(관절 이동 평균 ÷ 손 박스 긴 변 ÷ 시간 간격)의 정면 창 {c20['noise_percentile']}번째 백분위수 v0, "
         f"v_thr = {c20['k_v']} × v0. 그 구간이 시작될 때(±{c20['motion_window_s']} s) v가 v_thr를 넘어야 한다. 참고로 이 조건을 끈 전환 시각도 함께 적는다(진단 열, 가이드 판정에는 쓰지 않음).",
         f"- 가이드 판정에 쓰는 전환: {cfg['transition_method']}. 입력: 17 공식 실행의 검출 결과와 프레임 목록(새 검출 없음, 영상은 읽지 않음).", "",
         "## 2. 전환 시각 (초)", "",
         "| 영상 | 검출기 | 기존(θ2d) | linear_d | linear_d(움직임 끔, 진단) | d0 | d_thr (θ 환산) | v0 | v_thr | 건너뛴 구간(움직임 부족) | 수동 기준 | 기존−수동 | linear_d−수동 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in jobs:
        if "failed" in j:
            L.append(f"| {j['video']} | {j['detector']} | 분석 실패: {j['failed']} | | | | | | | | | | |")
            continue
        l, mv = j["linear_d"], man.get(j["video"])
        th = "-" if l.get("d_thr") is None else f"{l['d_thr']:.3f} ({math.degrees(math.acos(max(-1.0, min(1.0, 1 - l['d_thr'])))):.0f}°)"
        L.append(f"| {j['video']} | {j['detector']} | {pseg_t(j['legacy_theta']['t_transition'])} | {pseg_t(l['t_transition'])} | {pseg_t(j['linear_d_no_motion']['t_transition'])} | "
                 f"{'-' if l.get('d0') is None else format(l['d0'], '.3f')} | {th} | {'-' if l.get('v0') is None else format(l['v0'], '.3f')} | {'-' if l.get('v_thr') is None else format(l['v_thr'], '.3f')} | "
                 f"{', '.join(format(x, '.2f') for x in l.get('skipped_no_motion_s') or []) or '-'} | {'-' if mv is None else format(mv, '.2f')} | "
                 f"{pseg_manual_diff(j['legacy_theta']['t_transition'], mv)} | {pseg_manual_diff(l['t_transition'], mv)} |")
    L += ["", "## 3. 가이드 판정 (두 전환 시각으로 각각)", "",
          "| 영상 | 검출기 | 기존 방식 | 이유 | linear_d | 이유 | 전환 항목만 바뀐 것 |", "|---|---|---|---|---|---|---|"]
    n_same = n_chg = 0
    for j in jobs:
        if "failed" in j:
            continue
        a, b = j["legacy_theta"], j["linear_d"]
        chg = a["items"].get("transition") != b["items"].get("transition")
        n_chg += int(chg)
        n_same += int(not chg)
        L.append(f"| {j['video']} | {j['detector']} | {'충족' if a['guide'] else '미충족'} | {'; '.join(a['reasons']) or '-'} | {'충족' if b['guide'] else '미충족'} | "
                 f"{'; '.join(b['reasons']) or '-'} | {'전환 항목 통과 여부가 바뀜' if chg else '같음'} |")
    ng = sum(1 for j in jobs if "failed" not in j)
    L += ["", f"- 전환 항목({cfg['conditions']['transition_min_s']} s 이상) 판정이 바뀐 조합: {n_chg}/{ng}. 가이드 충족으로 바뀐 조합: "
          f"{sum(1 for j in jobs if 'failed' not in j and not j['legacy_theta']['guide'] and j['linear_d']['guide'])}/{ng}, 충족이 미충족으로 바뀐 조합: "
          f"{sum(1 for j in jobs if 'failed' not in j and j['legacy_theta']['guide'] and not j['linear_d']['guide'])}/{ng}.", "",
          "## 4. 수동 기준과의 비교", ""]
    have = {v: mv for v, mv in man.items() if mv is not None}
    if not have:
        L.append("- 수동 전환 시각(manual_transition_s)이 비어 있어 수동 기준이 없다. 임의로 채우지 않았다. 사용자가 yaml에 채우면 두 방식의 차이를 보고한다(목표: linear_d가 "
                 f"{tol} s 이내).")
    else:
        L += [f"- 목표: linear_d가 수동 기준과 {tol} s 이내.", "", "| 영상 | 검출기 | 수동 | 기존 | 기존−수동 | linear_d | linear_d−수동 | linear_d 목표 |", "|---|---|---|---|---|---|---|---|"]
        for j in jobs:
            mv = have.get(j["video"])
            if mv is None or "failed" in j:
                continue
            tl = j["linear_d"]["t_transition"]
            ok = "-" if tl is None else ("달성" if abs(tl - mv) <= tol else "미달")
            L.append(f"| {j['video']} | {j['detector']} | {mv:.2f} | {pseg_t(j['legacy_theta']['t_transition'])} | {pseg_manual_diff(j['legacy_theta']['t_transition'], mv)} | "
                     f"{pseg_t(tl)} | {pseg_manual_diff(tl, mv)} | {ok} |")
    L += ["", "## 5. f·판정이 17의 기존 결과와 같은지 (기본 설정: nominal, 선택 기능 꺼짐)", "",
          "| 영상 | 검출기 | f fixed_template | 17의 f | f 일치 | joint_delta 일치 | 판정 일치 | 기존 전환 시각 일치 |", "|---|---|---|---|---|---|---|---|"]
    for j in jobs:
        if "same_as_17" not in j:
            continue
        s, d = j["same_as_17"], j["default"]["solutions"]["fixed_template"]
        L.append(f"| {j['video']} | {j['detector']} | {pseg_fval(d)} | {'-' if s['f_17'] is None else format(s['f_17'], '.1f') + ' px'} | {'O' if s['f_fixed_template'] else 'X'} | "
                 f"{'O' if s['f_joint_delta'] else 'X'} | {'O' if s['verdict'] else 'X'} | {'O' if s['legacy_transition'] else 'X'} |")
    L += ["", f"- 모두 일치한 조합: {sum(1 for j in jobs if j.get('same_as_17', {}).get('all'))}/{sum(1 for j in jobs if 'same_as_17' in j)}", "",
          "## 6. 한계", "",
          "- 정면 창은 17의 front_window가 고른 창을 그대로 쓴다. 그 창이 이미 기울기 시작을 포함하면(0.5–3.0초 안에서 가장 덜 움직이는 1초가 3초 직전에 끝나는 경우) d0가 커져 d_thr가 높아진다.",
          "- 움직임 조건은 기본값(k_v = 3)에서 천천히 시작하는 기울임을 거부해 다음 기울임으로 건너뛸 수 있다. 위 표의 '건너뛴 구간'과 '움직임 끔' 열로 확인한다.",
          "- 수동 기준이 없으면 어느 전환 시각이 맞는지는 판단하지 않았다.", ""]
    path.write_text("\n".join(L) + "\n")


def pseg_summary(path: Path, jobs: list, cfg: dict):
    ref = float(cfg["reference_f_px"])
    rel = lambda sol: "-" if sol is None or sol["at_grid_edge"] else f"{100 * (sol['f'] / ref - 1):+.1f}%"      # noqa: E731
    L = ["# 실험 20 진단 변형 (summary.md에만 기록)", "", "**진단 결과이며 공식 f 추정값이 아님.** 전환 시각 검출만 바꾼 공식 결과(report.md)와 별도다. 1680과의 비교는 이 파일에만 적는다.", "",
         f"- 비교 기준: 사용자 체커보드 f = {ref:.0f} px (추정·판정에 쓰지 않음)",
         f"- auto_phase: 기울기 구간 시작을 {cfg['transition_method']}의 전환 시각으로 바꿈. max_tilt: 기울기 구간에서 θ2d > {cfg['phase_segmentation']['max_tilt_deg']:.0f}° 프레임을 f 추정에서 뺌. "
         "no_wrist: 템플릿과 f 추정 관절 집합에서 손목(0번)을 뺌.", "",
         "| 영상 | 검출기 | 변형 | 기울기 시작 (s) | 사용한 기울기 프레임 | f fixed_template | 1680 대비 | f joint_delta | 판정 | 날카로움 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for j in jobs:
        if "failed" in j:
            continue
        rows = [("기본(nominal)", j["default"])] + [(k, v) for k, v in j["variants"].items()]
        for name, v in rows:
            if v.get("skipped"):
                L.append(f"| {j['video']} | {j['detector']} | {name} | - | - | 실행 안 함: {v['skipped']} | | | | |")
                continue
            if v.get("failed"):
                L.append(f"| {j['video']} | {j['detector']} | {name} | - | - | 분석 실패: {v['failed']} | | | | |")
                continue
            ft, jd = v["solutions"]["fixed_template"], v["solutions"]["joint_delta"]
            nt = f"{v['n_tilt_used']}" + (f" (거르기 전 {v['n_tilt_before']} → {v['n_tilt_after']})" if v.get("n_tilt_before") is not None else "")
            L.append(f"| {j['video']} | {j['detector']} | {name} | {v['tilt_start_s']:.2f} | {nt} | {pseg_fval(ft)} | {rel(ft)} | {pseg_fval(jd)} | {v['verdict']} | {v['sharpness']:.3f} |")
    L += ["", "## 한계", "", "- 진단 변형이며 공식 추정 방식이 아니다. 영상 5개, 검출기 3종. 가이드를 충족한 영상은 없다.", "- 체커보드 f는 사용자 제공 값이며 이 비교에만 썼다.", ""]
    path.write_text("\n".join(L) + "\n")


def pseg_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    man = cfg.get("manual_transition_s") or {}
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    out = {"experiment": "20_phase_segmentation", "run": run_name, "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "transition_method": r["transition_method"], "note": PSEG_NOTE + " 기준 f와의 비교 수치는 summary.md에만 있다.", "videos": {}}
    for j in r["jobs"]:
        e = out["videos"].setdefault(j["video"], {"manual_transition_s": man.get(j["video"]), "detectors": {}})
        if "failed" in j:
            e["detectors"][j["detector"]] = {"failed": j["failed"]}
            continue
        d, l = j["default"], j["linear_d"]
        fs = lambda v: {"f_fixed_template_px": None if v["solutions"]["fixed_template"]["at_grid_edge"] else round(v["solutions"]["fixed_template"]["f"], 1),      # noqa: E731
                        "f_status": f"값 없음 ({v['solutions']['fixed_template']['side']})" if v["solutions"]["fixed_template"]["at_grid_edge"] else "value",
                        "f_joint_delta_px": None if v["solutions"]["joint_delta"]["at_grid_edge"] else round(v["solutions"]["joint_delta"]["f"], 1),
                        "verdict": v["verdict"], "tilt_start_s": round(v["tilt_start_s"], 3), "n_tilt_frames": v["n_tilt_used"]}
        e["detectors"][j["detector"]] = {
            "transition_legacy_theta_s": j["legacy_theta"]["t_transition"], "transition_linear_d_s": l["t_transition"],
            "transition_linear_d_no_motion_s": j["linear_d_no_motion"]["t_transition"],
            "d0": None if l.get("d0") is None else round(l["d0"], 5), "d_thr": None if l.get("d_thr") is None else round(l["d_thr"], 5),
            "v0": None if l.get("v0") is None else round(l["v0"], 5), "v_thr": None if l.get("v_thr") is None else round(l["v_thr"], 5),
            "guide_satisfied_legacy_theta": j["legacy_theta"]["guide"], "guide_satisfied_linear_d": l["guide"],
            "guide_reasons_legacy_theta": j["legacy_theta"]["reasons"], "guide_reasons_linear_d": l["reasons"],
            "default": fs(d), "same_as_17": j["same_as_17"]["all"],
            "variants": {k: ({"skipped": v["skipped"]} if v.get("skipped") else fs(v)) for k, v in j["variants"].items() if not v.get("failed")}}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def pseg_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/20_phase_segmentation.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list overriding yaml videos")
    ap.add_argument("--detectors", default=None, help="comma list overriding yaml detectors")
    ap.add_argument("--transition-method", default=None, choices=["legacy_theta", "linear_d"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(pseg_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.videos:
        cfg["videos"] = a.videos.split(",")
    if a.detectors:
        cfg["detectors"] = a.detectors.split(",")
    if a.transition_method:
        cfg["transition_method"] = a.transition_method
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"20 전환 시각 검출 교정: 영상 {cfg['videos']}, 검출기 {cfg['detectors']}, 전환 방식 {cfg['transition_method']} → {run_root}")
    t0 = time.time()
    args = [(v, d, str(hf.resolve(cfg["sources"][v])), cfg["pipeline"], cfg["phase_segmentation"], cfg["conditions"], cfg["detector_settings"][d]["conf_min"], cfg["transition_method"])
            for v in cfg["videos"] for d in cfg["detectors"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        jobs = list(ex.map(hf.pseg_job, args))
    man = cfg.get("manual_transition_s") or {}
    for j in jobs:
        if "failed" in j:
            log(f"  {j['video']} {j['detector']}: 분석 실패 ({j['failed']})")
            continue
        pseg_figure(run_root / "figures" / f"{Path(j['video']).stem}_{j['detector']}.png", j, man.get(j["video"]))
        g = lambda x: "충족" if x else "미충족"      # noqa: E731
        log(f"  {j['video']} {j['detector']}: 기존 {pseg_t(j['legacy_theta']['t_transition'])} s, linear_d {pseg_t(j['linear_d']['t_transition'])} s "
            f"(움직임 끔 {pseg_t(j['linear_d_no_motion']['t_transition'])} s), d0 {j['linear_d'].get('d0', float('nan')):.3f}, 가이드 {g(j['legacy_theta']['guide'])} → {g(j['linear_d']['guide'])}, "
            f"17과 일치 {'O' if j['same_as_17']['all'] else 'X'}")
    pseg_report(run_root / "report.md", jobs, cfg)
    pseg_summary(run_root / "summary.md", jobs, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"transition_method": cfg["transition_method"], "jobs": jobs}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_17": cfg["code_17"], "sources": cfg["sources"], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, jobs


# ====================================================================================================
# grev  (experiment 21: transition defaults, shooting-guide revision, wrist handling; new section)
# Same inputs as 20 (the 17 runs' detections, nothing is re-detected). Reports the transition under 20's and the new defaults, the revised shooting guide,
# and the f estimate under the three wrist modes (keep / exclude / downweight) side by side; the adoption rules and every comparison with the reference f
# are in summary.md only. The f estimate and the f verdict thresholds are 17's.
#
#     python HandCalib/run.py guide-revision --config HandCalib/configs/21_guide_revision.yaml [--videos A,B] [--detectors a,b] [--transition-method M]
#     python HandCalib/run.py guide-revision --config ... results-yaml --run runs/21_guide_revision/<UTC>
#     python HandCalib/run.py guide-revision --config ... overlay --run runs/21_guide_revision/<UTC> [--videos A,B] [--detector D]
# ====================================================================================================

GREV_NOTE = "전환 검출 기본값, 가이드 판정 기준, 손목 점 처리를 바꿔 다시 낸 실험이다. f 추정 방식(평면 템플릿 + 프로파일)과 f 판정 문턱값은 바꾸지 않았다."
GREV_ITEMS = [("box_short_ratio", "박스"), ("edge_rate", "가장자리"), ("transition", "전환"), ("tilt_frames", "기울기 프레임"), ("theta_tilt_median", "θ2d 중앙값"),
              ("axis_bins", "축 덮임"), ("finger_extension", "손가락 펴짐")]


def grev_spread(jobs: list, key) -> dict:
    """Per video: max - min of key(job) over the detectors; None when any detector has no value."""
    out = {}
    for v in sorted({j["video"] for j in jobs}):
        ts = [key(j) for j in jobs if j["video"] == v and "failed" not in j]
        out[v] = None if not ts or any(t is None for t in ts) else max(ts) - min(ts)
    return out


def grev_figure(path: Path, j: dict, manual):
    """theta2d, and d (raw and smoothed) with the new threshold, 20's threshold and d0; the transitions of the legacy rule, 20's default, 20 without the motion
    condition and the new default are marked; shaded: the new noise-floor window (blue) and the f estimate's front window (grey)."""
    plt = _plt()
    S = j["series"]
    t = np.array(S["t"])
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    n, t20 = j["transition"], j.get("transition_20")
    marks = [("legacy_theta", j["legacy_theta"]["t_transition"], "#c00", "-", "기존(θ2d)")]
    if t20:
        marks += [("t20", t20["t_transition"], "#c47a00", "-", "20 기본"), ("t20nm", t20.get("t_no_motion"), "#3a9a4a", "--", "20 움직임 끔")]
    marks += [("t21", n["t_transition"], "#2a6fbb", "-", "새 방식")]
    for ax in axs:
        if j["default"].get("front_window") and j["default"]["front_window"][0] is not None:
            ax.axvspan(*j["default"]["front_window"], color="#999", alpha=0.15)
        if n["front_window"][0] is not None:
            ax.axvspan(*n["front_window"], color="#2a6fbb", alpha=0.15)
        for k, tv, col, ls, lab in marks:
            if tv is not None:
                ax.axvline(tv, color=col, lw=1.4, ls=ls, label=lab if ax is axs[0] else None)
        if manual is not None:
            ax.axvline(manual, color="k", lw=1.2, ls=":", label="수동 기준" if ax is axs[0] else None)
        ax.grid(alpha=0.3)
    axs[0].plot(t, S["theta"], ".-", ms=3, lw=1, color="#2a6fbb")
    axs[0].set_ylabel("θ2d (°)")
    axs[0].legend(fontsize=7, loc="upper right")
    axs[1].plot(t, S["d"], ".", ms=3, color="#9bb7d9", label="d")
    axs[1].plot(t, S["d_smooth"], "-", lw=1.4, color="#2a6fbb", label="d (0.3 s 중앙값)")
    if n.get("d_thr") is not None:
        axs[1].axhline(n["d_thr"], color="#2a6fbb", lw=1.2, ls="-.", label=f"새 d_thr {n['d_thr']:.3f}" + (" (d0 상한 적용)" if n.get("d0_capped") else ""))
        axs[1].axhline(n["d0"], color="#888", lw=0.9, ls="--", label=f"d0 {n['d0']:.3f}")
    if t20 and t20.get("d_thr") is not None:
        axs[1].axhline(t20["d_thr"], color="#c47a00", lw=1.0, ls=":", label=f"20 d_thr {t20['d_thr']:.3f}")
    axs[1].set_ylabel("d = 1 − s2/s1")
    axs[1].set_xlabel("시간 (s)  (파랑: 새 정면 창, 회색: f 추정의 정면 창)")
    axs[1].legend(fontsize=7, loc="upper right")
    axs[0].set_title(f"{j['video']} / {j['detector']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def grev_pass(g, key):
    return "O" if g["items"].get(key) else "X"


def grev_report(path: Path, jobs: list, cfg: dict):
    p, cc, c20 = cfg["phase_segmentation"], cfg["conditions"], cfg["compare_20"]
    man = cfg.get("manual_transition_s") or {}
    tol = float(cfg["manual_tolerance_s"])
    ok = [j for j in jobs if "failed" not in j]
    cap = 1 - math.cos(math.radians(p["d0_cap_theta_deg"])) if p.get("d0_cap_theta_deg") is not None else None
    L = ["# 실험 21: 전환 검출 기본값, 가이드 기준, 손목 점 처리 수정", "", GREV_NOTE, "",
         "## 1. 바뀐 것", "",
         f"- 전환 검출 기본값: {cfg['transition_method']}, 움직임 조건 {'켬' if p['use_motion'] else '끔'}(20: 켬). k_d = {p['k_d']}, θ_min = {p['theta_min_deg']:.0f}°, 중앙값 필터 {p['median_window_s']} s, 지속 {p['sustain_s']} s는 20과 같다.",
         f"- 잡음 바닥용 정면 창: 키포인트가 가장 덜 움직이는 1초를 {p['front_search_s'][0]}–{p['front_search_s'][1]} s 안에서 고른다(20: 0.5–3.0 s). f 추정의 정면 창과 템플릿은 17의 것 그대로다.",
         f"- 잡음 바닥 상한: d_thr = max({p['k_d']} × min(d0, d0_cap), 1 − cos {p['theta_min_deg']:.0f}°), d0_cap = 1 − cos {p['d0_cap_theta_deg']:.0f}° = {cap:.3f}. d0가 상한을 넘으면 경고를 적는다.",
         f"- 가이드 기준: 손 박스/짧은 변 ≤ {cc['box_short_ratio_max']} (20: {c20['conditions']['box_short_ratio_max']}). θ2d p95 항목 대신 기울기 구간에서 θ2d ≤ {cc['tilt_max_deg']:.0f}°인 프레임 "
         f"≥ {cc['min_tilt_frames']}장, 그 프레임들의 θ2d 중앙값 {cc['theta_tilt_median_range_deg'][0]:.0f}–{cc['theta_tilt_median_range_deg'][1]:.0f}°. 전환 ≥ {cc['transition_min_s']} s는 새 전환 검출로 판단. 나머지는 그대로.",
         "  기울기 구간은 17의 f 추정이 쓰는 구간(3.0초 이후)이다.",
         f"- 손목 점: wrist_mode keep / exclude / downweight(가중치 {cfg['f_estimation']['wrist_weight']}) 세 방식을 모두 실행했다. f 추정(템플릿, PnP 비용, 불확실성)에만 적용하고 전환 검출과 가이드 판정은 같은 기본 관절 집합을 쓴다. "
         "downweight는 한 프레임의 재투영 비용을 관절 가중 평균으로 계산한다(자세는 모든 점으로 풀고 비용만 가중).", "",
         "## 2. 전환 시각 (초)", "",
         "| 영상 | 검출기 | 기존(θ2d) | 20 기본 | 20 움직임 끔 | 21 | 수동 기준 | 21−수동 | 20−수동 |", "|---|---|---|---|---|---|---|---|---|"]
    for j in jobs:
        if "failed" in j:
            L.append(f"| {j['video']} | {j['detector']} | 분석 실패: {j['failed']} | | | | | | |")
            continue
        mv, n, t20 = man.get(j["video"]), j["transition"], j["transition_20"]
        L.append(f"| {j['video']} | {j['detector']} | {pseg_t(j['legacy_theta']['t_transition'])} | {pseg_t(t20['t_transition'])} | {pseg_t(t20.get('t_no_motion'))} | {pseg_t(n['t_transition'])} | "
                 f"{'-' if mv is None else format(mv, '.2f')} | {pseg_manual_diff(n['t_transition'], mv)} | {pseg_manual_diff(t20['t_transition'], mv)} |")
    L += ["", "검출기 사이 최대−최소 편차 (초, 값이 없는 검출기가 있으면 '계산 불가'):", "", "| 영상 | 기존 | 20 기본 | 20 움직임 끔 | 21 |", "|---|---|---|---|---|"]
    sp = [grev_spread(ok, lambda j: j["legacy_theta"]["t_transition"]), grev_spread(ok, lambda j: j["transition_20"]["t_transition"]),
          grev_spread(ok, lambda j: j["transition_20"].get("t_no_motion")), grev_spread(ok, lambda j: j["transition"]["t_transition"])]
    for v in sorted({j["video"] for j in ok}):
        L.append(f"| {v} | " + " | ".join("계산 불가" if s[v] is None else f"{s[v]:.2f}" for s in sp) + " |")
    L += ["", "## 3. 잡음 바닥과 정면 창", "",
          "| 영상 | 검출기 | 새 정면 창 (s) | d0 | d0 상한 적용 | 새 d_thr (θ 환산) | 20 d0 | 20 d_thr | 경고 |", "|---|---|---|---|---|---|---|---|---|"]
    for j in ok:
        n, t20 = j["transition"], j["transition_20"]
        fw = n["front_window"]
        th = "-" if n.get("d_thr") is None else f"{n['d_thr']:.3f} ({math.degrees(math.acos(max(-1.0, min(1.0, 1 - n['d_thr'])))):.0f}°)"
        L.append(f"| {j['video']} | {j['detector']} | {'-' if fw[0] is None else f'{fw[0]:.2f}–{fw[1]:.2f}'} | {'-' if n.get('d0') is None else format(n['d0'], '.3f')} | "
                 f"{'예' if n.get('d0_capped') else '아니오'} | {th} | {'-' if t20.get('d0') is None else format(t20['d0'], '.3f')} | {'-' if t20.get('d_thr') is None else format(t20['d_thr'], '.3f')} | "
                 f"{'; '.join(n.get('warnings') or []) or '-'} |")
    L += ["", "## 4. 새 가이드 판정", "",
          "항목별 통과(O/X): " + ", ".join(k for _, k in GREV_ITEMS) + " 순서.", "",
          "| 영상 | 검출기 | 새 판정 | " + " | ".join(k for _, k in GREV_ITEMS) + " | θ2d ≤ 45° 프레임 / 기울기 프레임 | 중앙값(°) | 20 판정 | 20 판정과의 차이 | 이유(새) |",
          "|---|---|---|" + "---|" * len(GREV_ITEMS) + "---|---|---|---|---|"]
    for j in ok:
        n, g20 = j["transition"], j["transition_20"]
        m = n["metrics"]
        diff = "같음" if n["guide"] == g20["guide"] else ("미충족 → 충족" if n["guide"] else "충족 → 미충족")
        L.append(f"| {j['video']} | {j['detector']} | {'충족' if n['guide'] else '미충족'} | " + " | ".join(grev_pass(n, k) for k, _ in GREV_ITEMS) +
                 f" | {m['tilt_frames_le_max']} / {m['tilt_frames']} | {hf.cond_f(m['tilt_median_le_max_deg'], 0)} | {'충족' if g20['guide'] else '미충족'} | {diff} | {'; '.join(n['reasons']) or '-'} |")
    L += ["", f"- 새 기준으로 충족: {sum(1 for j in ok if j['transition']['guide'])}/{len(ok)} 조합, 20 기준으로 충족: {sum(1 for j in ok if j['transition_20']['guide'])}/{len(ok)} 조합.", "",
          "## 5. 수동 기준과의 비교", ""]
    have = {v: mv for v, mv in man.items() if mv is not None}
    if not have:
        L.append("- 수동 전환 시각(manual_transition_s)이 비어 있다. 임의로 채우지 않았다.")
    else:
        L += [f"- 목표: 새 방식이 수동 기준과 {tol} s 이내.", "", "| 영상 | 검출기 | 수동 | 기존−수동 | 20−수동 | 21−수동 | 21 목표 |", "|---|---|---|---|---|---|---|"]
        for j in ok:
            mv = have.get(j["video"])
            if mv is None:
                continue
            tn = j["transition"]["t_transition"]
            L.append(f"| {j['video']} | {j['detector']} | {mv:.2f} | {pseg_manual_diff(j['legacy_theta']['t_transition'], mv)} | {pseg_manual_diff(j['transition_20']['t_transition'], mv)} | "
                     f"{pseg_manual_diff(tn, mv)} | {'-' if tn is None else ('달성' if abs(tn - mv) <= tol else '미달')} |")
    L += ["", "## 6. 손목 점 처리별 f 추정", "",
          "f는 fixed_template(공식 추정), 판정은 17의 f 판정이다. 템플릿 범위는 정면 창 기준 점들의 긴 축 길이·폭(px, 손 박스 긴 변 대비 %).", "",
          "| 영상 | 검출기 | 방식 | f fixed_template | f joint_delta | f 판정 | 날카로움 | f 불확실성(상대) | 관절 집합 범위 | 템플릿 긴 축 | 템플릿 폭 | 점 수 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in ok:
        for mode, w in j["wrist"].items():
            if "solutions" not in w:
                L.append(f"| {j['video']} | {j['detector']} | {mode} | 분석 실패: {w.get('failed')} | | | | | | | | |")
                continue
            e = w["template_extent"]
            unc = w["f_rel_uncertainty"]
            L.append(f"| {j['video']} | {j['detector']} | {mode} | {pseg_fval(w['solutions']['fixed_template'])} | {pseg_fval(w['solutions']['joint_delta'])} | {w['verdict']} | {w['sharpness']:.3f} | "
                     f"{'계산 불가' if unc is None or not np.isfinite(unc) else format(unc, '.3f')} | {'계산 불가' if w['jointset_range'] is None or not np.isfinite(w['jointset_range']) else format(w['jointset_range'], '.3f')} | "
                     f"{e['long_px']:.0f} px ({e['long_pct_box']:.0f}%) | {e['width_px']:.0f} px ({e['width_pct_box']:.0f}%) | {e['n_points']} |")
    L += ["", "프레임별 f 신호와 잔차의 비율, 기준 f와의 비교, 채택 규칙 판정은 summary.md에만 적는다.", "",
          "## 7. 기존 결과와 같은지 (기본 설정, wrist_mode keep)", "",
          "| 영상 | 검출기 | f 17 일치 | joint_delta 일치 | 판정 일치 | keep = 기본 분석 | 20 f·판정 일치 | 20 전환·d0 일치 |", "|---|---|---|---|---|---|---|---|"]
    for j in ok:
        s, s20 = j.get("same_as_17"), j.get("same_as_20")
        ox = lambda b: "O" if b else "X"      # noqa: E731
        L.append(f"| {j['video']} | {j['detector']} | {ox(s['f_fixed_template'])} | {ox(s['f_joint_delta'])} | {ox(s['verdict'])} | {ox(s.get('keep_equals_default'))} | "
                 + (f"{ox(s20['f_fixed_template'] and s20['verdict'])} | {ox(s20.get('transition_linear_d') and s20.get('d0'))} |" if s20 else "- | - |"))
    L += ["", f"- 17과 모두 일치: {sum(1 for j in ok if j.get('same_as_17', {}).get('all'))}/{len(ok)}, 20과 모두 일치: {sum(1 for j in ok if j.get('same_as_20', {}).get('all'))}/{sum(1 for j in ok if 'same_as_20' in j)}", "",
          "## 8. 한계", "",
          "- 정면 창 범위(front_search_s)는 잡음 바닥 계산에만 쓴다. f 추정은 17의 정면 창과 3.0초 이후 기울기 구간을 그대로 쓴다.",
          "- 가이드의 기울기 구간은 17의 f 추정이 쓰는 구간(3.0초 이후)이라 새 전환 시각과 무관하다.",
          "- 수동 기준이 없으면 어느 전환 시각이 맞는지는 판단하지 않았다.",
          "- 영상 5개, 검출기 3종이다. 손목 처리의 채택 여부는 summary.md의 규칙 판정에 있고, 통과해도 탐색 결과다.", ""]
    path.write_text("\n".join(L) + "\n")


def grev_summary(path: Path, jobs: list, cfg: dict):
    ref = float(cfg["reference_f_px"])
    rules = cfg["wrist_adoption"]
    ok = [j for j in jobs if "failed" not in j]
    rel = lambda sol: "-" if sol is None or sol["at_grid_edge"] else f"{100 * (sol['f'] / ref - 1):+.1f}%"      # noqa: E731
    modes = list(cfg["f_estimation"]["compare_wrist_modes"])
    L = ["# 실험 21 손목 점 처리 비교와 채택 규칙 (summary.md에만 기록)", "",
         "**탐색 결과이며 공식 f 추정값이 아님.** 1680과의 비교, 프레임별 f 신호, 채택 규칙 판정은 이 파일에만 적는다.", "",
         f"- 비교 기준: 사용자 체커보드 f = {ref:.0f} px (추정·판정에 쓰지 않음). f 신호는 f를 {ref * 0.8:.0f} px와 {ref * 1.2:.0f} px로 둘 때(자세는 프레임마다 다시 풂) 재투영 점의 평균 이동이다.",
         "- 오차 = |f / 1680 − 1|, f는 fixed_template. 격자 경계에 붙은 f는 값이 없는 것으로 보고 가장 나쁜 오차로 센다.", "",
         "## 1. 조합별 f와 f 신호", "",
         "| 영상 | 검출기 | 방식 | f fixed_template | 1680 대비 | f joint_delta | 1680 대비 | f 신호(px) | 잔차(px) | 신호/잔차 중앙값 | 기울기 프레임 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in ok:
        for mode in modes:
            w = j["wrist"].get(mode)
            if not w or "solutions" not in w:
                L.append(f"| {j['video']} | {j['detector']} | {mode} | 분석 실패 | | | | | | | |")
                continue
            ft, jd, fs = w["solutions"]["fixed_template"], w["solutions"]["joint_delta"], w.get("f_signal") or {}
            nz = lambda x, d=2: "-" if x is None else format(x, f".{d}f")      # noqa: E731
            L.append(f"| {j['video']} | {j['detector']} | {mode} | {pseg_fval(ft)} | {rel(ft)} | {pseg_fval(jd)} | {rel(jd)} | {nz(fs.get('signal_px_median'))} | {nz(fs.get('residual_px_median'))} | "
                     f"{nz(fs.get('ratio_median'), 3)} | {fs.get('n_frames', '-')} |")
    L += ["", "## 2. 채택 규칙 판정 (keep 대비, 같은 조합끼리)", "",
          f"규칙(실행 전 yaml에 고정): ① f가 경계에 붙은 조합 수가 keep보다 늘지 않는다 ② 오차가 줄어든 조합 ≥ {rules['min_improved_pairs']} ③ 오차 중앙값이 keep보다 상대 {100 * rules['median_rel_drop']:.0f}% 이상 낮다 "
          f"④ 영상 하나를 빼고 계산해도(5가지) ②의 방향(줄어든 조합 > 늘어난 조합)과 ③의 방향(중앙값이 더 낮다)이 유지된다 ⑤ 세 검출기가 모두 {rules['worse_pp']:.0f}%p 이상 나빠지거나 값을 잃는 영상이 없다.", ""]
    for mode in [m for m in modes if m != "keep"]:
        a = hf.grev_adoption(jobs, ref, rules, mode)
        al, r = a["all"], a["rules"]
        pf = lambda b: "통과" if b else "실패"      # noqa: E731
        fm = lambda x: "경계(값 없음)" if not np.isfinite(x) else f"{100 * x:.1f}%"      # noqa: E731
        L += [f"### {mode}", "",
              f"- 조합 {a['n_pairs']}개. 경계에 붙은 조합: keep {a['edge_base']}, {mode} {a['edge_mode']}. 오차가 줄어든 조합 {al['improved']}, 늘어난 조합 {al['worsened']}. 오차 중앙값: keep {fm(al['median_base'])}, {mode} {fm(al['median_mode'])}.",
              f"- ① {pf(r['1_edge_not_more'])}, ② {pf(r['2_improved_pairs'])}, ③ {pf(r['3_median_drop'])}, ④ {pf(r['4_leave_one_out'])}, ⑤ {pf(r['5_no_video_all_worse'])}"
              + (f" (세 검출기 모두 나빠진 영상: {', '.join(a['worse_videos'])})" if a["worse_videos"] else ""), "",
              "| 뺀 영상 | 조합 | 줄어든 | 늘어난 | 중앙값 keep | 중앙값 " + mode + " | 방향 ② | 방향 ③ |", "|---|---|---|---|---|---|---|---|"]
        for v, x in a["leave_one_out"].items():
            L.append(f"| {v} | {x['n']} | {x['improved']} | {x['worsened']} | {fm(x['median_base'])} | {fm(x['median_mode'])} | {'유지' if x['direction_2'] else '깨짐'} | {'유지' if x['direction_3'] else '깨짐'} |")
        L += ["", f"- **{mode}: " + (f"모든 규칙 통과 → 다음 실험의 기본값 후보({rules['label']})" if a["candidate"] else "규칙 미통과 → 기본값 후보 아님") + "**", ""]
    L += ["## 한계", "",
          "- 규칙을 통과해도 이 5개 영상으로 고른 탐색 결과이다. 확정은 새로 찍은 영상 2~3개로 따로 확인해야 한다.",
          "- 체커보드 f는 사용자 제공 값이며 이 비교와 f 신호의 기준점에만 썼다. 같은 세션 체커보드(1646.5 px)와는 다른 값이다.", ""]
    path.write_text("\n".join(L) + "\n")


def grev_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    man = cfg.get("manual_transition_s") or {}
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=5: None if x is None else round(x, n)      # noqa: E731
    out = {"experiment": "21_guide_revision", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"],
           "transition_method": r["transition_method"], "note": GREV_NOTE + " 기준 f와의 비교 수치와 채택 규칙 판정은 summary.md에만 있다.", "videos": {}}
    for j in r["jobs"]:
        e = out["videos"].setdefault(j["video"], {"manual_transition_s": man.get(j["video"]), "detectors": {}})
        if "failed" in j:
            e["detectors"][j["detector"]] = {"failed": j["failed"]}
            continue
        n, t20 = j["transition"], j["transition_20"]
        wr = {}
        for mode, w in j["wrist"].items():
            if "solutions" not in w:
                wr[mode] = {"failed": w.get("failed")}
                continue
            ft, jd, ex = w["solutions"]["fixed_template"], w["solutions"]["joint_delta"], w["template_extent"]
            wr[mode] = {"f_fixed_template_px": None if ft["at_grid_edge"] else round(ft["f"], 1), "f_status": f"값 없음 ({ft['side']})" if ft["at_grid_edge"] else "value",
                        "f_joint_delta_px": None if jd["at_grid_edge"] else round(jd["f"], 1), "verdict": w["verdict"], "sharpness": rd(w["sharpness"], 4),
                        "f_rel_uncertainty": rd(w["f_rel_uncertainty"], 4), "jointset_range": rd(w["jointset_range"], 4), "n_tilt_frames": w["n_tilt_used"],
                        "template_long_px": rd(ex["long_px"], 1), "template_width_px": rd(ex["width_px"], 1), "template_long_pct_box": rd(ex["long_pct_box"], 1),
                        "template_width_pct_box": rd(ex["width_pct_box"], 1)}
        g = n["metrics"]
        e["detectors"][j["detector"]] = {
            "transition_legacy_theta_s": j["legacy_theta"]["t_transition"], "transition_20_s": t20["t_transition"], "transition_20_no_motion_s": t20.get("t_no_motion"),
            "transition_21_s": n["t_transition"], "front_window_21_s": [rd(x, 3) for x in n["front_window"]], "d0": rd(n.get("d0")), "d0_capped": n.get("d0_capped"),
            "d_thr": rd(n.get("d_thr")), "d0_20": rd(t20.get("d0")), "d_thr_20": rd(t20.get("d_thr")),
            "guide_satisfied_21": n["guide"], "guide_items_21": {k: bool(v) for k, v in n["items"].items()}, "guide_reasons_21": n["reasons"],
            "tilt_frames_le_45": g["tilt_frames_le_max"], "tilt_median_le_45_deg": rd(g["tilt_median_le_max_deg"], 1),
            "guide_satisfied_20": t20["guide"], "guide_reasons_20": t20["reasons"],
            "same_as_17": j["same_as_17"]["all"], "same_as_20": j.get("same_as_20", {}).get("all"), "wrist": wr}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def grev_overlay_lines(fr: dict, d_smooth, d_thr, t_new, t_20, t_legacy) -> list:
    """Text of one overlay frame: time (0.01 s), normalised-video frame number and frame-list position, smoothed d with its threshold, and the transition times."""
    tt = lambda x: "none" if x is None else f"{x:.2f} s"      # noqa: E731
    dv = "-" if d_smooth is None or not np.isfinite(d_smooth) else f"{d_smooth:.3f}"
    return [f"t {fr['time_s']:.2f} s   frame {fr['frame']} (#{fr['index']})", f"d {dv}   thr {'-' if d_thr is None else format(d_thr, '.3f')}",
            f"new {tt(t_new)}   20 {tt(t_20)}   legacy {tt(t_legacy)}"]


def grev_overlay(cfg: dict, run_dir: Path, videos: list, detector: str):
    """One overlay video per video (the detector's tracked keypoints, the segmentation numbers of the 21 run) in <run>/overlay/<video>_<detector>.mp4."""
    r = json.loads((run_dir / "results.json").read_text())
    out_dir = run_dir / "overlay"
    out_dir.mkdir(exist_ok=True)
    made = []
    for video in videos:
        job = next(j for j in r["jobs"] if j["video"] == video and j["detector"] == detector and "failed" not in j)
        vdir = hf.resolve(cfg["sources"][video])
        r17 = json.loads((vdir / detector / "results.json").read_text())
        src = Path(cfg["pipeline"]["input_dir"]).expanduser() / video
        tmp = Path(tempfile.mkdtemp(prefix="hfov21_"))
        try:
            norm = hf.normalize(src, tmp / "n.mkv", cfg["pipeline"])
            W, H = norm["output"]["display_w"], norm["output"]["display_h"]
            frames, k = hf.extract_frames(tmp / "n.mkv", tmp / "f", norm["output"]["fps"], None, cfg["pipeline"]["frames"]["detect_fps"])
            if [(x["index"], x["sha256"]) for x in frames] != [(x["index"], x["sha256"]) for x in r17["frames"]["list"]]:
                raise SystemExit(f"{video}: re-created frames differ from the run's frame list")
            times = np.array([x["time_s"] for x in frames])
            det = hf.load_detections(vdir / detector / "detections.npz")
            hf.check_alignment(frames, det)
            tc = dict(cfg["pipeline"]["tracking"], conf_min=float(cfg["detector_settings"][detector]["conf_min"]))
            tr = hf.track(det, times, (W, H), tc, front_start_s=float(cfg["pipeline"]["phases"]["front_skip_s"]))
            S = job["series"]
            ds = [np.nan if x is None else x for x in S["d_smooth"]]
            n = job["transition"]
            (tmp / "o").mkdir()
            for i, fr in enumerate(frames):
                img = cv2.imread(fr["png"], cv2.IMREAD_COLOR)
                lines = grev_overlay_lines(fr, ds[i], n.get("d_thr"), n["t_transition"], job["transition_20"]["t_transition"], job["legacy_theta"]["t_transition"])
                tag = f"{detector}   " + ("TILT (new)" if n["t_transition"] is not None and fr["time_s"] >= n["t_transition"] - 1e-9 else "front")
                kp = tr["kp"][i] if tr["present"][i] else None
                cv2.imwrite(str(tmp / "o" / f"o_{i:05d}.png"), overlay_draw(img, kp, tr["mask"][i], tr["box"][i] if tr["present"][i] else None, tag, lines))
            out = out_dir / f"{Path(video).stem}_{detector}.mp4"
            overlay_encode(tmp / "o", "o_%05d.png", norm["output"]["fps"] / k, out)
            made.append(str(out))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return made


def grev_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/21_guide_revision.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ov = sub.add_parser("overlay")
    ov.add_argument("--run", required=True)
    ov.add_argument("--videos", default=None)
    ov.add_argument("--detector", default=None)
    ap.add_argument("--videos", default=None, help="comma list overriding yaml videos")
    ap.add_argument("--detectors", default=None, help="comma list overriding yaml detectors")
    ap.add_argument("--transition-method", default=None, choices=["legacy_theta", "linear_d"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(grev_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.cmd == "overlay":
        vids = a.videos.split(",") if a.videos else cfg["videos"]
        made = grev_overlay(cfg, hf.resolve(a.run), vids, a.detector or cfg["overlay"]["detector"])
        print("\n".join(made))
        return made
    if a.videos:
        cfg["videos"] = a.videos.split(",")
    if a.detectors:
        cfg["detectors"] = a.detectors.split(",")
    if a.transition_method:
        cfg["transition_method"] = a.transition_method
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"21 가이드·전환·손목 처리 수정: 영상 {cfg['videos']}, 검출기 {cfg['detectors']}, 전환 방식 {cfg['transition_method']}, 손목 방식 {cfg['f_estimation']['compare_wrist_modes']} → {run_root}")
    t0 = time.time()
    ref20 = {}
    for rp in cfg.get("results_20") or []:
        pth = hf.resolve(rp)
        if pth.exists():
            for j in json.loads(pth.read_text())["jobs"]:
                ref20[(j["video"], j["detector"])] = j
    cjob = {k: cfg[k] for k in ("phase_segmentation", "conditions", "f_estimation", "reference_f_px", "compare_20", "transition_method") if k in cfg}
    args = [(v, d, str(hf.resolve(cfg["sources"][v])), cfg["pipeline"], cjob, cfg["detector_settings"][d]["conf_min"], ref20.get((v, d)))
            for v in cfg["videos"] for d in cfg["detectors"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        jobs = list(ex.map(hf.grev_job, args))
    man = cfg.get("manual_transition_s") or {}
    for j in jobs:
        if "failed" in j:
            log(f"  {j['video']} {j['detector']}: 분석 실패 ({j['failed']})")
            continue
        grev_figure(run_root / "figures" / f"{Path(j['video']).stem}_{j['detector']}.png", j, man.get(j["video"]))
        g = lambda x: "충족" if x else "미충족"      # noqa: E731
        log(f"  {j['video']} {j['detector']}: 기존 {pseg_t(j['legacy_theta']['t_transition'])} s, 20 {pseg_t(j['transition_20']['t_transition'])} s, 21 {pseg_t(j['transition']['t_transition'])} s, "
            f"d0 {j['transition'].get('d0', float('nan')):.3f}{' (상한 적용)' if j['transition'].get('d0_capped') else ''}, 가이드 20 {g(j['transition_20']['guide'])} → 21 {g(j['transition']['guide'])}, "
            f"17과 일치 {'O' if j['same_as_17']['all'] else 'X'}, 20과 일치 {'O' if j.get('same_as_20', {}).get('all') else 'X'}")
    grev_report(run_root / "report.md", jobs, cfg)
    grev_summary(run_root / "summary.md", jobs, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"transition_method": cfg["transition_method"], "jobs": jobs}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_20": cfg["code_20"], "sources": cfg["sources"], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, jobs


# ====================================================================================================
# fresp  (experiment 22: does the candidate respond to f?; new section)
# A1 estimator scale equivariance, A2 centre-cropped / enlarged frames (detector and AnyCalib re-run), A3 the candidate on every video of the range with the
# hand's box size. The candidate (mediapipe_hands + wrist_mode exclude, keep alongside) is fixed in the yaml. B (distance / lens series) is prepared in the yaml only.
#
#     python HandCalib/run.py f-response --config HandCalib/configs/22_f_response.yaml [--videos A,B] [--out DIR]
#     python HandCalib/run.py f-response --config ... results-yaml --run runs/22_f_response/<UTC>
# ====================================================================================================

FRESP_NOTE = "후보(mediapipe_hands + 손목 제외)가 f에 반응하는지 확인한 실험이다. f 추정 방식과 판정 문턱값은 바꾸지 않았다. 확대 검사의 통과는 f를 잰다는 근거가 아니다."


def fresp_ok(x):
    return "-" if x is None else ("통과" if x else "미통과")


def fresp_num(x, d=3):
    return "계산 불가" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def fresp_stats(xs, ys):
    """Spearman rank correlation, slope of f on the box ratio (px per 1.0) and the log-log slope; needs >= 3 videos with a value."""
    from scipy.stats import spearmanr
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3:
        return {"n": int(len(x)), "spearman": None, "slope_px": None, "loglog_slope": None}
    return {"n": int(len(x)), "spearman": float(spearmanr(x, y)[0]), "slope_px": float(np.polyfit(x, y, 1)[0]), "loglog_slope": float(np.polyfit(np.log(x), np.log(y), 1)[0])}


def fresp_a3_points(jobs, mode, exclude=()):
    """(video, front-window box ratio, f) of the videos with an f value; the videos in `exclude` (a3.exclude_from_comparison) are left out."""
    return [(j["video"], j["hand_box_short_ratio_front"], j["original"][mode]["solutions"]["fixed_template"]["f"]) for j in jobs
            if "failed" not in j and j["video"] not in set(exclude) and mode in (j.get("original") or {}) and "solutions" in j["original"][mode]
            and not j["original"][mode]["solutions"]["fixed_template"]["at_grid_edge"]]


def fresp_ref_f(cfg, video):
    """Reference f of a video: its discrimination metadata (other lens) when it has one, else the 1x checkerboard value. Summary use only."""
    m = (cfg.get("discrimination") or {}).get("videos", {}).get(video)
    return float(m["reference_f_px"]) if m and m.get("reference_f_px") else float(cfg["reference_f_px"])


def fresp_figures(out: Path, jobs: list, cfg: dict):
    plt = _plt()
    modes = cfg["f_estimation"]["compare_wrist_modes"]
    ok = [j for j in jobs if "failed" not in j]
    if cfg["a2"]["enabled"] and ok:
        series = [(m, f"손 {m}") for m in modes] + [("anycalib_pinhole", "AnyCalib pinhole"), ("anycalib_gen", "AnyCalib gen")]
        fig, axs = plt.subplots(1, len(series), figsize=(4.2 * len(series), 4), sharey=True)
        zs = [float(z) for z in cfg["a2"]["zoom_factors"]]
        for ax, (key, lab) in zip(axs, series):
            ax.plot(zs, zs, "k--", lw=1, label="f(s)/f(1.0) = s")
            for j in ok:
                e1 = j["a2"].get("1.0")
                if not e1:
                    continue
                xs, ys = [], []
                for z in zs:
                    e = j["a2"].get(str(z))
                    if not e or e.get("excluded"):
                        continue
                    if key.startswith("anycalib"):
                        a, b = (e.get("anycalib") or {}).get(key), (e1.get("anycalib") or {}).get(key)
                        r = a["median"] / b["median"] if a and b and a.get("median") and b.get("median") else None
                    else:
                        r = hf.fresp_ratio_status(((e.get("hand") or {}).get(key) or {}).get("fixed_template"), ((e1.get("hand") or {}).get(key) or {}).get("fixed_template"), z, 1.0)["ratio"]
                    if r is not None:
                        xs.append(z), ys.append(r)
                if xs:
                    ax.plot(xs, ys, "o-", ms=4, lw=1, label=Path(j["video"]).stem[4:])
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("확대 배율 s")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("f(s) / f(1.0)")
        axs[-1].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "a2_zoom_ratio.png", dpi=110)
        plt.close(fig)
    if cfg["a3"]["enabled"] and ok:
        fig, axs = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, m in zip(axs, modes):
            excl = cfg["a3"].get("exclude_from_comparison") or []
            for v, x, f in fresp_a3_points(ok, m):
                if v in excl:
                    ax.plot(x, f, "x", color="#888")
                else:
                    ax.plot(x, f, "o", color="#2a6fbb" if next(j for j in ok if j["video"] == v)["used_in_21"] else "#c47a00")
                ax.annotate(v[4:8], (x, f), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"손 {m}", fontsize=9)
            ax.set_xlabel("정면 창 손 박스 / 영상 짧은 변")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("추정 f (px)  (파랑: 21에서 사용, 주황: 미사용, 회색 x: 0.5x 렌즈·통계 제외)")
        fig.tight_layout()
        fig.savefig(out / "a3_box_vs_f.png", dpi=110)
        plt.close(fig)


def fresp_report(path: Path, rows: list, jobs: list, cfg: dict):
    modes = cfg["f_estimation"]["compare_wrist_modes"]
    a1c, a2c = cfg["a1"], cfg["a2"]
    notes = cfg.get("selection_notes") or {}
    ok = [j for j in jobs if "failed" not in j]
    L = ["# 실험 22: 후보가 실제로 f를 재는지 확인", "", FRESP_NOTE, "",
         "## 1. 대상 영상", "",
         f"- 범위: IMG_{cfg['target_videos']['number_range'][0]} 이상, 제외 목록 {cfg['target_videos']['exclude']}. 21의 손목 비교에 쓰인 영상과 쓰이지 않은 영상을 구분했다.", "",
         "| 영상 | 포함 | 이유 | 21의 손목 비교에 사용 | 비고 |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['video']} | {'포함' if r['included'] else '제외'} | {r['reason'] or '-'} | {'사용' if r['used_in_21'] else '미사용'} | {notes.get(r['video'], '-')} |")
    L += ["", "## 2. 고정한 후보와 방법", "",
          f"- 후보: {cfg['candidate']['detector']} + wrist_mode {cfg['candidate']['wrist_mode']}, f 추정 {cfg['candidate']['estimator']}, 17의 정면 창(phase_mode nominal). 비교용으로 keep도 함께 계산했다. 실행 전 yaml에 고정했다.",
          "- 검출: 17 공식 실행이 있는 영상은 그 검출 결과를 그대로 쓰고, 없는 영상(새 영상)은 정규화한 프레임에 mediapipe를 새로 돌렸다.",
          f"- A1: 원본 검출 키포인트와 박스를 화면 중심(W/2, H/2)에서 s배(s = {a1c['scales']}) 늘려 추정기에 넣는다(재검출 없음). 기대: f가 정확히 s배. 판정: 비율 오차 {100 * a1c['tolerance_rel']:.1f}% 이내.",
          f"- A2: 17의 규격 변환을 거친 프레임을 중심에서 (W/s) × (H/s) 잘라 W × H로 다시 키웠다({a2c['interpolation']}, s = {a2c['zoom_factors']}). 광축이 화면 중심에 그대로 있고 실제 f는 정확히 s배가 된다. "
          f"확대 프레임에 mediapipe와 AnyCalib(pinhole, gen)를 새로 돌렸다. 확대 후 가장자리에 닿는 프레임이 {100 * a2c['edge_rate_max']:.0f}%를 넘는 배율(1.0 대조군 제외)은 그 영상에서 뺐다.",
          f"- A2 판정: 손 방식 f(s)/f(1.0)이 s의 ±{100 * a2c['hand_tol']:.0f}% 안이면 '검출기 해상도 의존성 없음', AnyCalib는 ±{100 * a2c['anycalib_tol']:.0f}% 안이면 '확대에 반응함'. "
          "**손 방식의 통과는 크기 동변성 확인일 뿐 f를 잰다는 근거가 아니다.**", ""]
    if a1c["enabled"]:
        L += ["## 3. A1: 추정기의 크기 동변성", "",
              "f는 fixed_template(joint_delta도 함께 표시). '격자 이동' 열은 f 격자 범위도 s배로 옮긴 진단이다.", "",
              "| 영상 | 방식 | s | f(1) | f(s) | f(s)/f(1) | s 대비 오차 | 판정 | 격자 이동 시 s 대비 오차 | joint_delta s 대비 오차 |", "|---|---|---|---|---|---|---|---|---|---|"]
        n_pass = n_all = 0
        for j in ok:
            for s, byc in j["a1"].items():
                for mode in modes:
                    row = byc.get(mode) or {}
                    ft, jd = row.get("fixed_template") or {}, row.get("joint_delta") or {}
                    ok1 = None if ft.get("ratio_err") is None else abs(ft["ratio_err"]) <= a1c["tolerance_rel"]
                    if ok1 is not None:
                        n_all += 1
                        n_pass += int(ok1)
                    pc_ = lambda x: "계산 불가" if x is None else f"{100 * x:+.3f}%"      # noqa: E731
                    L.append(f"| {j['video']} | {mode} | {s} | {'-' if ft.get('f_1') is None else format(ft['f_1'], '.1f')} | {'-' if ft.get('f_s') is None else format(ft['f_s'], '.1f')} | "
                             f"{fresp_num(ft.get('ratio'), 4)} | {pc_(ft.get('ratio_err'))} | {fresp_ok(ok1)} | {pc_(ft.get('grid_scaled_ratio_err'))} | {pc_(jd.get('ratio_err'))} |")
        L += ["", f"- fixed_template 판정 가능한 조합 {n_all}개 중 통과 {n_pass}개. 경계에 붙은 f는 비율을 계산하지 않았다.",
              "- 미통과가 있으면 원인: f 격자는 s배로 옮겨지지 않고 같은 점에서 비용을 잰 뒤 포물선으로 보간한다. 비용 곡선이 거의 평평해서(날카로움 ≈ 1) 격자 점 위치가 달라지면 보간된 최솟값이 달라진다. "
              "격자도 s배로 옮기면 오차가 줄어드는지가 '격자 이동' 열이다.", ""]
    if a2c["enabled"]:
        L += ["## 4. A2: 잘라서 확대한 영상", "",
              "f(s)/f(1.0)에서 f(1.0)은 같은 자르기·키우기 경로를 거친 s = 1.0 대조군이다. 값 없음 = f가 격자 경계에 붙음(비율 계산 불가).", "",
              "| 영상 | s | 가장자리 접촉 | 대상 | f (px) | f(s)/f(1.0) | s 대비 오차 | 판정 |", "|---|---|---|---|---|---|---|---|"]
        for j in ok:
            e1 = j["a2"].get("1.0")
            for z in a2c["zoom_factors"]:
                e = j["a2"].get(str(z))
                if not e:
                    continue
                if e.get("excluded"):
                    L.append(f"| {j['video']} | {z} | {100 * e['edge_rate']:.1f}% | (제외) | {e['excluded']} | | | |")
                    continue
                zf = float(z)
                for mode in modes:
                    ft = ((e.get("hand") or {}).get(mode) or {}).get("fixed_template")
                    f1 = (((e1 or {}).get("hand") or {}).get(mode) or {}).get("fixed_template")
                    st = hf.fresp_ratio_status(ft, f1, zf, a2c["hand_tol"])
                    L.append(f"| {j['video']} | {z} | {100 * e['edge_rate']:.1f}% | 손 {mode} | {pseg_fval(ft) if ft else '-'} | {fresp_num(st['ratio'], 3)} | "
                             f"{'계산 불가' if st['ratio'] is None else format(100 * st['rel_to_s'], '+.1f') + '%'} | {fresp_ok(st['within'])} |")
                for m in ("anycalib_pinhole", "anycalib_gen"):
                    a, b = (e.get("anycalib") or {}).get(m), ((e1 or {}).get("anycalib") or {}).get(m)
                    if not a:
                        continue
                    r = a["median"] / b["median"] if b and b.get("median") else None
                    ok2 = None if r is None else abs(r / zf - 1.0) <= a2c["anycalib_tol"]
                    L.append(f"| {j['video']} | {z} | {100 * e['edge_rate']:.1f}% | {m} | {a['median']:.1f} | {fresp_num(r, 3)} | {'계산 불가' if r is None else format(100 * (r / zf - 1.0), '+.1f') + '%'} | {fresp_ok(ok2)} |")
        L += ["", "손 방식 / AnyCalib 별 통과 수 (s > 1 조합):", "", "| 대상 | 통과 / 판정 가능 | 영상 단위: 모든 배율 통과 |", "|---|---|---|"]
        for key in [f"손 {m}" for m in modes] + ["anycalib_pinhole", "anycalib_gen"]:
            tot = okc = vids_ok = vids_n = 0
            for j in ok:
                e1 = j["a2"].get("1.0")
                flags = []
                for z in a2c["zoom_factors"]:
                    if float(z) <= 1.0 or not j["a2"].get(str(z)) or j["a2"][str(z)].get("excluded"):
                        continue
                    e, zf = j["a2"][str(z)], float(z)
                    if key.startswith("손"):
                        mode = key[2:]
                        st = hf.fresp_ratio_status(((e.get("hand") or {}).get(mode) or {}).get("fixed_template"), (((e1 or {}).get("hand") or {}).get(mode) or {}).get("fixed_template"), zf, a2c["hand_tol"])
                        flags.append(st["within"])
                    else:
                        a, b = (e.get("anycalib") or {}).get(key), ((e1 or {}).get("anycalib") or {}).get(key)
                        flags.append(None if not (a and b and b.get("median")) else abs(a["median"] / b["median"] / zf - 1.0) <= a2c["anycalib_tol"])
                valid = [f for f in flags if f is not None]
                tot += len(valid)
                okc += sum(valid)
                if valid:
                    vids_n += 1
                    vids_ok += int(all(valid) and len(valid) == len(flags))
            L.append(f"| {key} | {okc} / {tot} | {vids_ok} / {vids_n} 영상 |")
        L += ["", "대조군(s = 1.0)이 원본 분석과 같은지:", "", "| 영상 | 방식 | 원본 f | s = 1.0 f | 같음 |", "|---|---|---|---|---|"]
        for j in ok:
            e1 = j["a2"].get("1.0")
            for mode in modes:
                o = (j.get("original") or {}).get(mode, {}).get("solutions", {}).get("fixed_template")
                c1 = (((e1 or {}).get("hand") or {}).get(mode) or {}).get("fixed_template")
                if o and c1:
                    L.append(f"| {j['video']} | {mode} | {pseg_fval(o)} | {pseg_fval(c1)} | {'O' if o['f'] == c1['f'] else 'X'} |")
        L.append("")
    if cfg["a3"]["enabled"]:
        L += ["## 5. A3: 후보의 원본 영상 결과", "",
              "| 영상 | 구분 | 정면 창 손 박스/짧은 변 | 방식 | f fixed_template | f joint_delta | f 판정 | 날카로움 | f 불확실성(상대) | 관절 집합 범위 |", "|---|---|---|---|---|---|---|---|---|---|"]
        excl = cfg["a3"].get("exclude_from_comparison") or []
        meta = (cfg.get("discrimination") or {}).get("videos") or {}
        for grp, sel in (("21 사용", lambda j: j["used_in_21"] and j["video"] not in excl), ("21 미사용", lambda j: not j["used_in_21"] and j["video"] not in excl),
                         ("0.5x 렌즈 (B, 통계 제외)", lambda j: j["video"] in excl)):
            for j in ok:
                if not sel(j):
                    continue
                for mode in modes:
                    w = j["original"].get(mode, {})
                    if "solutions" not in w:
                        L.append(f"| {j['video']} | {grp} | {j['hand_box_short_ratio_front']:.3f} | {mode} | 분석 실패 | | | | | |")
                        continue
                    L.append(f"| {j['video']} | {grp} | {j['hand_box_short_ratio_front']:.3f} | {mode} | {pseg_fval(w['solutions']['fixed_template'])} | {pseg_fval(w['solutions']['joint_delta'])} | {w['verdict']} | "
                             f"{w['sharpness']:.3f} | {fresp_num(w['f_rel_uncertainty'])} | {fresp_num(w['jointset_range'])} |")
        L += ["", f"손 박스 크기와 추정 f의 관계 (기록용, 영상 수가 적어 상관만으로 결론 내지 않는다). f 값이 있는 영상만 쓰고, {', '.join(excl) or '(없음)'}는 뺐다.", "",
              "| 방식 | 영상 수 | Spearman | 기울기(px / 박스 비율 1.0) | log-log 기울기 (1이면 f가 손 크기에 비례) |", "|---|---|---|---|---|"]
        for m in modes:
            pts = fresp_a3_points(ok, m, excl)
            st = fresp_stats([p[1] for p in pts], [p[2] for p in pts])
            L.append(f"| {m} | {st['n']} | {fresp_num(st['spearman'])} | {fresp_num(st['slope_px'], 0)} | {fresp_num(st['loglog_slope'])} |")
        L += ["", "산점도: figures/a3_box_vs_f.png", ""]
    if (cfg.get("discrimination") or {}).get("enabled") and cfg["a3"]["enabled"]:
        lj = hf.fresp_lens_judgment(ok, cfg["discrimination"]["videos"], cfg["discrimination"]["rules"], cfg["candidate"]["wrist_mode"], modes)
        L += ["## 6. B: 렌즈 변화 판정", "",
              f"- 1배 추정값 = 1배 영상 {len(lj['one_x_videos'])}개({', '.join(v[4:8] for v in lj['one_x_videos'])})의 f 중앙값(후보 설정, 값 없음은 뺌). 추정 비율 = 0.5x 영상의 추정 f ÷ 1배 추정값.",
              f"- 판정은 후보({cfg['candidate']['wrist_mode']})에만 적용한다. 기준 비율과 허용 범위, 기준 f와의 비교는 summary.md에만 적는다.", "",
              "| 영상 | 방식 | 0.5x 영상의 f | 1배 추정값(중앙값) | 1배 f 값 수 | 추정 비율 | 판정 |", "|---|---|---|---|---|---|---|"]
        for v, bm in lj["by_video"].items():
            for mode in modes:
                x = bm[mode]
                L.append(f"| {v} | {mode} | {'-' if x['f_numerator_px'] is None else format(x['f_numerator_px'], '.1f') + ' px'} | {'-' if x['f_1x_median_px'] is None else format(x['f_1x_median_px'], '.1f') + ' px'} | "
                         f"{x['n_1x_values']}/{x['n_1x_videos']} | {fresp_num(x['ratio'])} | {x['verdict'] if x['judged'] else '(판정 대상 아님)'}{' — ' + x['unavailable'] if x['unavailable'] else ''} |")
        L.append("")
    L += ["## 7. 기존 결과와 같은지", "",
          "| 영상 | 21의 keep·exclude f와 판정 일치 |", "|---|---|"]
    for j in ok:
        s21 = j.get("same_as_21")
        L.append(f"| {j['video']} | " + ("21 사용 영상이 아님(비교 대상 없음)" if s21 is None else ", ".join(f"{k} {'O' if v else 'X'}" for k, v in s21.items() if k != "all")) + " |")
    L += ["", f"- 21과 모두 일치한 영상: {sum(1 for j in ok if j.get('same_as_21', {}).get('all'))}/{sum(1 for j in ok if 'same_as_21' in j)}", "",
          "## 8. 한계", "",
          "- 영상 수가 적고(9개) 한 대의 폰, 한 가지 렌즈로 찍은 영상이다. 후보는 21에서 사후 탐색으로 골랐다.",
          "- 확대 검사(A2)의 손 방식 통과는 검출기가 확대된 같은 장면에서 같은 비율의 키포인트를 낸다는 확인이지, 방법이 f를 잰다는 증거가 아니다. 손 크기가 같은 비율로 커지면 f도 같은 비율로 커지는 방식이면 통과한다.",
          "- 렌즈 변화 판정(B)의 0.5x 영상은 같은 폰의 초광각 렌즈로 찍었다. 이 방법은 렌즈 왜곡이 없다고 가정하는데 초광각은 왜곡이 커서, 그 영상의 추정 f에는 f 변화와 왜곡 가정 위반이 섞여 있다. "
          "'f 변화에 반응함'이 나와도 f를 정확히 쟀다는 근거가 아니다.",
          "- 거리 변화 영상으로 하는 판별 테스트(B의 distance_series)는 준비만 했다(yaml의 discrimination). 영상이 들어온 뒤 따로 실행한다.", ""]
    path.write_text("\n".join(L) + "\n")


def fresp_summary(path: Path, jobs: list, cfg: dict):
    ref = float(cfg["reference_f_px"])
    modes = cfg["f_estimation"]["compare_wrist_modes"]
    ok = [j for j in jobs if "failed" not in j]
    excl = cfg["a3"].get("exclude_from_comparison") or []
    rel = lambda sol, r: None if sol is None or sol["at_grid_edge"] else sol["f"] / r - 1.0      # noqa: E731
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    main = [j for j in ok if j["video"] not in excl]
    L = ["# 실험 22 기준 f와의 비교 (summary.md에만 기록)", "", "**탐색 결과이며 공식 f 추정값이 아님.** 기준 f와의 비교와 상대 오차는 이 파일에만 적는다.", "",
         f"- 비교 기준: 1배 렌즈 사용자 체커보드 f = {ref:.0f} px (추정·판정에 쓰지 않음). 같은 세션 체커보드(1646.5 px)는 기준 불확실이다.",
         f"- 0.5x 렌즈 영상({', '.join(excl) or '없음'})은 이 표의 1배 기준 비교, |오차| 중앙값에서 뺐다. 그 영상은 자기 렌즈의 기준 f로 4절에 따로 적는다.", "",
         "## 1. 영상별 오차 (원본 영상, fixed_template, 1배 영상)", "",
         "| 영상 | 21 사용 | " + " | ".join(f"{m} f | {m} 오차" for m in modes) + " |", "|---|---|" + "---|---|" * len(modes)]
    for j in main:
        cells = []
        for m in modes:
            sol = ((j.get("original") or {}).get(m) or {}).get("solutions", {}).get("fixed_template")
            cells += [pseg_fval(sol) if sol else "-", pct(rel(sol, ref)) if sol else "-"]
        L.append(f"| {j['video']} | {'사용' if j['used_in_21'] else '미사용'} | " + " | ".join(cells) + " |")
    L += ["", "## 2. |오차| 중앙값 (21에서 사용한 영상 / 사용하지 않은 영상)", "",
          "값이 있는 영상만으로 계산했고, 값 없음(격자 경계) 영상 수를 함께 적는다.", "",
          "| 방식 | 구분 | 영상 수 | 값 있음 | 값 없음 | \\|오차\\| 중앙값 | 부호 포함 중앙값 | 영상별 오차 |", "|---|---|---|---|---|---|---|---|"]
    for m in modes:
        for grp, flag in (("21 사용", True), ("21 미사용", False)):
            js = [j for j in main if j["used_in_21"] == flag]
            es = [rel(((j.get("original") or {}).get(m) or {}).get("solutions", {}).get("fixed_template"), ref) for j in js]
            v = [e for e in es if e is not None]
            L.append(f"| {m} | {grp} | {len(js)} | {len(v)} | {len(js) - len(v)} | {'계산 불가' if not v else format(100 * float(np.median(np.abs(v))), '.1f') + '%'} | "
                     f"{'계산 불가' if not v else pct(float(np.median(v)))} | " + ", ".join(f"{j['video'][4:8]} {pct(e)}" if e is not None else f"{j['video'][4:8]} 값 없음" for j, e in zip(js, es)) + " |")
    L += ["", "## 3. 확대 영상: 실제 f(= s × 영상 렌즈의 기준 f) 대비 오차", ""]
    if cfg["a2"]["enabled"]:
        L += ["| 영상 | 기준 f | s | " + " | ".join(f"손 {m}" for m in modes) + " | AnyCalib pinhole | AnyCalib gen |", "|---|---|---|" + "---|" * (len(modes) + 2)]
        for j in ok:
            rf = fresp_ref_f(cfg, j["video"])
            for z in cfg["a2"]["zoom_factors"]:
                e = j["a2"].get(str(z))
                if not e or e.get("excluded"):
                    continue
                zf = float(z)
                cells = []
                for m in modes:
                    sol = ((e.get("hand") or {}).get(m) or {}).get("fixed_template")
                    cells.append("-" if sol is None else ("값 없음" if sol["at_grid_edge"] else f"{sol['f']:.0f} ({100 * (sol['f'] / (zf * rf) - 1):+.1f}%)"))
                for k in ("anycalib_pinhole", "anycalib_gen"):
                    a = (e.get("anycalib") or {}).get(k)
                    cells.append("-" if not a else f"{a['median']:.0f} ({100 * (a['median'] / (zf * rf) - 1):+.1f}%)")
                L.append(f"| {j['video']} | {rf:.0f} | {z} | " + " | ".join(cells) + " |")
    disc = cfg.get("discrimination") or {}
    if disc.get("enabled") and cfg["a3"]["enabled"]:
        lj = hf.fresp_lens_judgment(ok, disc["videos"], disc["rules"], cfg["candidate"]["wrist_mode"], modes)
        rl = disc["rules"]["lens_series"]
        L += ["", "## 4. 0.5x 렌즈 영상 (B, 렌즈 변화 판정)", ""]
        for v, bm in lj["by_video"].items():
            mv = disc["videos"][v]
            j = next(x for x in ok if x["video"] == v)
            rf = float(mv["reference_f_px"])
            L += [f"### {v} ({mv['lens']}, 약 {mv['distance_cm']} cm, 기준 f {rf:.0f} px)", "",
                  f"- 판정 규칙(결과를 보기 전 고정): 1배 추정값 = 1배 영상 {len(lj['one_x_videos'])}개 f의 중앙값, 추정 비율 = {v}의 f ÷ 1배 추정값, 기준 비율 = {rf:.0f} ÷ {float(rl['reference_f_1x_px']):.0f}, "
                  f"추정 비율이 기준 비율의 ±{100 * lj['tolerance']:.0f}% 안이면 \"f 변화에 반응함\". 후보에만 적용.", "",
                  "| 방식 | 0.5x 영상 f | 880 대비 | 1배 추정값 | 추정 비율 | 기준 비율 | 허용 범위 | 판정 |", "|---|---|---|---|---|---|---|---|"]
            for mode in modes:
                x = bm[mode]
                sol = ((j.get("original") or {}).get(mode) or {}).get("solutions", {}).get("fixed_template")
                L.append(f"| {mode} | {pseg_fval(sol) if sol else '-'} | {pct(rel(sol, rf)) if sol else '-'} | {'-' if x['f_1x_median_px'] is None else format(x['f_1x_median_px'], '.1f') + ' px'} | "
                         f"{fresp_num(x['ratio'])} | {x['reference_ratio']:.3f} | {x['range'][0]:.3f}–{x['range'][1]:.3f} | "
                         f"{(x['verdict'] if x['judged'] else '(판정 대상 아님) ' + ('범위 안' if x['within'] else '범위 밖' if x['within'] is not None else '계산 불가'))} |")
            w = (j.get("original") or {}).get(cfg["candidate"]["wrist_mode"]) or {}
            e1 = (j.get("a2") or {}).get("1.0") or {}
            L += ["", f"- 후보 {cfg['candidate']['wrist_mode']}: f {pseg_fval(w['solutions']['fixed_template']) if 'solutions' in w else '-'}, 판정 {w.get('verdict', '-')}, 날카로움 {fresp_num(w.get('sharpness'))}, "
                  f"불확실성 {fresp_num(w.get('f_rel_uncertainty'))}, 정면 창 손 박스 {j['hand_box_short_ratio_front']:.3f}",
                  "- AnyCalib (s = 1.0 대조군, 원본과 같은 프레임) 중앙값과 기준 f와의 상대 차이: " + (", ".join(
                      f"{k} {a['median']:.1f} px ({100 * (a['median'] / rf - 1):+.1f}%)" for k, a in (e1.get("anycalib") or {}).items()) or "실행 안 함"), ""]
    L += ["", "## 한계", "",
          "- 영상 9개, 한 대의 폰. 후보는 21에서 사후 탐색으로 골랐고, 21 미사용 영상 중 7181은 20의 진단 결과를 이미 본 영상이다.",
          "- 렌즈 변화 판정은 0.5x 영상 하나(약 30 cm)에 대한 것이다. 1배 영상은 거리가 달라(손 크기가 다름) 1배 추정값 자체가 퍼져 있다.",
          "- 0.5x 영상은 같은 폰의 초광각 렌즈로 찍었다. 이 방법과 AnyCalib pinhole은 왜곡이 없다고 가정하는데 초광각은 왜곡이 커서, 추정 f에는 f 변화와 왜곡 가정 위반이 섞여 있다.",
          "- 체커보드 f는 사용자 제공 값이며 이 비교에만 썼다.", ""]
    path.write_text("\n".join(L) + "\n")


def fresp_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None else round(x, n)      # noqa: E731
    fv = lambda sol: ({"f_px": None, "status": f"값 없음 ({sol['side']})"} if sol["at_grid_edge"] else {"f_px": round(sol["f"], 1), "status": "value"})      # noqa: E731
    out = {"experiment": "22_f_response", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": FRESP_NOTE + " 기준 f와의 비교 수치는 summary.md에만 있다.",
           "candidate": cfg["candidate"], "videos": {}}
    a1t, a2c = cfg["a1"]["tolerance_rel"], cfg["a2"]
    for j in r["jobs"]:
        e = {"used_in_21": j["used_in_21"]}
        if "failed" in j:
            out["videos"][j["video"]] = {**e, "failed": j["failed"]}
            continue
        e["hand_box_short_ratio_front"] = rd(j.get("hand_box_short_ratio_front"))
        e["detections"] = j.get("detections")
        if j.get("original"):
            e["original"] = {md: ({"failed": w.get("failed")} if "solutions" not in w else
                                  {"f_fixed_template": fv(w["solutions"]["fixed_template"]), "f_joint_delta": fv(w["solutions"]["joint_delta"]), "verdict": w["verdict"],
                                   "sharpness": rd(w["sharpness"]), "f_rel_uncertainty": rd(w["f_rel_uncertainty"]), "jointset_range": rd(w["jointset_range"])})
                             for md, w in j["original"].items()}
        if j.get("same_as_21"):
            e["same_as_21"] = j["same_as_21"]["all"]
        if j.get("a1"):
            e["a1"] = {s: {md: {"ratio": rd((row.get("fixed_template") or {}).get("ratio"), 5), "ratio_err": rd((row.get("fixed_template") or {}).get("ratio_err"), 5),
                               "pass": None if (row.get("fixed_template") or {}).get("ratio_err") is None else abs(row["fixed_template"]["ratio_err"]) <= a1t,
                               "grid_scaled_ratio_err": rd((row.get("fixed_template") or {}).get("grid_scaled_ratio_err"), 6)}
                            for md, row in byc.items()} for s, byc in j["a1"].items()}
        if j.get("a2"):
            e1 = j["a2"].get("1.0") or {}
            a2 = {}
            for z, ez in j["a2"].items():
                if ez.get("excluded"):
                    a2[z] = {"edge_rate": rd(ez["edge_rate"]), "excluded": ez["excluded"]}
                    continue
                zf = float(z)
                row = {"edge_rate": rd(ez["edge_rate"])}
                for md, hv in (ez.get("hand") or {}).items():
                    sol = hv.get("fixed_template")
                    st = hf.fresp_ratio_status(sol, ((e1.get("hand") or {}).get(md) or {}).get("fixed_template"), zf, a2c["hand_tol"]) if sol else {"ratio": None, "within": None}
                    row[f"hand_{md}"] = {**(fv(sol) if sol else {}), "ratio_to_1.0": rd(st["ratio"]), "within_tolerance": st["within"]}
                for k2, a in (ez.get("anycalib") or {}).items():
                    b = (e1.get("anycalib") or {}).get(k2)
                    rr = a["median"] / b["median"] if b and b.get("median") else None
                    row[k2] = {"f_median_px": rd(a["median"], 1), "ratio_to_1.0": rd(rr), "within_tolerance": None if rr is None else abs(rr / zf - 1.0) <= a2c["anycalib_tol"]}
                a2[z] = row
            e["a2"] = a2
        out["videos"][j["video"]] = e
    disc = cfg.get("discrimination") or {}
    if disc.get("enabled") and cfg["a3"]["enabled"]:
        lj = hf.fresp_lens_judgment([j for j in r["jobs"] if "failed" not in j], disc["videos"], disc["rules"], cfg["candidate"]["wrist_mode"], cfg["f_estimation"]["compare_wrist_modes"])
        out["lens_series"] = {"one_x_videos": lj["one_x_videos"], "videos": {v: {"lens": disc["videos"][v]["lens"], "distance_cm": disc["videos"][v]["distance_cm"], "modes": {
            md: {"judged": x["judged"], "f_px": None if x["f_numerator_px"] is None else round(x["f_numerator_px"], 1),
                 "f_1x_median_px": None if x["f_1x_median_px"] is None else round(x["f_1x_median_px"], 1), "n_1x_values": x["n_1x_values"],
                 "estimated_ratio": rd(x["ratio"]), "verdict": x["verdict"], "unavailable": x["unavailable"]} for md, x in bm.items()}} for v, bm in lj["by_video"].items()}}
    rules = copy.deepcopy(disc.get("rules"))
    if rules:
        rules["lens_series"].pop("reference_f_1x_px", None)                  # reference f values stay out of the results file
    out["discrimination_rules"] = rules
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def fresp_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/22_f_response.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list narrowing the included videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(fresp_results_yaml(hf.resolve(a.run), cfg))
        return None
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    rows = hf.fresp_target_videos(cfg["target_videos"]["input_dir"], cfg["target_videos"], cfg["used_in_21"])
    vids = [r["video"] for r in rows if r["included"] and (not a.videos or r["video"] in a.videos.split(","))]
    log(f"22 f 반응 확인: 포함 영상 {vids} (제외 {[r['video'] for r in rows if not r['included']]}), 후보 {cfg['candidate']['detector']} + {cfg['candidate']['wrist_mode']} → {run_root}")
    t0 = time.time()
    ref21 = {}
    for rp in cfg.get("results_21") or []:
        pth = hf.resolve(rp)
        if pth.exists():
            for j in json.loads(pth.read_text())["jobs"]:
                if j["detector"] == cfg["candidate"]["detector"]:
                    ref21[j["video"]] = j
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "f_estimation", "phase_segmentation", "a1", "a2", "a3", "detector_settings") if k in cfg}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    n_det = int(cfg["a2"].get("n_workers", 1))
    gpus = list(cfg["a2"].get("gpu_ids") or [])
    log(f"  CPU: 논리 {os.cpu_count()}개, 검출 n_workers {n_det} (절반 {os.cpu_count() // 2} 이하), 분석 workers {cfg['workers']}, AnyCalib GPU {gpus or '(상속)'}")
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            t1 = time.time()
            preps = list(ex.map(hf.fresp_prepare, [(v, cfg["target_videos"]["input_dir"], cfg["pipeline"], cjob, str(tmp_root)) for v in vids]))
            log(f"  1단계 프레임 준비 {time.time() - t1:.0f} s")
            dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
            tasks = []
            for p_ in preps:
                if p_["video"] not in (cfg.get("sources") or {}):
                    tasks.append(((p_["video"], "orig"), p_["frames"]))
                tasks += [((p_["video"], z), fr) for z, fr in p_["zoom"].items()]
            t1 = time.time()
            npz = hf.fresp_detect_many(tasks, dset, run_root / "detection_cache", n_det, log)
            log(f"  2단계 mediapipe 검출 {len(tasks)}개 목록, n_workers {n_det}: {time.time() - t1:.0f} s")
            args = [(p_["video"], p_, str(hf.resolve(cfg["sources"][p_["video"]])) if p_["video"] in (cfg.get("sources") or {}) else None, npz.get((p_["video"], "orig")),
                     {z: npz[(p_["video"], z)] for z in p_["zoom"]}, cfg["pipeline"], cjob, str(tmp_root), ref21.get(p_["video"]), p_["video"] in set(cfg["used_in_21"]),
                     (gpus[i % len(gpus)] if gpus else None)) for i, p_ in enumerate(preps)]
            t1 = time.time()
            with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex3:      # one video per process: a process keeps one CUDA device
                jobs = list(ex3.map(hf.fresp_job, args))
            log(f"  3단계 분석 {time.time() - t1:.0f} s")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)                # the enlarged frames never stay
    for j in jobs:
        if "failed" in j:
            log(f"  {j['video']}: 분석 실패 ({j['failed']})")
            continue
        o = j.get("original") or {}
        fx = lambda m: pseg_fval(o[m]["solutions"]["fixed_template"]) if m in o and "solutions" in o[m] else "-"      # noqa: E731
        log(f"  {j['video']}: 검출 {j['detections']}, 손 박스 {j['hand_box_short_ratio_front']:.3f}, f exclude {fx('exclude')}, keep {fx('keep')}, 21과 일치 {j.get('same_as_21', {}).get('all', '-')}")
    fresp_figures(run_root / "figures", jobs, cfg)
    fresp_report(run_root / "report.md", rows, jobs, cfg)
    fresp_summary(run_root / "summary.md", jobs, cfg)
    with (run_root / "per_frame.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "index", "time_s", "present", "box_short_ratio", "theta2d_deg", "role"])
        for j in jobs:
            ft = j.get("frames_table")
            if ft:
                for i in range(len(ft["index"])):
                    wr.writerow([j["video"], ft["index"][i], f"{ft['time_s'][i]:.4f}", int(ft["present"][i]), f"{ft['box_short_ratio'][i]:.4f}",
                                 "" if ft["theta2d_deg"][i] is None else f"{ft['theta2d_deg'][i]:.2f}", ft["role"][i]])
    for j in jobs:
        j.pop("frames_table", None)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": rows, "jobs": jobs}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_21": cfg["code_21"], "sources": cfg["sources"], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, jobs


# ====================================================================================================
# ferr  (experiment 23: why do the per-video errors differ?; new section)
# Diagnostics of the candidate (mediapipe_hands + wrist_mode exclude + fixed_template) from cached detections only: video-level factors, frame-level preferred f from every tilt
# frame's own cost curve, and f re-estimated on subsets fixed in the yaml. Reference f values (1680 / 880) appear in summary.md, the pull diagnostic and the factor-vs-error
# figure only; report.md, per_frame.csv and the results yaml hold values without them.
#
#     python HandCalib/run.py error-factors --config HandCalib/configs/23_error_factors.yaml [--videos A,B] [--out DIR]
#     python HandCalib/run.py error-factors --config ... results-yaml --run runs/23_error_factors/<UTC>
# ====================================================================================================

FERR_NOTE = "후보 방법의 영상별 오차가 어떤 요인과 함께 움직이는지 찾은 탐색 결과이다. 원인 확정이 아니며, 확정은 통제 촬영이 필요하다."
FERR_FACTORS = [   # (key, group, label, definition)
    ("jd_delta_deg", "정면", "JD 정면 기울기 δ (°)", "joint_delta가 추정한 정면 창 손의 기울기(정면이 아니라고 본 정도)"),
    ("jd_phi_deg", "정면", "JD 기울기 방향 φ (°)", "그 기울기의 방향(템플릿 좌표, 0–180°)"),
    ("d0", "정면", "정면 잡음 바닥 d0", "정면 창 프레임의 d = 1 − s2/s1(템플릿→프레임 아핀) 95번째 백분위수"),
    ("front_jitter_rel_pct", "정면", "정면 키포인트 떨림 (% 박스)", "정면 창에서 프레임 키포인트가 템플릿(닮음 변환)에서 벗어나는 RMS ÷ 손 박스 긴 변"),
    ("finger_ext_min", "정면", "손가락 펴짐 (최소)", "정면 창 MCP→끝 직선 거리 ÷ 세 마디 길이 합, 손가락별 중앙값의 최솟값"),
    ("n_tilt_frames", "동작", "FT 기울기 프레임 수", "f 추정에 쓰인 기울기 구간 프레임"),
    ("theta_median_deg", "동작", "θ2d 중앙값 (°)", "기울기 프레임의 θ2d = arccos(s2/s1)"),
    ("theta_p90_deg", "동작", "θ2d 90번째 백분위수 (°)", "위와 같음"),
    ("theta_over_45_frac", "동작", "θ2d > 45° 비율", "기울기 프레임 중 45° 초과 비율"),
    ("frac_pitch", "동작", "숙임 비율", "분류된 프레임 중. 기울기(회전)축과 손의 긴 축 사이 각도 > 60° (손 폭 방향 축 둘레, 손끝이 멀어짐/가까워짐)"),
    ("frac_roll", "동작", "굴림 비율", "분류된 프레임 중. 그 각도 < 30° (손의 긴 축 둘레로 돌림)"),
    ("frac_mixed", "동작", "혼합 비율", "분류된 프레임 중. 30–60°"),
    ("frac_unclassified", "동작", "미분류 비율", "θ2d가 작아(10° 미만) 축을 정할 수 없는 프레임 비율(기울기 프레임 전체 기준)"),
    ("axis_bins", "동작", "기울기 축 덮임 (/6)", "θ2d ≥ 10° 프레임의 이미지 기울기 방향이 30° 구간 6개 중 2장 이상인 구간 수"),
    ("hand_box_short_ratio_front", "크기·위치", "정면 창 손 박스 / 짧은 변", "정면 창 손 박스 긴 변 ÷ 영상 짧은 변"),
    ("center_offset_pct", "크기·위치", "손 중심 화면 중심 거리 (% 대각선 절반)", "기울기 프레임 손 박스 중심이 화면 중심에서 떨어진 거리의 중앙값"),
    ("edge_rate", "크기·위치", "가장자리 접촉 비율", "검출된 프레임 중 손 박스가 영상 가장자리(2 px 이내)에 닿은 비율"),
    ("bone_ratio_pct", "손 모양 변화", "마디 길이 비 변화 (%)", "기울기 프레임에서 최적 평면 기울기를 되돌린 뒤 손가락 마디·MCP 줄의 길이 ÷ 템플릿 길이 − 1 의 절댓값 중앙값(마디별, 그다음 프레임별 중앙값)"),
    ("spread_change_deg", "손 모양 변화", "손가락 벌어짐 변화 (°)", "검지(MCP→끝)와 새끼 방향 사이 각도가 정면 창 값에서 변한 정도의 절댓값 중앙값"),
    ("wrist_change_deg", "손 모양 변화", "손목 꺾임 변화 (°)", "손목→중지 MCP 방향과 MCP 줄의 수직 방향 사이 각도가 정면 창 값에서 변한 정도의 절댓값 중앙값"),
    ("overlap_rel", "손 모양 변화", "손가락 겹침 (최소 거리 / 박스)", "이웃 손가락의 PIP·DIP·끝 관절 사이 최소 거리 ÷ 손 박스 긴 변(프레임별, 기울기 프레임 중앙값). 작을수록 겹침"),
    ("speed", "동작", "키포인트 속도 (1/s)", "관절 이동 평균 ÷ 손 박스 긴 변 ÷ 시간 간격(기울기 프레임 중앙값)"),
    ("reproj_rms_rel_pct", "검출·추정", "재투영 RMS (% 박스)", "추정 f에서 쓰인 프레임의 재투영 오차 RMS ÷ 손 박스 긴 변"),
    ("detector_diff_wilor_yolo", "검출·추정", "검출기 차이: mediapipe↔wilor_yolo (% 박스)", "같은 기울기 프레임의 추정 관절 평균 거리 ÷ mediapipe 손 박스 긴 변의 중앙값"),
    ("detector_diff_rtmpose_hand", "검출·추정", "검출기 차이: mediapipe↔rtmpose (% 박스)", "위와 같음"),
    ("sharpness", "검출·추정", "날카로움", "min(비용(1.15 f), 비용(f/1.15)) ÷ 비용(f)"),
    ("bootstrap_rel_std", "검출·추정", "부트스트랩 불확실성 (상대 표준편차)", "프레임 블록 부트스트랩 f의 상대 표준편차"),
    ("f_signal_ratio", "검출·추정", "f 신호 ÷ 잔차 중앙값", "기울기 프레임마다 f를 0.8·1.2배로 둘 때 재투영 점의 평균 이동 ÷ 그 프레임의 잔차 RMS의 중앙값. 기준점은 영상 자신의 추정 f"),
]
FERR_FRAME_FEATURES = [("theta2d_deg", "θ2d (°)"), ("axis_angle_deg", "회전축–긴 축 각도 (°)"), ("direction_deg", "이미지 기울기 방향 (°)"), ("resid_rel_pct", "재투영 잔차 (% 박스)"),
                       ("bone_ratio_pct", "마디 길이 비 변화 (%)"), ("spread_change_deg", "벌어짐 변화 (°)"), ("wrist_change_deg", "손목 꺾임 변화 (°)"),
                       ("overlap_rel", "손가락 겹침 (최소 거리/박스)"), ("speed", "속도 (1/s)"), ("center_offset_pct", "손 위치 (% 대각선 절반)")]


def ferr_flat(j: dict) -> dict:
    """The factor values of one video as {key: float | None} (the detector differences flattened)."""
    f = dict(j["factors"])
    for d, v in (f.pop("detector_diff_pct", None) or {}).items():
        f[f"detector_diff_{d}"] = None if v is None else v["median_pct"]
    return f


def ferr_err(j: dict, ref: float):
    ft = j["ft"]
    return None if ft["at_grid_edge"] else ft["f"] / ref - 1.0


def ferr_ref(cfg: dict, video: str) -> float:
    return float(cfg["reference_f"].get(video, cfg["reference_f"]["default"]))


def ferr_edges(vals, step, lo=0.0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    hi = float(np.ceil((v.max() + 1e-9) / step) * step) if len(v) else lo + step
    return list(np.arange(lo, hi + step / 2, step))


def ferr_quantile_edges(vals, n):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < n:
        return []
    e = sorted(set(np.quantile(v, np.linspace(0, 1, n + 1)).tolist()))
    if len(e) > 1:
        e[-1] += 1e-9
    return e


def ferr_pooled_tables(frames: list, fc: dict) -> list:
    """[(title, rows)] of the preferred f per feature bin for the given frames (a list of frame dicts)."""
    f = [r["f_pref"] for r in frames]
    ed = [r["pref_at_edge"] for r in frames]
    out = [("θ2d 구간 (°)", hf.ferr_bin_table([r["theta2d_deg"] for r in frames], f, ed, edges=ferr_edges([r["theta2d_deg"] for r in frames], fc["theta_bin_deg"]))),
           ("기울기 종류", hf.ferr_bin_table([r["kind"] for r in frames], f, ed, labels=["숙임", "굴림", "혼합", "미분류"])),
           ("이미지 기울기 방향 구간 (°)", hf.ferr_bin_table([r["direction_deg"] for r in frames], f, ed, edges=ferr_edges([r["direction_deg"] for r in frames], fc["direction_bin_deg"])))]
    for key, lab in FERR_FRAME_FEATURES:
        if key in ("theta2d_deg", "axis_angle_deg", "direction_deg"):
            continue
        e = ferr_quantile_edges([r[key] for r in frames], int(fc["n_quantiles"]))
        if len(e) > 2:
            out.append((f"{lab} (분위 구간)", hf.ferr_bin_table([r[key] for r in frames], f, ed, edges=e)))
    return out


def ferr_table_md(rows: list) -> list:
    L = ["| 구간 | 프레임 수 | 경계에 붙은 프레임 | 값 있는 프레임 | 선호 f 중앙값 (px) |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['n']} | {r['n_edge']} | {r['n_with_f']} | {'-' if r['median_f'] is None else format(r['median_f'], '.0f')} |")
    return L


def ferr_fmt(key, x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "계산 불가"
    if key in ("n_tilt_frames", "axis_bins"):
        return f"{x:.0f}"
    if key.startswith("frac_") or key in ("theta_over_45_frac", "edge_rate"):
        return f"{100 * x:.0f}%" if key.startswith("frac_") or key == "theta_over_45_frac" else f"{100 * x:.1f}%"
    if key in ("d0", "sharpness", "hand_box_short_ratio_front", "bootstrap_rel_std", "f_rel_uncertainty", "overlap_rel", "f_signal_ratio", "speed"):
        return f"{x:.3f}"
    return f"{x:.1f}"


def ferr_report(path: Path, jobs: list, cfg: dict):
    fc = cfg["factors"]
    ok = [j for j in jobs if "failed" not in j]
    lens = set(cfg["lens_videos"])
    main = [j for j in ok if j["video"] not in lens]
    L = ["# 실험 23: 영상별 오차가 왜 다른가 (진단)", "", FERR_NOTE, "",
         "후보: mediapipe_hands + wrist_mode exclude + fixed_template (22와 같음). 새 검출 없이 캐시한 검출을 읽었다. 기준 f와의 비교는 이 파일에 없다(summary.md).", "",
         "## 1. 요인 정의", "", "| 묶음 | 요인 | 정의 |", "|---|---|---|"]
    for k, g, lab, d in FERR_FACTORS:
        L.append(f"| {g} | {lab} | {d} |")
    L += ["", f"- 기울기 종류: 기울기 축(회전축)은 템플릿에서 가장 줄어드는 방향(아핀 선형부의 작은 특이값 방향 v)에 수직이다. 회전축과 손의 긴 축(템플릿 첫 주성분) 사이 각도 = 90° − angle(v, 긴 축). "
          f"< {fc['roll_max_deg']:.0f}° 굴림, > {fc['pitch_min_deg']:.0f}° 숙임, 그 사이 혼합. θ2d < {fc['class_min_theta_deg']:.0f}° 프레임은 미분류.",
          "- 손 모양 변화 지표는 프레임을 템플릿에 맞춘 최적 아핀(평면 기울기)을 되돌린 점으로 계산한다. 강체 평면 손이면 0에 가깝고, 손가락이 휘거나 벌어지면 커진다.",
          "- 7184(0.5x 렌즈)는 아래 표에 열로 넣었지만 순위 상관과 프레임 구간 통합 표에서는 뺐다.", "",
          "## 2. 영상 단위 요인 표", "",
          "| 요인 | " + " | ".join(j["video"][4:8] + ("(0.5x)" if j["video"] in lens else "") for j in ok) + " |", "|---|" + "---|" * len(ok)]
    flat = {j["video"]: ferr_flat(j) for j in ok}
    for k, g, lab, d in FERR_FACTORS:
        L.append(f"| {lab} | " + " | ".join(ferr_fmt(k, flat[j["video"]].get(k)) for j in ok) + " |")
    L += ["", "검출기 차이는 17 실행에 wilor_yolo, rtmpose 검출이 있는 영상만 계산한다(7184는 없음 → 계산 불가, 새 검출을 하지 않았다).", "",
          "## 3. 프레임 단위: 선호 f 구간별 표", "",
          "기울기 구간 프레임마다 그 프레임 하나의 비용 곡선(같은 f 격자)에서 최솟값 위치를 선호 f로 했다(fixed_template와 같은 포물선 보간). 곡선 최솟값이 격자 경계면 '경계에 붙은 프레임'으로 세고 중앙값에서 뺐다.", ""]
    frames_main = [dict(r, video=j["video"]) for j in main for r in j["frames"]]
    L += [f"### 1배 영상 {len(main)}개 통합 (프레임 {len(frames_main)}장, 경계에 붙은 프레임 {sum(r['pref_at_edge'] for r in frames_main)}장)", ""]
    for title, rows in ferr_pooled_tables(frames_main, fc):
        L += [f"**{title}**", ""] + ferr_table_md(rows) + [""]
    L += ["### 영상별", ""]
    for j in ok:
        fr = j["frames"]
        L += [f"**{j['video']}** (기울기 프레임 {len(fr)}장, 경계에 붙은 {sum(r['pref_at_edge'] for r in fr)}장)", ""]
        f_ = [r["f_pref"] for r in fr]
        e_ = [r["pref_at_edge"] for r in fr]
        L += ferr_table_md(hf.ferr_bin_table([r["theta2d_deg"] for r in fr], f_, e_, edges=ferr_edges([r["theta2d_deg"] for r in fr], fc["theta_bin_deg"]))) + [""]
        L += ferr_table_md(hf.ferr_bin_table([r["kind"] for r in fr], f_, e_, labels=["숙임", "굴림", "혼합", "미분류"])) + [""]
    L += ["## 4. 부분집합으로 다시 추정한 f (fixed_template)", "",
          f"정면 창 프레임 + 아래 기울기 프레임만으로 추정했다. 부분집합은 실행 전 yaml에 고정했고 5개를 모두 적는다. 기울기 프레임이 {fc['min_subset_frames']}장 미만이면 계산하지 않는다. 가장 좋은 것만 고르지 않는다.", "",
          "| 영상 | 전체 기울기 프레임(FT) | " + " | ".join(s["name"] for s in cfg["subsets"]) + " |", "|---|---|" + "---|" * len(cfg["subsets"])]

    def subcell(x):
        if x is None:
            return "-"
        if x.get("skipped"):
            return f"계산 안 함: {x['skipped']}"
        return (f"값 없음 ({x['side']})" if x["at_grid_edge"] else f"{x['f']:.1f} px") + f" ({x['n_tilt']}장)"
    for j in ok:
        L.append(f"| {j['video']} | {subcell(j['subset_all'])} | " + " | ".join(subcell(j["subsets"].get(s["name"])) for s in cfg["subsets"]) + " |")
    L += ["", "## 5. 확인", "",
          "| 영상 | 22의 f·판정과 일치 | 프레임별 비용 곡선 평균 = FT 비용 곡선 (최대 상대 차이) | 다시 만든 비용 = 모델 비용 | 전체 부분집합 = FT |", "|---|---|---|---|---|"]
    for j in ok:
        cc, mc = j["curve_mean_check"], j["cost_check"]
        L.append(f"| {j['video']} | {'-' if 'same_as_22' not in j else ('O' if j['same_as_22'] else 'X')} | {cc['max_rel_diff']:.2e} ({cc['n_frames']}장) | "
                 f"{'O' if abs(mc['model_cost'] - mc['rebuilt_cost']) <= 1e-9 * max(1.0, abs(mc['model_cost'])) else 'X'} | {'O' if j['subset_all']['f'] == j['ft']['f'] else 'X'} |")
    L += ["", "## 6. 한계", "",
          "- 영상 8개(1배)와 한 대의 폰. 요인 하나하나가 영상 수준에서 서로 얽혀 있어 순위 상관으로 원인을 가릴 수 없다.",
          "- 프레임 하나의 비용 곡선은 대부분 평평하거나 경계로 간다. 선호 f는 프레임의 정보 정도에 따라 크게 흔들리므로 구간별 중앙값은 경향만 보여준다.",
          "- 후보 설정과 부분집합 목록은 결과를 보고 바꾸지 않았다. 이 결과로 새 방법을 채택하지 않는다.", ""]
    path.write_text("\n".join(L) + "\n")


def ferr_pull_summary(j: dict, cfg: dict) -> dict:
    """Pull of every tilt frame (cost at the reference f - cost at the estimate, frame alone): the share of the tilt total carried by the top frames and the features of the top
    frames against the rest."""
    rows = j["frames"]
    pulls = np.array([r["pull"] for r in rows])
    n_top = max(1, int(round(float(cfg["factors"]["pull_top_fraction"]) * len(rows))))
    order = np.argsort(-pulls, kind="stable")
    top = [rows[k] for k in order[:n_top]]
    rest = [rows[k] for k in order[n_top:]]
    pos = float(pulls[pulls > 0].sum())
    feat = {}
    for key, lab in FERR_FRAME_FEATURES:
        a, b = [r[key] for r in top], [r[key] for r in rest]
        feat[key] = {"top": float(np.nanmedian(a)) if a else None, "rest": float(np.nanmedian(b)) if b else None}
    kinds = {k: (sum(r["kind"] == k for r in top) / len(top) if top else None) for k in ("숙임", "굴림", "혼합", "미분류")}
    kinds_rest = {k: (sum(r["kind"] == k for r in rest) / len(rest) if rest else None) for k in ("숙임", "굴림", "혼합", "미분류")}
    return {"n_tilt": len(rows), "n_top": n_top, "total_tilt": float(pulls.sum()), "front_total": j["pull_front_total"], "n_front": j["n_front_frames"],
            "n_positive": int((pulls > 0).sum()), "top_share_of_positive": float(pulls[order[:n_top]][pulls[order[:n_top]] > 0].sum() / pos) if pos > 0 else None,
            "features": feat, "kinds_top": kinds, "kinds_rest": kinds_rest}


def ferr_pull_md(jobs: list, cfg: dict, title: str) -> list:
    ok = [j for j in jobs if "failed" not in j]
    fc = cfg["factors"]
    L = [title, "", f"프레임마다 비용(기준 f) − 비용(추정 f)를 구했다(그 프레임 하나의 재투영 비용). 양수가 클수록 그 프레임은 추정값을 기준 f에서 멀리 끌어간 쪽이다. 상위 {100 * fc['pull_top_fraction']:.0f}% 프레임을 나머지와 비교한다.", "",
         "| 영상 | 기준 f | 정면 창 합 | 기울기 합 | 기울기 중 끌림>0 프레임 | 상위 프레임이 차지한 양의 끌림 비율 |", "|---|---|---|---|---|---|"]
    ps = {}
    for j in ok:
        p = ferr_pull_summary(j, cfg)
        ps[j["video"]] = p
        L.append(f"| {j['video']} | {ferr_ref(cfg, j['video']):.0f} | {p['front_total']:.4g} ({p['n_front']}장) | {p['total_tilt']:.4g} ({p['n_tilt']}장) | {p['n_positive']} | "
                 f"{'-' if p['top_share_of_positive'] is None else format(100 * p['top_share_of_positive'], '.0f') + '%'} |")
    L += ["", "### 상위 끌림 프레임의 특징 (영상별 중앙값, 상위 / 나머지)", "",
          "| 영상 | " + " | ".join(lab for _, lab in FERR_FRAME_FEATURES) + " | 상위의 숙임/굴림/혼합 비율 | 나머지의 숙임/굴림/혼합 비율 |", "|---|" + "---|" * (len(FERR_FRAME_FEATURES) + 2)]
    for j in ok:
        p = ps[j["video"]]
        fm = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.3g}"      # noqa: E731
        kt = "/".join("-" if p["kinds_top"][k] is None else f"{100 * p['kinds_top'][k]:.0f}" for k in ("숙임", "굴림", "혼합"))
        kr = "/".join("-" if p["kinds_rest"][k] is None else f"{100 * p['kinds_rest'][k]:.0f}" for k in ("숙임", "굴림", "혼합"))
        L.append(f"| {j['video']} | " + " | ".join(f"{fm(p['features'][k]['top'])} / {fm(p['features'][k]['rest'])}" for k, _ in FERR_FRAME_FEATURES) + f" | {kt}% | {kr}% |")
    frames = [dict(r, video=j["video"]) for j in ok if j["video"] not in set(cfg["lens_videos"]) for r in j["frames"]]
    L += ["", f"### 특징 구간별 평균 끌림 (1배 영상 {len(set(r['video'] for r in frames))}개 통합)", ""]
    for name, key, edges in (("θ2d 구간 (°)", "theta2d_deg", ferr_edges([r["theta2d_deg"] for r in frames], fc["theta_bin_deg"])), ("기울기 종류", "kind", None)):
        L += [f"**{name}**", "", "| 구간 | 프레임 수 | 평균 끌림 | 양의 끌림 프레임 비율 |", "|---|---|---|---|"]
        cats = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)] if edges else ["숙임", "굴림", "혼합", "미분류"]
        for cat in cats:
            sel = [r for r in frames if (cat[0] <= r[key] < cat[1] if edges else r[key] == cat)]
            L.append(f"| {('%g–%g' % cat) if edges else cat} | {len(sel)} | {'-' if not sel else format(float(np.mean([r['pull'] for r in sel])), '.3g')} | "
                     f"{'-' if not sel else format(100 * float(np.mean([r['pull'] > 0 for r in sel])), '.0f') + '%'} |")
        L.append("")
    return L


def ferr_summary(path: Path, jobs: list, cfg: dict):
    from scipy.stats import spearmanr
    ok = [j for j in jobs if "failed" not in j]
    lens = set(cfg["lens_videos"])
    main = [j for j in ok if j["video"] not in lens]
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    L = ["# 실험 23 기준 f와의 비교, 요인 상관, 끌림, 부분집합 (summary.md)", "", f"**{FERR_NOTE} 모든 분석이 탐색용이다.**", "",
         "기준 f(1배 1680 px, 0.5x 880 px)는 사용자 체커보드 값이며 추정에 쓰지 않았다. 1배 영상 8개로만 상관을 낸다.", "",
         "## 1. 영상별 FT 오차", "", "| 영상 | 기준 f | FT f | 오차 | 판정 |", "|---|---|---|---|---|"]
    for j in ok:
        ft = j["ft"]
        L.append(f"| {j['video']}{' (0.5x)' if j['video'] in lens else ''} | {ferr_ref(cfg, j['video']):.0f} | {'값 없음 (' + ft['side'] + ')' if ft['at_grid_edge'] else format(ft['f'], '.1f') + ' px'} | {pct(ferr_err(j, ferr_ref(cfg, j['video'])))} | {ft['verdict']} |")
    err = {j["video"]: ferr_err(j, ferr_ref(cfg, j["video"])) for j in main}
    L += ["", "## 2. 요인과 오차의 순위 상관 (Spearman, 탐색용)", "",
          f"영상 {len(main)}개뿐이라 상관만으로 결론 내지 않는다. 한 요인이 오차와 함께 움직이는 것은 인과가 아니다(요인끼리도 얽혀 있다). 값이 없는 영상은 뺐다.", "",
          "| 묶음 | 요인 | 영상 수 | ρ (부호 포함 오차) | ρ (절댓값 오차) |", "|---|---|---|---|---|"]
    flat = {j["video"]: ferr_flat(j) for j in main}
    for k, g, lab, d in FERR_FACTORS:
        xs = [(flat[v].get(k), e) for v, e in err.items() if flat[v].get(k) is not None and np.isfinite(flat[v].get(k)) and e is not None]
        if len(xs) < 4 or len({x for x, _ in xs}) < 2:
            L.append(f"| {g} | {lab} | {len(xs)} | 계산 불가 | 계산 불가 |")
            continue
        x = [a for a, _ in xs]
        L.append(f"| {g} | {lab} | {len(xs)} | {spearmanr(x, [e for _, e in xs])[0]:+.2f} | {spearmanr(x, [abs(e) for _, e in xs])[0]:+.2f} |")
    L += [""] + ferr_pull_md(jobs, cfg, "## 3. 끌림 분석 (기준 f 사용: summary.md와 진단 파일에만)")
    L += ["## 4. 부분집합별 오차 (fixed_template, 5개 모두)", "",
          "| 영상 | 기준 f | 전체(FT) | " + " | ".join(s["name"] for s in cfg["subsets"]) + " |", "|---|---|---|" + "---|" * len(cfg["subsets"])]

    def ecell(x, ref):
        if x is None:
            return "-"
        if x.get("skipped"):
            return f"계산 안 함 ({x['skipped']})"
        return ("값 없음 (" + x["side"] + ")") if x["at_grid_edge"] else f"{x['f']:.0f} ({100 * (x['f'] / ref - 1):+.1f}%, {x['n_tilt']}장)"
    for j in ok:
        ref = ferr_ref(cfg, j["video"])
        L.append(f"| {j['video']}{' (0.5x)' if j['video'] in lens else ''} | {ref:.0f} | {ecell(j['subset_all'], ref)} | " + " | ".join(ecell(j["subsets"].get(s["name"]), ref) for s in cfg["subsets"]) + " |")
    L += ["", "부분집합 5개를 모두 적었다. 이 결과로 새 방법을 채택하지 않는다.", "",
          "## 한계", "", "- 영상 8개(1배), 한 대의 폰, 프레임은 같은 영상 안에서 서로 독립이 아니다.", "- 7184(0.5x)는 왜곡이 큰 초광각이라 따로 보며, 요인 상관에는 넣지 않았다.", ""]
    path.write_text("\n".join(L) + "\n")


def ferr_figures(out: Path, jobs: list, cfg: dict):
    plt = _plt()
    ok = [j for j in jobs if "failed" not in j]
    lens = set(cfg["lens_videos"])
    main = [j for j in ok if j["video"] not in lens]
    err = {j["video"]: ferr_err(j, ferr_ref(cfg, j["video"])) for j in main}
    flat = {j["video"]: ferr_flat(j) for j in main}
    n = len(FERR_FACTORS)
    cols = 5
    fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(3.4 * cols, 2.9 * ((n + cols - 1) // cols)))
    for ax, (k, g, lab, d) in zip(axs.ravel(), FERR_FACTORS):
        for v, e in err.items():
            x = flat[v].get(k)
            if x is not None and np.isfinite(x) and e is not None:
                ax.plot(x, 100 * e, "o", ms=4, color="#2a6fbb")
                ax.annotate(v[4:8], (x, 100 * e), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_title(lab, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.supylabel("FT 오차 (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "factor_vs_error.png", dpi=100)
    plt.close(fig)
    cols = 3
    fig, axs = plt.subplots((len(ok) + cols - 1) // cols, cols, figsize=(4.6 * cols, 3.2 * ((len(ok) + cols - 1) // cols)), squeeze=False)
    colors = {"숙임": "#c00", "굴림": "#2a6fbb", "혼합": "#3a9a4a", "미분류": "#999"}
    for ax, j in zip(axs.ravel(), ok):
        for kind, col in colors.items():
            sel = [r for r in j["frames"] if r["kind"] == kind]
            ax.plot([r["time_s"] for r in sel if not r["pref_at_edge"]], [r["f_pref"] for r in sel if not r["pref_at_edge"]], ".", ms=4, color=col, label=kind)
            ax.plot([r["time_s"] for r in sel if r["pref_at_edge"]], [r["f_pref"] for r in sel if r["pref_at_edge"]], "x", ms=3, color=col, alpha=0.5)
        ax.axhline(j["ft"]["f"], color="k", lw=0.8, ls="--")
        ax.set_title(j["video"] + (" (0.5x)" if j["video"] in lens else ""), fontsize=8)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
    axs.ravel()[0].legend(fontsize=6, loc="upper right")
    fig.supxlabel("시간 (s)  (점: 프레임별 선호 f, x: 격자 경계, 점선: 영상의 FT f, 색: 기울기 종류)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "frame_preferred_f.png", dpi=100)
    plt.close(fig)
    frames = [r for j in main for r in j["frames"] if not r["pref_at_edge"]]
    fc = cfg["factors"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    edges = ferr_edges([r["theta2d_deg"] for r in frames], fc["theta_bin_deg"])
    data = [[r["f_pref"] for r in frames if lo <= r["theta2d_deg"] < hi] for lo, hi in zip(edges[:-1], edges[1:])]
    lab0 = [f"{lo:g}–{hi:g}" for (lo, hi), d in zip(zip(edges[:-1], edges[1:]), data) if d]
    axs[0].boxplot([d for d in data if d], showfliers=False)
    axs[0].set_xticks(range(1, len(lab0) + 1))
    axs[0].set_xticklabels(lab0)
    axs[0].set_xlabel("θ2d 구간 (°)")
    axs[0].set_ylabel("프레임별 선호 f (px, 경계 제외)")
    kd = [[r["f_pref"] for r in frames if r["kind"] == k] for k in ("숙임", "굴림", "혼합", "미분류")]
    lab1 = [k for k, d in zip(("숙임", "굴림", "혼합", "미분류"), kd) if d]
    axs[1].boxplot([d for d in kd if d], showfliers=False)
    axs[1].set_xticks(range(1, len(lab1) + 1))
    axs[1].set_xticklabels(lab1)
    axs[1].set_xlabel("기울기 종류")
    for ax in axs:
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "preferred_f_by_bins.png", dpi=110)
    plt.close(fig)


def ferr_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    fv = lambda x: ({"skipped": x["skipped"]} if x.get("skipped") else ({"f_px": None, "status": f"값 없음 ({x['side']})", "n_tilt_frames": x["n_tilt"]} if x["at_grid_edge"] else      # noqa: E731
                                                                          {"f_px": round(x["f"], 1), "status": "value", "n_tilt_frames": x["n_tilt"]}))
    out = {"experiment": "23_error_factors", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"],
           "note": FERR_NOTE + " 기준 f와의 비교와 순위 상관은 summary.md에만 있다.", "candidate": cfg["candidate"], "videos": {}}
    for j in r["jobs"]:
        if "failed" in j:
            out["videos"][j["video"]] = {"failed": j["failed"]}
            continue
        flat = ferr_flat(j)
        out["videos"][j["video"]] = {
            "lens_series": j["video"] in set(cfg["lens_videos"]), "detections": j["detections"], "same_as_22": j.get("same_as_22"),
            "ft": {"f_px": None if j["ft"]["at_grid_edge"] else round(j["ft"]["f"], 1), "status": "value" if not j["ft"]["at_grid_edge"] else f"값 없음 ({j['ft']['side']})", "verdict": j["ft"]["verdict"]},
            "factors": {k: rd(v) for k, v in flat.items()}, "curve_mean_check_max_rel_diff": rd(j["curve_mean_check"]["max_rel_diff"], 12),
            "subsets": {"all_tilt_frames": fv(j["subset_all"]), **{name: fv(x) for name, x in j["subsets"].items()}},
            "frames": {"n_tilt": len(j["frames"]), "n_pref_at_grid_edge": sum(fr["pref_at_edge"] for fr in j["frames"]),
                       "kinds": {k: sum(fr["kind"] == k for fr in j["frames"]) for k in ("숙임", "굴림", "혼합", "미분류")}}}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def ferr_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/23_error_factors.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list overriding yaml videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(ferr_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.videos:
        cfg["videos"] = a.videos.split(",")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"23 영상별 오차 요인 진단: 영상 {cfg['videos']}, 후보 {cfg['candidate']['detector']} + {cfg['candidate']['wrist_mode']} → {run_root}")
    t0 = time.time()
    ref22 = {}
    p22 = hf.resolve(cfg["results_22"])
    if p22.exists():
        ref22 = {j["video"]: j for j in json.loads(p22.read_text())["jobs"]}
    tmp_root = run_root / "_tmp"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "compare_detectors", "factors", "subsets", "detector_settings")}
    args = [(v, str(hf.resolve(cfg["sources"][v])) if v in (cfg.get("sources") or {}) else None, cfg["input_dir"], cfg["pipeline"], cjob, str(hf.resolve(cfg["cache_22"])),
             str(tmp_root), ref22.get(v), ferr_ref(cfg, v)) for v in cfg["videos"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            jobs = list(ex.map(hf.ferr_video, args))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    for j in jobs:
        if "failed" in j:
            log(f"  {j['video']}: 분석 실패 ({j['failed']})")
            continue
        ft = j["ft"]
        log(f"  {j['video']}: FT {'값 없음' if ft['at_grid_edge'] else format(ft['f'], '.1f') + ' px'}, 판정 {ft['verdict']}, 기울기 프레임 {len(j['frames'])}장, "
            f"22와 일치 {j.get('same_as_22', '-')}, 곡선 평균 차 {j['curve_mean_check']['max_rel_diff']:.1e}")
    ferr_figures(run_root / "figures", jobs, cfg)
    ferr_report(run_root / "report.md", jobs, cfg)
    ferr_summary(run_root / "summary.md", jobs, cfg)
    (run_root / "diagnostic_pull.md").write_text("\n".join(ferr_pull_md(jobs, cfg, "# 실험 23 끌림 진단 (기준 f 사용, 진단 파일)")) + "\n")
    cols = ["time_s", "theta2d_deg", "kind", "axis_angle_deg", "direction_deg", "resid_rel_pct", "bone_ratio_pct", "spread_change_deg", "wrist_change_deg", "overlap_rel", "speed",
            "center_offset_pct", "f_pref", "pref_at_edge", "sharpness"]
    with (run_root / "per_frame.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "index"] + cols)
        for j in jobs:
            for r in j.get("frames") or []:
                wr.writerow([j["video"], r["index"]] + [f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols])
    with (run_root / "pull_per_frame.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "index", "pull"])
        for j in jobs:
            for r in j.get("frames") or []:
                wr.writerow([j["video"], r["index"], f"{r['pull']:.6g}"])
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"jobs": jobs}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_22": cfg["code_22"], "sources": cfg["sources"], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, jobs


# ====================================================================================================
# long  (experiment 25: the candidate on long new videos; new section)
# Order of the run (fixed in the specification): 1 code committed; 2 `bias`: the common bias b of the baseline videos (22's 1x videos, cached detections) is computed and written
# to the yaml and a log, then committed; 3 the new videos are detected and estimated; 4 H1-H4 are judged. New videos never enter b or the candidate.
#
#     python HandCalib/run.py long-video-check --config HandCalib/configs/25_long_video_check.yaml bias
#     python HandCalib/run.py long-video-check --config HandCalib/configs/25_long_video_check.yaml [--videos A,B]
#     python HandCalib/run.py long-video-check --config ... results-yaml --run runs/25_long_video_check/<UTC>
# ====================================================================================================

LONG_NOTE = "긴 새 영상으로 후보 방법(mediapipe_hands + 손목 제외 + fixed_template)을 다시 평가한 검증 실험이다. 후보 설정과 판정 문턱값은 바꾸지 않았다. 기준 f와의 비교는 summary.md에만 있다."


def long_fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def long_fval(r):
    """'값 없음 (상한/하한)' for a grid-edge solution, else 'f px'."""
    return "값 없음 (" + str(r["side"]) + ")" if r["at_grid_edge"] else f"{r['f']:.1f} px"


def long_patch_yaml(path: Path, values: dict):
    """Writes the baseline values into the `baseline:` block of the config (text edit that keeps comments)."""
    t = path.read_text()
    for k, v in values.items():
        t, n = re.subn(rf"^(  {k}:)[^\n#]*(#[^\n]*)?$", lambda m: f"{m.group(1)} {v}" + (f"  {m.group(2)}" if m.group(2) else ""), t, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"baseline key not found in the yaml: {k}")
    path.write_text(t)


def long_base_stats(jobs: list, ref: float) -> dict:
    ok = [j for j in jobs if "ft" in j and not j["ft"]["at_grid_edge"]]
    e = np.array([j["ft"]["f"] / ref - 1.0 for j in ok])
    lf = np.log([j["ft"]["f"] for j in ok])
    return {"n": len(ok), "median_abs_err": float(np.median(np.abs(e))) if len(e) else None, "median_signed_err": float(np.median(e)) if len(e) else None,
            "log_f_sd": float(np.std(lf, ddof=1)) if len(lf) > 1 else None, "boot_log_std_median": float(np.median([j["ft"]["boot_log_std"] for j in ok])) if ok else None}


def long_bias_main(cfg: dict, cfg_path: Path, log) -> dict:
    """Step 2: the baseline videos only. Returns the baseline record (also written next to the run folder and into the yaml)."""
    hy = cfg["hypotheses"]
    ref = float(hy["reference_f_px"])
    run_root = hf.resolve(cfg["output"]["root"]) / f"{utc_stamp()}_bias"
    run_root.mkdir(parents=True)
    log2 = Log(run_root / "log.txt")
    log2(f"25 편향 보정값 b: 기존 영상 {cfg['baseline_videos']}만 사용(새 영상 없음) → {run_root}")
    cjob = {k: cfg[k] for k in ("candidate", "exclusion", "hypotheses", "detector_settings", "guide", "compare_estimates", "anycalib_guard") if k in cfg}
    cjob["compare_estimates"] = False
    args = [(v, "baseline", str(hf.resolve(cfg["sources"][v])), None, None, cfg["pipeline"], cjob, str(run_root), None) for v in cfg["baseline_videos"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        jobs = list(ex.map(hf.long_video, args))
    bad = [j["video"] for j in jobs if "ft" not in j]
    if bad:
        raise SystemExit(f"baseline analysis failed: {bad}")
    grids = [np.array(j["curve"]["grid"]) for j in jobs]
    assert all(np.allclose(grids[0], g) for g in grids), "the f grids differ"
    curves = np.array([j["curve"]["cost"] for j in jobs])
    pooled = hf.long_pooled_estimate(curves, grids[0])
    if pooled["at_grid_edge"]:
        raise SystemExit("the pooled baseline estimate is on the f grid edge: b is not defined")
    b = float(np.log(pooled["f"] / ref))
    st = long_base_stats(jobs, ref)
    # re-check against 22 and 23
    r22 = {j["video"]: j for j in json.loads(hf.resolve(cfg["results_22"]).read_text())["jobs"]}
    r23 = {j["video"]: j for j in json.loads(hf.resolve(cfg["results_23"]).read_text())["jobs"]}
    for j in jobs:
        s22 = r22[j["video"]]["original"][cfg["candidate"]["wrist_mode"]]["solutions"]["fixed_template"]["f"]
        j["same_as_22"] = bool(s22 == j["ft"]["f"])
        j["rel_std_as_23"] = bool(abs(r23[j["video"]]["factors"]["bootstrap_rel_std"] - j["ft"]["boot_rel_std"]) < 1e-12)
    rec = {"videos": jobs, "pooled": pooled, "b": b, "reference_f_px": ref, "stats": st, "grid": grids[0].tolist()}
    jp = run_root / "baseline.json"
    jp.write_text(json.dumps(hf.jsonable(rec), ensure_ascii=False, indent=1))
    log2(f"합동 추정 f = {pooled['f']:.2f} px, b = log(f_합동 / {ref:.0f}) = {b:+.6f} (f_합동 / 기준 = {np.exp(b):.4f})")
    log2(f"기존 {st['n']}개 영상: |오차| 중앙값 {100 * st['median_abs_err']:.1f}%, 부호 포함 중앙값 {100 * st['median_signed_err']:+.1f}%, log f 표준편차 {st['log_f_sd']:.3f}, 부트스트랩 로그 SD 중앙값 {st['boot_log_std_median']:.4f}")
    log2(f"22와 같은 f: {sum(j['same_as_22'] for j in jobs)}/{len(jobs)}, 23과 같은 부트스트랩 상대 표준편차: {sum(j['rel_std_as_23'] for j in jobs)}/{len(jobs)}")
    long_patch_yaml(cfg_path, {"bias_b": f"{b!r}", "json": str(jp.relative_to(hf.PROJECT_ROOT)) if str(jp).startswith(str(hf.PROJECT_ROOT)) else str(jp), "bootstrap_log_std_median": repr(st["boot_log_std_median"]),
                               "log_f_sd_between_videos": repr(st["log_f_sd"]), "pooled_f_px": repr(pooled["f"])})
    log2(f"yaml에 기록: {cfg_path}")
    log2.close()
    return rec


def long_figures(out: Path, base: dict, new_ok: list, pooled_new: dict | None, cfg: dict):
    plt = _plt()
    bj = [j for j in base["videos"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = 0
    ticks = []
    for grp, js, col in (("기존", bj, "#2a6fbb"), ("새 영상", new_ok, "#c47a00")):
        for j in js:
            f, s = j["ft"]["f"], j["ft"]["boot_log_std"]
            if j["ft"]["at_grid_edge"]:
                ax.plot(x, f, "x", color=col)
            else:
                ax.errorbar(x, f, yerr=[[f - f * np.exp(-s)], [f * np.exp(s) - f]], fmt="o", color=col, capsize=3)
            ticks.append((x, j["video"][4:8] + ("\n새" if grp == "새 영상" else "")))
            x += 1
        x += 1
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=8)
    ax.set_ylabel("FT f (px), 막대 = 블록 부트스트랩 ±1σ(로그)")
    ax.set_title("영상별 f와 부트스트랩 구간 (파랑: 기존 8개, 주황: 새 영상)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "f_with_bootstrap.png", dpi=110)
    plt.close(fig)
    grid = np.array(base["grid"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j in bj:
        c = np.array(j["curve"]["cost"]); ax.plot(grid, c / c.min(), color="#9bb7d9", lw=0.9)
    ax.plot(grid, np.array([j["curve"]["cost"] for j in bj]).sum(0) / np.array([j["curve"]["cost"] for j in bj]).sum(0).min(), color="#2a6fbb", lw=2.2, label="기존 8개 합동")
    for j in new_ok:
        c = np.array(j["curve"]["cost"]); ax.plot(grid, c / c.min(), color="#f0b66a", lw=0.9)
    if new_ok:
        pc_ = np.array([j["curve"]["cost"] for j in new_ok]).sum(0)
        ax.plot(grid, pc_ / pc_.min(), color="#c47a00", lw=2.2, label="새 영상 합동")
    ax.set_xscale("log")
    ax.set_xlabel("f (px)")
    ax.set_ylabel("비용 / 최솟값")
    ax.set_ylim(0.99, 1.15)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pooled_cost_curves.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bt = [x for j in bj for x in j["motion"]["theta_tilt"]]
    nt = [x for j in new_ok for x in j["motion"]["theta_tilt"]]
    bins = np.arange(0, 91, 5)
    ax.hist(bt, bins=bins, density=True, alpha=0.55, color="#2a6fbb", label=f"기존 8개 (프레임 {len(bt)})")
    if nt:
        ax.hist(nt, bins=bins, density=True, alpha=0.55, color="#c47a00", label=f"새 영상 (프레임 {len(nt)})")
    ax.set_xlabel("θ2d (°)")
    ax.set_ylabel("밀도")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "theta_distribution.png", dpi=110)
    plt.close(fig)


def long_report(path: Path, listing: list, new_all: list, new_ok: list, base: dict, cfg: dict):
    ex = cfg["exclusion"]
    L = ["# 실험 25: 긴 영상으로 후보 방법 재검증", "", LONG_NOTE, "",
         "후보: mediapipe_hands + wrist_mode exclude + fixed_template (22와 같음). 새 영상은 후보를 고를 때 쓰지 않았다(검증). 비교용(판정에 쓰지 않음): keep FT, exclude JD, 21 가이드, AnyCalib.", "",
         "## 1. 영상 목록과 제외", "", f"제외 규칙(실행 전 고정): 유효한 정면 창(1초) 없음, 가장자리 접촉 프레임 > {100 * ex['edge_rate_max']:.0f}%, 손 검출 실패 프레임 > {100 * ex['detect_fail_max']:.0f}%.", "",
         "| 영상 | 길이(초) | 해상도(처리 후) | 검출된 프레임 비율 | 가장자리 접촉 비율 | 정면 창 | 포함 | 이유 |", "|---|---|---|---|---|---|---|---|"]
    for j in new_all:
        fw = j.get("front_window") or [None, None]
        L.append(f"| {j['video']} | {j['duration_s']:.1f} | {j['size'][0]}x{j['size'][1]} | {100 * j['detect_rate']:.1f}% | {100 * j['edge_rate']:.1f}% | "
                 f"{'-' if fw[0] is None else f'{fw[0]:.2f}–{fw[1]:.2f} s'} | {'제외' if j['excluded'] else ('포함' if 'ft' in j else '분석 실패')} | {'; '.join(j['excluded']) or j.get('failed', '-')} |")
    L += ["", "## 2. 영상별 촬영 지표", "",
          "| 영상 | 영상 길이(초) | 기울기 구간 길이(초) | 사용한 기울기 프레임 | θ2d 중앙값(°) | 90번째 백분위수(°) | 45° 초과 비율 | 최댓값(°) |", "|---|---|---|---|---|---|---|---|"]
    for j in new_ok:
        m = j["motion"]
        L.append(f"| {j['video']} | {j['duration_s']:.1f} | {m['tilt_duration_s']:.1f} | {m['n_tilt_frames']} | {long_fmt(m['theta_median_deg'], 1)} | {long_fmt(m['theta_p90_deg'], 1)} | {100 * m['theta_over_45_frac']:.0f}% | {long_fmt(m['theta_max_deg'], 1)} |")
    L += ["", "기존 8개 영상(참고): 기울기 구간 길이 중앙값 %.1f초, 기울기 프레임 중앙값 %d장." % (np.median([j["motion"]["tilt_duration_s"] for j in base["videos"]]), int(np.median([j["motion"]["n_tilt_frames"] for j in base["videos"]]))), "",
          "## 3. 21의 새 가이드 판정과 전환 시각 (linear_d)", "",
          "| 영상 | 전환 시각(초) | 가이드 | 항목별 통과 | 이유 |", "|---|---|---|---|---|"]
    for j in new_ok:
        g = j.get("guide") or {}
        if "guide" not in g:
            L.append(f"| {j['video']} | - | 계산 안 함 | | {g.get('failed', '비교용 계산 끔')} |")
            continue
        L.append(f"| {j['video']} | {long_fmt(g['t_transition'], 2)} | {'충족' if g['guide'] else '미충족'} | " + ", ".join(f"{k}:{'O' if v else 'X'}" for k, v in g["items"].items()) + f" | {'; '.join(g['reasons']) or '-'} |")
    L += ["", "## 4. f, 판정, 날카로움, 불확실성 (후보: 손목 제외 fixed_template)", "",
          "| 영상 | f | f 판정 | 날카로움 | 블록 부트스트랩 로그 SD | 상대 SD(22·23과 같은 지표) | 결합 불확실성 |", "|---|---|---|---|---|---|---|"]
    for j in new_ok:
        f = j["ft"]
        L.append(f"| {j['video']} | {long_fval(f)} | {f['verdict']} | {f['sharpness']:.3f} | {long_fmt(f['boot_log_std'], 4)} | {long_fmt(f['boot_rel_std'], 4)} | {long_fmt(f['f_rel_uncertainty'], 3)} |")
    L += ["", "블록 부트스트랩 로그 SD = 영상 안에서 10프레임 블록을 복원 추출해 다시 추정한 f(30회)의 로그 표준편차. 22·23의 상대 SD(표준편차 ÷ 평균)와 값이 거의 같다.", "",
          "## 5. 한계", "", "- 새 영상 수가 적으면 H2, H3은 판정하지 않는다(지시문). 영상 한두 개의 판정은 증거가 약하다.", "- 기준 f와의 비교는 summary.md에만 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def long_summary(path: Path, new_all: list, new_ok: list, base: dict, h: dict, cfg: dict, extra: dict):
    ref = float(cfg["hypotheses"]["reference_f_px"])
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    L = ["# 실험 25 summary: 기준 f와의 비교, H1~H4", "", f"**{LONG_NOTE}**", "",
         f"- 기준 f = {ref:.0f} px (1배 렌즈, 사용자 체커보드 값; 추정에 쓰지 않음). 새 영상 {len(new_ok)}개 사용, 제외 {sum(1 for j in new_all if j['excluded'])}개.",
         f"- 편향 보정값 b = {cfg['baseline']['bias_b']:+.6f} (기존 8개 영상의 합동 추정으로 실행 전에 고정, 새 영상은 쓰지 않음; exp(b) = {np.exp(cfg['baseline']['bias_b']):.4f})", "",
         "## 1. 영상별 오차", "", "| 영상 | FT f | 오차 | 보정 후 f (f / exp(b)) | 보정 후 오차 | 판정 |", "|---|---|---|---|---|---|"]
    b = float(cfg["baseline"]["bias_b"])
    for j in new_ok:
        f = j["ft"]
        L.append(f"| {j['video']} | {long_fval(f)} | {pct(None if f['at_grid_edge'] else f['f'] / ref - 1)} | {'-' if f['at_grid_edge'] else format(f['f'] / np.exp(b), '.1f') + ' px'} | "
                 f"{pct(None if f['at_grid_edge'] else f['f'] / np.exp(b) / ref - 1)} | {f['verdict']} |")
    st_new = long_base_stats(new_ok, ref)
    st_base = long_base_stats(base["videos"], ref)
    L += ["", f"- 새 영상 |오차| 중앙값 {pct(st_new['median_abs_err']).replace('+', '')}, 부호 포함 중앙값 {pct(st_new['median_signed_err'])} / 기존 8개: |오차| 중앙값 {pct(st_base['median_abs_err']).replace('+', '')}, 부호 포함 {pct(st_base['median_signed_err'])}", "",
          "## 2. H1~H4 판정 (규칙과 문턱값은 실행 전 yaml에 고정)", ""]
    h1, h2, h3, h4 = h["H1"], h["H2"], h["H3"], h["H4"]
    L += [f"**H1 (데이터가 늘면 잡음이 줄어든다): {h1['verdict']}**  — 새 영상 부트스트랩 로그 SD 중앙값 {long_fmt(h1.get('new_median'), 4)}, 기존 중앙값 {long_fmt(h1.get('base_median'), 4)}, 비율 {long_fmt(h1.get('ratio_observed'), 3)} (지지 기준 ≤ {h1['ratio_used']})",
          f"  - 기록용: 영상 사이 log f 표준편차 새 {long_fmt(extra.get('new_log_f_sd'), 3)} (영상 {len(new_ok)}개) / 기존 {long_fmt(st_base['log_f_sd'], 3)}; 기울기 구간 길이 중앙값 새 {long_fmt(extra.get('new_tilt_len'), 1)}초 / 기존 {long_fmt(extra.get('base_tilt_len'), 1)}초 (비 {long_fmt(extra.get('tilt_len_ratio'), 2)})", "",
          f"**H2 (공통 편향이 있다): {h2['verdict']}**  — 새 영상 합동 f = {long_fmt(extra.get('pooled_new_f'), 1)} px, log(f_합동/{ref:.0f})의 95% 구간 [{long_fmt(h2['ci'][0], 4)}, {long_fmt(h2['ci'][1], 4)}] (영상 단위 부트스트랩 {cfg['hypotheses']['bootstrap_n']}회, 시드 {cfg['hypotheses']['bootstrap_seed']})" + (f" — {h2['reason']}" if h2.get('reason') else ""), "",
          f"**H3 (기존 영상으로 정한 보정이 새 영상에서도 통한다): {h3['verdict']}**  — 보정 전 |오차| 중앙값 {long_fmt(h3['median_abs_before'], 4)} → 보정 후 {long_fmt(h3['median_abs_after'], 4)}, |오차|가 줄어든 영상 {long_fmt(h3['fraction_reduced'], 2)} (n = {h3['n']})" + (f" — {h3['reason']}" if h3.get('reason') else ""), "",
          f"**H4 (작은 기울기 프레임을 빼면 나아진다, θ2d ≥ {cfg['hypotheses']['h4_theta_min_deg']:.0f}°): {h4['verdict']}**  — " + (f"|오차| 중앙값 {long_fmt(h4['median_abs_orig'], 4)} → {long_fmt(h4['median_abs_h4'], 4)}, 개선된 영상 {long_fmt(h4['fraction_improved'], 2)} (n = {h4['n']})" if h4.get("n") else "계산한 영상이 없음"), "",
          "## 3. 기록용 (판정 없음)", "", "### 불확실성 보정 상태 (|log 오차|가 부트스트랩 로그 SD 안에 드는 영상 비율)", "",
          "| 대상 | 보정 | 1σ 안 | 2σ 안 | 영상 수 |", "|---|---|---|---|---|"]
    for name, cal in extra["calibration"].items():
        L.append(f"| {name} | {cal['label']} | {long_fmt(cal['within_1sigma'], 2)} | {long_fmt(cal['within_2sigma'], 2)} | {cal['n']} |")
    L += ["", "### 비교용 추정 (새 영상, 판정에 쓰지 않음)", "", "| 영상 | keep FT | 오차 | exclude JD | 오차 | AnyCalib pinhole | 오차 | AnyCalib gen | 오차 |", "|---|---|---|---|---|---|---|---|---|"]
    for j in new_ok:
        k = j.get("keep") or {}
        kf = k.get("ft")
        ac = j.get("anycalib") or {}
        e = lambda sol: ("값 없음", "-") if sol is None or sol.get("at_grid_edge") else (f"{sol['f']:.1f} px", pct(sol["f"] / ref - 1))      # noqa: E731
        a1, a2 = ac.get("anycalib_pinhole"), ac.get("anycalib_gen")
        cell = lambda a: ("-", "-") if not a else (f"{a['median']:.1f} px", pct(a["median"] / ref - 1))      # noqa: E731
        c = [e(kf), e(j["jd"]), cell(a1), cell(a2)]
        L.append(f"| {j['video']} | " + " | ".join(f"{x} | {y}" for x, y in c) + " |")
    L += ["", "## 4. 기존 8개와의 비교", "", "| 영상 | FT f | 오차 | 블록 부트스트랩 로그 SD | 기울기 구간(초) | 기울기 프레임 |", "|---|---|---|---|---|---|"]
    for j in base["videos"]:
        f = j["ft"]
        L.append(f"| {j['video']} | {long_fval(f)} | {pct(None if f['at_grid_edge'] else f['f'] / ref - 1)} | {long_fmt(f['boot_log_std'], 4)} | {j['motion']['tilt_duration_s']:.1f} | {j['motion']['n_tilt_frames']} |")
    L += ["", "## 한계", "", "- 새 영상이 적을수록 H1~H4의 증거가 약하다. H2, H3은 새 영상 %d개 미만이면 판정하지 않는다." % int(cfg["hypotheses"]["min_new_videos_h2_h3"]),
          "- 기준 f 1680은 사용자 제공 값이며 원본 보정 파일은 서버에 없다.", ""]
    path.write_text("\n".join(L) + "\n")


def long_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "25_long_video_check", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": LONG_NOTE, "candidate": cfg["candidate"], "videos": {}}
    for j in r["new"]:
        e = {"duration_s": rd(j["duration_s"], 2), "detect_rate": rd(j["detect_rate"]), "edge_rate": rd(j["edge_rate"]), "excluded": j["excluded"] or None}
        if "ft" in j:
            f = j["ft"]
            mo = j["motion"]
            e.update({"ft": {"f_px": None if f["at_grid_edge"] else round(f["f"], 1), "status": "value" if not f["at_grid_edge"] else f"값 없음 ({f['side']})", "verdict": f["verdict"], "sharpness": rd(f["sharpness"]),
                             "boot_log_std": rd(f["boot_log_std"], 5), "boot_rel_std": rd(f["boot_rel_std"], 5), "f_rel_uncertainty": rd(f["f_rel_uncertainty"])},
                      "motion": {k: rd(v, 3) if isinstance(v, float) else v for k, v in mo.items() if k != "theta_tilt"},
                      "h4_variant": {k: (rd(v, 1) if k == "f" else v) for k, v in j["h4"].items()},
                      "guide": {k: v for k, v in (j.get("guide") or {}).items() if k in ("t_transition", "guide", "reasons", "items")},
                      "comparison": {"keep_ft_px": None if not j.get("keep") or "ft" not in j["keep"] or j["keep"]["ft"]["at_grid_edge"] else round(j["keep"]["ft"]["f"], 1),
                                     "exclude_jd_px": None if j["jd"]["at_grid_edge"] else round(j["jd"]["f"], 1),
                                     "anycalib_pinhole_px": rd(((j.get("anycalib") or {}).get("anycalib_pinhole") or {}).get("median"), 1), "anycalib_gen_px": rd(((j.get("anycalib") or {}).get("anycalib_gen") or {}).get("median"), 1)}})
        out["videos"][j["video"]] = e
    h = r["hypotheses"]
    out["hypotheses"] = {"H1": {k: rd(v, 5) if isinstance(v, float) else v for k, v in h["H1"].items()}, "H2": {"verdict": h["H2"]["verdict"], "reason": h["H2"].get("reason")}, "H3": {"verdict": h["H3"]["verdict"], "reason": h["H3"].get("reason")},
                         "H4": {"verdict": h["H4"]["verdict"], "n": h["H4"].get("n")}}
    out["bias_correction_fixed_before_new_videos"] = True
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def long_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/25_long_video_check.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("bias")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list overriding yaml new_videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg_path = Path(a.config)
    text = cfg_path.read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(long_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.cmd == "bias":
        return long_bias_main(cfg, cfg_path, print)
    if cfg["baseline"]["bias_b"] is None:
        raise SystemExit("run `bias` first: b (step 2) must be fixed and committed before any new video is processed")
    new_list = a.videos.split(",") if a.videos else list(cfg["new_videos"])
    if not new_list:
        found = hf.long_find_new_videos(cfg["input_dir"], cfg["after_number"])
        print(f"new_videos is empty. Hand videos above IMG_{cfg['after_number']}: " + (", ".join(f"{x['video']} ({x['duration_s']:.1f} s, {x['width']}x{x['height']})" for x in found) or "(none)"))
        print("Confirm with the user, then put the file names into new_videos of the yaml and commit it before the run.")
        return found
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    hy = cfg["hypotheses"]
    ref = float(hy["reference_f_px"])
    b = float(cfg["baseline"]["bias_b"])
    base = json.loads(hf.resolve(cfg["baseline"]["json"]).read_text())
    log(f"25 긴 영상 재검증: 새 영상 {new_list}, 기존 {len(base['videos'])}개(고정된 baseline.json), b = {b:+.6f} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "exclusion", "hypotheses", "detector_settings", "guide", "compare_estimates", "anycalib_guard") if k in cfg}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    gpus = list(cfg.get("gpu_ids") or [])
    log(f"  CPU: 논리 {os.cpu_count()}개, 검출 n_workers {cfg['n_workers']} (절반 {os.cpu_count() // 2} 이하), AnyCalib GPU {gpus or '(상속)'}")
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(v, cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for v in new_list]))
            dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
            npz = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            args = [(p["video"], "new", None, p, npz[(p["video"], "orig")], cfg["pipeline"], cjob, str(tmp_root), gpus[i % len(gpus)] if gpus else None) for i, p in enumerate(preps)]
            with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex3:
                new_all = list(ex3.map(hf.long_video, args))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    new_ok = [j for j in new_all if "ft" in j]
    for j in new_all:
        if j["excluded"]:
            log(f"  {j['video']}: 제외 — {'; '.join(j['excluded'])}")
        elif "ft" in j:
            log(f"  {j['video']}: 길이 {j['duration_s']:.1f} s, 기울기 프레임 {j['motion']['n_tilt_frames']}장, FT {long_fval(j['ft'])}, 판정 {j['ft']['verdict']}, 부트스트랩 로그 SD {long_fmt(j['ft']['boot_log_std'], 4)}")
        else:
            log(f"  {j['video']}: 분석 실패 ({j.get('failed')})")
    # ---- hypotheses
    base_std = [j["ft"]["boot_log_std"] for j in base["videos"]]
    h1 = hf.long_h1([j["ft"]["boot_log_std"] for j in new_ok], base_std, float(hy["h1_ratio"]))
    grid = np.array(base["grid"])
    pooled_new = None
    if new_ok:
        curves_new = np.array([j["curve"]["cost"] for j in new_ok])
        pooled_new = hf.long_pooled_estimate(curves_new, grid)
        boot = hf.long_pooled_bootstrap(curves_new, grid, int(hy["bootstrap_n"]), int(hy["bootstrap_seed"]))
        ci = hf.long_log_ci(boot["f"], ref) if not pooled_new["at_grid_edge"] else (float("nan"), float("nan"))
    else:
        boot, ci = None, (float("nan"), float("nan"))
    h2 = hf.long_h2(ci, len(new_ok), int(hy["min_new_videos_h2_h3"]))
    vals = [j for j in new_ok if not j["ft"]["at_grid_edge"]]
    h3 = hf.long_h3([j["ft"]["f"] for j in vals], ref, b, len(new_ok), int(hy["min_new_videos_h2_h3"]), float(hy["fraction"]))
    pairs = [(j["ft"]["f"], j["h4"]["f"]) for j in vals if "f" in j["h4"] and not j["h4"]["at_grid_edge"]]
    h4 = hf.long_h4([p[0] for p in pairs], [p[1] for p in pairs], ref, float(hy["h4_gain"]), float(hy["fraction"]))
    h = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    for k, v in h.items():
        log(f"  {k}: {v['verdict']}")
    st_base = long_base_stats(base["videos"], ref)
    lf_new = np.log([j["ft"]["f"] for j in vals])
    extra = {"new_log_f_sd": float(np.std(lf_new, ddof=1)) if len(lf_new) > 1 else None, "pooled_new_f": pooled_new["f"] if pooled_new else None,
             "new_tilt_len": float(np.median([j["motion"]["tilt_duration_s"] for j in new_ok])) if new_ok else None,
             "base_tilt_len": float(np.median([j["motion"]["tilt_duration_s"] for j in base["videos"]])), "calibration": {}}
    extra["tilt_len_ratio"] = extra["new_tilt_len"] / extra["base_tilt_len"] if extra["new_tilt_len"] else None
    for name, js in (("새 영상", vals), ("기존 8개", [j for j in base["videos"] if not j["ft"]["at_grid_edge"]])):
        for label, bb in (("보정 전", 0.0), ("보정 후(b 뺌)", b)):
            c_ = hf.long_calibration([j["ft"]["f"] for j in js], [j["ft"]["boot_log_std"] for j in js], ref, bb)
            extra["calibration"][f"{name} {label}"] = {**c_, "label": label}
    long_figures(run_root / "figures", base, new_ok, pooled_new, cfg)
    long_report(run_root / "report.md", [], new_all, new_ok, base, cfg)
    long_summary(run_root / "summary.md", new_all, new_ok, base, h, cfg, extra)
    with (run_root / "per_frame.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "index", "time_s", "theta2d_deg", "in_theta_min_subset"])
        for j in new_ok:
            for r_ in j.get("frames") or []:
                wr.writerow([j["video"], r_["index"], f"{r_['time_s']:.4f}", f"{r_['theta2d_deg']:.2f}", int(r_["in_h4_subset"])])
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"new": new_all, "hypotheses": h, "pooled_new": pooled_new, "bootstrap": boot, "extra": extra, "baseline_json": cfg["baseline"]["json"]}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_23": cfg["code_23"], "sources": cfg["sources"], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, new_all, h


# ====================================================================================================
# gen  (experiment 26: rating v2, generalisation groups, trajectory effect of the f error; new section)
#
#     python HandCalib/run.py generalization --config HandCalib/configs/26_generalization.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py generalization --config ... results-yaml --run runs/26_generalization/<UTC>
# The videos (file, group, person_id, phone, lens, glove, ref_f) come from the yaml. Every group G1-G5 can be processed; only the groups present are judged.
# ====================================================================================================

GEN_NOTE = ("새 판정 규칙(v2), 일반화 그룹, f 오차가 손 궤적에 주는 영향을 본 실험이다. 후보 설정(mediapipe_hands + 손목 제외 + fixed_template)과 모든 문턱값은 실행 전에 고정했고 바꾸지 않았다. "
            "기준 f(ref_f)와의 비교는 summary.md에만 있다.")
GEN_GROUP_KO = {"G1": "다른 사람", "G2": "다른 폰", "G3": "다른 렌즈(0.5배)", "G4": "장갑", "G5": "보정 + 작업 구간"}


def gen_orient(size) -> str:
    """'가로' for a processed video wider than tall (or square), '세로' for a portrait one (size = [width, height] after rotation)."""
    return "-" if not size else ("가로" if size[0] >= size[1] else "세로")


def gen_traj_variants(res: dict, cfg: dict, pooled_f: float | None) -> dict:
    """Trajectories (smoothed) of one video for the reference f and every comparison f: {name: {"f", "P" (n, 3)}} and the time vector."""
    tcfg = cfg["trajectory"]
    rows = res["traj"]["rows"]
    if not rows:
        return {}
    t = np.array([r["time_s"] for r in rows])
    u, v = np.array([r["u"] for r in rows]), np.array([r["v"] for r in rows])
    ell = res["traj"]["ell_front_px"] * np.array([r["size_ratio"] for r in rows])
    W, H = res["size"]
    L = float(tcfg["hand_length_mm"])
    fref = res["ref_f"]
    fs = {"ref": fref, **{f"eps_{int(e):+d}": fref * (1 + e / 100.0) for e in tcfg["epsilons_pct"]}}
    if "ft" in res and not res["ft"]["at_grid_edge"]:
        fs["est"] = res["ft"]["f"]
    if pooled_f is not None:
        fs["pooled"] = pooled_f
    return {"t": t, "variants": {k: {"f": f, "P": hf.gen_smooth_traj(t, hf.gen_trajectory(u, v, ell, f, L, W, H), float(tcfg["median_window_s"]))} for k, f in fs.items()}}


def gen_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/26_generalization.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the yaml entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(gen_results_yaml(hf.resolve(a.run), cfg))
        return None
    entries = list(cfg.get("videos") or [])
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    if not entries:
        found = hf.long_find_new_videos(cfg["input_dir"], cfg["after_number"])
        print(f"videos is empty. Hand videos above IMG_{cfg['after_number']}: " + (", ".join(f"{x['video']} ({x['duration_s']:.1f} s, {x['width']}x{x['height']})" for x in found) or "(none)"))
        print("Confirm with the user, then put the entries (file, group, person_id, phone, lens, glove, ref_f) into videos of the yaml and commit it before the run.")
        return found
    entries = hf.gen_check_videos(entries)
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"26 일반화 확인: 영상 {[e['file'] for e in entries]}, 그룹 {sorted({e['group'] for e in entries})} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "trajectory", "calib_end_default_s") if k in cfg}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            npz = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], cfg["detector_settings"][cfg["candidate"]["detector"]], run_root / "detection_cache", int(cfg["n_workers"]), log)
            jobs_args = [(e, p, npz[(p["video"], "orig")], cfg["pipeline"], cjob) for e, p in zip(entries, preps)]
            results = list(ex.map(hf.gen_video, jobs_args))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    v2 = cfg["rating_v2"]
    for r in results:
        if r["excluded"]:
            log(f"  {r['video']} ({r['group']}): 제외 — {'; '.join(r['excluded'])} (손 검출 실패 {100 * (1 - r['detect_rate']):.1f}%)")
        elif "ft" in r:
            r["rating"] = hf.gen_rating(r["motion"]["tilt_duration_s"], r["motion"]["theta_max_deg"], r["ft"]["boot_log_std"], v2)
            log(f"  {r['video']} ({r['group']}, {r['person_id']}): 기울기 {r['motion']['tilt_duration_s']:.1f}초, 최대 θ2d {r['motion']['theta_max_deg']:.0f}°, f {format(r['ft']['f'], '.1f') + ' px' if not r['ft']['at_grid_edge'] else '값 없음'}, "
                f"로그 SD {long_fmt(r['ft']['boot_log_std'], 4)}, 등급 {r['rating']}")
        else:
            log(f"  {r['video']}: 분석 실패 ({r.get('failed')})")
    ok = [r for r in results if "ft" in r]
    # ---- groups
    groups = {}
    for g in hf.GEN_GROUPS:
        gv = [r for r in results if r["group"] == g]
        gok = [r for r in gv if "ft" in r]
        if g == "G5":
            groups[g] = {"verdict": "판정 없음 (궤적 분석용)" if gv else "영상 없음", "n_videos": len(gok), "n_excluded": len(gv) - len(gok)}
            continue
        if not gv:
            groups[g] = {"verdict": "영상 없음", "n_videos": 0, "n_excluded": 0}
            continue
        ref = float(np.mean([r["ref_f"] for r in gv]))
        grid = np.array(gok[0]["curve"]["grid"]) if gok else np.array([])
        d = hf.gen_group_verdict([r["curve"]["cost"] for r in gok], grid, [None if r["ft"]["at_grid_edge"] else r["ft"]["f"] for r in gok], ref, cfg["generalization"], len(gv) - len(gok), len(gv)) if gok else \
            {"verdict": "판단 불가", "n_videos": 0, "n_excluded": len(gv), "n_total": len(gv), "ref_f": ref, "reason": "분석된 영상이 없음"}
        d["detect_fail_rates"] = [round(1 - r["detect_rate"], 4) for r in gv]
        if g == "G1" and gok:
            d["by_person"] = {}
            for pid in sorted({r["person_id"] for r in gok}):
                pv = [r for r in gok if r["person_id"] == pid]
                d["by_person"][pid] = {"n_videos": len(pv), "pooled_f": hf.long_pooled_estimate([r["curve"]["cost"] for r in pv], grid)["f"], "at_grid_edge": hf.long_pooled_estimate([r["curve"]["cost"] for r in pv], grid)["at_grid_edge"]}
        groups[g] = d
        log(f"  {g} ({GEN_GROUP_KO[g]}): {d['verdict']}" + (f" (합동 f {d['pooled_f']:.1f} px)" if "pooled_f" in d else ""))
    rows = [{"video": r["video"], "rating": r["rating"], "f": None if r["ft"]["at_grid_edge"] else r["ft"]["f"], "ref_f": r["ref_f"], "sigma": r["ft"]["boot_log_std"]} for r in ok]
    ver = hf.gen_verify(rows, cfg["verification_v2"])
    log(f"  판정 v2 검증: {ver['verdict']}" + (f" ({ver.get('reason')})" if ver.get("reason") else ""))
    # ---- existing videos, rated after the fact
    ex_rows = hf.gen_existing_rows(cfg)
    for r in ex_rows:
        r["rating"] = hf.gen_rating(r["tilt_s"], r["theta_max"], r["sigma"], v2)
    # ---- trajectories
    traj = {}
    for r in ok:
        gp = groups.get(r["group"], {})
        pooled = gp.get("pooled_f") if gp.get("verdict") in ("통과", "실패", "판단 불가") and not gp.get("at_grid_edge") else None
        tv = gen_traj_variants(r, cfg, pooled)
        if tv:
            traj[r["video"]] = {"t": tv["t"], "variants": tv["variants"], "metrics": {k: hf.gen_step_metrics(tv["t"], tv["variants"]["ref"]["P"], x["P"], cfg["trajectory"]) for k, x in tv["variants"].items() if k != "ref"}}
    gen_figures(run_root / "figures", results, ex_rows, groups, traj, cfg)
    gen_report(run_root / "report.md", results, ex_rows, traj, cfg)
    gen_summary(run_root / "summary.md", results, ex_rows, groups, ver, traj, cfg)
    with (run_root / "per_frame.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "index", "time_s", "theta2d_deg", "centre_u", "centre_v", "size_ratio", "segment"])
        for r in ok:
            for q in r["traj"]["rows"]:
                wr.writerow([r["video"], q["index"], f"{q['time_s']:.4f}", "" if q["theta2d_deg"] is None else f"{q['theta2d_deg']:.2f}", f"{q['u']:.2f}", f"{q['v']:.2f}", f"{q['size_ratio']:.5f}", r["traj"]["segment"]])
    with (run_root / "trajectory.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        names = None
        for vid, tj in traj.items():
            if names is None:
                names = list(tj["variants"])
                wr.writerow(["video", "time_s"] + [f"{n}_{c}" for n in names for c in ("X_mm", "Y_mm", "Z_mm")])
            for i, ti in enumerate(tj["t"]):
                wr.writerow([vid, f"{ti:.4f}"] + [f"{tj['variants'][n]['P'][i][k]:.3f}" for n in names if n in tj["variants"] for k in range(3)])
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "groups": groups, "verification_v2": ver, "existing": ex_rows,
                                                                "trajectory_metrics": {k: v["metrics"] for k, v in traj.items()}}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_25": cfg["code_25"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, groups, ver


def gen_figures(out: Path, results: list, ex_rows: list, groups: dict, traj: dict, cfg: dict):
    plt = _plt()
    ok = [r for r in results if "ft" in r]
    gs = [g for g in ("G1", "G2", "G3", "G4") if any(r["group"] == g for r in ok)]
    if gs:
        fig, axs = plt.subplots(1, len(gs), figsize=(4.4 * len(gs), 4), squeeze=False)
        for ax, g in zip(axs[0], gs):
            vs = [r for r in ok if r["group"] == g]
            for i, r in enumerate(vs):
                f, sg = r["ft"]["f"], r["ft"]["boot_log_std"]
                if r["ft"]["at_grid_edge"]:
                    ax.plot(i, f, "x", color="#c00")
                else:
                    ax.errorbar(i, f, yerr=[[f - f * np.exp(-sg)], [f * np.exp(sg) - f]], fmt="o", color="#2a6fbb", capsize=3)
            gp = groups.get(g, {})
            if "pooled_f" in gp and not gp["at_grid_edge"]:
                ax.axhspan(*gp["interval_95"], color="#c47a00", alpha=0.18)
                ax.axhline(gp["pooled_f"], color="#c47a00", lw=1.5, label=f"합동 f {gp['pooled_f']:.0f} px (95% 구간)")
                ax.legend(fontsize=7)
            ax.set_xticks(range(len(vs)))
            ax.set_xticklabels([r["video"][4:8] for r in vs], fontsize=8)
            ax.set_title(f"{g} {GEN_GROUP_KO[g]}", fontsize=9)
            ax.set_ylabel("f (px), 막대 = 로그 SD ±1σ")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "f_by_group.png", dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    cols = {"사용 가능": "#3a9a4a", "참고": "#c47a00", "사용 불가": "#c00"}
    pts = [(r["rating"], abs(r["ft"]["f"] / r["ref_f"] - 1) * 100, "새") for r in ok if not r["ft"]["at_grid_edge"]] + \
          [(r["rating"], abs(r["f"] / r["ref_f"] - 1) * 100, "기존") for r in ex_rows if not r["at_grid_edge"]]
    for k, name in enumerate(("사용 가능", "참고", "사용 불가")):
        for src, mk in (("기존", "o"), ("새", "D")):
            ys = [y for rt, y, s_ in pts if rt == name and s_ == src]
            ax.plot([k + (0.12 if src == "새" else -0.12)] * len(ys) + np.random.default_rng(0).uniform(-0.05, 0.05, len(ys)), ys, mk, color=cols[name], alpha=0.8, label=f"{src} 영상" if k == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["사용 가능", "참고", "사용 불가"])
    ax.set_ylabel("|f 오차| (%) (ref_f 대비)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "error_by_rating.png", dpi=110)
    plt.close(fig)
    if traj:
        eps = [int(e) for e in cfg["trajectory"]["epsilons_pct"]]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
        for vid, tj in traj.items():
            for ax, key in zip(axs, ("magnitude_err_median_pct", "direction_err_median_deg", "depth_err_median_mm")):
                ax.plot(eps, [tj["metrics"][f"eps_{e:+d}"][key] if tj["metrics"][f"eps_{e:+d}"][key] is not None else np.nan for e in eps], "o-", ms=4, label=vid[4:8])
        for ax, ttl in zip(axs, ("이동량 오차 중앙값 (%)", "방향 오차 중앙값 (°)", "깊이 이동 오차 중앙값 (mm)")):
            ax.set_xlabel("ε (f = ref_f × (1+ε), %)")
            ax.set_title(ttl, fontsize=9)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "trajectory_eps_error.png", dpi=110)
        plt.close(fig)
        vid, tj = next(iter(traj.items()))
        fig = plt.figure(figsize=(6.4, 5))
        ax = fig.add_subplot(111, projection="3d")
        for name, col in (("ref", "k"), ("eps_-20", "#2a6fbb"), ("eps_+20", "#c47a00")):
            P = tj["variants"][name]["P"]
            ax.plot(P[:, 0], P[:, 2], -P[:, 1], color=col, lw=1.2, label=("기준 f" if name == "ref" else name.replace("eps_", "ε ") + "%"))
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z 깊이 (mm)")
        ax.set_zlabel("−Y (mm)")
        ax.set_title(f"{vid} 손 중심 궤적", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "trajectory_3d.png", dpi=110)
        plt.close(fig)


def gen_report(path: Path, results: list, ex_rows: list, traj: dict, cfg: dict):
    v2 = cfg["rating_v2"]
    ok = [r for r in results if "ft" in r]
    L = ["# 실험 26: 판정 v2, 일반화, f 오차의 궤적 영향", "", GEN_NOTE, "",
         f"후보: {cfg['candidate']['detector']} + wrist_mode {cfg['candidate']['wrist_mode']} + {cfg['candidate']['estimator']} (22·25와 같음). 사람은 person_id로만 적고, 폰은 main / teammate로만 적는다.", "",
         "## 1. 영상 목록과 제외", "", f"제외 규칙(실행 전 고정): 유효한 정면 창(1초) 없음, 가장자리 접촉 프레임 > {100 * cfg['exclusion']['edge_rate_max']:.0f}%, 손 검출 실패 프레임 > {100 * cfg['exclusion']['detect_fail_max']:.0f}%.", "",
         "| 그룹 | 영상 | 방향 | person_id | 폰 | 렌즈 | 장갑 | 길이(초) | 검출 실패율 | 가장자리 접촉 | 포함 | 이유 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['group']} {GEN_GROUP_KO[r['group']]} | {r['video']} | {gen_orient(r['size'])} | {r['person_id']} | {r['phone']} | {r['lens']} | {'예' if r['glove'] else '아니오'} | {r['duration_s']:.1f} | {100 * (1 - r['detect_rate']):.1f}% | "
                 f"{100 * r['edge_rate']:.1f}% | {'제외' if r['excluded'] else ('포함' if 'ft' in r else '분석 실패')} | {'; '.join(r['excluded']) or r.get('failed', '-')} |")
    L += ["", "## 2. 영상별 촬영 지표와 신뢰 등급(v2)", "",
          f"등급 규칙: 사용 가능 = 기울기 구간 ≥ {v2['tilt_min_s']:.0f}초, θ2d 최댓값 ≥ {v2['theta_max_min_deg']:.0f}°, 부트스트랩 로그 SD ≤ {v2['usable_log_std_max']}; 참고 = 촬영 조건 둘은 만족하고 로그 SD ≤ {v2['reference_log_std_max']}; 그 외 사용 불가.", "",
          "| 영상 | 방향 | 기울기 구간(초) | 기울기 프레임 | θ2d 중앙값 / 90번째 / 최댓값(°) | 정면 창 손 박스/짧은 변 | 가이드 v2 | 전환 시각(초) | 부트스트랩 로그 SD | 등급 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        m, g = r["motion"], r.get("guide") or {}
        L.append(f"| {r['video']} | {gen_orient(r['size'])} | {m['tilt_duration_s']:.1f} | {m['n_tilt_frames']} | {long_fmt(m['theta_median_deg'], 1)} / {long_fmt(m['theta_p90_deg'], 1)} / {long_fmt(m['theta_max_deg'], 1)} | {r['hand_box_short_ratio_front']:.3f} | "
                 f"{('충족' if g.get('guide') else '미충족') if 'guide' in g else '-'}{(': ' + '; '.join(g['reasons'])) if g.get('reasons') else ''} | {long_fmt(g.get('t_transition'), 2)} | {long_fmt(r['ft']['boot_log_std'], 4)} | **{r['rating']}** |")
    L += ["", "## 3. 초점거리와 불확실성 (값만)", "", "| 영상 | 방향 | f (px) | f 판정(추정기) | 부트스트랩 로그 SD | 상대 SD(22·25와 같은 지표) |", "|---|---|---|---|---|---|"]
    for r in ok:
        f = r["ft"]
        L.append(f"| {r['video']} | {gen_orient(r['size'])} | {'값 없음 (' + str(f['side']) + ')' if f['at_grid_edge'] else format(f['f'], '.1f')} | {f['verdict']} | {long_fmt(f['boot_log_std'], 4)} | {long_fmt(f['boot_rel_std'], 4)} |")
    L += ["", "## 4. 궤적 지표: f = ref_f × (1+ε)로 바꿨을 때 (기준 궤적 = f가 ref_f일 때)", "",
          f"손 중심 궤적을 {cfg['trajectory']['median_window_s']}초 중앙값 필터로 평활화하고 {cfg['trajectory']['step_s']}초 간격 이동 벡터를 비교했다(기준 이동 {cfg['trajectory']['min_step_mm']:.0f} mm 미만은 제외). "
          "X, Y는 f와 무관하고 Z만 f에 비례하므로 오차는 깊이 이동에서 생긴다.", ""]
    for vid, tj in traj.items():
        r = next(x for x in ok if x["video"] == vid)
        L += [f"**{vid}** (구간: {r['traj']['segment']}, 프레임 {len(tj['t'])}장, 이동 벡터 {next(iter(tj['metrics'].values()))['n_steps']}개)", "",
              "| 비교 | 이동량 오차 중앙값 / 90번째(%) | 방향 오차 중앙값 / 90번째(°) | 깊이 이동 오차 중앙값 / 90번째(mm) | 기준 이동의 깊이 성분(중앙값) |", "|---|---|---|---|---|"]
        for k, m in tj["metrics"].items():
            if k in ("est", "pooled"):
                continue
            L.append(f"| {k.replace('eps_', 'ε ')}% | {long_fmt(m['magnitude_err_median_pct'], 1)} / {long_fmt(m['magnitude_err_p90_pct'], 1)} | {long_fmt(m['direction_err_median_deg'], 1)} / {long_fmt(m['direction_err_p90_deg'], 1)} | "
                     f"{long_fmt(m['depth_err_median_mm'], 1)} / {long_fmt(m['depth_err_p90_mm'], 1)} | {long_fmt(m['depth_share_median'], 2)} |")
        L.append("")
    L += ["## 5. 한계", "",
          "- 궤적은 기울기 구간(손을 제자리에서 기울이는 구간)으로 계산했다. 제자리 회전이라 실제 작업보다 깊이 이동이 작아서, 깊이 방향 오차를 과소평가할 수 있다.",
          "- 0.5배 렌즈 영상의 수치는 1배 영상과 직접 비교하기 어렵다(렌즈가 다르고 왜곡이 크다).",
          "- 손 길이 L은 입력값(hand_length_mm)이고, 손 크기는 정면 창 대비 비율로 구한 값이라 절대 깊이에는 L과 정면 창의 크기 오차가 그대로 들어간다.",
          "- ref_f 대비 비교, 그룹 판정의 근거 표, 추정 f·합동 f로 만든 궤적의 오차는 summary.md에 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def gen_summary(path: Path, results: list, ex_rows: list, groups: dict, ver: dict, traj: dict, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    ok = [r for r in results if "ft" in r]
    L = ["# 실험 26 summary", "", f"**{GEN_NOTE}**", "", "## 판정", "",
         f"- **판정 v2 검증: {ver['verdict']}**" + (f" — {ver['reason']}" if ver.get("reason") else "") + f" (사용 가능 {ver['n_usable']} / 새 영상 {ver['n_videos']})"]
    for g in hf.GEN_GROUPS:
        d = groups.get(g, {"verdict": "영상 없음"})
        extra = ""
        if "pooled_f" in d:
            extra = f" — 합동 f {d['pooled_f']:.1f} px (95% 구간 {d['interval_95'][0]:.0f}–{d['interval_95'][1]:.0f}), ref_f {d['ref_f']:.0f}, 합동이 ±{100 * cfg['generalization']['pooled_tol']:.0f}% 안: {'예' if d['pooled_within_tol'] else '아니오'}, " \
                    f"영상 중 ±{100 * cfg['generalization']['video_tol']:.0f}% 안 {100 * d['video_share_within_tol']:.0f}% (영상 {d['n_videos']}개, 제외 {d['n_excluded']}개)"
        elif d.get("reason"):
            extra = f" — {d['reason']}"
        L.append(f"- **{g} {GEN_GROUP_KO[g]}: {d['verdict']}**{extra}")
    L += ["", "## 1. 영상별 ref_f 대비 오차", "", "| 영상 | 방향 | 그룹 | ref_f | f | 오차 | 로그 SD | 2σ 안 | 등급 | 가이드 v2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        f = r["ft"]
        e = None if f["at_grid_edge"] else f["f"] / r["ref_f"] - 1
        w = None if f["at_grid_edge"] or not f["boot_log_std"] else abs(np.log(f["f"] / r["ref_f"])) <= cfg["verification_v2"]["within_sigma"] * f["boot_log_std"]
        L.append(f"| {r['video']} | {gen_orient(r['size'])} | {r['group']} | {r['ref_f']:.0f} | {'값 없음' if f['at_grid_edge'] else format(f['f'], '.1f')} | {pct(e)} | {long_fmt(f['boot_log_std'], 4)} | {'-' if w is None else ('예' if w else '아니오')} | {r['rating']} | "
                 f"{('충족' if (r.get('guide') or {}).get('guide') else '미충족') if 'guide' in (r.get('guide') or {}) else '-'} |")
    L += ["", "## 2. 그룹 합동 추정 (근거 표)", "", "| 그룹 | 영상 | 제외 | 합동 f | 95% 구간 | ref_f 대비 | 영상 중 ±15% 안 | 검출 실패율(영상별) | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
    for g, d in groups.items():
        if "pooled_f" in d:
            L.append(f"| {g} | {d['n_videos']} | {d['n_excluded']} | {'값 없음' if d['at_grid_edge'] else format(d['pooled_f'], '.1f')} | {d['interval_95'][0]:.0f}–{d['interval_95'][1]:.0f} | {pct(None if d['at_grid_edge'] else d['pooled_f'] / d['ref_f'] - 1)} | "
                     f"{100 * d['video_share_within_tol']:.0f}% | {', '.join(f'{100 * x:.1f}%' for x in d['detect_fail_rates'])} | {d['verdict']} |")
        elif d.get("n_videos", 0) or d.get("n_excluded", 0):
            L.append(f"| {g} | {d.get('n_videos', 0)} | {d.get('n_excluded', 0)} | - | - | - | - | {', '.join(f'{100 * x:.1f}%' for x in d.get('detect_fail_rates', []))} | {d['verdict']} |")
        else:
            L.append(f"| {g} | 0 | 0 | - | - | - | - | - | 영상 없음 |")
    for g, d in groups.items():
        if d.get("by_person"):
            L += ["", f"{g} 사람별 합동 f: " + ", ".join(f"{k} {'값 없음' if v['at_grid_edge'] else format(v['pooled_f'], '.1f')} px ({v['n_videos']}개)" for k, v in d["by_person"].items())]
    L += ["", "## 3. 기존 영상과의 비교 (사후 기술, 검증 아님)", "",
          "기존 영상 22개 중 22·25의 1배 14개와 7184를 같은 v2 규칙으로 등급을 매겼다. 판정 v2를 기존 영상으로 검증한 것이 아니다.", "",
          "| 영상 | 출처 | ref_f | f | 오차 | 기울기 구간(초) | θ2d 최댓값(°) | 로그 SD | 등급 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in ex_rows:
        L.append(f"| {r['video']} | {r['source']} | {r['ref_f']:.0f} | {'값 없음' if r['at_grid_edge'] else format(r['f'], '.1f')} | {pct(None if r['at_grid_edge'] else r['f'] / r['ref_f'] - 1)} | {r['tilt_s']:.1f} | "
                 f"{long_fmt(r['theta_max'], 0)} | {long_fmt(r['sigma'], 4)}{' (상대 SD)' if r.get('sigma_is_rel_std') else ''} | {r['rating']} |")
    lens_old = [r for r in ex_rows if r["source"] == "22 (0.5x)"]
    lens_new = [r for r in ok if r["lens"] == "0.5x"]
    if lens_old and lens_new:
        o = lens_old[0]
        L += ["", f"### 0.5배(약 30 cm): 실험 22의 {o['video']}와 이번 G3 영상을 나란히", "",
              "| 영상 | 출처 | 방향 | 길이 / 기울기 구간(초) | f | ref_f 대비 | 로그 SD | θ2d 최댓값(°) | 등급 |", "|---|---|---|---|---|---|---|---|---|",
              f"| {o['video']} | 실험 22 | {gen_orient(o.get('size'))} | {long_fmt(o.get('duration_s'), 1)} / {o['tilt_s']:.1f} | {format(o['f'], '.1f')} | {pct(o['f'] / o['ref_f'] - 1)} | {long_fmt(o['sigma'], 4)} (상대 SD) | {long_fmt(o['theta_max'], 0)} | {o['rating']} |"]
        for r in lens_new:
            f = r["ft"]
            L.append(f"| {r['video']} | 실험 26 (G3) | {gen_orient(r['size'])} | {r['duration_s']:.1f} / {r['motion']['tilt_duration_s']:.1f} | {'값 없음' if f['at_grid_edge'] else format(f['f'], '.1f')} | "
                     f"{pct(None if f['at_grid_edge'] else f['f'] / r['ref_f'] - 1)} | {long_fmt(f['boot_log_std'], 4)} | {long_fmt(r['motion']['theta_max_deg'], 0)} | {r['rating']} |")
        vals = [r["ft"]["f"] for r in lens_new if not r["ft"]["at_grid_edge"]]
        if vals:
            L += ["", f"- 이번 G3 영상 {len(vals)}개의 f 중앙값 {np.median(vals):.1f} px, 7184 {o['f']:.1f} px. 길이는 7184가 짧다(처리 프레임 {o.get('n_frames', '?')}장, 15 fps 기준 약 {long_fmt(o.get('duration_s'), 1)}초)."]
    L += ["", "## 4. 궤적 영향: 추정 f와 그룹 합동 f", "", "| 영상 | 비교 | 사용한 f (px) | 이동량 오차 중앙값 / 90번째(%) | 방향 오차 중앙값 / 90번째(°) | 깊이 이동 오차 중앙값 / 90번째(mm) |", "|---|---|---|---|---|---|"]
    tol = cfg["trajectory"]["tolerance"]
    for vid, tj in traj.items():
        for k in ("est", "pooled"):
            if k in tj["metrics"]:
                m = tj["metrics"][k]
                L.append(f"| {vid} | {'이 영상의 추정 f' if k == 'est' else '그룹 합동 f'} | {tj['variants'][k]['f']:.1f} | {long_fmt(m['magnitude_err_median_pct'], 1)} / {long_fmt(m['magnitude_err_p90_pct'], 1)} | "
                         f"{long_fmt(m['direction_err_median_deg'], 1)} / {long_fmt(m['direction_err_p90_deg'], 1)} | {long_fmt(m['depth_err_median_mm'], 1)} / {long_fmt(m['depth_err_p90_mm'], 1)} |")
    L += ["", "## 5. 허용 오차 비교", ""]
    if tol.get("motion_pct") is None or tol.get("direction_deg") is None:
        L.append("- 허용 오차(tolerance)가 비어 있어 값만 보고하고 판정하지 않았다.")
    else:
        L += [f"- 허용 오차: 이동량 {tol['motion_pct']}%, 방향 {tol['direction_deg']}° (중앙값이 이 값을 넘으면 초과; 90번째 백분위수도 함께 표시)", "", "| 영상 | 비교 | 이동량 중앙값 초과 | 이동량 90번째 초과 | 방향 중앙값 초과 | 방향 90번째 초과 |", "|---|---|---|---|---|---|"]
        for vid, tj in traj.items():
            for k, m in tj["metrics"].items():
                if m["magnitude_err_median_pct"] is None:
                    continue
                yn = lambda x, t: "초과" if x > t else "이내"      # noqa: E731
                L.append(f"| {vid} | {k.replace('eps_', 'ε ')} | {yn(m['magnitude_err_median_pct'], tol['motion_pct'])} | {yn(m['magnitude_err_p90_pct'], tol['motion_pct'])} | "
                         f"{yn(m['direction_err_median_deg'], tol['direction_deg'])} | {yn(m['direction_err_p90_deg'], tol['direction_deg'])} |")
    L += ["", "## 한계", "", "- 판정 v2 검증은 \"사용 가능\" 영상이 5개 미만이면 판단하지 않는다(규칙). 그룹 판정은 영상이 3개 미만이면 판단하지 않는다.",
          "- 기준 f는 사용자 제공 값이며 원본 보정 파일은 서버에 없다. 0.5배는 880 px, 1배는 1680 px.", "- 궤적은 기울기 구간(제자리 회전)이라 깊이 이동이 실제 작업보다 작다.", ""]
    path.write_text("\n".join(L) + "\n")


def gen_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "26_generalization", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": GEN_NOTE, "candidate": cfg["candidate"], "videos": {}}
    for v in r["videos"]:
        e = {"group": v["group"], "person_id": v["person_id"], "phone": v["phone"], "lens": v["lens"], "glove": v["glove"], "duration_s": rd(v["duration_s"], 2), "detect_fail_rate": rd(1 - v["detect_rate"]),
             "edge_rate": rd(v["edge_rate"]), "excluded": v["excluded"] or None}
        if "ft" in v:
            f, mo, g = v["ft"], v["motion"], v.get("guide") or {}
            e.update({"f_px": None if f["at_grid_edge"] else round(f["f"], 1), "status": "value" if not f["at_grid_edge"] else f"값 없음 ({f['side']})", "f_verdict_estimator": f["verdict"], "rating_v2": v["rating"],
                      "boot_log_std": rd(f["boot_log_std"], 5), "tilt_duration_s": rd(mo["tilt_duration_s"], 2), "n_tilt_frames": mo["n_tilt_frames"], "theta_median_deg": rd(mo["theta_median_deg"], 1),
                      "theta_p90_deg": rd(mo["theta_p90_deg"], 1), "theta_max_deg": rd(mo["theta_max_deg"], 1), "hand_box_short_ratio_front": rd(v["hand_box_short_ratio_front"]),
                      "guide_v2": {k: g.get(k) for k in ("guide", "reasons", "t_transition")} if "guide" in g else None})
        out["videos"][v["video"]] = e
    out["verification_v2"] = {"verdict": r["verification_v2"]["verdict"], "n_usable": r["verification_v2"]["n_usable"], "n_videos": r["verification_v2"]["n_videos"], "reason": r["verification_v2"].get("reason")}
    out["groups"] = {g: {"verdict": d["verdict"], "n_videos": d.get("n_videos"), "n_excluded": d.get("n_excluded"), "pooled_f_px": None if d.get("pooled_f") is None or d.get("at_grid_edge") else round(d["pooled_f"], 1),
                         "interval_95_px": None if "interval_95" not in d else [round(x, 1) for x in d["interval_95"]]} for g, d in r["groups"].items()}
    out["trajectory_eps_metrics"] = {vid: {k: {a: rd(b, 3) for a, b in m_.items()} for k, m_ in ms.items() if k.startswith("eps_")} for vid, ms in r["trajectory_metrics"].items()}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# ext  (experiment 27: main-camera 1x extension, G6 distance and G7 left hand; new section)
#
#     python HandCalib/run.py main-camera-extension --config HandCalib/configs/27_main_camera_extension.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py main-camera-extension --config ... results-yaml --run runs/27_main_camera_extension/<UTC>
# The videos (file, group, distance_cm, hand, ref_f) come from the yaml; the per-video processing is 26's (gen_video via ext_video).
# ====================================================================================================

EXT_NOTE = ("1배 주 카메라 조건에서 손과 카메라의 거리(G6)와 왼손(G7)을 확인한 실험이다. 후보 설정(mediapipe_hands + 손목 제외 + fixed_template)과 모든 문턱값은 실행 전에 고정했고 바꾸지 않았다. "
            "기준 f(ref_f)와의 비교는 summary.md에만 있다.")
EXT_GROUP_KO = {"G6": "거리 변화(오른손)", "G7": "왼손"}
EXT_HAND_KO = {"left": "왼손", "right": "오른손"}


def ext_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/27_main_camera_extension.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the yaml entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(ext_results_yaml(hf.resolve(a.run), cfg))
        return None
    entries = hf.ext_check_videos(list(cfg.get("videos") or []))
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"27 1배 조건 확장: 영상 {[e['file'] for e in entries]}, 그룹 {sorted({e['group'] for e in entries})} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "session") if k in cfg}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            npz = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], cfg["detector_settings"][cfg["candidate"]["detector"]], run_root / "detection_cache", int(cfg["n_workers"]), log)
            jobs_args = [(e, p, npz[(p["video"], "orig")], cfg["pipeline"], cjob) for e, p in zip(entries, preps)]
            results = list(ex.map(hf.ext_video, jobs_args))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    v2 = cfg["rating_v2"]
    rounds = hf.ext_rounds(entries)
    for r in results:
        r["round"] = rounds.get(r["video"])
        tag = f"{r['group']}, {r['distance_cm']:g} cm, {EXT_HAND_KO[r['hand']]}"
        if r["excluded"]:
            log(f"  {r['video']} ({tag}): 제외 — {'; '.join(r['excluded'])} (손 검출 실패 {100 * (1 - r['detect_rate']):.1f}%)")
        elif "ft" in r:
            r["rating"] = hf.gen_rating(r["motion"]["tilt_duration_s"], r["motion"]["theta_max_deg"], r["ft"]["boot_log_std"], v2)
            log(f"  {r['video']} ({tag}): 기울기 {r['motion']['tilt_duration_s']:.1f}초, 최대 θ2d {r['motion']['theta_max_deg']:.0f}°, f {format(r['ft']['f'], '.1f') + ' px' if not r['ft']['at_grid_edge'] else '값 없음'}, "
                f"로그 SD {long_fmt(r['ft']['boot_log_std'], 4)}, 등급 {r['rating']}")
        else:
            log(f"  {r['video']}: 분석 실패 ({r.get('failed')})")
    ok = [r for r in results if "ft" in r]
    grid = np.array(ok[0]["curve"]["grid"]) if ok else np.array([])
    if any(not np.allclose(np.array(r["curve"]["grid"]), grid) for r in ok):
        raise SystemExit("비용 곡선의 f 격자가 영상마다 다릅니다")
    # ---- G6 distance
    g6 = [r for r in results if r["group"] == "G6"]
    groups = {}
    if g6:
        by_d = {d: {"curves": [r["curve"]["cost"] for r in g6 if r["distance_cm"] == d and "ft" in r], "n_total": sum(1 for r in g6 if r["distance_cm"] == d)} for d in sorted({r["distance_cm"] for r in g6})}
        groups["G6"] = hf.ext_distance_verdict(by_d, grid, float(np.mean([r["ref_f"] for r in g6])), cfg["distance"])
    else:
        groups["G6"] = {"verdict": "영상 없음", "distances": {}}
    log(f"  G6 ({EXT_GROUP_KO['G6']}): {groups['G6']['verdict']}" + (f" (최대/최소 {groups['G6']['max_over_min']:.3f})" if groups["G6"].get("max_over_min") else "") + (f" — {groups['G6']['reason']}" if groups["G6"].get("reason") else ""))
    # ---- G7 left hand (26's group verdict)
    g7 = [r for r in results if r["group"] == "G7"]
    g7ok = [r for r in g7 if "ft" in r]
    if not g7:
        groups["G7"] = {"verdict": "영상 없음", "n_videos": 0, "n_excluded": 0}
    elif g7ok:
        groups["G7"] = hf.gen_group_verdict([r["curve"]["cost"] for r in g7ok], grid, [None if r["ft"]["at_grid_edge"] else r["ft"]["f"] for r in g7ok], float(np.mean([r["ref_f"] for r in g7])),
                                            cfg["generalization"], len(g7) - len(g7ok), len(g7))
    else:
        groups["G7"] = {"verdict": "판단 불가", "n_videos": 0, "n_excluded": len(g7), "n_total": len(g7), "reason": "분석된 영상이 없음"}
    groups["G7"]["detect_fail_rates"] = [round(1 - r["detect_rate"], 4) for r in g7]
    log(f"  G7 ({EXT_GROUP_KO['G7']}): {groups['G7']['verdict']}" + (f" (합동 f {groups['G7']['pooled_f']:.1f} px)" if "pooled_f" in groups["G7"] else ""))
    # ---- rule v2 verification on all new videos
    rows = [{"video": r["video"], "rating": r["rating"], "f": None if r["ft"]["at_grid_edge"] else r["ft"]["f"], "ref_f": r["ref_f"], "sigma": r["ft"]["boot_log_std"]} for r in ok]
    ver = hf.gen_verify(rows, cfg["verification_v2"])
    log(f"  판정 v2 검증: {ver['verdict']}" + (f" ({ver.get('reason')})" if ver.get("reason") else ""))
    # ---- record-only quantities and the comparison with 25
    rec = hf.ext_distance_records(results, entries, grid, cfg["distance"]) if ok else {}
    hand = hf.ext_hand_compare(results, grid, cfg["generalization"], float(cfg["hand_compare"]["right_distance_cm"])) if ok else {}
    ex25 = [r for r in hf.gen_existing_rows(cfg) if r["source"] == "25 (1x)"]
    for r in ex25:
        r["rating"] = hf.gen_rating(r["tilt_s"], r["theta_max"], r["sigma"], v2)
    ext_figures(run_root / "figures", results, groups, cfg)
    ext_report(run_root / "report.md", results, groups, rec, hand, cfg)
    ext_summary(run_root / "summary.md", results, groups, ver, rec, hand, ex25, cfg)
    with (run_root / "per_frame.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "index", "time_s", "theta2d_deg", "centre_u", "centre_v", "size_ratio", "segment"])
        for r in ok:
            for q in r["traj"]["rows"]:
                wr.writerow([r["video"], q["index"], f"{q['time_s']:.4f}", "" if q["theta2d_deg"] is None else f"{q['theta2d_deg']:.2f}", f"{q['u']:.2f}", f"{q['v']:.2f}", f"{q['size_ratio']:.5f}", r["traj"]["segment"]])
    for r in results:
        r.pop("traj", None)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "groups": groups, "verification_v2": ver, "records": rec, "hand_compare": hand, "existing_25": ex25}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_26": cfg["code_26"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, groups, ver


def ext_dynamics_lines(results: list, cfg: dict) -> list:
    """Table of the motion indicators (G7 left hand beside the G6 right-hand videos at hand_compare.right_distance_cm), per video and as group medians; values only."""
    d0 = float(cfg["hand_compare"]["right_distance_cm"])
    sets = [("오른손", [r for r in results if r["group"] == "G6" and "dynamics" in r and abs(r["distance_cm"] - d0) < 1e-9]), ("왼손", [r for r in results if r["group"] == "G7" and "dynamics" in r])]
    if not all(rs for _, rs in sets):
        return []
    L = ["| 손 | 영상 | 기울기 구간(초) | θ2d 중앙값 / 최댓값(°) | 기울기 축 덮임 (6구간 중 / 구간별 프레임 수) | 손 움직임 속도 중앙값 / 90번째 (손 박스 길이/초) | 마디 길이 비 변화 중앙값 / 90번째 (%) | 손가락 벌어짐 변화 중앙값 / 90번째 (°) | 부트스트랩 로그 SD |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, rs in sets:
        for r in rs:
            m, dy = r["motion"], r["dynamics"]
            L.append(f"| {name} | {r['video']} | {m['tilt_duration_s']:.1f} | {long_fmt(m['theta_median_deg'], 1)} / {long_fmt(m['theta_max_deg'], 1)} | {dy['axis_bins']} / {dy['axis_counts']} | "
                     f"{long_fmt(dy['speed']['median'], 2)} / {long_fmt(dy['speed']['p90'], 2)} | {long_fmt(dy['bone_ratio_pct']['median'], 2)} / {long_fmt(dy['bone_ratio_pct']['p90'], 2)} | "
                     f"{long_fmt(dy['spread_change_deg']['median'], 2)} / {long_fmt(dy['spread_change_deg']['p90'], 2)} | {long_fmt(r['ft']['boot_log_std'], 4)} |")
        med = lambda g: np.nanmedian([g(r) if g(r) is not None else np.nan for r in rs])      # noqa: E731
        L.append(f"| **{name} 중앙값** | {len(rs)}개 | {med(lambda r: r['motion']['tilt_duration_s']):.1f} | {med(lambda r: r['motion']['theta_median_deg']):.1f} / {med(lambda r: r['motion']['theta_max_deg']):.1f} | "
                 f"{med(lambda r: r['dynamics']['axis_bins']):.1f} | {med(lambda r: r['dynamics']['speed']['median']):.2f} / {med(lambda r: r['dynamics']['speed']['p90']):.2f} | "
                 f"{med(lambda r: r['dynamics']['bone_ratio_pct']['median']):.2f} / {med(lambda r: r['dynamics']['bone_ratio_pct']['p90']):.2f} | "
                 f"{med(lambda r: r['dynamics']['spread_change_deg']['median']):.2f} / {med(lambda r: r['dynamics']['spread_change_deg']['p90']):.2f} | {med(lambda r: r['ft']['boot_log_std']):.4f} |")
    return L


def ext_figures(out: Path, results: list, groups: dict, cfg: dict):
    plt = _plt()
    ok = [r for r in results if "ft" in r]
    g6 = [r for r in ok if r["group"] == "G6"]
    cols = {1: "#2a6fbb", 2: "#c47a00", 3: "#3a9a4a"}
    if g6:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for r in g6:
            f, sg, d = r["ft"]["f"], r["ft"]["boot_log_std"], r["distance_cm"] + (r["round"] - 2) * 0.9
            if r["ft"]["at_grid_edge"]:
                ax.plot(d, f, "x", color="#c00")
            else:
                ax.errorbar(d, f, yerr=[[f - f * np.exp(-sg)], [f * np.exp(sg) - f]], fmt="o", color=cols.get(r["round"], "#555"), capsize=3, label=f"{r['round']}회차" if r["video"] == next(x["video"] for x in g6 if x["round"] == r["round"]) else None)
        for d, row in groups["G6"].get("distances", {}).items():
            if "pooled_f" in row and not row["at_grid_edge"]:
                ax.add_patch(plt.Rectangle((float(d) - 3.2, row["interval_95"][0]), 6.4, row["interval_95"][1] - row["interval_95"][0], color="#555", alpha=0.15))
                ax.hlines(row["pooled_f"], float(d) - 3.2, float(d) + 3.2, color="#333", lw=2)
        ax.set_xlabel("손-카메라 거리 (cm)  (점 = 영상, 막대 = 로그 SD ±1σ, 굵은 선·띠 = 거리별 합동 f와 95% 구간)", fontsize=8)
        ax.set_ylabel("f (px)")
        ax.set_xticks(sorted({r["distance_cm"] for r in g6}))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "f_by_distance.png", dpi=110)
        plt.close(fig)
    gs = [g for g in ("G6", "G7") if any(r["group"] == g for r in ok)]
    if gs:
        fig, axs = plt.subplots(1, len(gs), figsize=(4.6 * len(gs), 4), squeeze=False)
        for ax, g in zip(axs[0], gs):
            vs = [r for r in ok if r["group"] == g]
            for i, r in enumerate(vs):
                f, sg = r["ft"]["f"], r["ft"]["boot_log_std"]
                if r["ft"]["at_grid_edge"]:
                    ax.plot(i, f, "x", color="#c00")
                else:
                    ax.errorbar(i, f, yerr=[[f - f * np.exp(-sg)], [f * np.exp(sg) - f]], fmt="o", color="#2a6fbb", capsize=3)
            gp = groups.get(g, {})
            if g == "G7" and "pooled_f" in gp and not gp["at_grid_edge"]:
                ax.axhspan(*gp["interval_95"], color="#c47a00", alpha=0.18)
                ax.axhline(gp["pooled_f"], color="#c47a00", lw=1.5, label=f"합동 f {gp['pooled_f']:.0f} px (95% 구간)")
                ax.legend(fontsize=7)
            ax.set_xticks(range(len(vs)))
            ax.set_xticklabels([f"{r['video'][4:8]}\n{r['distance_cm']:g}cm" for r in vs], fontsize=7)
            ax.set_title(f"{g} {EXT_GROUP_KO[g]}", fontsize=9)
            ax.set_ylabel("f (px), 막대 = 로그 SD ±1σ")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "f_by_group.png", dpi=110)
        plt.close(fig)


def ext_report(path: Path, results: list, groups: dict, rec: dict, hand: dict, cfg: dict):
    v2 = cfg["rating_v2"]
    ok = [r for r in results if "ft" in r]
    fmtf = lambda r: "값 없음 (" + str(r["ft"]["side"]) + ")" if r["ft"]["at_grid_edge"] else format(r["ft"]["f"], ".1f")      # noqa: E731
    L = ["# 실험 27: 1배 조건 확장 (거리 변화, 왼손)", "", EXT_NOTE, "",
         f"후보: {cfg['candidate']['detector']} + wrist_mode {cfg['candidate']['wrist_mode']} + {cfg['candidate']['estimator']} (25·26과 같음). 모든 영상은 1배 주 카메라, 같은 세션이다.", "",
         "## 1. 영상 목록과 제외", "", f"제외 규칙(실행 전 고정): 유효한 정면 창(1초) 없음, 가장자리 접촉 프레임 > {100 * cfg['exclusion']['edge_rate_max']:.0f}%, 손 검출 실패 프레임 > {100 * cfg['exclusion']['detect_fail_max']:.0f}%.", "",
         "| 그룹 | 영상 | 거리(cm) | 손 | 회차 | 방향 | 길이(초) | 검출 실패율 | 가장자리 접촉 | 포함 | 이유 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['group']} {EXT_GROUP_KO[r['group']]} | {r['video']} | {r['distance_cm']:g} | {EXT_HAND_KO[r['hand']]} | {r['round'] or '-'} | {gen_orient(r['size'])} | {r['duration_s']:.1f} | {100 * (1 - r['detect_rate']):.1f}% | "
                 f"{100 * r['edge_rate']:.1f}% | {'제외' if r['excluded'] else ('포함' if 'ft' in r else '분석 실패')} | {'; '.join(r['excluded']) or r.get('failed', '-')} |")
    L += ["", "## 2. 영상별 촬영 지표와 신뢰 등급(v2)", "",
          f"등급 규칙: 사용 가능 = 기울기 구간 ≥ {v2['tilt_min_s']:.0f}초, θ2d 최댓값 ≥ {v2['theta_max_min_deg']:.0f}°, 부트스트랩 로그 SD ≤ {v2['usable_log_std_max']}; 참고 = 촬영 조건 둘은 만족하고 로그 SD ≤ {v2['reference_log_std_max']}; 그 외 사용 불가.", "",
          "| 영상 | 거리(cm) | 손 | 기울기 구간(초) | 기울기 프레임 | θ2d 중앙값 / 90번째 / 최댓값(°) | 정면 창 손 박스/짧은 변 | 가이드 v2 | 전환 시각(초) | 부트스트랩 로그 SD | 등급 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        m, g = r["motion"], r.get("guide") or {}
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {EXT_HAND_KO[r['hand']]} | {m['tilt_duration_s']:.1f} | {m['n_tilt_frames']} | {long_fmt(m['theta_median_deg'], 1)} / {long_fmt(m['theta_p90_deg'], 1)} / {long_fmt(m['theta_max_deg'], 1)} | "
                 f"{r['hand_box_short_ratio_front']:.3f} | {('충족' if g.get('guide') else '미충족') if 'guide' in g else '-'}{(': ' + '; '.join(g['reasons'])) if g.get('reasons') else ''} | {long_fmt(g.get('t_transition'), 2)} | "
                 f"{long_fmt(r['ft']['boot_log_std'], 4)} | **{r['rating']}** |")
    L += ["", "## 3. 초점거리와 불확실성 (값만)", "", "| 영상 | 거리(cm) | 손 | 회차 | f (px) | f 판정(추정기) | 부트스트랩 로그 SD | 상대 SD(22·25·26과 같은 지표) |", "|---|---|---|---|---|---|---|---|"]
    for r in ok:
        f = r["ft"]
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {EXT_HAND_KO[r['hand']]} | {r['round'] or '-'} | {fmtf(r)} | {f['verdict']} | {long_fmt(f['boot_log_std'], 4)} | {long_fmt(f['boot_rel_std'], 4)} |")
    L += ["", "## 4. 합동 추정과 기록용 값 (값만)", ""]
    if groups.get("G6", {}).get("distances"):
        L += ["거리별 합동 추정(영상별 fixed_template 비용 곡선의 합, 영상 단위 부트스트랩 95% 구간):", "", "| 거리(cm) | 영상 | 제외 | 합동 f (px) | 95% 구간 (px) |", "|---|---|---|---|---|"]
        for d, row in groups["G6"]["distances"].items():
            L.append(f"| {float(d):g} | {row['n_videos']} | {row['n_excluded']} | {'-' if 'pooled_f' not in row else ('값 없음 (' + str(row['side']) + ')' if row['at_grid_edge'] else format(row['pooled_f'], '.1f'))} | "
                     f"{'-' if 'interval_95' not in row else format(row['interval_95'][0], '.0f') + '–' + format(row['interval_95'][1], '.0f')} |")
        if groups["G6"].get("max_over_min"):
            L += ["", f"- 거리별 합동 f의 최대 ÷ 최소 = {groups['G6']['max_over_min']:.3f}"]
    if rec:
        L += ["", f"- log f와 log 거리의 기울기: 영상 단위 {long_fmt(rec.get('slope_video'), 3)} (점 {rec['n_points']}개), 거리별 합동 f 단위 {long_fmt(rec.get('slope_pooled'), 3)}. 순위 상관(Spearman): 영상 단위 {long_fmt(rec.get('spearman_video'), 3)}, 합동 단위 {long_fmt(rec.get('spearman_pooled'), 3)}.",
              "  양의 기울기·상관은 가까울수록가 아니라 멀수록 f가 커지는 경향, 음의 기울기·상관은 가까울수록 f가 커지는 경향이다.", "",
              "촬영 순서(회차)별 f (회차 k = 각 거리의 k번째 영상):", "", "| 회차 | 영상 | 합동 f (px) | 영상별 f (px) |", "|---|---|---|---|"]
        for k, row in rec["rounds"].items():
            L.append(f"| {k} | {', '.join(row['videos'])} | {long_fmt(row['pooled_f'], 1)} | {', '.join(long_fmt(x, 1) for x in row['f_videos'])} |")
    if hand and "pooled_f" in hand.get("left", {}) and "pooled_f" in hand.get("right", {}):
        L += ["", "왼손(G7)과 같은 거리의 오른손(G6) 영상:", "", "| 손 | 영상 | 합동 f (px) | 95% 구간 (px) | 영상별 f (px) |", "|---|---|---|---|---|"]
        for name in ("right", "left"):
            h = hand[name]
            L.append(f"| {EXT_HAND_KO[name]} | {', '.join(h['videos'])} | {long_fmt(h['pooled_f'], 1)} | {h['interval_95'][0]:.0f}–{h['interval_95'][1]:.0f} | {', '.join(long_fmt(x, 1) for x in h['f_videos'])} |")
        L += ["", f"- 합동 f의 비 (왼손 ÷ 오른손) = {long_fmt(hand['ratio_left_over_right'], 3)}"]
    dyn = ext_dynamics_lines(results, cfg)
    if dyn:
        L += ["", "왼손(G7)과 같은 세션 오른손 60 cm(G6)의 동작 지표 (기록용, 실험 23과 같은 정의, 기울기 구간 프레임 기준):", ""] + dyn
    L += ["", "## 5. 한계", "",
          "- 이 보고서는 값만 적는다. 기준 f(ref_f)와의 비교, 판정의 근거 표, 실험 25와의 비교는 summary.md에 있다.",
          "- 영상은 모두 같은 세션·같은 폰이라 거리와 촬영 순서의 효과가 세션 효과와 섞여 있다.",
          "- 거리별 영상이 3개씩(회차 3)뿐이라 합동 추정의 구간이 넓다.", ""]
    path.write_text("\n".join(L) + "\n")


def ext_summary(path: Path, results: list, groups: dict, ver: dict, rec: dict, hand: dict, ex25: list, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    ok = [r for r in results if "ft" in r]
    dcfg, gcfg = cfg["distance"], cfg["generalization"]
    g6, g7 = groups["G6"], groups["G7"]
    ref6 = g6.get("ref_f")
    l6 = f"- **G6 거리 ({EXT_GROUP_KO['G6']}): {g6['verdict']}**"
    if g6.get("reason"):
        l6 += f" — {g6['reason']}"
    elif g6.get("distances"):
        l6 += " — 거리별 합동 f " + ", ".join(f"{float(d):g} cm {format(r['pooled_f'], '.1f')} px" for d, r in g6["distances"].items() if "pooled_f" in r and not r["at_grid_edge"])
        l6 += f", 최대/최소 {long_fmt(g6.get('max_over_min'), 3)} (기준 ≤ {dcfg['ratio_max']}), ref_f {ref6:.0f}, 모두 ±{100 * dcfg['pooled_tol']:.0f}% 안: {'예' if all(r.get('pooled_within_tol') for r in g6['distances'].values()) else '아니오'}"
    l7 = f"- **G7 왼손: {g7['verdict']}**"
    if "pooled_f" in g7:
        l7 += (f" — 합동 f {g7['pooled_f']:.1f} px (95% 구간 {g7['interval_95'][0]:.0f}–{g7['interval_95'][1]:.0f}), ref_f {g7['ref_f']:.0f}, 합동이 ±{100 * gcfg['pooled_tol']:.0f}% 안: {'예' if g7['pooled_within_tol'] else '아니오'}, "
               f"영상 중 ±{100 * gcfg['video_tol']:.0f}% 안 {100 * g7['video_share_within_tol']:.0f}% (영상 {g7['n_videos']}개, 제외 {g7['n_excluded']}개)")
    elif g7.get("reason"):
        l7 += f" — {g7['reason']}"
    L = ["# 실험 27 summary", "", f"**{EXT_NOTE}**", "", "## 판정", "", l6, l7,
         f"- **판정 v2 검증: {ver['verdict']}**" + (f" — {ver['reason']}" if ver.get("reason") else "") + f" (사용 가능 {ver['n_usable']} / 새 영상 {ver['n_videos']})"
         + ("" if "within_share" not in ver else f", 사용 가능 영상 중 2σ 안 {100 * ver['within_share']:.0f}% (기준 ≥ {100 * cfg['verification_v2']['within_share_min']:.0f}%), |오차| 중앙값 {100 * ver['median_abs_err']:.1f}% (기준 ≤ {100 * cfg['verification_v2']['median_abs_err_max']:.0f}%)"),
         "", "## 1. 영상별 ref_f 대비 오차", "", "| 영상 | 그룹 | 거리(cm) | 손 | 회차 | ref_f | f | 오차 | 로그 SD | 2σ 안 | 등급 | 가이드 v2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "ft" not in r:
            L.append(f"| {r['video']} | {r['group']} | {r['distance_cm']:g} | {EXT_HAND_KO[r['hand']]} | {r['round'] or '-'} | {r['ref_f']:.0f} | 제외: {'; '.join(r['excluded']) or r.get('failed', '-')} | - | - | - | - | - |")
            continue
        f, sg = r["ft"], r["ft"]["boot_log_std"]
        err = None if f["at_grid_edge"] else f["f"] / r["ref_f"] - 1.0
        inside = None if f["at_grid_edge"] else abs(np.log(f["f"] / r["ref_f"])) <= float(cfg["verification_v2"]["within_sigma"]) * sg
        L.append(f"| {r['video']} | {r['group']} | {r['distance_cm']:g} | {EXT_HAND_KO[r['hand']]} | {r['round'] or '-'} | {r['ref_f']:.0f} | {'값 없음 (' + str(f['side']) + ')' if f['at_grid_edge'] else format(f['f'], '.1f')} | {pct(err)} | "
                 f"{sg:.4f} | {'-' if inside is None else ('예' if inside else '아니오')} | {r['rating']} | {'충족' if (r.get('guide') or {}).get('guide') else '미충족'} |")
    L += ["", "## 2. G6 거리별 합동 추정 (판정 근거)", ""]
    if g6.get("distances"):
        L += [f"통과 조건(실행 전 고정): 세 거리의 합동 f가 모두 ref_f의 ±{100 * dcfg['pooled_tol']:.0f}% 안, 최대 ÷ 최소 ≤ {dcfg['ratio_max']}, 거리마다 포함 영상 ≥ {dcfg['min_videos_per_distance']}개.", "",
              "| 거리(cm) | 영상 | 제외 | 합동 f (px) | 95% 구간 (px) | ref_f 대비 | 구간이 ref_f를 포함 | ±10% 안 |", "|---|---|---|---|---|---|---|---|"]
        for d, row in g6["distances"].items():
            if "pooled_f" not in row:
                L.append(f"| {float(d):g} | {row['n_videos']} | {row['n_excluded']} | - | - | - | - | - |")
                continue
            lo, hi = row["interval_95"]
            L.append(f"| {float(d):g} | {row['n_videos']} | {row['n_excluded']} | {'값 없음 (' + str(row['side']) + ')' if row['at_grid_edge'] else format(row['pooled_f'], '.1f')} | {lo:.0f}–{hi:.0f} | "
                     f"{pct(None if row['at_grid_edge'] else row['pooled_f'] / ref6 - 1.0)} | {'예' if lo <= ref6 <= hi else '아니오'} | {'예' if row.get('pooled_within_tol') else '아니오'} |")
        L += ["", f"- 최대 ÷ 최소 = {long_fmt(g6.get('max_over_min'), 3)}"]
    if rec:
        L += ["", "기록용(판정 아님): log f와 log 거리의 기울기와 순위 상관, 촬영 순서별 f.", "",
              f"- 기울기 (영상 단위, 점 {rec['n_points']}개): {long_fmt(rec.get('slope_video'), 3)}; 거리별 합동 f 단위: {long_fmt(rec.get('slope_pooled'), 3)}",
              f"- Spearman (영상 단위): {long_fmt(rec.get('spearman_video'), 3)}; 거리별 합동 f 단위: {long_fmt(rec.get('spearman_pooled'), 3)}",
              "- 부호: 양수는 멀수록 f가 커지는 경향, 음수는 가까울수록 f가 커지는 경향이다.", "",
              "| 회차 | 영상 | 합동 f (px) | ref_f 대비 | 영상별 f (px) |", "|---|---|---|---|---|"]
        for k, row in rec["rounds"].items():
            L.append(f"| {k} | {', '.join(row['videos'])} | {long_fmt(row['pooled_f'], 1)} | {pct(None if row['pooled_f'] is None or ref6 is None else row['pooled_f'] / ref6 - 1.0)} | {', '.join(long_fmt(x, 1) for x in row['f_videos'])} |")
    L += ["", "## 3. G7 왼손 (판정 근거)와 같은 세션 오른손 60 cm 비교", ""]
    if "pooled_f" in g7:
        L += [f"통과 조건(26의 그룹 판정과 같음): 합동 f가 ref_f의 ±{100 * gcfg['pooled_tol']:.0f}% 안, 영상의 {100 * gcfg['video_fraction']:.0f}% 이상이 ±{100 * gcfg['video_tol']:.0f}% 안, 영상 ≥ {gcfg['min_videos']}개.", ""]
    if hand and "pooled_f" in hand.get("left", {}) and "pooled_f" in hand.get("right", {}):
        L += ["| 손 | 영상 | 합동 f (px) | 95% 구간 (px) | ref_f 대비 | 영상별 f (오차) |", "|---|---|---|---|---|---|"]
        for name in ("right", "left"):
            h = hand[name]
            ref_h = float(np.mean([r["ref_f"] for r in results if r["video"] in h["videos"]]))
            L.append(f"| {EXT_HAND_KO[name]} | {', '.join(h['videos'])} | {h['pooled_f']:.1f} | {h['interval_95'][0]:.0f}–{h['interval_95'][1]:.0f} | {pct(h['pooled_f'] / ref_h - 1.0)} | "
                     f"{', '.join('-' if x is None else format(x, '.1f') + ' (' + pct(x / ref_h - 1.0) + ')' for x in h['f_videos'])} |")
        L += ["", f"- 합동 f의 비 (왼손 ÷ 오른손 60 cm) = {long_fmt(hand['ratio_left_over_right'], 3)}"]
    dyn = ext_dynamics_lines(results, cfg)
    if dyn:
        L += ["", "동작 지표 비교 (기록용, 판정에 쓰지 않음; 정의는 실험 23과 같다: 속도 = 관절 평균 이동 / 손 박스 긴 변 / 초, 마디 길이 비 변화 = 손가락 마디·MCP 연결 길이가 정면 템플릿 대비 바뀐 비율의 중앙값, "
                  "손가락 벌어짐 변화 = 검지-새끼 방향 각도가 정면 창 중앙값과 다른 정도):", ""] + dyn
    if (cfg.get("notes") or {}).get("G7"):
        L += ["", f"촬영자 메모: {cfg['notes']['G7']} (촬영자 구두 보고이며 분석에 쓰지 않았다.)"]
    L += ["", "## 4. 판정 v2 검증 (G6 + G7 새 영상 전체)", "",
          f"규칙(26의 4-2): 사용 가능 영상 ≥ {cfg['verification_v2']['min_usable']}개이고, 그중 |log(f/ref_f)| ≤ {cfg['verification_v2']['within_sigma']:g}σ인 비율 ≥ {100 * cfg['verification_v2']['within_share_min']:.0f}%, |오차| 중앙값 ≤ {100 * cfg['verification_v2']['median_abs_err_max']:.0f}%.", "",
          "| 등급 | 영상 수 | 영상 |", "|---|---|---|"]
    for name in ("사용 가능", "참고", "사용 불가"):
        vs = [r["video"] for r in ok if r["rating"] == name]
        L.append(f"| {name} | {len(vs)} | {', '.join(vs) or '-'} |")
    L += ["", f"- 결과: {ver['verdict']}" + (f" ({ver['reason']})" if ver.get("reason") else "")]
    L += ["", "## 5. 실험 25와의 비교 (사후 기술, 같은 v2 규칙으로 매긴 등급)", "",
          "실험 25는 1배 긴 영상 6개(IMG_7190~7195)다. 영상이 서로 다르므로 영상별이 아니라 그룹 요약을 나란히 둔다.", "",
          "| 묶음 | 영상 | f 중앙값 (px) | |오차| 중앙값 | ±15% 안 | 로그 SD 중앙값 | 사용 가능 / 참고 / 사용 불가 |", "|---|---|---|---|---|---|---|"]

    def summarize(label, rows):
        rows = [x for x in rows if not x["at_grid_edge"]]
        if not rows:
            return f"| {label} | 0 | - | - | - | - | - |"
        e = np.array([abs(x["f"] / x["ref_f"] - 1.0) for x in rows])
        rt = [x["rating"] for x in rows]
        return (f"| {label} | {len(rows)} | {np.median([x['f'] for x in rows]):.1f} | {100 * np.median(e):.1f}% | {100 * np.mean(e <= float(gcfg['video_tol'])):.0f}% | {np.median([x['sigma'] for x in rows]):.4f} | "
                f"{rt.count('사용 가능')} / {rt.count('참고')} / {rt.count('사용 불가')} |")
    new_rows = [{"f": r["ft"]["f"], "at_grid_edge": r["ft"]["at_grid_edge"], "ref_f": r["ref_f"], "sigma": r["ft"]["boot_log_std"], "rating": r["rating"], "group": r["group"], "distance_cm": r["distance_cm"]} for r in ok]
    L.append(summarize("실험 25 (1배, 긴 영상)", ex25))
    L.append(summarize("실험 27 전체 (G6 + G7)", new_rows))
    L.append(summarize("실험 27 G6 (오른손, 50 / 60 / 75 cm)", [x for x in new_rows if x["group"] == "G6"]))
    for d in sorted({x["distance_cm"] for x in new_rows if x["group"] == "G6"}):
        L.append(summarize(f"  G6 {d:g} cm", [x for x in new_rows if x["group"] == "G6" and x["distance_cm"] == d]))
    L.append(summarize("실험 27 G7 (왼손, 60 cm)", [x for x in new_rows if x["group"] == "G7"]))
    L += ["", "## 한계", "",
          "- 판정 v2 검증은 \"사용 가능\" 영상이 5개 미만이면 판단하지 않는다(규칙). G7은 영상이 3개 미만, G6은 거리마다 포함 영상이 2개 미만이면 판단하지 않는다.",
          "- 기준 f는 사용자 제공 값(1배 1680 px)이며 원본 보정 파일은 서버에 없다.",
          "- 영상은 모두 같은 세션·같은 폰이다. 거리·촬영 순서·손의 효과가 세션 효과와 섞여 있고, 이 결과는 이 세션에 대한 것이다.",
          "- 기울기·순위 상관은 점이 9개(거리 3종)라 기록용이다. 판정에 쓰지 않았다.", ""]
    path.write_text("\n".join(L) + "\n")


def ext_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "27_main_camera_extension", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": EXT_NOTE, "candidate": cfg["candidate"], "videos": {}}
    for v in r["videos"]:
        e = {"group": v["group"], "distance_cm": v["distance_cm"], "hand": v["hand"], "round": v["round"], "duration_s": rd(v["duration_s"], 2), "detect_fail_rate": rd(1 - v["detect_rate"]),
             "edge_rate": rd(v["edge_rate"]), "excluded": v["excluded"] or None}
        if "ft" in v:
            f, mo, g = v["ft"], v["motion"], v.get("guide") or {}
            e.update({"f_px": None if f["at_grid_edge"] else round(f["f"], 1), "status": "value" if not f["at_grid_edge"] else f"값 없음 ({f['side']})", "f_verdict_estimator": f["verdict"], "rating_v2": v["rating"],
                      "boot_log_std": rd(f["boot_log_std"], 5), "tilt_duration_s": rd(mo["tilt_duration_s"], 2), "n_tilt_frames": mo["n_tilt_frames"], "theta_median_deg": rd(mo["theta_median_deg"], 1),
                      "theta_p90_deg": rd(mo["theta_p90_deg"], 1), "theta_max_deg": rd(mo["theta_max_deg"], 1), "hand_box_short_ratio_front": rd(v["hand_box_short_ratio_front"]),
                      "guide_v2": {k: g.get(k) for k in ("guide", "reasons", "t_transition")} if "guide" in g else None})
            dy = v.get("dynamics")
            if dy:
                e["dynamics"] = {"axis_bins": dy["axis_bins"], "axis_counts": dy["axis_counts"], **{k: {"median": rd(dy[k]["median"], 3), "p90": rd(dy[k]["p90"], 3)} for k in ("speed", "bone_ratio_pct", "spread_change_deg")}}
        out["videos"][v["video"]] = e
    out["verification_v2"] = {"verdict": r["verification_v2"]["verdict"], "n_usable": r["verification_v2"]["n_usable"], "n_videos": r["verification_v2"]["n_videos"], "reason": r["verification_v2"].get("reason")}
    g6, g7 = r["groups"]["G6"], r["groups"]["G7"]
    pf = lambda d: None if d.get("pooled_f") is None or d.get("at_grid_edge") else round(d["pooled_f"], 1)      # noqa: E731
    out["groups"] = {"G6": {"verdict": g6["verdict"], "reason": g6.get("reason"), "max_over_min": rd(g6.get("max_over_min"), 4),
                            "distances": {f"{float(d):g}": {"n_videos": x["n_videos"], "n_excluded": x["n_excluded"], "pooled_f_px": pf(x), "interval_95_px": None if "interval_95" not in x else [round(q, 1) for q in x["interval_95"]]}
                                          for d, x in g6.get("distances", {}).items()}},
                     "G7": {"verdict": g7["verdict"], "reason": g7.get("reason"), "n_videos": g7.get("n_videos"), "n_excluded": g7.get("n_excluded"), "pooled_f_px": pf(g7),
                            "interval_95_px": None if "interval_95" not in g7 else [round(q, 1) for q in g7["interval_95"]]}}
    rec = r.get("records") or {}
    out["records"] = {"slope_log_f_vs_log_distance": {"video": rd(rec.get("slope_video"), 4), "pooled": rd(rec.get("slope_pooled"), 4)}, "spearman": {"video": rd(rec.get("spearman_video"), 4), "pooled": rd(rec.get("spearman_pooled"), 4)},
                      "pooled_f_by_round_px": {str(k): None if x["pooled_f"] is None else round(x["pooled_f"], 1) for k, x in (rec.get("rounds") or {}).items()}}
    h = r.get("hand_compare") or {}
    out["hand_compare"] = {name: {"videos": h[name]["videos"], "pooled_f_px": pf(h[name]), "interval_95_px": None if "interval_95" not in h[name] else [round(q, 1) for q in h[name]["interval_95"]]} for name in ("right", "left") if name in h}
    out["hand_compare"]["ratio_left_over_right"] = rd(h.get("ratio_left_over_right"), 4)
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# wide  (experiment 29: 0.5x lens at about 60 cm, hypothesis "the method always reads about 62 cm"; new section)
#
#     python HandCalib/run.py wide-60cm --config HandCalib/configs/29_wide_60cm.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py wide-60cm --config ... results-yaml --run runs/29_wide_60cm/<UTC>
# The videos (file, group, lens, distance_cm, hand, ref_f) come from the yaml; the per-video processing is 26's (gen_video via wide_video); AnyCalib is a comparison only.
# ====================================================================================================

WIDE_NOTE = ("0.5배 렌즈로 손을 약 60 cm에 두고 찍은 영상에서 \"방법이 늘 약 62 cm로 읽는다\"(추정 f = ref_f × 62.5 ÷ 거리)는 가설을 확인한 실험이다. 후보 설정(mediapipe_hands + 손목 제외 + fixed_template)과 "
             "판정 범위는 실행 전에 고정했고 바꾸지 않았다. 기준 f(ref_f)와의 비교와 암시 거리는 summary.md에만 있다.")


def wide_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/29_wide_60cm.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the yaml entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(wide_results_yaml(hf.resolve(a.run), cfg))
        return None
    entries = hf.wide_check_videos(list(cfg.get("videos") or []))
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"29 0.5배 60 cm: 영상 {[e['file'] for e in entries]} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "session") if k in cfg}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    acs = cfg.get("anycalib_compare") or {}
    ac_res = {}
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            npz = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], cfg["detector_settings"][cfg["candidate"]["detector"]], run_root / "detection_cache", int(cfg["n_workers"]), log)
            jobs_args = [(e, p, npz[(p["video"], "orig")], cfg["pipeline"], cjob) for e, p in zip(entries, preps)]
            results = list(ex.map(hf.wide_video, jobs_args))
        if acs.get("enabled"):
            if acs.get("gpu_ids") and "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(acs["gpu_ids"][0])
            for r, p in zip(results, preps):
                if r["excluded"]:
                    continue
                with hf.fresp_gpu_lock(run_root / "gpu.lock"):
                    hf.fresp_thermal_wait(float(acs["gpu_pause_c"]), float(acs["gpu_resume_c"]), log=log)
                    ac = hf.anycalib_run(p["frames"], copy.deepcopy(cfg["pipeline"]["anycalib"]), float(cfg["pipeline"]["phases"]["front_phase_s"]), lambda m="": None)
                ac_res[r["video"]] = {m: {"median": v["summary"].get("median"), "n": v["summary"].get("n"), "iqr": v["summary"].get("iqr")} for m, v in ac["models"].items()}
                log(f"  {r['video']} AnyCalib: " + ", ".join(f"{m} {v['median']:.1f} px" for m, v in ac_res[r["video"]].items()))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    v2 = cfg["rating_v2"]
    for r in results:
        if r["excluded"]:
            log(f"  {r['video']} ({r['distance_cm']:g} cm): 제외 — {'; '.join(r['excluded'])} (손 검출 실패 {100 * (1 - r['detect_rate']):.1f}%)")
        elif "ft" in r:
            r["rating"] = hf.gen_rating(r["motion"]["tilt_duration_s"], r["motion"]["theta_max_deg"], r["ft"]["boot_log_std"], v2)
            r["anycalib"] = ac_res.get(r["video"])
            r["implied_distance_cm"] = hf.wide_implied_distance(None if r["ft"]["at_grid_edge"] else r["ft"]["f"], r["distance_cm"], r["ref_f"])
            log(f"  {r['video']} ({r['distance_cm']:g} cm): 기울기 {r['motion']['tilt_duration_s']:.1f}초, 최대 θ2d {r['motion']['theta_max_deg']:.0f}°, f {format(r['ft']['f'], '.1f') + ' px' if not r['ft']['at_grid_edge'] else '값 없음'}, "
                f"로그 SD {long_fmt(r['ft']['boot_log_std'], 4)}, 등급 {r['rating']}, 정면 떨림 {r['front_jitter_pct']:.2f}%")
        else:
            log(f"  {r['video']}: 분석 실패 ({r.get('failed')})")
    ok = [r for r in results if "ft" in r]
    grid = np.array(ok[0]["curve"]["grid"]) if ok else np.array([])
    if any(not np.allclose(np.array(r["curve"]["grid"]), grid) for r in ok):
        raise SystemExit("비용 곡선의 f 격자가 영상마다 다릅니다")
    hy = cfg["hypothesis"]
    pooled = hf.ext_pooled([r["curve"]["cost"] for r in ok], grid, hy)
    pooled["n_excluded"] = len(results) - len(ok)
    verdict = hf.wide_hypothesis_verdict(pooled, hy)
    ref_f = float(np.mean([r["ref_f"] for r in results]))
    dist = float(np.mean([r["distance_cm"] for r in results]))
    verdict["ref_f"], verdict["distance_cm"] = ref_f, dist
    if "pooled_f" in verdict and not verdict["at_grid_edge"]:
        verdict["implied_distance_cm"] = hf.wide_implied_distance(verdict["pooled_f"], dist, ref_f)
        verdict["implied_distance_interval_cm"] = [hf.wide_implied_distance(x, dist, ref_f) for x in verdict["interval_95"]]
    log(f"  가설 판정: {verdict['verdict']}" + (f" (합동 f {verdict['pooled_f']:.1f} px)" if "pooled_f" in verdict else f" ({verdict.get('reason')})"))
    table = wide_condition_table(cfg, verdict)
    wide_figures(run_root / "figures", results, verdict, table, cfg)
    wide_report(run_root / "report.md", results, verdict, cfg)
    wide_summary(run_root / "summary.md", results, verdict, table, cfg)
    for r in results:
        r.pop("traj", None)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "hypothesis": verdict, "conditions": table}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_27": cfg["code_27"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, verdict


def wide_condition_table(cfg: dict, verdict: dict) -> list:
    """Rows of the comparison table: condition, actual distance (cm), pooled f (px), reference f, implied distance (cm). 1x 50 / 60 / 75 cm from 27, 0.5x 30 cm from 26, 0.5x 60 cm from this run.
    Rows whose stored result is not on disk are skipped."""
    ex = cfg.get("existing") or {}
    rows = []
    p27 = hf.resolve(ex["results_27"]) if ex.get("results_27") else None
    if p27 is not None and p27.exists():
        for d, x in json.loads(p27.read_text())["groups"]["G6"]["distances"].items():
            if x.get("pooled_f") is not None and not x.get("at_grid_edge"):
                rows.append({"condition": f"1배 {float(d):g} cm (실험 27)", "distance_cm": float(d), "pooled_f": x["pooled_f"], "ref_f": float(ex["ref_f_27"]),
                             "implied_distance_cm": hf.wide_implied_distance(x["pooled_f"], float(d), float(ex["ref_f_27"]))})
    p26 = hf.resolve(ex["results_26"]) if ex.get("results_26") else None
    if p26 is not None and p26.exists():
        g = json.loads(p26.read_text())["groups"][ex["group_26"]]
        if g.get("pooled_f") is not None and not g.get("at_grid_edge"):
            rows.append({"condition": "0.5배 30 cm (실험 26)", "distance_cm": float(ex["distance_26_cm"]), "pooled_f": g["pooled_f"], "ref_f": float(ex["ref_f_26"]),
                         "implied_distance_cm": hf.wide_implied_distance(g["pooled_f"], float(ex["distance_26_cm"]), float(ex["ref_f_26"]))})
    if verdict.get("pooled_f") is not None and not verdict.get("at_grid_edge"):
        rows.append({"condition": "0.5배 60 cm (이번)", "distance_cm": verdict["distance_cm"], "pooled_f": verdict["pooled_f"], "ref_f": verdict["ref_f"], "implied_distance_cm": verdict["implied_distance_cm"]})
    return rows


def wide_figures(out: Path, results: list, verdict: dict, table: list, cfg: dict):
    plt = _plt()
    ok = [r for r in results if "ft" in r]
    if ok:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        lo, hi = verdict["support_range"]
        ax.axhspan(lo, hi, color="#3a9a4a", alpha=0.15, label=f"가설 지지 범위 {lo:.0f}–{hi:.0f} px")
        ax.axhline(verdict["predicted_f"], color="#3a9a4a", ls="--", lw=1, label=f"가설 예측 {verdict['predicted_f']:.0f} px")
        for k, r in enumerate(ok):
            f, sg = r["ft"]["f"], r["ft"]["boot_log_std"]
            if r["ft"]["at_grid_edge"]:
                ax.plot(k, f, "x", color="#c00")
            else:
                ax.errorbar(k, f, yerr=[[f - f * np.exp(-sg)], [f * np.exp(sg) - f]], fmt="o", color="#2a6fbb", capsize=3)
        if "pooled_f" in verdict and not verdict["at_grid_edge"]:
            ax.axhspan(*verdict["interval_95"], color="#c47a00", alpha=0.18)
            ax.axhline(verdict["pooled_f"], color="#c47a00", lw=1.5, label=f"합동 f {verdict['pooled_f']:.0f} px (95% 구간)")
        ax.set_xticks(range(len(ok)))
        ax.set_xticklabels([r["video"][4:8] for r in ok])
        ax.set_ylabel("f (px), 막대 = 로그 SD ±1σ")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "f_by_video.png", dpi=110)
        plt.close(fig)
    if table:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        xs = [r["distance_cm"] for r in table]
        ax.plot([0, 80], [0, 80], color="#999", lw=1, label="실제 거리와 같게 읽음")
        ax.axhline(float(cfg["hypothesis"]["ref_distance_cm"]), color="#c47a00", ls="--", lw=1, label=f"가설: 늘 {float(cfg['hypothesis']['ref_distance_cm']):g} cm")
        for r in table:
            ax.plot(r["distance_cm"], r["implied_distance_cm"], "o", color="#2a6fbb")
            ax.annotate(r["condition"].replace(" (실험", "\n(실험").replace(" (이번)", "\n(이번)"), (r["distance_cm"], r["implied_distance_cm"]), fontsize=6, xytext=(4, -16), textcoords="offset points")
        ax.set_xlabel("실제 거리 (cm)")
        ax.set_ylabel("암시 거리 = 거리 × f ÷ ref_f (cm)")
        ax.set_xlim(20, 80)
        ax.set_ylim(20, 80)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "implied_distance.png", dpi=110)
        plt.close(fig)


def wide_report(path: Path, results: list, verdict: dict, cfg: dict):
    v2 = cfg["rating_v2"]
    ok = [r for r in results if "ft" in r]
    L = ["# 실험 29: 0.5배 렌즈, 60 cm", "", WIDE_NOTE, "",
         f"후보: {cfg['candidate']['detector']} + wrist_mode {cfg['candidate']['wrist_mode']} + {cfg['candidate']['estimator']} (25·26·27과 같음). 영상은 모두 같은 폰, 0.5배 렌즈, 오른손, 세로 영상이다.", "",
         "## 1. 영상 목록, 검출 상태, 제외", "", f"제외 규칙(실행 전 고정, 27과 같음): 유효한 정면 창(1초) 없음, 가장자리 접촉 프레임 > {100 * cfg['exclusion']['edge_rate_max']:.0f}%, 손 검출 실패 프레임 > {100 * cfg['exclusion']['detect_fail_max']:.0f}%.", "",
         "| 영상 | 거리(cm) | 방향 | 길이(초) | 처리 해상도 | 검출 실패율 | 가장자리 접촉 | 포함 | 이유 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {gen_orient(r['size'])} | {r['duration_s']:.1f} | {r['size'][0]}×{r['size'][1]} | {100 * (1 - r['detect_rate']):.1f}% | {100 * r['edge_rate']:.1f}% | "
                 f"{'제외' if r['excluded'] else ('포함' if 'ft' in r else '분석 실패')} | {'; '.join(r['excluded']) or r.get('failed', '-')} |")
    L += ["", "## 2. 영상별 촬영 지표와 신뢰 등급(v2)", "",
          f"등급 규칙: 사용 가능 = 기울기 구간 ≥ {v2['tilt_min_s']:.0f}초, θ2d 최댓값 ≥ {v2['theta_max_min_deg']:.0f}°, 부트스트랩 로그 SD ≤ {v2['usable_log_std_max']}; 참고 = 촬영 조건 둘은 만족하고 로그 SD ≤ {v2['reference_log_std_max']}; 그 외 사용 불가.", "",
          "| 영상 | 기울기 구간(초) | 기울기 프레임 | θ2d 중앙값 / 최댓값(°) | 정면 창 손 박스/짧은 변 | 정면 창 키포인트 떨림(손 박스 대비 %) | 가이드 v2 | 부트스트랩 로그 SD | 등급 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        m, g = r["motion"], r.get("guide") or {}
        L.append(f"| {r['video']} | {m['tilt_duration_s']:.1f} | {m['n_tilt_frames']} | {long_fmt(m['theta_median_deg'], 1)} / {long_fmt(m['theta_max_deg'], 1)} | {r['hand_box_short_ratio_front']:.3f} | {r['front_jitter_pct']:.2f} | "
                 f"{('충족' if g.get('guide') else '미충족') if 'guide' in g else '-'}{(': ' + '; '.join(g['reasons'])) if g.get('reasons') else ''} | {long_fmt(r['ft']['boot_log_std'], 4)} | **{r['rating']}** |")
    L += ["", "## 3. 초점거리와 불확실성 (값만)", "", "| 영상 | f (px) | f 판정(추정기) | 부트스트랩 로그 SD | 상대 SD(22·25·26·27과 같은 지표) |", "|---|---|---|---|---|"]
    for r in ok:
        f = r["ft"]
        L.append(f"| {r['video']} | {'값 없음 (' + str(f['side']) + ')' if f['at_grid_edge'] else format(f['f'], '.1f')} | {f['verdict']} | {long_fmt(f['boot_log_std'], 4)} | {long_fmt(f['boot_rel_std'], 4)} |")
    if "pooled_f" in verdict:
        L += ["", f"합동 추정(영상별 fixed_template 비용 곡선의 합, 영상 단위 부트스트랩 95% 구간): {'값 없음 (' + str(verdict['side']) + ')' if verdict['at_grid_edge'] else format(verdict['pooled_f'], '.1f') + ' px'} "
                  f"(구간 {verdict['interval_95'][0]:.0f}–{verdict['interval_95'][1]:.0f} px, 영상 {verdict['n_videos']}개, 제외 {verdict['n_excluded']}개)"]
    L += ["", "## 4. 한계", "",
          "- 이 보고서는 값만 적는다. 기준 f(ref_f) 대비 오차, 암시 거리, 가설 판정의 근거, 다른 실험과의 비교, AnyCalib 비교는 summary.md에 있다.",
          "- 영상은 3개이고 모두 같은 세션이다. 0.5배 렌즈는 왜곡이 커서 pinhole을 가정하는 이 방법에는 f 변화와 왜곡 가정 위반이 섞여 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def wide_summary(path: Path, results: list, verdict: dict, table: list, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    ok = [r for r in results if "ft" in r]
    hy = cfg["hypothesis"]
    lo, hi = verdict["support_range"]
    line = f"- **가설 판정: {verdict['verdict']}**"
    if "pooled_f" in verdict:
        line += (f" — 합동 f {'값 없음 (' + str(verdict['side']) + ')' if verdict['at_grid_edge'] else format(verdict['pooled_f'], '.1f') + ' px'} (95% 구간 {verdict['interval_95'][0]:.0f}–{verdict['interval_95'][1]:.0f}), "
                 f"예측 {verdict['predicted_f']:.0f} px, 지지 범위 {lo:.0f}–{hi:.0f}, 반박 범위 ≥ {verdict['refute_above']:.0f} 또는 ≤ {verdict['refute_below']:.0f} (영상 {verdict['n_videos']}개, 제외 {verdict['n_excluded']}개)")
    elif verdict.get("reason"):
        line += f" — {verdict['reason']}"
    L = ["# 실험 29 summary", "", f"**{WIDE_NOTE}**", "", "## 판정", "", line, "",
         f"가설: 방법이 거리와 상관없이 늘 약 {float(hy['ref_distance_cm']):g} cm로 읽는다 → 추정 f = ref_f × {float(hy['ref_distance_cm']):g} ÷ 거리. 이번 조건(ref_f {verdict['ref_f']:.0f}, 거리 {verdict['distance_cm']:g} cm)의 예측은 {verdict['predicted_f']:.0f} px.", "",
         "## 1. 영상별 ref_f 대비 오차와 암시 거리", "", "암시 거리 = 실제 거리 × f ÷ ref_f (cm).", "",
         "| 영상 | 방향 | ref_f | f | 오차 | 암시 거리(cm) | 로그 SD | 등급 | 가이드 v2 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "ft" not in r:
            L.append(f"| {r['video']} | {gen_orient(r['size'])} | {r['ref_f']:.0f} | 제외: {'; '.join(r['excluded']) or r.get('failed', '-')} | - | - | - | - | - |")
            continue
        f = r["ft"]
        L.append(f"| {r['video']} | {gen_orient(r['size'])} | {r['ref_f']:.0f} | {'값 없음 (' + str(f['side']) + ')' if f['at_grid_edge'] else format(f['f'], '.1f')} | "
                 f"{pct(None if f['at_grid_edge'] else f['f'] / r['ref_f'] - 1.0)} | {long_fmt(r.get('implied_distance_cm'), 1)} | {f['boot_log_std']:.4f} | {r['rating']} | {'충족' if (r.get('guide') or {}).get('guide') else '미충족'} |")
    L += ["", "## 2. 합동 추정 (판정 근거)", ""]
    if "pooled_f" in verdict and not verdict["at_grid_edge"]:
        L += [f"| 합동 f (px) | 95% 구간 (px) | ref_f 대비 | 암시 거리(cm) | 암시 거리 구간(cm) | 예측 {verdict['predicted_f']:.0f} 대비 |", "|---|---|---|---|---|---|",
              f"| {verdict['pooled_f']:.1f} | {verdict['interval_95'][0]:.0f}–{verdict['interval_95'][1]:.0f} | {pct(verdict['pooled_f'] / verdict['ref_f'] - 1.0)} | {verdict['implied_distance_cm']:.1f} | "
              f"{verdict['implied_distance_interval_cm'][0]:.1f}–{verdict['implied_distance_interval_cm'][1]:.1f} | {pct(verdict['pooled_f'] / verdict['predicted_f'] - 1.0)} |"]
    L += ["", "## 3. 조건별 합동 f와 암시 거리", "", "| 조건 | 실제 거리(cm) | 합동 f (px) | ref_f | 암시 거리(cm) |", "|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['condition']} | {r['distance_cm']:g} | {r['pooled_f']:.1f} | {r['ref_f']:.0f} | {r['implied_distance_cm']:.1f} |")
    L += ["", f"- 가설이 맞다면 모든 조건의 암시 거리가 약 {float(hy['ref_distance_cm']):g} cm로 같아야 한다.", "", "## 4. AnyCalib 비교 (판정에 쓰지 않음)", ""]
    ac = [r for r in ok if r.get("anycalib")]
    if ac:
        L += ["실험 25와 같은 설정(pinhole·gen, 모든 프레임). 값은 영상 중앙값.", "", "| 영상 | AnyCalib pinhole (ref_f 대비) | AnyCalib gen (ref_f 대비) | 손 방법 f (ref_f 대비) |", "|---|---|---|---|"]
        for r in ac:
            p, g = r["anycalib"]["anycalib_pinhole"]["median"], r["anycalib"]["anycalib_gen"]["median"]
            f = None if r["ft"]["at_grid_edge"] else r["ft"]["f"]
            L.append(f"| {r['video']} | {p:.1f} ({pct(p / r['ref_f'] - 1.0)}) | {g:.1f} ({pct(g / r['ref_f'] - 1.0)}) | {long_fmt(f, 1)} ({pct(None if f is None else f / r['ref_f'] - 1.0)}) |")
        L += ["", f"- 영상 {len(ac)}개 중앙값: pinhole {np.median([r['anycalib']['anycalib_pinhole']['median'] for r in ac]):.1f} px, gen {np.median([r['anycalib']['anycalib_gen']['median'] for r in ac]):.1f} px"]
    else:
        L += ["AnyCalib을 돌리지 않았다."]
    L += ["", "## 한계", "",
          "- 영상은 3개이고 모두 같은 세션·같은 폰·같은 거리다. 판정은 이 조건에 대한 것이다. 제외된 영상이 있으면 합동 추정은 남은 영상으로만 구한다(영상 2개 미만이면 판단 불가).",
          "- 기준 f는 사용자 제공 값(0.5배 880 px, 1배 1680 px)이며 원본 보정 파일은 서버에 없다.",
          "- 0.5배 렌즈는 왜곡이 커서 pinhole을 가정하는 이 방법에는 f 변화와 왜곡 가정 위반이 섞여 있다. 가설은 단일 상수(약 62.5 cm)만 검정하며 원인을 가려내지는 않는다.",
          "- 가설 지지·반박은 이 영상 3개의 합동 f 한 값으로만 판정한다. 영상별 값과 구간은 근거 표에 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def wide_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "29_wide_60cm", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": WIDE_NOTE, "candidate": cfg["candidate"], "videos": {}}
    for v in r["videos"]:
        e = {"group": v["group"], "lens": v["lens"], "distance_cm": v["distance_cm"], "hand": v["hand"], "size": v["size"], "duration_s": rd(v["duration_s"], 2), "detect_fail_rate": rd(1 - v["detect_rate"]),
             "edge_rate": rd(v["edge_rate"]), "excluded": v["excluded"] or None}
        if "ft" in v:
            f, mo, g = v["ft"], v["motion"], v.get("guide") or {}
            e.update({"f_px": None if f["at_grid_edge"] else round(f["f"], 1), "status": "value" if not f["at_grid_edge"] else f"값 없음 ({f['side']})", "f_verdict_estimator": f["verdict"], "rating_v2": v["rating"],
                      "boot_log_std": rd(f["boot_log_std"], 5), "front_jitter_pct": rd(v["front_jitter_pct"], 3), "tilt_duration_s": rd(mo["tilt_duration_s"], 2), "n_tilt_frames": mo["n_tilt_frames"],
                      "theta_median_deg": rd(mo["theta_median_deg"], 1), "theta_max_deg": rd(mo["theta_max_deg"], 1), "hand_box_short_ratio_front": rd(v["hand_box_short_ratio_front"]),
                      "guide_v2": {k: g.get(k) for k in ("guide", "reasons", "t_transition")} if "guide" in g else None, "principal_point_px": v["principal_point"]})
            if v.get("anycalib"):
                e["anycalib_median_px"] = {k.replace("anycalib_", ""): rd(x["median"], 1) for k, x in v["anycalib"].items()}
        out["videos"][v["video"]] = e
    h = r["hypothesis"]
    out["hypothesis"] = {"verdict": h["verdict"], "reason": h.get("reason"), "predicted_f_px": h["predicted_f"], "support_range_px": [round(x, 1) for x in h["support_range"]],
                         "refute_above_px": round(h["refute_above"], 1), "refute_below_px": round(h["refute_below"], 1), "n_videos": h["n_videos"], "n_excluded": h["n_excluded"],
                         "pooled_f_px": None if h.get("pooled_f") is None or h.get("at_grid_edge") else round(h["pooled_f"], 1),
                         "interval_95_px": None if "interval_95" not in h else [round(x, 1) for x in h["interval_95"]]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# jset  (experiment 30: uncertainty joint sets that fit the wrist-excluded candidate; new section)
#
#     python HandCalib/run.py jointset-sets --config HandCalib/configs/30_jointset_sets.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py jointset-sets --config ... results-yaml --run runs/30_jointset_sets/<UTC>
# Every video of 22-29 is analysed from its stored detections (no detector run). f and the other diagnostics are the candidate's; only the joint-set check of the verdict is replaced.
# ====================================================================================================

JSET_NOTE = ("후보 방법(mediapipe_hands + 손목 제외 + fixed_template)과 f는 그대로 두고, 판정의 jointset_range가 쓰는 불확실성 관절 집합만 손목 제외에 맞게 바꿨을 때 판정이 어떻게 달라지는지 본 실험이다. "
             "집합과 채택 규칙은 실행 전에 고정했고 바꾸지 않았다.")


def jset_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/30_jointset_sets.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the yaml entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(jset_results_yaml(hf.resolve(a.run), cfg))
        return None
    entries = list(cfg["videos"])
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"30 불확실성 집합 교체: 영상 {len(entries)}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cjob = {k: cfg[k] for k in ("candidate", "detector_settings", "jointset") if k in cfg}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = [hf.jset_find_detection(p["frames"], dset, cfg["detections"]["search_dirs"], e.get("source17"), cfg["candidate"]["detector"]) for e, p in zip(entries, preps)]
            results = list(ex.map(hf.jset_video, [(e, p, f["npz"], cfg["pipeline"], cjob) for e, p, f in zip(entries, preps, found)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    for r, f in zip(results, found):
        r["detection_source"] = f["source"]
        if "f" in r:
            log(f"  {r['video']}: f {r['f']:.1f} px, jointset_range 새 {r['jointset_range_new']:.3f} / 옛 {r['jointset_range_old']:.3f}, 판정 새 {r['verdict_new']} / 옛 집합 {r['verdict_old_sets']}")
        else:
            log(f"  {r['video']}: 분석 실패 ({r.get('failed')})")
    ok = [r for r in results if "f" in r]
    vcfg = cfg["pipeline"]["verdict"]
    lim = float(vcfg["rough"]["jointset_range_max"])
    dec = cfg["decision"]
    pairs_new = sum(len(r["jointsets_new"]) for r in ok)
    edge_new = sum(len(r["jointset_edge_new"]) for r in ok)
    edge_old = sum(len(r["jointset_edge_old"]) for r in ok)
    pairs_old = sum(len(r["jointsets_old"]) for r in ok)
    summ = {"n_videos": len(ok), "limit": lim, "within_new": sum(1 for r in ok if r["jointset_range_new"] <= lim), "within_old": sum(1 for r in ok if r["jointset_range_old"] <= lim),
            "edge_pairs_new": edge_new, "pairs_new": pairs_new, "edge_pairs_old": edge_old, "pairs_old": pairs_old,
            "verdict_counts_new": {v: sum(1 for r in ok if r["verdict_new"] == v) for v in ("GOOD", "ROUGH", "REJECT")},
            "verdict_counts_old_sets": {v: sum(1 for r in ok if r["verdict_old_sets"] == v) for v in ("GOOD", "ROUGH", "REJECT")},
            "median_range_new": float(np.median([r["jointset_range_new"] for r in ok])) if ok else None, "median_range_old": float(np.median([r["jointset_range_old"] for r in ok])) if ok else None}
    summ["within_share_new"] = summ["within_new"] / max(len(ok), 1)
    summ["edge_share_new"] = edge_new / max(pairs_new, 1)
    summ["adopted"] = bool(summ["within_share_new"] >= float(dec["min_within_share"]) and summ["edge_share_new"] <= float(dec["max_edge_share"]))
    summ["decision"] = "교체 채택" if summ["adopted"] else "교체해도 도움이 되지 않음"
    log(f"  결정: {summ['decision']} (새 범위가 한계 안인 영상 {summ['within_new']}/{len(ok)}, 격자 경계 집합 {edge_new}/{pairs_new})")
    jset_figures(run_root / "figures", ok, lim)
    jset_report(run_root / "report.md", results, summ, cfg)
    jset_summary(run_root / "summary.md", results, summ, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "summary": summ}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_29": cfg["code_29"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, summ


def jset_figures(out: Path, ok: list, lim: float):
    plt = _plt()
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for r in ok:
        ax.plot(r["jointset_range_old"], r["jointset_range_new"], "o", color="#2a6fbb" if r["lens"] == "1x" else "#c47a00", alpha=0.8)
    ax.axhline(lim, color="#c00", ls="--", lw=1, label=f"ROUGH 한계 {lim:g}")
    ax.plot([0, max(max(r["jointset_range_old"], r["jointset_range_new"]) for r in ok) * 1.05], [0, max(max(r["jointset_range_old"], r["jointset_range_new"]) for r in ok) * 1.05], color="#999", lw=1)
    ax.set_xlabel("jointset_range (옛 집합: palm5, palm_proximal9, flat17 − 손목)")
    ax.set_ylabel("jointset_range (새 집합)")
    ax.set_title("파랑 1배 / 주황 0.5배", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "jointset_range_old_vs_new.png", dpi=110)
    plt.close(fig)


def jset_report(path: Path, results: list, summ: dict, cfg: dict):
    ok = [r for r in results if "f" in r]
    L = ["# 실험 30: 불확실성 관절 집합 교체", "", JSET_NOTE, "",
         f"새 집합: {', '.join(f'{k} {v}' for k, v in cfg['jointset']['sets'].items())}, 그리고 flat17(손목 제외 16점). 옛 집합: palm5, palm_proximal9, flat17에서 손목을 뺀 것(25~29가 쓴 것).", "",
         "## 1. 영상별 값", "", "| 영상 | 렌즈 | f (px) | 검출 출처 | 부트스트랩 상대 SD | 일관성 범위 | 날카로움 | 새 집합 f (prox8 / index_middle8 / ring_pinky8 / flat17) | 새 jointset_range | 옛 집합 f (palm5 / palm_proximal9 / flat17) | 옛 jointset_range |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        mt = r["metrics"]
        jn, jo = r["jointsets_new"], r["jointsets_old"]
        L.append(f"| {r['video']} | {r['lens']} | {r['f']:.1f} | {r['detection_source'].split(' ')[0]} | {long_fmt(mt['bootstrap_rel_std'], 4)} | {long_fmt(mt['consistency_range'], 3)} | {long_fmt(mt['sharpness'], 4)} | "
                 f"{' / '.join(format(jn[k], '.0f') for k in cfg['jointset']['uncertainty_sets'])} | {r['jointset_range_new']:.3f} | {' / '.join(format(v, '.0f') for v in jo.values())} | {r['jointset_range_old']:.3f} |")
    L += ["", "## 2. 판정", "", "| 영상 | 판정 (새 집합) | 실패한 ROUGH 항목 | 실패한 GOOD 항목 | 판정 (옛 집합) |", "|---|---|---|---|---|"]
    for r in ok:
        L.append(f"| {r['video']} | {r['verdict_new']} | {', '.join(r['failed_rough_new']) or '-'} | {', '.join(r['failed_good_new']) or '-'} | {r['verdict_old_sets']} |")
    L += ["", "## 3. 한계", "", "- 영상 33개의 값만 적는다. 기준 f와의 비교는 summary.md에 없다(이 실험은 판정 항목만 본다).", "- 같은 검출·같은 후보 설정이므로 f는 25~29의 값과 같다.", ""]
    path.write_text("\n".join(L) + "\n")


def jset_summary(path: Path, results: list, summ: dict, cfg: dict):
    ok = [r for r in results if "f" in r]
    dec = cfg["decision"]
    L = ["# 실험 30 summary", "", f"**{JSET_NOTE}**", "", "## 판정", "",
         f"- **불확실성 집합 교체: {summ['decision']}** — 새 jointset_range가 ROUGH 한계({summ['limit']:g}) 안인 영상 {summ['within_new']}/{summ['n_videos']} ({100 * summ['within_share_new']:.0f}%, 채택 기준 ≥ {100 * float(dec['min_within_share']):.0f}%), "
         f"격자 경계에 붙은 (영상, 집합) {summ['edge_pairs_new']}/{summ['pairs_new']} ({100 * summ['edge_share_new']:.0f}%, 채택 기준 ≤ {100 * float(dec['max_edge_share']):.0f}%)",
         f"- 옛 집합: 한계 안 {summ['within_old']}/{summ['n_videos']}, 격자 경계 집합 {summ['edge_pairs_old']}/{summ['pairs_old']}",
         f"- jointset_range 중앙값: 새 {summ['median_range_new']:.3f}, 옛 {summ['median_range_old']:.3f}", "",
         "## 1. 판정 분포", "", "| 집합 | GOOD | ROUGH | REJECT |", "|---|---|---|---|",
         f"| 새 집합 | {summ['verdict_counts_new']['GOOD']} | {summ['verdict_counts_new']['ROUGH']} | {summ['verdict_counts_new']['REJECT']} |",
         f"| 옛 집합 | {summ['verdict_counts_old_sets']['GOOD']} | {summ['verdict_counts_old_sets']['ROUGH']} | {summ['verdict_counts_old_sets']['REJECT']} |", "",
         "## 2. 실패한 항목 (새 집합)", ""]
    for kind, key in (("ROUGH", "failed_rough_new"), ("GOOD", "failed_good_new")):
        cnt = {}
        for r in ok:
            for k in r[key]:
                cnt[k] = cnt.get(k, 0) + 1
        L.append(f"- {kind} 기준에서 실패한 항목과 영상 수: " + (", ".join(f"{k} {v}개" for k, v in sorted(cnt.items(), key=lambda x: -x[1])) or "없음"))
    L += ["", "## 3. 영상별 jointset_range (새 / 옛)와 판정", "", "| 영상 | 렌즈 | 새 | 옛 | 판정 (새) | 실패한 ROUGH 항목 |", "|---|---|---|---|---|---|"]
    for r in ok:
        L.append(f"| {r['video']} | {r['lens']} | {r['jointset_range_new']:.3f} | {r['jointset_range_old']:.3f} | {r['verdict_new']} | {', '.join(r['failed_rough_new']) or '-'} |")
    L += ["", "## 한계", "", "- f와 다른 진단값은 바뀌지 않는다. 바뀌는 것은 jointset_range와 그것이 들어가는 판정 라벨뿐이다.",
          "- 판정 라벨이 실제 오차를 맞히는지는 이 실험이 아니라 실험 31에서 본다.", ""]
    path.write_text("\n".join(L) + "\n")


def jset_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "30_jointset_sets", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": JSET_NOTE, "candidate": cfg["candidate"],
           "jointset": cfg["jointset"], "decision": cfg["decision"], "videos": {}}
    for v in r["videos"]:
        if "f" not in v:
            out["videos"][v["video"]] = {"failed": v.get("failed")}
            continue
        mt = v["metrics"]
        out["videos"][v["video"]] = {"lens": v["lens"], "f_px": round(v["f"], 1), "detection_source": v["detection_source"].split(" ")[0], "bootstrap_rel_std": rd(mt["bootstrap_rel_std"], 5), "boot_log_std": rd(v["boot_log_std"], 5),
                                     "consistency_range": rd(mt["consistency_range"], 4), "sharpness": rd(mt["sharpness"], 5), "reproj_rms_rel": rd(mt["reproj_rms_rel"], 5), "theta_p90_deg": rd(mt["theta_p90"], 1),
                                     "axis_bins": mt["axis_bins6"], "n_target_frames": mt["n_target_frames"], "front_jitter_rel": rd(v["front_jitter_rel"], 6), "tilt_duration_s": rd(v["tilt_duration_s"], 2),
                                     "jointsets_new_px": {k: round(x, 1) for k, x in v["jointsets_new"].items()}, "jointset_range_new": rd(v["jointset_range_new"]), "jointset_edge_new": v["jointset_edge_new"],
                                     "jointsets_old_px": {k: round(x, 1) for k, x in v["jointsets_old"].items()}, "jointset_range_old": rd(v["jointset_range_old"]), "jointset_edge_old": v["jointset_edge_old"],
                                     "verdict_new": v["verdict_new"], "failed_rough_new": v["failed_rough_new"], "failed_good_new": v["failed_good_new"], "verdict_old_sets": v["verdict_old_sets"]}
    out["summary"] = {k: (rd(x, 4) if isinstance(x, float) else x) for k, x in r["summary"].items()}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# command line
# ====================================================================================================

COMMANDS = {"protocol": protocol_main, "frame-filter": ff_main, "same-session": session_main, "overlay": overlay_main, "phase-segmentation": pseg_main, "guide-revision": grev_main, "f-response": fresp_main, "error-factors": ferr_main, "long-video-check": long_main, "generalization": gen_main, "main-camera-extension": ext_main, "wide-60cm": wide_main, "jointset-sets": jset_main}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit("usage: run.py {" + ",".join(COMMANDS) + "} [options]   (see the module docstring)")
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
