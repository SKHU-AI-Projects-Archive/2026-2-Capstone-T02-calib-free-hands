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
    python HandCalib/run.py verdict-calibration --config HandCalib/configs/31_verdict_calibration.yaml [results-yaml --run DIR]
    python HandCalib/run.py printed-hand --config HandCalib/configs/34_printed_hand.yaml [--videos A,B] [results-yaml --run DIR]
    python HandCalib/run.py hand-markers --config HandCalib/configs/35_hand_markers.yaml [precheck | --videos A,B | results-yaml --run DIR]

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
# cal  (experiment 31: do the verdict's diagnostics predict the real f error? calibration protocol; new section)
#
#     python HandCalib/run.py verdict-calibration --config HandCalib/configs/31_verdict_calibration.yaml [--out DIR]
#     python HandCalib/run.py verdict-calibration --config ... results-yaml --run runs/31_verdict_calibration/<UTC>
# Reads experiment 30's per-video results (candidate f and every diagnostic) and the user-provided reference f stored with them; no video is processed.
# ====================================================================================================

CAL_NOTE = ("검증된 진단값(부트스트랩 SD, 일관성 범위, 날카로움 등)이 실제 f 오차를 예측하는지 보고, 예측하는 것이 있으면 판정 문턱을 다시 정하는 실험이다. "
            "기준 f(ref_f)는 사용자 제공 값이며 판정 규칙과 문턱 선택 절차는 실행 전에 고정했고 바꾸지 않았다. 오차에 대한 근거는 summary.md에만 있다.")
CAL_PRED_KO = {"boot_log_std": "부트스트랩 로그 SD", "bootstrap_rel_std": "부트스트랩 상대 SD", "consistency_range": "일관성 범위", "jointset_range_new": "관절 집합 범위(새 집합)", "sharpness": "날카로움",
               "reproj_rms_rel": "재투영 RMS(상대)", "front_jitter_rel": "정면 창 떨림(상대)", "theta_p90": "θ2d 90번째(°)", "theta_max": "θ2d 최댓값(°)", "tilt_duration_s": "기울기 구간(초)",
               "hand_box_short_ratio_front": "정면 창 손 박스/짧은 변", "n_target_frames": "사용 프레임 수"}


def cal_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/31_verdict_calibration.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(cal_results_yaml(hf.resolve(a.run), cfg))
        return None
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    t0 = time.time()
    src = json.loads(hf.resolve(cfg["source"]["results_30"]).read_text())
    rows = hf.cal_rows(src["videos"], cfg)
    res = hf.cal_analyse(rows, cfg)
    log(f"31 판정 문턱 재보정: 영상 {len(rows)}개 (보정 {res['n_calibration']}개, 제외·보류 {res['n_held_out']}개), 정확(오차 ≤ {100 * float(cfg['accurate_tol']):.0f}%) {res['n_accurate']}개")
    for k, d in res["predictors"].items():
        if "skipped" in d:
            log(f"  {k}: {d['skipped']}")
        else:
            log(f"  {k}: ρ {d['rho']:+.2f} (구간 {d['ci'][0]:+.2f}~{d['ci'][1]:+.2f}), 정보 있음 {d['informative']}, LOO 균형 정확도 {long_fmt(d['loo']['balanced_accuracy'], 2)}")
    log(f"  결정: {res['decision']}" + (f" ({', '.join(res['useful_predictors'])})" if res["useful_predictors"] else ""))
    cal_figures(run_root / "figures", rows, res, cfg)
    cal_report(run_root / "report.md", rows, res, cfg)
    cal_summary(run_root / "summary.md", rows, res, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"rows": rows, "analysis": res}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_30": cfg["code_30"], "source": cfg["source"], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, rows, res


def cal_figures(out: Path, rows: list, res: dict, cfg: dict):
    plt = _plt()
    names = [k for k, d in res["predictors"].items() if "skipped" not in d]
    if not names:
        return
    ncol = 4
    nrow = (len(names) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
    for ax, k in zip(axs.ravel(), names):
        d = res["predictors"][k]
        for r in rows:
            if r[k] is None or r["excluded"] or r["at_grid_edge"]:
                continue
            ax.plot(r[k], 100 * r["error"], "o" if r["lens"] == cfg["calibration_lens"] else "s", color="#2a6fbb" if r["lens"] == cfg["calibration_lens"] else "#c47a00", alpha=0.8)
        ax.axhline(100 * float(cfg["accurate_tol"]), color="#999", ls="--", lw=1)
        ax.set_xlabel(CAL_PRED_KO.get(k, k), fontsize=8)
        ax.set_ylabel("|f 오차| (%)", fontsize=8)
        ax.set_title(f"ρ {d['rho']:+.2f}" + (" (정보 있음)" if d["informative"] else ""), fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "predictors_vs_error.png", dpi=105)
    plt.close(fig)


def cal_report(path: Path, rows: list, res: dict, cfg: dict):
    names = list(cfg["predictors"])
    L = ["# 실험 31: 판정 진단값 재보정", "", CAL_NOTE, "",
         "## 1. 영상별 진단값 (값만)", "", "| 영상 | 렌즈 | 제외 | " + " | ".join(CAL_PRED_KO.get(k, k) for k in names) + " | 판정(새 집합) |", "|---|---|---|" + "---|" * len(names) + "---|"]
    for r in rows:
        L.append(f"| {r['video']} | {r['lens']} | {'예' if r['excluded'] else '-'} | " + " | ".join(long_fmt(r[k], 4) for k in names) + f" | {r['verdict_new']} |")
    L += ["", "## 2. 한계", "", "- 이 보고서는 진단값만 적는다. 오차와의 상관, 분류 성능, 결정은 summary.md에 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def cal_summary(path: Path, rows: list, res: dict, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    L = ["# 실험 31 summary", "", f"**{CAL_NOTE}**", "", "## 판정", "",
         f"- **판정 문턱 재보정: {res['decision']}**" + (f" — 유용한 진단값: {', '.join(CAL_PRED_KO.get(k, k) for k in res['useful_predictors'])}" if res["useful_predictors"] else
                                                      f" — 정보가 있는 진단값 {sum(1 for d in res['predictors'].values() if d.get('informative'))}개, 그중 leave-one-out 균형 정확도 ≥ {float(cfg['loo_ba_min']):.2f}인 것이 없다"),
         f"- 보정 영상 {res['n_calibration']}개({cfg['calibration_lens']}, 제외·격자 경계 제외), 그중 정확(오차 ≤ {100 * float(cfg['accurate_tol']):.0f}%) {res['n_accurate']}개. 보류 영상(다른 렌즈) {res['n_held_out']}개.",
         f"- 규칙(실행 전 고정): 정보 있음 = 기대 부호의 Spearman |ρ| ≥ {cfg['rho_min']}이고 영상 단위 부트스트랩 95% 구간이 0을 포함하지 않음; 유용 = 정보 있음이고 leave-one-out 균형 정확도 ≥ {cfg['loo_ba_min']}.", "",
         "## 1. 진단값과 오차의 상관 (보정 영상)", "", "| 진단값 | 기대 부호 | ρ | 95% 구간 | 정보 있음 | 전체 자료 문턱 (정확 = 이쪽) | LOO 균형 정확도 | 유용 |", "|---|---|---|---|---|---|---|---|"]
    for k, d in res["predictors"].items():
        if "skipped" in d:
            L.append(f"| {CAL_PRED_KO.get(k, k)} | - | - | - | - | - | - | {d['skipped']} |")
            continue
        sign = {1: "+", -1: "−", 0: "없음"}[d["expected_sign"]]
        L.append(f"| {CAL_PRED_KO.get(k, k)} | {sign} | {d['rho']:+.2f} | {d['ci'][0]:+.2f} ~ {d['ci'][1]:+.2f} | {'예' if d['informative'] else '아니오'} | "
                 f"{'-' if d['full_data_threshold'] is None else format(d['full_data_threshold'], '.4g') + (' 이하' if d['good_below'] else ' 이상')} | {long_fmt(d['loo']['balanced_accuracy'], 2)} | {'예' if d['useful'] else '아니오'} |")
    L += ["", "## 2. 지금 쓰는 분류기의 성능 (보정 영상, 정확 = 오차 ≤ 한계)", "", "| 분류기 | 균형 정확도 | 민감도 | 특이도 | 참양성/거짓음성/참음성/거짓양성 |", "|---|---|---|---|---|"]
    for name, b in (("판정 v2 '사용 가능'", res["baselines"]["rating_v2_usable"]), ("옛 판정 ROUGH 이상 (새 집합)", res["baselines"]["verdict_rough_or_better"]), ("항상 '정확'이라고 답함", res["baselines"]["always_accurate"])):
        L.append(f"| {name} | {long_fmt(b['balanced_accuracy'], 2)} | {long_fmt(b['sensitivity'], 2)} | {long_fmt(b['specificity'], 2)} | {b['tp']}/{b['fn']}/{b['tn']}/{b['fp']} |")
    if res["proposal"]:
        L += ["", "## 3. 제안 문턱 (적용하지 않음)", ""] + [f"- {CAL_PRED_KO.get(k, k)}: {p['threshold']:.4g} {'이하' if p['accurate_if'] == 'below' else '이상'}이면 정확으로 본다." for k, p in res["proposal"].items()]
        if res.get("held_out_check"):
            L += ["", "다른 렌즈(0.5배)에서의 확인(문턱을 정하는 데 쓰지 않음):"] + [f"- {CAL_PRED_KO.get(k, k)}: 균형 정확도 {long_fmt(b['balanced_accuracy'], 2)} (참양성 {b['tp']}, 거짓음성 {b['fn']}, 참음성 {b['tn']}, 거짓양성 {b['fp']})" for k, b in res["held_out_check"].items()]
    L += ["", "## 4. 영상별 오차와 진단값", "", "| 영상 | 렌즈 | f | 오차 | 정확 | 제외 | 부트스트랩 로그 SD | 일관성 범위 | 날카로움 | 등급 v2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rt = hf.gen_rating(r.get("tilt_duration_s"), r.get("theta_max"), r.get("boot_log_std"), cfg["rating_v2"])
        L.append(f"| {r['video']} | {r['lens']} | {r['f']:.1f} | {pct(r['signed_error'])} | {'예' if r['accurate'] else '아니오'} | {'예' if r['excluded'] else '-'} | {long_fmt(r['boot_log_std'], 4)} | {long_fmt(r['consistency_range'], 3)} | {long_fmt(r['sharpness'], 4)} | {rt} |")
    L += ["", "## 한계", "", "- 보정 영상은 1배 한 폰·한 세션 계열이라 문턱은 이 조건에 대한 것이다. 보정 영상 수가 작아 문턱 선택이 불안정하다(leave-one-out 문턱을 results.json에 적었다).",
          "- 기준 f는 사용자 제공 값이며 원본 보정 파일은 서버에 없다. 0.5배 영상은 문턱 선택에 쓰지 않았다.", "- 이 절차는 진단값이 오차를 \"예측하는지\"만 본다. 오차를 줄이는 방법은 아니다.", ""]
    path.write_text("\n".join(L) + "\n")


def cal_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    a = r["analysis"]
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "31_verdict_calibration", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": CAL_NOTE,
           "rules": {k: cfg[k] for k in ("accurate_tol", "calibration_lens", "rho_min", "loo_ba_min", "bootstrap_n", "bootstrap_seed")},
           "n_calibration_videos": a["n_calibration"], "decision": a["decision"], "useful_predictors": a["useful_predictors"],
           "proposal": {k: {"accurate_if": p["accurate_if"], "threshold": rd(p["threshold"], 6)} for k, p in a["proposal"].items()}}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# pr  (experiment 34: printed hand, is the f-versus-distance slope still there when the hand cannot change shape?; new section)
#
#     python HandCalib/run.py printed-hand --config HandCalib/configs/34_printed_hand.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py printed-hand --config ... results-yaml --run runs/34_printed_hand/<UTC>
# The printed-hand videos are detected anew (mediapipe, CPU); the real-hand videos of 27 (compare_videos) are read from their stored detections and only used for the comparison.
# ====================================================================================================

PR_NOTE = ("프린트한 손(평면, 모양이 변하지 않음)을 같은 카메라·같은 거리 50 / 60 / 75 cm에서 찍은 영상으로, 손 모양 변화가 없어도 추정 f가 거리에 반비례하는지 본 실험이다. "
           "후보 설정(mediapipe_hands + 손목 제외 + fixed_template)과 판정 규칙은 실행 전에 고정했고 바꾸지 않았다. 기준 f와의 비교는 summary.md에만 있다.")


def pr_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/34_printed_hand.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the yaml entries (printed and real)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(pr_results_yaml(hf.resolve(a.run), cfg))
        return None
    printed = [{**e, "kind": "printed"} for e in cfg["videos"]]
    real = [{**e, "kind": "real"} for e in cfg.get("compare_videos") or []]
    if a.videos:
        keep = set(a.videos.split(","))
        printed, real = [e for e in printed if e["file"] in keep], [e for e in real if e["file"] in keep]
    entries = printed + real
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"34 프린트 손: 프린트 영상 {len(printed)}개, 비교용 실제 손 영상 {len(real)}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    base = {k: cfg[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "session") if k in cfg}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            by_file = dict(zip([e["file"] for e in entries], preps))
            npz = {}
            if printed:
                found = hf.fresp_detect_many([((e["file"], "orig"), by_file[e["file"]]["frames"]) for e in printed], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
                npz.update({e["file"]: found[(e["file"], "orig")] for e in printed})
            for e in real:
                npz[e["file"]] = hf.jset_find_detection(by_file[e["file"]]["frames"], dset, cfg["detections"]["search_dirs"], None, cfg["candidate"]["detector"])["npz"]
            jobs = [(e, by_file[e["file"]], npz[e["file"]], cfg["pipeline"], {**base, **({"overlay_dir": str(run_root / "overlays")} if e["kind"] == "printed" else {})}) for e in entries]
            results = list(ex.map(hf.pr_video, jobs))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    v2 = cfg["rating_v2"]
    for r in results:
        tag = f"{r['kind']} {r['distance_cm']:g} cm"
        if r["excluded"]:
            log(f"  {r['video']} ({tag}): 제외 — {'; '.join(r['excluded'])} (손 검출 실패 {100 * (1 - r['detect_rate']):.1f}%)")
        elif "ft" in r:
            r["rating"] = hf.gen_rating(r["motion"]["tilt_duration_s"], r["motion"]["theta_max_deg"], r["ft"]["boot_log_std"], v2)
            log(f"  {r['video']} ({tag}): f {format(r['ft']['f'], '.1f') + ' px' if not r['ft']['at_grid_edge'] else '값 없음'} (keep {r['keep']['f']:.1f}), 로그 SD {long_fmt(r['ft']['boot_log_std'], 4)}, "
                f"θ2d 최댓값 {r['motion']['theta_max_deg']:.0f}°, 두 손 이상 프레임 {100 * r['multi_hand']['share']:.1f}%, RMS 정면/기울기 {long_fmt(r['phase_rms_pct']['front'], 3)}/{long_fmt(r['phase_rms_pct']['tilt'], 3)}%")
        else:
            log(f"  {r['video']}: 분석 실패 ({r.get('failed')})")
    ok = [r for r in results if "ft" in r]
    grid = np.array(ok[0]["curve"]["grid"]) if ok else np.array([])
    if any(not np.allclose(np.array(r["curve"]["grid"]), grid) for r in ok):
        raise SystemExit("비용 곡선의 f 격자가 영상마다 다릅니다")
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    ana = {}
    for kind in ("printed", "real"):
        rs = [r for r in ok if r["kind"] == kind]
        if not rs:
            continue
        ds = sorted({r["distance_cm"] for r in rs})
        cur = {d: [r["curve"]["cost"] for r in rs if r["distance_cm"] == d] for d in ds}
        curk = {d: [r["keep"]["curve"] for r in rs if r["distance_cm"] == d] for d in ds}
        sl, slk = hf.pr_slope(cur, grid, bn, bs), hf.pr_slope(curk, grid, bn, bs)
        per = {d: {**hf.ext_pooled(cur[d], grid, {"bootstrap_n": bn, "bootstrap_seed": bs}), "n_total": sum(1 for r in results if r["kind"] == kind and r["distance_cm"] == d)} for d in ds}
        ana[kind] = {"slope": sl, "slope_keep": slk, "per_distance": per}
    if "printed" in ana:
        nv = {d: p["n_videos"] for d, p in ana["printed"]["per_distance"].items()}
        for d in ref:
            nv.setdefault(d, 0)
        verdict = hf.pr_verdict(ana["printed"]["slope"], ref, cfg["verdict"], nv)
    else:
        verdict = {"verdict": "판단 불가", "reason": "프린트 영상이 없음"}
    verdict["ref_f_by_distance"] = ref
    ana["checkerboard_slope"] = hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)])
    ana["verdict"] = verdict
    log(f"  판정: {verdict['verdict']}" + (f" (기울기 {verdict['slope']:+.3f}, 구간 {verdict['interval_95'][0]:+.3f} ~ {verdict['interval_95'][1]:+.3f})" if verdict.get("slope") is not None and verdict.get("interval_95") else f" ({verdict.get('reason')})"))
    pr_figures(run_root / "figures", ana, ref, cfg)
    pr_report(run_root / "report.md", results, ana, cfg)
    pr_summary(run_root / "summary.md", results, ana, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": [{k: v for k, v in r.items() if k != "traj"} for r in results], "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_31": cfg["code_31"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def pr_figures(out: Path, ana: dict, ref: dict, cfg: dict):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ds = sorted(ref)
    series = [("체커보드 (같은 날 거리 실험)", ds, [ref[d] for d in ds], "#555", "s")]
    for kind, label, col in (("real", "실제 손 (실험 27)", "#c47a00"), ("printed", "프린트 손", "#2a6fbb")):
        if kind in ana:
            pf = ana[kind]["slope"]["pooled_f"]
            series.append((label, sorted(pf), [pf[d] for d in sorted(pf)], col, "o"))
    for label, xs_, fs, col, mk in series:
        ax.plot(xs_, fs, mk + "-", color=col, label=label)
    anchor = (60.0, ana["printed"]["slope"]["pooled_f"][60.0]) if "printed" in ana and 60.0 in ana["printed"]["slope"]["pooled_f"] else (60.0, ref.get(60.0, 1650.0))
    xs = np.array([min(ds) * 0.9, max(ds) * 1.1])
    ax.plot(xs, anchor[1] * (xs / anchor[0]) ** 0.0, ":", color="#999", label="기울기 0")
    ax.plot(xs, anchor[1] * (xs / anchor[0]) ** -1.0, "--", color="#999", label="기울기 -1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손(판)-카메라 거리 (cm)")
    ax.set_ylabel("합동 f (px)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)


def pr_report(path: Path, results: list, ana: dict, cfg: dict):
    ok = [r for r in results if "ft" in r]
    L = ["# 실험 34: 프린트 손", "", PR_NOTE, "",
         f"후보: {cfg['candidate']['detector']} + wrist_mode {cfg['candidate']['wrist_mode']} + {cfg['candidate']['estimator']} (실험 27과 같음). 프린트 손 영상과 비교용 실제 손 영상(실험 27의 오른손 G6, 저장된 검출) 모두 같은 처리이다.", "",
         "## 1. 영상 목록, 검출 상태, 제외", "", f"제외 규칙(27과 같음): 유효한 정면 창 없음, 가장자리 접촉 프레임 > {100 * cfg['exclusion']['edge_rate_max']:.0f}%, 손 검출 실패 프레임 > {100 * cfg['exclusion']['detect_fail_max']:.0f}%.", "",
         "| 종류 | 영상 | 거리(cm) | 길이(초) | 검출 실패율 | 가장자리 접촉 | 두 손 이상 프레임 | 포함 | 이유 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {'프린트' if r['kind'] == 'printed' else '실제 손'} | {r['video']} | {r['distance_cm']:g} | {r['duration_s']:.1f} | {100 * (1 - r['detect_rate']):.1f}% | {100 * r['edge_rate']:.1f}% | "
                 f"{100 * r['multi_hand']['share']:.1f}% | {'제외' if r['excluded'] else ('포함' if 'ft' in r else '분석 실패')} | {'; '.join(r['excluded']) or r.get('failed', '-')} |")
    L += ["", "## 2. 영상별 촬영 지표와 진단값", "",
          "| 영상 | 거리(cm) | 기울기 구간(초) | θ2d 중앙값 / 최댓값(°) | 손 박스 긴 변(px) | 날카로움 | 부트스트랩 로그 SD | 재투영 RMS 정면 / 기울기 (박스 대비 %) | 마디 길이 비 변화 중앙값(%) | 손가락 벌어짐 변화 중앙값(°) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        m, d, rm = r["motion"], r["dynamics"], r["phase_rms_pct"]
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {m['tilt_duration_s']:.1f} | {long_fmt(m['theta_median_deg'], 1)} / {long_fmt(m['theta_max_deg'], 1)} | {r['box_long_median_px']:.0f} | {r['ft']['sharpness']:.4f} | {long_fmt(r['ft']['boot_log_std'], 4)} | "
                 f"{long_fmt(rm['front'], 3)} / {long_fmt(rm['tilt'], 3)} | {long_fmt(d['bone_ratio_pct']['median'], 2)} | {long_fmt(d['spread_change_deg']['median'], 2)} |")
    L += ["", "## 3. 초점거리와 거리별 합동 (값만)", "", "| 영상 | 거리(cm) | f 제외(후보) (px) | f keep (px) | f 판정(추정기) |", "|---|---|---|---|---|"]
    for r in ok:
        f = r["ft"]
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {'값 없음 (' + str(f['side']) + ')' if f['at_grid_edge'] else format(f['f'], '.1f')} | {'값 없음' if r['keep']['at_grid_edge'] else format(r['keep']['f'], '.1f')} | {f['verdict']} |")
    for kind, name in (("printed", "프린트 손"), ("real", "실제 손")):
        if kind not in ana:
            continue
        L += ["", f"{name}의 거리별 합동 f (영상별 fixed_template 비용 곡선의 합, 영상 단위 부트스트랩 95% 구간):", "", "| 거리(cm) | 영상 | 합동 f (px) | 95% 구간 (px) |", "|---|---|---|---|"]
        for d, p in ana[kind]["per_distance"].items():
            L.append(f"| {float(d):g} | {p['n_videos']} / {p['n_total']} | {format(p['pooled_f'], '.1f') if 'pooled_f' in p else '-'} | {'-' if 'interval_95' not in p else format(p['interval_95'][0], '.0f') + '–' + format(p['interval_95'][1], '.0f')} |")
        sl = ana[kind]["slope"]
        L.append("")
        L.append(f"- log f와 log 거리의 기울기: {long_fmt(sl['slope'], 3)}" + ("" if sl["interval_95"] is None else f" (영상 단위 부트스트랩 95% 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f}, 사용 {sl['n_boot_used']}회)") + f"; keep 모드 {long_fmt(ana[kind]['slope_keep']['slope'], 3)}")
    L += ["", "## 4. 한계", "", "- 이 보고서는 값만 적는다. 기준 f와의 비교, 판정, 실험 27·체커보드와의 비교는 summary.md에 있다.",
          "- 프린트 손 영상은 거리마다 2개씩이라 부트스트랩 구간이 거칠다(거리 안에서 영상 2개를 복원 추출하면 조합이 3가지뿐이다).", ""]
    path.write_text("\n".join(L) + "\n")


def pr_summary(path: Path, results: list, ana: dict, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vd, vcfg = ana["verdict"], cfg["verdict"]
    ref = vd["ref_f_by_distance"]
    line = f"- **판정: {vd['verdict']}**"
    if vd.get("slope") is not None and vd.get("interval_95"):
        line += f" — 기울기 {vd['slope']:+.3f}, 95% 구간 {vd['interval_95'][0]:+.3f} ~ {vd['interval_95'][1]:+.3f} (반비례 없음: 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 합동 f가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음: 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래)"
        if vd.get("within_f_tol"):
            line += "; 합동 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in vd["within_f_tol"].items())
    elif vd.get("reason"):
        line += f" — {vd['reason']}"
    L = ["# 실험 34 summary", "", f"**{PR_NOTE}**", "", "## 판정", "", line, ""]
    pr = ana.get("printed")
    L += ["## 1. 프린트 손: 거리별 합동 f와 기준 f", "", "| 거리(cm) | 영상 | 합동 f (px) | 95% 구간 (px) | 같은 거리 기준 f | 기준 f 대비 | 1680 대비 |", "|---|---|---|---|---|---|---|"]
    if pr:
        for d, p in pr["per_distance"].items():
            pf = p.get("pooled_f")
            L.append(f"| {float(d):g} | {p['n_videos']} / {p['n_total']} | {format(pf, '.1f') if pf else '-'} | {'-' if 'interval_95' not in p else format(p['interval_95'][0], '.0f') + '–' + format(p['interval_95'][1], '.0f')} | {ref[float(d)]:.1f} | "
                     f"{pct(None if not pf else pf / ref[float(d)] - 1.0)} | {pct(None if not pf else pf / 1680.0 - 1.0)} |")
        sl = pr["slope"]
        L += ["", f"- 기울기(log f 대 log 거리): {long_fmt(sl['slope'], 3)}" + ("" if sl["interval_95"] is None else f", 95% 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f} (영상 단위 부트스트랩 {sl['n_boot_used']}회, 시드 {cfg['bootstrap']['seed']})"),
              f"- keep 모드 기울기: {long_fmt(pr['slope_keep']['slope'], 3)}" + ("" if pr["slope_keep"]["interval_95"] is None else f" (구간 {pr['slope_keep']['interval_95'][0]:+.3f} ~ {pr['slope_keep']['interval_95'][1]:+.3f})")]
    L += ["", "## 2. 프린트 손, 실제 손(실험 27), 체커보드의 비교 (기록용)", "", "| 거리(cm) | 프린트 손 합동 f | 실제 손 합동 f (27) | 체커보드 f |", "|---|---|---|---|"]
    for d in sorted(ref):
        a_ = pr["slope"]["pooled_f"].get(d) if pr else None
        b_ = ana["real"]["slope"]["pooled_f"].get(d) if "real" in ana else None
        L.append(f"| {d:g} | {format(a_, '.1f') if a_ else '-'} | {format(b_, '.1f') if b_ else '-'} | {ref[d]:.1f} |")
    L += ["", "| 대상 | 기울기 (제외 모드) | 95% 구간 | keep 모드 기울기 |", "|---|---|---|---|"]
    for kind, name in (("printed", "프린트 손"), ("real", "실제 손 (실험 27 영상)")):
        if kind in ana:
            sl = ana[kind]["slope"]
            L.append(f"| {name} | {long_fmt(sl['slope'], 3)} | {'-' if sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} | {long_fmt(ana[kind]['slope_keep']['slope'], 3)} |")
    L.append(f"| 체커보드 | {long_fmt(ana['checkerboard_slope'], 3)} | - | - |")
    L += ["", "![거리 대 합동 f](figures/f_vs_distance_loglog.png)", "", "## 3. 재투영 RMS와 손 모양 변화: 프린트 손 대 실제 손 (같은 거리, 영상 중앙값, 기록용)", "",
          "| 거리(cm) | 종류 | 영상 | 재투영 RMS 정면 (%) | 재투영 RMS 기울기 (%) | 마디 길이 비 변화 (%) | 손가락 벌어짐 변화 (°) | θ2d 최댓값 (°) | 손 박스 긴 변 (px) |", "|---|---|---|---|---|---|---|---|---|"]
    ok = [r for r in results if "ft" in r]
    med = lambda vals: float(np.median([v for v in vals if v is not None])) if any(v is not None for v in vals) else None      # noqa: E731
    for d in sorted(ref):
        for kind, name in (("printed", "프린트"), ("real", "실제 손")):
            rs = [r for r in ok if r["kind"] == kind and r["distance_cm"] == d]
            if not rs:
                continue
            L.append(f"| {d:g} | {name} | {len(rs)} | {long_fmt(med([r['phase_rms_pct']['front'] for r in rs]), 3)} | {long_fmt(med([r['phase_rms_pct']['tilt'] for r in rs]), 3)} | {long_fmt(med([r['dynamics']['bone_ratio_pct']['median'] for r in rs]), 2)} | "
                     f"{long_fmt(med([r['dynamics']['spread_change_deg']['median'] for r in rs]), 2)} | {long_fmt(med([r['motion']['theta_max_deg'] for r in rs]), 1)} | {long_fmt(med([r['box_long_median_px'] for r in rs]), 0)} |")
    for kind, name in (("printed", "프린트 손"), ("real", "실제 손")):
        rs = [r for r in ok if r["kind"] == kind]
        if rs:
            L.append(f"| 전체 | {name} | {len(rs)} | {long_fmt(med([r['phase_rms_pct']['front'] for r in rs]), 3)} | {long_fmt(med([r['phase_rms_pct']['tilt'] for r in rs]), 3)} | {long_fmt(med([r['dynamics']['bone_ratio_pct']['median'] for r in rs]), 2)} | "
                     f"{long_fmt(med([r['dynamics']['spread_change_deg']['median'] for r in rs]), 2)} | {long_fmt(med([r['motion']['theta_max_deg'] for r in rs]), 1)} | {long_fmt(med([r['box_long_median_px'] for r in rs]), 0)} |")
    L += ["", "프린트 손이 실제로 더 평평하고 단단하게 측정되면 재투영 RMS와 마디 길이 비·손가락 벌어짐 변화가 실제 손보다 작아야 한다.", "",
          "## 4. 영상별 f와 오차", "", "| 영상 | 종류 | 거리(cm) | f (px) | 같은 거리 기준 f 대비 | 1680 대비 | 로그 SD | 날카로움 | 두 손 이상 프레임 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "ft" not in r:
            L.append(f"| {r['video']} | {r['kind']} | {r['distance_cm']:g} | 제외: {'; '.join(r['excluded']) or r.get('failed', '-')} | - | - | - | - | {100 * r['multi_hand']['share']:.1f}% |")
            continue
        f = r["ft"]
        L.append(f"| {r['video']} | {'프린트' if r['kind'] == 'printed' else '실제 손'} | {r['distance_cm']:g} | {'값 없음' if f['at_grid_edge'] else format(f['f'], '.1f')} | "
                 f"{pct(None if f['at_grid_edge'] else f['f'] / ref[r['distance_cm']] - 1.0)} | {pct(None if f['at_grid_edge'] else f['f'] / 1680.0 - 1.0)} | {long_fmt(f['boot_log_std'], 4)} | {f['sharpness']:.4f} | {100 * r['multi_hand']['share']:.1f}% |")
    L += ["", "## 한계", "", "- 프린트 손 영상은 거리마다 2개씩이라 기울기의 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 기준 f는 같은 날 체커보드 거리 실험의 값(사용자 제공)이며 보정 파일은 서버에 없다.",
          "- 프린트 손이 \"더 평평하고 단단하다\"는 것은 위 3절의 지표로 확인한 범위에서만 말할 수 있다. 판에 인쇄한 사진은 손의 3차원 구조가 없는 대신, 실제 손 사진의 검출기 반응과 판 표면의 반사 같은 새 요인이 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def pr_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "34_printed_hand", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": PR_NOTE, "candidate": cfg["candidate"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        e = {"kind": v["kind"], "distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "detect_fail_rate": rd(1 - v["detect_rate"]), "edge_rate": rd(v["edge_rate"]), "excluded": v["excluded"] or None,
             "multi_hand_share": rd(v["multi_hand"]["share"])}
        if "ft" in v:
            f, mo = v["ft"], v["motion"]
            e.update({"f_px": None if f["at_grid_edge"] else round(f["f"], 1), "status": "value" if not f["at_grid_edge"] else f"값 없음 ({f['side']})", "f_keep_px": None if v["keep"]["at_grid_edge"] else round(v["keep"]["f"], 1),
                      "sharpness": rd(f["sharpness"], 5), "boot_log_std": rd(f["boot_log_std"], 5), "tilt_duration_s": rd(mo["tilt_duration_s"], 2), "theta_median_deg": rd(mo["theta_median_deg"], 1),
                      "theta_max_deg": rd(mo["theta_max_deg"], 1), "box_long_median_px": rd(v["box_long_median_px"], 1), "reproj_rms_front_pct": rd(v["phase_rms_pct"]["front"], 4),
                      "reproj_rms_tilt_pct": rd(v["phase_rms_pct"]["tilt"], 4), "bone_ratio_pct_median": rd(v["dynamics"]["bone_ratio_pct"]["median"], 3),
                      "spread_change_deg_median": rd(v["dynamics"]["spread_change_deg"]["median"], 3), "rating_v2": v.get("rating")})
        out["videos"][v["video"]] = e
    a = r["analysis"]
    out["by_kind"] = {k: {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in a[k]["slope"]["pooled_f"].items()}, "slope": rd(a[k]["slope"]["slope"]), "slope_interval_95": None if a[k]["slope"]["interval_95"] is None else [round(x, 4) for x in a[k]["slope"]["interval_95"]],
                           "slope_keep": rd(a[k]["slope_keep"]["slope"])} for k in ("printed", "real") if k in a}
    vd = a["verdict"]
    out["verdict"] = {"verdict": vd["verdict"], "reason": vd.get("reason"), "slope": rd(vd.get("slope")), "interval_95": None if not vd.get("interval_95") else [round(x, 4) for x in vd["interval_95"]]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# mk  (experiment 35: hand markers, does the slope survive when the points are measured by image processing?; new section)
#
#     python HandCalib/run.py hand-markers --config HandCalib/configs/35b_hand_markers_subset.yaml [--videos A.MOV,B.MOV] [--out DIR] precheck
#     python HandCalib/run.py hand-markers --config ... [--videos A.MOV,B.MOV] [--out DIR] pairing          (pairing-quality check, no f)
#     python HandCalib/run.py hand-markers --config ... [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py hand-markers --config ... results-yaml --run runs/35_hand_markers/<UTC>
# precheck: marker statistics, per-marker detection rates and the reference-frame picture of every video, no analysis; the official run needs every distance to have a video that passes the rule.
# The MediaPipe hand (box and naming only) is detected anew, CPU. markers.used_ids: the markers (= MediaPipe joint numbers) that enter the estimate; without it all markers are used (35's first config).
# ====================================================================================================

MK_NOTE = ("실제 손의 손바닥 면에 붙인 표시 점을 학습 모델 대신 영상 처리로 측정해 같은 fixed_template 계산에 넣으면, 손 모양 변화와 비평면성이 그대로인 실제 손에서도 거리 반비례가 사라지는지 본 실험이다. "
           "쓰는 점, 표시 검출 설정, 판정 규칙은 실행 전에 고정했고 바꾸지 않았다. 기준 f와의 비교는 summary.md에만 있다.")
MK_METHODS = {"markers": "표시 점", "mediapipe8": "MediaPipe 같은 관절 (주 비교)", "mediapipe16": "MediaPipe 손목 제외 16점 (기록용)", "markers_nearfree": "표시 점, 2% 근접 프레임 제외 (참고용, 판정에 쓰지 않음)"}


def mk_interpret(marker_verdict: str, mp_verdict: str) -> str:
    """Interpretation fixed before the run: markers 반비례 없음 + MediaPipe 반비례 있음 -> 원인은 검출 모델; both 반비례 있음 -> 원인은 손 자체; otherwise 해석 보류."""
    if marker_verdict == "반비례 없음" and mp_verdict == "반비례 있음":
        return "원인은 검출 모델"
    if marker_verdict == "반비례 있음" and mp_verdict == "반비례 있음":
        return "원인은 손 자체"
    return "해석 보류"


def mk_norm(r: dict, key: str):
    """One method's result of a video in a common shape (f, at_grid_edge, sharpness, boot_log_std, curve cost, phase RMS, theta, used frames) or None when it was excluded / failed."""
    if key == "markers":
        x = r["marker"]
        return x if "f" in x else None
    if key == "mediapipe8":
        x = r.get("mediapipe8") or {}
        return x if "f" in x else None
    if key == "markers_nearfree":
        x = r.get("marker_nearfree") or {}
        return x if "f" in x else None
    x = r["mediapipe"]
    if "ft" not in x:
        return None
    return {"f": x["ft"]["f"], "at_grid_edge": x["ft"]["at_grid_edge"], "side": x["ft"]["side"], "sharpness": x["ft"]["sharpness"], "boot_log_std": x["ft"]["boot_log_std"], "curve": x["curve"], "phase_rms_pct": x["phase_rms_pct"],
            "theta_median": x["motion"]["theta_median_deg"], "theta_max": x["motion"]["theta_max_deg"], "n_used_frames": x["motion"]["n_tilt_frames"]}


def mk_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/35b_hand_markers_subset.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("precheck")
    sub.add_parser("pairing")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the yaml entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(mk_results_yaml(hf.resolve(a.run), cfg))
        return None
    if a.cmd == "pairing":
        return mk_pairing_main(a, cfg)
    pre = a.cmd == "precheck"
    entries = list(cfg["videos"])
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / (utc_stamp() + ("_precheck" if pre else ""))
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    used = cfg["markers"].get("used_ids")
    log(f"35 손 표시 점 {'사전 확인' if pre else '공식 실행'}: 영상 {len(entries)}개, 표시 {cfg['markers']['color']} {cfg['markers']['count']}개" + (f", 추정에 쓰는 점 {used}" if used else "") + f" → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    base = {k: cfg[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "session", "markers", "marker_exclusion", "precheck") if k in cfg}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    infos = [hf.probe(Path(cfg["input_dir"]).expanduser() / e["file"]) for e in entries]
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = hf.fresp_detect_many([((e["file"], "orig"), p["frames"]) for e, p in zip(entries, preps)], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            cj = {**base, **({"out_dir": str(run_root / "figures")} if pre else {"overlay_dir": str(run_root / "overlays")})}
            results = list(ex.map(hf.mk_precheck_video if pre else hf.mk_video, [(e, p, found[(e["file"], "orig")], cfg["pipeline"], cj) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    if pre:
        ok_by_d = {d: [r["video"] for r in results if r["distance_cm"] == d and r.get("passes")] for d in sorted(ref)}
        proceed = all(len(v) >= 1 for v in ok_by_d.values())
        mk_precheck_report(run_root / "precheck.md", results, infos, ok_by_d, proceed, cfg)
        (run_root / "precheck.json").write_text(json.dumps(hf.jsonable({"videos": results, "passing_by_distance": ok_by_d, "official_run_ok": proceed}), ensure_ascii=False, indent=1))
        for r in results:
            log(f"  {r['video']} ({r['distance_cm']:g} cm): " + ("기준 프레임 없음" if "tilt_valid_ratio" not in r else f"기울기 구간 유효 {100 * r['tilt_valid_ratio']:.1f}%, 0.5~1.5초 성공 {r['window_success']}/{r['window_frames']}") + f" → {'통과' if r.get('passes') else '탈락'}")
        log(f"  거리별 통과 영상: {ok_by_d} → 공식 실행 {'진행 가능' if proceed else '하지 않음 (보고하고 멈춤)'}")
        log(f"완료 {time.time() - t0:.0f} s → {run_root}")
        log.close()
        return run_root, results, proceed
    v2 = cfg["rating_v2"]
    for r in results:
        mkr = r["marker"]
        if "f" in mkr:
            mkr["rating"] = hf.gen_rating(mkr["tilt_duration_s"], mkr["theta_max"], mkr["boot_log_std"], v2)
        parts = []
        for key in MK_METHODS:
            x = mk_norm(r, key)
            parts.append(f"{MK_METHODS[key].split(' ')[0]}{'(8)' if key == 'mediapipe8' else ''} " + (f"f {x['f']:.1f} px" if x else "제외"))
        line = f"  {r['video']} ({r['distance_cm']:g} cm): " + " / ".join(parts)
        if mkr.get("excluded"):
            line += f" [표시 제외: {'; '.join(mkr['excluded'])}]"
        if r.get("diag"):
            line += f" / 원근 반영률 k {long_fmt(r['diag']['k']['k'], 2)}, 평면 예측과의 차이 중앙값 {long_fmt(r['diag']['gap_median_pct'], 2)}%"
        log(line)
    grid = None
    for r in results:
        for key in MK_METHODS:
            x = mk_norm(r, key)
            if x and grid is None:
                grid = np.array(x["curve"]["grid"])
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    ana = {}
    for key in MK_METHODS:
        inc = [(r["distance_cm"], mk_norm(r, key)["curve"]["cost"]) for r in results if mk_norm(r, key)]
        ds = sorted({d for d, _ in inc})
        if not ds or grid is None:
            ana[key] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
            continue
        cur = {d: [c for dd, c in inc if dd == d] for d in ds}
        sl = hf.pr_slope(cur, grid, bn, bs)
        per = {d: {**hf.ext_pooled(cur[d], grid, {"bootstrap_n": bn, "bootstrap_seed": bs}), "n_total": sum(1 for r in results if r["distance_cm"] == d)} for d in ds}
        nv = {d: len(cur.get(d, [])) for d in ref}
        ana[key] = {"slope": sl, "per_distance": per, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        log(f"  {key}: {ana[key]['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else ""))
    main = "mediapipe8" if used else "mediapipe16"
    ana["interpretation"] = mk_interpret(ana["markers"]["verdict"]["verdict"], ana[main]["verdict"]["verdict"])
    ana["main_comparison"] = main
    ana["checkerboard_slope"] = hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)])
    ana["diag"] = mk_diag_summary(results)
    log(f"  해석: {ana['interpretation']}")
    mk_figures(run_root / "figures", results, ana, ref, cfg)
    mk_report(run_root / "report.md", results, ana, cfg)
    mk_summary(run_root / "summary.md", results, ana, ref, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_34": cfg["code_34"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def mk_pairing_main(a, cfg: dict):
    """hand-markers pairing: the pairing-quality check of every video (statistics and the 10-panel pictures); computes no f, slope or error."""
    entries = list(cfg["videos"])
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / (utc_stamp() + "_pairing")
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"35 짝짓기 품질 확인: 영상 {len(entries)}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    base = {k: cfg[k] for k in ("candidate", "exclusion", "detector_settings", "guide", "session", "markers", "marker_exclusion", "precheck") if k in cfg}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = hf.fresp_detect_many([((e["file"], "orig"), p["frames"]) for e, p in zip(entries, preps)], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            cj = {**base, "out_dir": str(run_root / "figures")}
            results = list(ex.map(hf.mk_pairing_video, [(e, p, found[(e["file"], "orig")], cfg["pipeline"], cj) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    L = ["# 실험 35 짝짓기 품질 확인", "", "f, 기울기, 오차는 계산하지 않았다. 쓰인 8개 점이 예측 위치에서 떨어진 거리(px)와, 무시한 덩어리가 쓰인 점에서 손 박스의 2% 안에 있었던 프레임 비율을 본다. "
         "예측 위치 = 기준 프레임의 점을 마지막으로 통과한 프레임의 호모그래피로 옮긴 위치. 그림: figures/pairing_IMG_xxxx.png (프레임 10장, 시드 42; 초록 = 쓰인 8개 점(번호), 주황 x = 무시한 덩어리, 하늘색 원 = 예측 위치).", "",
         "## 쓰인 8개 점과 예측 위치의 거리 (px)", "", "| 영상 | 거리(cm) | 구분 | 프레임 | 점 위치 수 | 중앙값 (px) | 95번째 백분위수 (px) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if "failed" in r:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | - | - | - | - | {r['failed']} |")
            continue
        for key, name in (("with_ignored", "무시한 덩어리가 있는 프레임"), ("without_ignored", "무시한 덩어리가 없는 프레임")):
            x = r[key]
            L.append(f"| {r['video']} | {r['distance_cm']:g} | {name} | {x['frames']} | {x['n_points']} | {long_fmt(x['median_px'], 2)} | {long_fmt(x['p95_px'], 2)} |")
    L += ["", "## 무시한 덩어리가 쓰인 점에서 손 박스의 2% 안에 있었던 프레임 (잘못 짝지어질 위험)", "", "| 영상 | 거리(cm) | 유지된 프레임 | 무시한 덩어리가 있는 프레임 | 그중 2% 안에 있었던 프레임 | 비율 | 그림 칸 수 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if "failed" in r:
            continue
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {r['kept_frames']} | {r['with_ignored']['frames']} | {r['near_2pct_frames']} | {'-' if r['near_2pct_share'] is None else format(100 * r['near_2pct_share'], '.1f') + '%'} | {r.get('panels', 0)} |")
    L += [""]
    (run_root / "pairing.md").write_text("\n".join(L) + "\n")
    (run_root / "pairing.json").write_text(json.dumps(hf.jsonable({"videos": results}), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results


def mk_precheck_report(path: Path, results: list, infos: list, ok_by_d: dict, proceed: bool, cfg: dict):
    sub = cfg["markers"].get("used_ids") is not None
    pc = cfg["precheck"]
    rule = (f"통과 규칙: 기울기 구간(3초 이후)에서 사용 표시 {len(cfg['markers']['used_ids'])}개 중 {100 * float(cfg['markers']['min_markers_ratio']):.0f}% 이상이 잡힌 프레임의 비율 ≥ {100 * float(pc['min_tilt_valid_ratio']):.0f}%이고 정면 창에 사용 표시가 모두 보이는 기준 프레임이 있을 것; 거리마다 통과한 영상이 1개 이상이어야 공식 실행."
            if sub else f"모두 잡힌 비율이 {100 * float(pc['min_all_found_ratio']):.0f}% 미만인 영상이 있으면 공식 실행을 하지 않는다.")
    L = ["# 실험 35 사전 확인", "", f"표시: {cfg['markers']['color']}, {cfg['markers']['count']}개" + (f", 추정에 쓰는 점 {cfg['markers']['used_ids']}" if sub else "") + f". {rule}", "",
         "| 영상 | 거리(cm) | 길이(초) | 저장 해상도 | 회전 | 처리 후 해상도 | 손 검출 프레임 | 프레임당 검출 표시 수(중앙값) | 기울기 구간 유효 프레임 | 0.5~1.5초 성공 프레임 | 기준 프레임 시각(초) | 통과 | 버려진 이유 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, i in zip(results, infos):
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {i['duration_s']:.1f} | {i['width']}x{i['height']} | {i['rotation_deg']}° | {r['size'][0]}x{r['size'][1]} | {r['hand_frames']} / {r['n_frames']} | {r['n_detected_median']:.0f} | "
                 f"{'-' if 'tilt_valid_ratio' not in r else format(100 * r['tilt_valid_ratio'], '.1f') + '%'} | {'-' if 'window_success' not in r else str(r['window_success']) + ' / ' + str(r['window_frames'])} | {long_fmt(r.get('ref_time_s'), 1)} | "
                 f"{'통과' if r.get('passes') else '탈락'} | {r.get('failed') or ', '.join(f'{k} {v}' for k, v in (r.get('reasons') or {}).items()) or '-'} |")
    L += ["", f"거리별 통과 영상: " + ", ".join(f"{d:g} cm {v or '없음'}" for d, v in ok_by_d.items()) + f" → 공식 실행 {'진행 가능' if proceed else '하지 않음'}.", "",
          "## 버려진 이유별 프레임 수", "", "| 영상 | 손 검출 프레임 | 유지된 프레임 | 손 박스 없음 | 표시가 너무 적음 | 짝 찾기 실패 | 호모그래피 검사 탈락 | 표시가 너무 많음 | 무시한 덩어리 수 중앙값 | 무시한 덩어리가 있는 프레임 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rs = r.get("reasons") or {}
        kept = None if "valid_ratio" not in r else int(round(r["valid_ratio"] * r["hand_frames"]))
        L.append(f"| {r['video']} | {r['hand_frames']} | {'-' if kept is None else kept} | {rs.get('no_hand_box', 0)} | {rs.get('too_few_markers', 0)} | {rs.get('match_failed', 0)} | {rs.get('homography_failed', 0)} | {rs.get('too_many_markers', 0)} | "
                 f"{long_fmt(r.get('ignored_median'), 1)} | {'-' if r.get('ignored_share') is None else format(100 * r['ignored_share'], '.0f') + '%'} |")
    L += ["", "## 표시 번호별 검출률 (기울기 구간, 손이 잡힌 프레임 기준, 관절 번호 = 표시 이름)", "",
          "표시를 추적한 비율이다(최소 표시 비율을 0.3으로 낮춘 추적, 호모그래피 검사는 켬). * = 추정에 쓰는 점.", ""]
    used = set(int(x) for x in cfg["markers"].get("used_ids") or [])
    vids = [r for r in results if r.get("per_marker_rate")]
    if vids:
        L += ["| 표시 번호 | " + " | ".join(r["video"][4:8] for r in vids) + " |", "|---|" + "---|" * len(vids)]
        for j in range(21):
            L.append(f"| {j}{'*' if j in used else ''} | " + " | ".join(("-" if j not in r["per_marker_rate"] else f"{100 * r['per_marker_rate'][j]:.0f}%") for r in vids) + " |")
    L += ["", "정면 기준 프레임 그림: figures/reference_IMG_xxxx.png (표시 위치와 이름이 붙은 MediaPipe 관절 번호).", ""]
    path.write_text("\n".join(L) + "\n")


def mk_diag_summary(results: list) -> dict:
    """Pooled diagnostics of the planar-hand check: perspective reflection k (all frames and per distance) and the planar gap in tilt bins and per distance."""
    out = {"k": {}, "gap_by_theta": {}, "gap_by_distance": {}}
    rows = [(r["distance_cm"], fr) for r in results if r.get("diag") for fr in r["diag"]["frames"]]
    if not rows:
        return out
    def kof(sel):
        a_ = np.array([fr["p_mp"] for _, fr in sel]).reshape(-1, 2)
        b_ = np.array([fr["p_mk"] for _, fr in sel]).reshape(-1, 2)
        return hf.mk_perspective_k(a_, b_)
    out["k"]["all"] = kof(rows)
    for d in sorted({d for d, _ in rows}):
        out["k"][f"{d:g}"] = kof([x for x in rows if x[0] == d])
    for lo, hi in ((0, 20), (20, 40), (40, 90)):
        g = [fr["gap_pct"] for _, fr in rows if fr["theta2d_deg"] is not None and lo <= fr["theta2d_deg"] < hi and np.isfinite(fr["gap_pct"])]
        out["gap_by_theta"][f"{lo}-{hi}"] = {"n": len(g), "median_pct": float(np.median(g)) if g else None}
    for d in sorted({d for d, _ in rows}):
        g = [fr["gap_pct"] for dd, fr in rows if dd == d and np.isfinite(fr["gap_pct"])]
        out["gap_by_distance"][f"{d:g}"] = {"n": len(g), "median_pct": float(np.median(g)) if g else None}
    return out


def mk_figures(out: Path, results: list, ana: dict, ref: dict, cfg: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="체커보드 (같은 날 거리 실험)")
    anchor = None
    for key, col in (("mediapipe16", "#999"), ("mediapipe8", "#c47a00"), ("markers", "#2a6fbb")):
        sl = ana.get(key, {}).get("slope")
        if sl:
            pf = sl["pooled_f"]
            ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o-" if key != "mediapipe16" else "x--", color=col, label=MK_METHODS[key])
            if key == "markers" and 60.0 in pf:
                anchor = (60.0, pf[60.0])
    anchor = anchor or (60.0, ref.get(60.0, 1650.0))
    xs = np.array([min(ds) * 0.9, max(ds) * 1.1])
    ax.plot(xs, anchor[1] * (xs / anchor[0]) ** 0.0, ":", color="#bbb", label="기울기 0")
    ax.plot(xs, anchor[1] * (xs / anchor[0]) ** -1.0, "--", color="#bbb", label="기울기 -1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("합동 f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)
    rows = [(r["distance_cm"], fr) for r in results if r.get("diag") for fr in r["diag"]["frames"] if fr["theta2d_deg"] is not None and np.isfinite(fr["gap_pct"])]
    if rows:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        for d, col in zip(sorted({d for d, _ in rows}), ("#2a6fbb", "#c47a00", "#3a9a4a", "#999")):
            sel = [fr for dd, fr in rows if dd == d]
            ax.plot([f["theta2d_deg"] for f in sel], [f["gap_pct"] for f in sel], ".", color=col, alpha=0.4, label=f"{d:g} cm")
        ax.set_xlabel("θ2d (표시 점 기준, °)")
        ax.set_ylabel("MediaPipe 관절과 평면 예측의 차이 (손 박스 대비 %)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "planar_gap_vs_tilt.png", dpi=110)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for d, col in zip(sorted({d for d, _ in rows}), ("#2a6fbb", "#c47a00", "#3a9a4a", "#999")):
            sel = [fr for dd, fr in rows if dd == d]
            ax.plot([f["p_mk"][0] for f in sel] + [f["p_mk"][1] for f in sel], [f["p_mp"][0] for f in sel] + [f["p_mp"][1] for f in sel], ".", color=col, alpha=0.3, label=f"{d:g} cm")
        lim = max(1e-3, np.nanmax(np.abs([f["p_mk"] for _, f in rows])))
        ax.plot([-lim, lim], [-lim, lim], color="#999", lw=1, label="k = 1")
        ax.set_xlabel("원근 성분 (표시 점의 호모그래피 h31, h32)")
        ax.set_ylabel("원근 성분 (MediaPipe 관절)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "perspective_scatter.png", dpi=110)
        plt.close(fig)


def mk_report(path: Path, results: list, ana: dict, cfg: dict):
    used = cfg["markers"].get("used_ids")
    L = ["# 실험 35: 손 표시 점", "", MK_NOTE, "", f"표시: {cfg['markers']['color']} {cfg['markers']['count']}개" + (f", 추정에 쓰는 점 {used}" if used else "") + ". MediaPipe 분석은 같은 계산에 관절을 넣었다(주 비교: 같은 관절, 기록용: 실험 27의 후보 설정).", "",
         "## 1. 영상별 표시 통계", "", "| 영상 | 거리(cm) | 기울기 구간 유효 프레임 | 0.5~1.5초 성공 프레임 | 포함 | 이유 |", "|---|---|---|---|---|---|"]
    for r in results:
        st, mkr = r["marker_stats"], r["marker"]
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {long_fmt(None if st.get('tilt_valid_ratio') is None else 100 * st['tilt_valid_ratio'], 1)}% | {st['window_success']} / {st['window_frames']} | {'포함' if 'f' in mkr else '제외'} | {'; '.join(mkr.get('excluded') or []) or mkr.get('failed') or '-'} |")
    L += ["", "## 2. 영상별 값", "", "| 영상 | 거리(cm) | 방법 | 사용 프레임 | θ2d 중앙값 / 최댓값(°) | f (px) | 날카로움 | 부트스트랩 로그 SD | 재투영 RMS 정면 / 기울기 (박스 대비 %) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for key in MK_METHODS:
            x = mk_norm(r, key)
            if x:
                L.append(f"| {r['video']} | {r['distance_cm']:g} | {MK_METHODS[key]} | {x['n_used_frames']} | {long_fmt(x['theta_median'], 1)} / {long_fmt(x['theta_max'], 1)} | {'값 없음' if x['at_grid_edge'] else format(x['f'], '.1f')} | {x['sharpness']:.4f} | "
                         f"{long_fmt(x['boot_log_std'], 4)} | {long_fmt(x['phase_rms_pct']['front'], 3)} / {long_fmt(x['phase_rms_pct']['tilt'], 3)} |")
    L += ["", "## 3. 거리별 합동 f (값만)", ""]
    for key in MK_METHODS:
        a_ = ana[key]
        if "slope" not in a_:
            continue
        L += [f"{MK_METHODS[key]}:", "", "| 거리(cm) | 영상 | 합동 f (px) | 95% 구간 (px) |", "|---|---|---|---|"]
        for d, p in a_["per_distance"].items():
            L.append(f"| {float(d):g} | {p['n_videos']} / {p['n_total']} | {format(p['pooled_f'], '.1f') if 'pooled_f' in p else '-'} | {'-' if 'interval_95' not in p else format(p['interval_95'][0], '.0f') + '–' + format(p['interval_95'][1], '.0f')} |")
        sl = a_["slope"]
        L += ["", f"- 기울기 {long_fmt(sl['slope'], 3)}" + ("" if sl["interval_95"] is None else f", 95% 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f}"), ""]
    L += ["## 4. 한계", "", "- 이 보고서는 값만 적는다. 기준 f와의 비교, 판정, 해석은 summary.md에 있다. 손 모양 변화 지표(실험 23의 정의)는 손가락 관절 번호로 정의되어 표시 점에는 정의되지 않는다.", ""]
    path.write_text("\n".join(L) + "\n")


def mk_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    main = ana["main_comparison"]
    def vline(key, name):
        v = ana[key]["verdict"]
        s_ = f"- **{name} 판정: {v['verdict']}**"
        if v.get("slope") is not None and v.get("interval_95"):
            s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
            if v.get("within_f_tol"):
                s_ += "; 합동 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
        elif v.get("reason"):
            s_ += f" — {v['reason']}"
        return s_
    L = ["# 실험 35 summary", "", f"**{MK_NOTE}**", "", "## 판정", "", vline("markers", "표시 점"), vline(main, MK_METHODS[main]),
         f"- **해석: {ana['interpretation']}** (규칙: 표시 점 반비례 없음 + MediaPipe 반비례 있음 → 원인은 검출 모델; 둘 다 반비례 있음 → 원인은 손 자체; 그 외 해석 보류. MediaPipe는 {MK_METHODS[main]})",
         f"- 판정 규칙(실행 전 고정): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 합동 f가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 거리에 영상이 없으면 판단 불가.", ""]
    others = [k for k in MK_METHODS if k not in ("markers", main)]
    L += ["기록용 판정: " + ", ".join(f"{MK_METHODS[k]} {ana[k]['verdict']['verdict']}" for k in others), ""]
    L += ["## 1. 거리별 합동 f와 기준 f", "", "| 방법 | 거리(cm) | 영상 | 합동 f (px) | 95% 구간 (px) | 같은 거리 기준 f | 기준 f 대비 | 1680 대비 |", "|---|---|---|---|---|---|---|---|"]
    for key in MK_METHODS:
        for d, p in (ana[key].get("per_distance") or {}).items():
            pf = p.get("pooled_f")
            L.append(f"| {MK_METHODS[key]} | {float(d):g} | {p['n_videos']} / {p['n_total']} | {format(pf, '.1f') if pf else '-'} | {'-' if 'interval_95' not in p else format(p['interval_95'][0], '.0f') + '–' + format(p['interval_95'][1], '.0f')} | {ref[float(d)]:.1f} | "
                     f"{pct(None if not pf else pf / ref[float(d)] - 1.0)} | {pct(None if not pf else pf / 1680.0 - 1.0)} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|"]
    for key in MK_METHODS:
        sl = ana[key].get("slope")
        if sl:
            L.append(f"| {MK_METHODS[key]} | {long_fmt(sl['slope'], 3)} | {'-' if sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |")
    r27 = hf.resolve(cfg["compare"]["results_27"])
    pf27 = {}
    if r27.exists():
        g6 = json.loads(r27.read_text())["groups"]["G6"]["distances"]
        pf27 = {float(d): x["pooled_f"] for d, x in g6.items() if x.get("pooled_f")}
        L.append(f"| 실제 손, 표시 없음, 16점 (실험 27) | {long_fmt(hf.ext_loglog_slope(sorted(pf27), [pf27[d] for d in sorted(pf27)]), 3)} | - |")
    L.append(f"| 체커보드 | {long_fmt(ana['checkerboard_slope'], 3)} | - |")
    L += ["", "거리별 비교 (합동 f, px):", "", "| 거리(cm) | " + " | ".join(MK_METHODS[k] for k in MK_METHODS) + " | 실제 손 27 (표시 없음, 16점) | 체커보드 |", "|---|" + "---|" * (len(MK_METHODS) + 2)]
    for d in sorted(ref):
        cells = []
        for key in MK_METHODS:
            v = (ana[key].get("slope") or {}).get("pooled_f", {}).get(d)
            cells.append(format(v, ".1f") if v else "-")
        L.append(f"| {d:g} | " + " | ".join(cells) + f" | {format(pf27[d], '.1f') if d in pf27 else '-'} | {ref[d]:.1f} |")
    L += ["", "![거리 대 합동 f](figures/f_vs_distance_loglog.png)", "", "## 2. 표시 점과 MediaPipe의 재투영 RMS와 날카로움 (기록용)", "",
          "| 거리(cm) | 방법 | 영상 | 재투영 RMS 정면 (%) | 재투영 RMS 기울기 (%) | 날카로움 | 부트스트랩 로그 SD |", "|---|---|---|---|---|---|---|"]
    med = lambda v: float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None      # noqa: E731
    for d in sorted(ref):
        for key in MK_METHODS:
            xs_ = [mk_norm(r, key) for r in results if r["distance_cm"] == d]
            xs_ = [x for x in xs_ if x]
            if xs_:
                L.append(f"| {d:g} | {MK_METHODS[key]} | {len(xs_)} | {long_fmt(med([x['phase_rms_pct']['front'] for x in xs_]), 3)} | {long_fmt(med([x['phase_rms_pct']['tilt'] for x in xs_]), 3)} | {long_fmt(med([x['sharpness'] for x in xs_]), 4)} | {long_fmt(med([x['boot_log_std'] for x in xs_]), 4)} |")
    dg = ana["diag"]
    L += ["", "## 3. MediaPipe의 원근 반영률 (기록용, 근사값)", "",
          "표시 점으로 프레임마다 기준 프레임 → 현재의 호모그래피를 구하고, 정면 기준 프레임의 MediaPipe 관절(5~20)을 그 호모그래피로 옮긴 위치를 \"손이 평면이라면 있어야 할 관절 위치\"로 두었다. 손은 완전한 평면이 아니므로 이 비교는 근사값이다. "
          "k는 두 점 집합의 호모그래피 원근 성분(정규화한 h31, h32)을 회귀한 기울기이다(1이면 원근을 그대로 반영, 0이면 반영 안 함).", ""]
    if dg.get("k"):
        L += ["| 구분 | 원근 반영률 k | 상관 | 프레임 수 |", "|---|---|---|---|"] + [f"| {'전체' if k_ == 'all' else k_ + ' cm'} | {long_fmt(v['k'], 2)} | {long_fmt(v['corr'], 2)} | {v['n']} |" for k_, v in dg["k"].items()]
        L += ["", "MediaPipe 관절과 평면 예측의 차이(손 박스 대비 %, 중앙값):", "", "| 구분 | 프레임 수 | 차이 중앙값 (%) |", "|---|---|---|"]
        L += [f"| θ2d {k_}° | {v['n']} | {long_fmt(v['median_pct'], 2)} |" for k_, v in dg["gap_by_theta"].items()] + [f"| {k_} cm | {v['n']} | {long_fmt(v['median_pct'], 2)} |" for k_, v in dg["gap_by_distance"].items()]
        L += ["", "![평면 예측과의 차이](figures/planar_gap_vs_tilt.png)", "", "![원근 성분](figures/perspective_scatter.png)"]
    L += ["", "## 4. 영상별 f와 오차", "", "| 영상 | 거리(cm) | 방법 | f (px) | 같은 거리 기준 f 대비 | 1680 대비 | 로그 SD | 사용 프레임 |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        d = r["distance_cm"]
        for key in MK_METHODS:
            x = mk_norm(r, key)
            if x:
                L.append(f"| {r['video']} | {d:g} | {MK_METHODS[key]} | {format(x['f'], '.1f')} | {pct(x['f'] / ref[d] - 1.0)} | {pct(x['f'] / 1680.0 - 1.0)} | {long_fmt(x['boot_log_std'], 4)} | {x['n_used_frames']} |")
            else:
                why = ("; ".join(r["marker"].get("excluded") or []) or r["marker"].get("failed") or "제외") if key in ("markers", "markers_nearfree") else ("; ".join((r.get("mediapipe8") or r["mediapipe"]).get("excluded") or []) or "제외")
                L.append(f"| {r['video']} | {d:g} | {MK_METHODS[key]} | 제외: {why} | - | - | - | - |")
    L += ["", "## 한계", "", "- 영상은 거리마다 2개씩이라 기울기 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.", "- 표시 점도 손의 3차원 구조(손바닥 곡면, 손가락 굽힘)를 그대로 갖는다. 표시가 검출기에 주는 영향은 MediaPipe 결과와 실험 27의 같은 거리 영상을 비교해서만 짐작할 수 있다.",
          "- 기준 f는 같은 날 체커보드 거리 실험의 값(사용자 제공)이며 보정 파일은 서버에 없다.", ""]
    path.write_text("\n".join(L) + "\n")


def mk_results_yaml(run_dir: Path, cfg: dict):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "35_hand_markers", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": MK_NOTE, "candidate": cfg["candidate"], "markers": cfg["markers"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "marker_all_found_ratio": rd(v["marker_stats"]["all_found_ratio"]), "marker_valid_ratio": rd(v["marker_stats"]["valid_ratio"]),
             "marker_tilt_valid_ratio": rd(v["marker_stats"].get("tilt_valid_ratio")),
             "marker_detection_rate_by_joint": {str(k): rd(x, 3) for k, x in (v["marker_stats"].get("per_marker_rate") or {}).items()}}
        for key in MK_METHODS:
            x = mk_norm(v, key)
            if x:
                e[key] = {"f_px": None if x["at_grid_edge"] else round(x["f"], 1), "sharpness": rd(x["sharpness"], 5), "boot_log_std": rd(x["boot_log_std"], 5), "n_used_frames": x["n_used_frames"],
                          "theta_median_deg": rd(x["theta_median"], 1), "theta_max_deg": rd(x["theta_max"], 1), "reproj_rms_front_pct": rd(x["phase_rms_pct"]["front"], 4), "reproj_rms_tilt_pct": rd(x["phase_rms_pct"]["tilt"], 4)}
            else:
                e[key] = {"excluded": (v["marker"].get("excluded") or v["marker"].get("failed")) if key in ("markers", "markers_nearfree") else ((v.get("mediapipe8") or v["mediapipe"]).get("excluded") or "제외")}
        if v.get("diag"):
            e["perspective_k"] = rd(v["diag"]["k"]["k"], 4)
            e["planar_gap_median_pct"] = rd(v["diag"]["gap_median_pct"], 4)
        out["videos"][v["video"]] = e
    a = r["analysis"]
    out["by_method"] = {k: {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in a[k]["slope"]["pooled_f"].items()}, "slope": rd(a[k]["slope"]["slope"]),
                            "slope_interval_95": None if a[k]["slope"]["interval_95"] is None else [round(x, 4) for x in a[k]["slope"]["interval_95"]]} for k in MK_METHODS if "slope" in a[k]}
    out["verdict"] = {k: {"verdict": a[k]["verdict"]["verdict"], "reason": a[k]["verdict"].get("reason")} for k in MK_METHODS}
    out["main_comparison"] = a["main_comparison"]
    out["interpretation"] = a["interpretation"]
    out["perspective_k"] = {k: rd(v["k"], 4) for k, v in a["diag"].get("k", {}).items()}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# command line
# ====================================================================================================

# ====================================================================================================
# mo  (experiment 39: markers only, no hand detector; the f-versus-distance question of 35 asked of the markers alone)
#     python HandCalib/run.py markers-only --config HandCalib/configs/39_markers_only.yaml [--videos A.MOV,B.MOV] [--out DIR] precheck
#     python HandCalib/run.py markers-only --config ... [--videos ...] [--out DIR]
#     python HandCalib/run.py markers-only --config ... results-yaml --run runs/39_markers_only/<UTC>
# ====================================================================================================
MO_NOTE = ("손 검출기(MediaPipe) 없이 손바닥에 붙인 표시 점만으로 f를 추정해, 거리를 바꿔 찍은 영상에서 f가 거리에 반비례하는지 본 실험이다. 실험 35의 계산(fixed_template, 정면 창, 기울기 구간, 부트스트랩)과 판정 규칙을 그대로 쓰고, "
           "표시 검출, 이름 붙이기, 길이 기준만 손 박스 대신 표시 점에서 얻는다. 쓰는 점, 검출 설정, 판정 규칙은 실행 전에 고정했고 바꾸지 않았다. 기준 f와의 비교는 summary.md에만 있다.")


def mo_context_pooled(cfg: dict) -> dict:
    """Pooled f of earlier marker runs for the comparison table of summary.md (read only): {label: {distance: f}}."""
    out = {}
    for label, p in (cfg.get("compare") or {}).items():
        q = hf.resolve(p)
        if q.exists():
            sl = (json.loads(q.read_text()).get("analysis", {}).get("markers", {}).get("slope") or {}).get("pooled_f") or {}
            out[label] = {float(d): f for d, f in sl.items() if f}
    return out


def mo_precheck_report(path: Path, results: list, infos: list, ok_by_d: dict, proceed: bool, cfg: dict):
    pc, mc = cfg["precheck"], cfg["markers"]
    used = [int(x) for x in mc["used_ids"]]
    L = ["# 실험 39 사전 확인 (점 전용)", "",
         f"표시: {mc['color']}, {mc['count']}개, 추정에 쓰는 점 {used}. 통과 규칙: 정면 창(0.5~1.5초)에 사용 표시가 모두 이름 붙는 프레임이 있고, 기울기 구간(3초 이후) 전체 프레임 중 사용 표시 {len(used)}개의 "
         f"{100 * float(mc['min_markers_ratio']):.0f}% 이상이 잡힌 프레임의 비율 ≥ {100 * float(pc['min_tilt_valid_ratio']):.0f}%; 거리마다 통과한 영상이 1개 이상이어야 공식 실행.", "",
         "| 영상 | 거리(cm) | 길이(초) | 저장 해상도 | 회전 | 처리 후 해상도 | 프레임당 검출 표시 수(중앙값) | 이름 붙은 정면 창 프레임 | 기울기 구간 유효 프레임 | 0.5~1.5초 성공 프레임 | 기준 프레임 시각(초) | 통과 | 버려진 이유 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, i in zip(results, infos):
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {i['duration_s']:.1f} | {i['width']}x{i['height']} | {i['rotation_deg']}° | {r['size'][0]}x{r['size'][1]} | {r['n_detected_median']:.0f} | {r['window_named']} / {r['window_frames']} | "
                 f"{'-' if 'tilt_valid_ratio' not in r else format(100 * r['tilt_valid_ratio'], '.1f') + '%'} | {'-' if 'window_success' not in r else str(r['window_success']) + ' / ' + str(r['window_frames_50_150'])} | {long_fmt(r.get('ref_time_s'), 1)} | "
                 f"{'통과' if r.get('passes') else '탈락'} | {r.get('failed') or ', '.join(f'{k} {v}' for k, v in (r.get('reasons') or {}).items()) or '-'} |")
    L += ["", "거리별 통과 영상: " + ", ".join(f"{d:g} cm {v or '없음'}" for d, v in ok_by_d.items()) + f" → 공식 실행 {'진행 가능' if proceed else '하지 않음'}.", "",
          "## 기준 프레임의 이름 붙이기 (저장된 점 배치에 맞춘 결과)", "", "| 영상 | 이름 붙은 점 | 배율 (px / 배치 단위) | 맞춤 잔차 RMS (px) | 무시한 검출 수 중앙값 | 무시한 검출이 있는 프레임 |", "|---|---|---|---|---|---|"]
    for r in results:
        nm = r.get("naming") or {}
        L.append(f"| {r['video']} | {nm.get('n_named', '-')} | {long_fmt(nm.get('scale_px'), 1)} | {long_fmt(nm.get('rms_px'), 2)} | {long_fmt(r.get('ignored_median'), 1)} | "
                 f"{'-' if r.get('ignored_share') is None else format(100 * r['ignored_share'], '.0f') + '%'} |")
    L += ["", "## 표시 번호별 검출률 (기울기 구간 전체 프레임 기준, 관절 번호 = 표시 이름)", "", "표시를 추적한 비율이다(최소 표시 비율을 0.3으로 낮춘 추적, 호모그래피 검사는 켬). * = 추정에 쓰는 점.", ""]
    vids = [r for r in results if r.get("per_marker_rate")]
    if vids:
        L += ["| 표시 번호 | " + " | ".join(r["video"][4:8] for r in vids) + " |", "|---|" + "---|" * len(vids)]
        for j in range(21):
            L.append(f"| {j}{'*' if j in used else ''} | " + " | ".join(("-" if j not in r["per_marker_rate"] else f"{100 * r['per_marker_rate'][j]:.0f}%") for r in vids) + " |")
    L += ["", "정면 기준 프레임 그림: figures/reference_IMG_xxxx.png (청록 원 = 검출한 표시와 배치가 준 번호, 주황 x = 이름 없이 무시한 검출).", ""]
    path.write_text("\n".join(L) + "\n")


def mo_figures(out: Path, results: list, ana: dict, ref: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="체커보드 (같은 날 거리 실험)")
    sl = ana.get("markers", {}).get("slope")
    anchor = (60.0, ref.get(60.0, 1650.0))
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o-", color="#2a6fbb", label="표시 점 (점 전용)")
        anchor = (60.0, pf[60.0]) if 60.0 in pf else anchor
    xs = np.array([min(ds) * 0.9, max(ds) * 1.1])
    ax.plot(xs, anchor[1] * (xs / anchor[0]) ** 0.0, ":", color="#bbb", label="기울기 0")
    ax.plot(xs, anchor[1] * (xs / anchor[0]) ** -1.0, "--", color="#bbb", label="기울기 -1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("합동 f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)
    inc = [r for r in results if "f" in r["marker"]]
    if inc:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for r, col in zip(inc, ("#2a6fbb", "#5aa0e0", "#c47a00", "#e0a540", "#3a9a4a", "#7fcf8a")):
            c = r["marker"]["curve"]
            cost = np.array(c["cost"])
            ax.plot(c["grid"], cost / cost.min(), color=col, label=f"{r['video'][4:8]} ({r['distance_cm']:g} cm)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f (px)")
        ax.set_ylabel("비용 / 최솟값")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(out / "cost_curves.png", dpi=110)
        plt.close(fig)


def mo_report(path: Path, results: list, cfg: dict):
    mc = cfg["markers"]
    L = ["# 실험 39: 점 전용 (손 검출기 없음)", "", MO_NOTE, "", f"표시: {mc['color']} {mc['count']}개, 추정에 쓰는 점 {[int(x) for x in mc['used_ids']]}. 표시 검출: Lab a* 채널의 국소 붉기(중앙값 {mc['median_ksize']} px 창 대비) ≥ {mc['contrast_min']:g}, 면적 {mc['area_min']:g}~{mc['area_max']:g} px.", "",
         "## 1. 영상별 표시 통계", "", "| 영상 | 거리(cm) | 기울기 구간 유효 프레임 | 0.5~1.5초 성공 프레임 | 포함 | 이유 |", "|---|---|---|---|---|---|"]
    for r in results:
        st, mkr = r["marker_stats"], r["marker"]
        L.append(f"| {r['video']} | {r['distance_cm']:g} | {long_fmt(None if st.get('tilt_valid_ratio') is None else 100 * st['tilt_valid_ratio'], 1)}% | {st.get('window_success')} / {st.get('window_frames_50_150')} | {'포함' if 'f' in mkr else '제외'} | {'; '.join(mkr.get('excluded') or []) or mkr.get('failed') or '-'} |")
    L += ["", "## 2. 영상별 값", "", "| 영상 | 거리(cm) | 사용 프레임 | θ2d 중앙값 / 최댓값(°) | f (px) | 날카로움 | 부트스트랩 로그 SD | 재투영 RMS 정면 / 기울기 (박스 대비 %) |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r["marker"]
        if "f" in x:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | {x['n_used_frames']} | {long_fmt(x['theta_median'], 1)} / {long_fmt(x['theta_max'], 1)} | {'값 없음' if x['at_grid_edge'] else format(x['f'], '.1f')} | {x['sharpness']:.4f} | "
                     f"{long_fmt(x['boot_log_std'], 4)} | {long_fmt(x['phase_rms_pct']['front'], 3)} / {long_fmt(x['phase_rms_pct']['tilt'], 3)} |")
        else:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | - | - | - | - |")
    L += ["", "![비용 곡선](figures/cost_curves.png)", "", "## 한계", "",
          "- 영상은 거리마다 2개씩이라 기울기 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 영상은 태블릿 화면에 띄운 평면 손 그림이다. 손의 3차원 구조(손바닥 곡면, 손가락 굽힘)가 없어서 \"원인은 손 자체\" 같은 해석에는 쓰지 않는다.",
          "- 표시 검출 기준(국소 붉기 문턱, 면적)과 이름 붙이기 설정은 이 영상들의 첫 프레임을 보고 정했고, 점 배치 템플릿은 같은 세션의 50 cm 영상 두 개의 기준 프레임에서 만들었다.", ""]
    path.write_text("\n".join(L) + "\n")


def mo_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["markers"]["verdict"]
    s_ = f"- **판정 (점 전용): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 합동 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    L = ["# 실험 39 summary", "", f"**{MO_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 35와 같음): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 합동 f가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 거리에 영상이 없으면 판단 불가.",
         "- 영상은 태블릿 화면의 평면 손 그림이라 이 판정을 \"원인은 손 자체\" 같은 해석에 쓰지 않는다.", "",
         "## 1. 거리별 합동 f와 기준 f", "", "| 거리(cm) | 영상 | 합동 f (px) | 95% 구간 (px) | 같은 거리 기준 f | 기준 f 대비 | 1680 대비 |", "|---|---|---|---|---|---|---|"]
    for d, p in (ana["markers"].get("per_distance") or {}).items():
        pf = p.get("pooled_f")
        L.append(f"| {float(d):g} | {p['n_videos']} / {p['n_total']} | {format(pf, '.1f') if pf else '-'} | {'-' if 'interval_95' not in p else format(p['interval_95'][0], '.0f') + '–' + format(p['interval_95'][1], '.0f')} | {ref[float(d)]:.1f} | "
                 f"{pct(None if not pf else pf / ref[float(d)] - 1.0)} | {pct(None if not pf else pf / 1680.0 - 1.0)} |")
    sl = ana["markers"].get("slope")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| 표시 점 (점 전용) | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |",
          f"| 체커보드 | {long_fmt(ana['checkerboard_slope'], 3)} | - |"]
    ctx = mo_context_pooled(cfg)
    names = {"35c": "실험 35c 표시 점 (실제 손, 손 박스 사용)", "38": "실험 38 표시 점 (태블릿 손 그림, 손 박스 사용)"}
    L += ["", "거리별 비교 (합동 f, px):", "", "| 거리(cm) | 표시 점 (점 전용) | " + " | ".join(names.get(k, k) for k in ctx) + " | 체커보드 |", "|---|---|" + "---|" * (len(ctx) + 1)]
    for d in sorted(ref):
        mine = (sl or {}).get("pooled_f", {}).get(d)
        L.append(f"| {d:g} | {format(mine, '.1f') if mine else '-'} | " + " | ".join((format(c[d], '.1f') if d in c else '-') for c in ctx.values()) + f" | {ref[d]:.1f} |" if ctx else f"| {d:g} | {format(mine, '.1f') if mine else '-'} | {ref[d]:.1f} |")
    L += ["", "![거리 대 합동 f](figures/f_vs_distance_loglog.png)", "", "## 2. 재투영 RMS와 날카로움 (기록용)", "", "| 거리(cm) | 영상 | 재투영 RMS 정면 (%) | 재투영 RMS 기울기 (%) | 날카로움 | 부트스트랩 로그 SD |", "|---|---|---|---|---|---|"]
    med = lambda v_: float(np.median(v_)) if len(v_) else None      # noqa: E731
    for d in sorted(ref):
        xs_ = [r["marker"] for r in results if r["distance_cm"] == d and "f" in r["marker"]]
        if xs_:
            L.append(f"| {d:g} | {len(xs_)} | {long_fmt(med([x['phase_rms_pct']['front'] for x in xs_]), 3)} | {long_fmt(med([x['phase_rms_pct']['tilt'] for x in xs_]), 3)} | {long_fmt(med([x['sharpness'] for x in xs_]), 4)} | {long_fmt(med([x['boot_log_std'] for x in xs_]), 4)} |")
    L += ["", "![비용 곡선](figures/cost_curves.png)", "", "## 3. 영상별 f와 오차", "", "| 영상 | 거리(cm) | f (px) | 같은 거리 기준 f 대비 | 1680 대비 | 로그 SD | 사용 프레임 | θ2d 최댓값(°) |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x = r["distance_cm"], r["marker"]
        if "f" in x:
            L.append(f"| {r['video']} | {d:g} | {format(x['f'], '.1f')} | {pct(x['f'] / ref[d] - 1.0)} | {pct(x['f'] / 1680.0 - 1.0)} | {long_fmt(x['boot_log_std'], 4)} | {x['n_used_frames']} | {long_fmt(x['theta_max'], 1)} |")
        else:
            L.append(f"| {r['video']} | {d:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | - | - | - | - |")
    L += ["", "## 한계", "", "- 영상은 거리마다 2개씩이라 기울기 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 기준 f는 실험 35c 때 체커보드 거리 실험의 값(사용자 제공)이며 이번 영상 세션에서 새로 재지 않았다.",
          "- 표시 검출 기준과 이름 붙이기 설정은 이 영상들의 첫 프레임을 보고 정했고(예비 확인), 점 배치 템플릿은 같은 세션의 50 cm 영상 두 개에서 만들었다.", ""]
    path.write_text("\n".join(L) + "\n")


def mo_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/39_markers_only.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("precheck")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(mo_results_yaml(hf.resolve(a.run), cfg))
        return None
    pre = a.cmd == "precheck"
    entries = list(cfg["videos"])
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / (utc_stamp() + ("_precheck" if pre else ""))
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    mc = cfg["markers"]
    log(f"39 점 전용 {'사전 확인' if pre else '공식 실행'}: 영상 {len(entries)}개, 표시 {mc['color']} {mc['count']}개, 추정에 쓰는 점 {mc['used_ids']} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    infos = [hf.probe(Path(cfg["input_dir"]).expanduser() / e["file"]) for e in entries]
    c = {k: cfg[k] for k in ("exclusion", "markers", "marker_exclusion", "precheck", "naming", "layout")}
    if pre:
        c["out_dir"] = str(run_root / "figures")
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            results = list(ex.map(hf.mo_precheck_video if pre else hf.mo_video, [(e, p, cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    if pre:
        ok_by_d = {d: [r["video"] for r in results if r["distance_cm"] == d and r.get("passes")] for d in sorted(ref)}
        proceed = all(len(v_) >= 1 for v_ in ok_by_d.values())
        mo_precheck_report(run_root / "precheck.md", results, infos, ok_by_d, proceed, cfg)
        (run_root / "precheck.json").write_text(json.dumps(hf.jsonable({"videos": results, "passing_by_distance": ok_by_d, "official_run_ok": proceed}), ensure_ascii=False, indent=1))
        for r in results:
            log(f"  {r['video']} ({r['distance_cm']:g} cm): " + ("기준 프레임 없음" if "tilt_valid_ratio" not in r else f"기울기 구간 유효 {100 * r['tilt_valid_ratio']:.1f}%, 0.5~1.5초 성공 {r['window_success']}/{r['window_frames_50_150']}") + f" → {'통과' if r.get('passes') else '탈락'}")
        log(f"  거리별 통과 영상: {ok_by_d} → 공식 실행 {'진행 가능' if proceed else '하지 않음 (보고하고 멈춤)'}")
        log(f"완료 {time.time() - t0:.0f} s → {run_root}")
        log.close()
        return run_root, results, proceed
    v2 = cfg["rating_v2"]
    for r in results:
        mkr = r["marker"]
        if "f" in mkr:
            mkr["rating"] = hf.gen_rating(mkr["tilt_duration_s"], mkr["theta_max"], mkr["boot_log_std"], v2)
        log(f"  {r['video']} ({r['distance_cm']:g} cm): " + (f"표시 f {mkr['f']:.1f} px" if "f" in mkr else "표시 제외") + (f" [제외: {'; '.join(mkr['excluded'])}]" if mkr.get("excluded") else "") + (f" [{mkr['failed']}]" if mkr.get("failed") else ""))
    grid = next((np.array(r["marker"]["curve"]["grid"]) for r in results if "f" in r["marker"]), None)
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [(r["distance_cm"], r["marker"]["curve"]["cost"]) for r in results if "f" in r["marker"]]
    ds = sorted({d for d, _ in inc})
    ana = {}
    if not ds or grid is None:
        ana["markers"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        cur = {d: [cc for dd, cc in inc if dd == d] for d in ds}
        sl = hf.pr_slope(cur, grid, bn, bs)
        per = {d: {**hf.ext_pooled(cur[d], grid, {"bootstrap_n": bn, "bootstrap_seed": bs}), "n_total": sum(1 for r in results if r["distance_cm"] == d)} for d in ds}
        nv = {d: len(cur.get(d, [])) for d in ref}
        ana["markers"] = {"slope": sl, "per_distance": per, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        log(f"  markers: {ana['markers']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else ""))
    ana["checkerboard_slope"] = hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)])
    mo_figures(run_root / "figures", results, ana, ref)
    mo_report(run_root / "report.md", results, cfg)
    mo_summary(run_root / "summary.md", results, ana, ref, cfg)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_39": cfg["code_39"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def mo_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "39_markers_only", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": MO_NOTE, "markers": cfg["markers"], "naming": cfg["naming"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v["marker"]
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "marker_valid_ratio": rd(v["marker_stats"].get("valid_ratio")), "marker_tilt_valid_ratio": rd(v["marker_stats"].get("tilt_valid_ratio")),
             "marker_detection_rate_by_joint": {str(k): rd(q, 3) for k, q in (v["marker_stats"].get("per_marker_rate") or {}).items()}}
        if "f" in x:
            e["markers"] = {"f_px": None if x["at_grid_edge"] else round(x["f"], 1), "sharpness": rd(x["sharpness"], 5), "boot_log_std": rd(x["boot_log_std"], 5), "n_used_frames": x["n_used_frames"],
                            "theta_median_deg": rd(x["theta_median"], 1), "theta_max_deg": rd(x["theta_max"], 1), "reproj_rms_front_pct": rd(x["phase_rms_pct"]["front"], 4), "reproj_rms_tilt_pct": rd(x["phase_rms_pct"]["tilt"], 4)}
        else:
            e["markers"] = {"excluded": x.get("excluded") or x.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]["markers"]
    if "slope" in a:
        out["by_method"] = {"markers": {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in a["slope"]["pooled_f"].items()}, "slope": rd(a["slope"]["slope"]),
                                        "slope_interval_95": None if a["slope"]["interval_95"] is None else [round(q, 4) for q in a["slope"]["interval_95"]]}}
    out["verdict"] = {"markers": {"verdict": a["verdict"]["verdict"], "reason": a["verdict"].get("reason")}}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# ft  (experiment 40: the supervisor's free template on the marker tracks of 39; the question of 39 (f versus distance) and the comparison with 39's fixed template)
#     python HandCalib/run.py free-template --config HandCalib/configs/40_free_template.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py free-template --config ... results-yaml --run runs/40_free_template/<UTC>
# ====================================================================================================
FT_KEYS = ("exclusion", "markers", "marker_exclusion", "naming", "layout", "free_template")      # the config sections ft_video needs
FT_NOTE = ("실험 39의 점 추적 결과(같은 코드와 설정)를 교수님이 보내주신 free_template(템플릿을 f마다 다시 추정하고 번들 조정으로 다듬는 방식)에 넣어 f를 구하고, 실험 39의 fixed_template 값과 비교한 실험이다. "
           "호출은 pb, idx = free_problem(trk, (W, H), ecfg, huber_rel, ids) 와 estimate_run(pb, 'free_template', grid, ecfg, sigma2) 이고, 실험 39와 같은 사용 표시 8개, 판정 규칙을 쓴다. "
           "쓰는 점, 설정, 판정 규칙은 실행 전에 고정했고 바꾸지 않았다. 기준 f와의 비교는 summary.md에만 있다.")


def ft_prior(cfg: dict) -> dict:
    """Per-video f and pooled f of experiment 39 (read only): {'video': {video: f}, 'pooled': {distance: f}}."""
    q = hf.resolve((cfg.get("compare") or {}).get("39", "x"))
    if not q.exists():
        return {"video": {}, "pooled": {}}
    r = json.loads(q.read_text())
    sl = (r.get("analysis", {}).get("markers", {}).get("slope") or {}).get("pooled_f") or {}
    return {"video": {v["video"]: v["marker"]["f"] for v in r["videos"] if "f" in v["marker"]}, "pooled": {float(d): f for d, f in sl.items() if f}}


def ft_figures(out: Path, results: list, ana: dict, ref: dict, prior: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="체커보드 (같은 날 거리 실험)")
    if prior["pooled"]:
        ax.plot(sorted(prior["pooled"]), [prior["pooled"][d] for d in sorted(prior["pooled"])], "o-", color="#2a6fbb", label="실험 39 (fixed_template)")
    sl = ana.get("markers", {}).get("slope")
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o--", color="#c47a00", label="free_template (격자 비용 곡선 합동)")
    for r in results:
        if "f" in r["marker"]:
            ax.plot([r["distance_cm"]], [r["marker"]["f"]], "x", color="#c47a00", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)
    inc = [r for r in results if "f" in r["marker"]]
    if inc:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for r, col in zip(inc, ("#2a6fbb", "#5aa0e0", "#c47a00", "#e0a540", "#3a9a4a", "#7fcf8a")):
            c = r["marker"]["curve"]
            cost = np.array(c["cost"])
            ax.plot(c["grid"], cost / cost.min(), color=col, label=f"{r['video'][4:8]} ({r['distance_cm']:g} cm)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f (px)")
        ax.set_ylabel("비용 / 최솟값 (free_template)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(out / "cost_curves.png", dpi=110)
        plt.close(fig)


def ft_report(path: Path, results: list, cfg: dict):
    ft = cfg["free_template"]
    L = ["# 실험 40: free_template (점 전용 추적 위에서)", "", FT_NOTE, "", f"추정에 쓰는 점 {[int(x) for x in cfg['markers']['used_ids']]}, free_template(alt_iters {ft['alt_iters']}, refine {ft['refine']}, max_frames {ft['max_frames']}), f 격자 {cfg['pipeline']['estimate']['f_grid_n']}점.", "",
         "## 1. 영상별 값", "", "| 영상 | 거리(cm) | 포함 | f 번들 조정 (px) | f 격자 최솟값 (px) | 격자 가장자리 | 번들 조정이 격자 한계에 붙음 | 날카로움 | 번들 조정 RMS (px) | 사용 프레임 | 걸린 시간(초) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r["marker"]
        if "f" in x:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 포함 | {x['f']:.1f} | {x['f_grid']:.1f} | {'예' if x['at_grid_edge'] else '아니오'} | {'예' if x['ba_at_bound'] else '아니오'} | {long_fmt(x['sharpness'], 4)} | {long_fmt(x['ba_rms_px'], 3)} | {x['n_frames_in_problem']} | {x['elapsed_s']:.0f} |")
        else:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | - | - | - | - | - | - | - |")
    L += ["", "![비용 곡선](figures/cost_curves.png)", "", "## 한계", "",
          "- 영상은 태블릿 화면에 띄운 평면 손 그림이고 거리마다 2개씩이다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- free_template은 템플릿 점 좌표를 같은 프레임들로 다시 추정하므로 f와 템플릿이 서로 일부 바뀔 수 있다.",
          "- 교수님 코드는 그대로 실행되지 않아 여섯 곳을 고쳐서 썼다(방법은 그대로): least_squares import, np.c -> np.c_, _backproject -> backproject, rodrigues -> _rodrigues, np.r -> np.r_, Problem.new -> Problem.__new__.", ""]
    path.write_text("\n".join(L) + "\n")


def ft_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict, prior: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["markers"]["verdict"]
    s_ = f"- **판정 (free_template, 격자 비용 곡선 합동 f 기준): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 합동 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    sl = ana["markers"].get("slope")
    L = ["# 실험 40 summary", "", f"**{FT_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 39와 같음): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 합동 f가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 거리에 영상이 없으면, 또는 합동 f가 격자 가장자리면 판단 불가.",
         "- 영상은 태블릿 화면의 평면 손 그림이라 이 판정을 \"원인은 손 자체\" 같은 해석에 쓰지 않는다.", "",
         "## 1. 거리별 합동 f: free_template과 실험 39 (fixed_template)", "", "| 거리(cm) | 영상 | free_template 합동 f (px) | 95% 구간 (px) | 실험 39 합동 f (px) | 같은 거리 기준 f | free_template 기준 f 대비 | 실험 39 기준 f 대비 |", "|---|---|---|---|---|---|---|---|"]
    for d, p in (ana["markers"].get("per_distance") or {}).items():
        pf, p39 = p.get("pooled_f"), prior["pooled"].get(float(d))
        L.append(f"| {float(d):g} | {p['n_videos']} / {p['n_total']} | {format(pf, '.1f') if pf else '-'}{' (격자 가장자리)' if p.get('at_grid_edge') else ''} | {'-' if 'interval_95' not in p else format(p['interval_95'][0], '.0f') + '–' + format(p['interval_95'][1], '.0f')} | "
                 f"{format(p39, '.1f') if p39 else '-'} | {ref[float(d)]:.1f} | {pct(None if not pf else pf / ref[float(d)] - 1.0)} | {pct(None if not p39 else p39 / ref[float(d)] - 1.0)} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| free_template | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |"]
    p39 = prior["pooled"]
    if len(p39) >= 2:
        L.append(f"| 실험 39 (fixed_template) | {long_fmt(hf.ext_loglog_slope(sorted(p39), [p39[d] for d in sorted(p39)]), 3)} | - |")
    L += [f"| 체커보드 | {long_fmt(hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)]), 3)} | - |", "", "![거리 대 f](figures/f_vs_distance_loglog.png)", "",
          "## 2. 영상별 f: free_template과 실험 39", "", "| 영상 | 거리(cm) | free_template f 번들 조정 (px) | f 격자 최솟값 (px) | 실험 39 f (px) | free ÷ 실험 39 (번들 조정) | 기준 f 대비 (번들 조정) | 기준 f 대비 (실험 39) | 1680 대비 (번들 조정) | 격자 가장자리 | 번들 조정이 격자 한계에 붙음 | 날카로움 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x = r["distance_cm"], r["marker"]
        f39 = prior["video"].get(r["video"])
        if "f" in x:
            L.append(f"| {r['video']} | {d:g} | {x['f']:.1f} | {x['f_grid']:.1f} | {format(f39, '.1f') if f39 else '-'} | {'-' if not f39 else format(x['f'] / f39, '.3f')} | {pct(x['f'] / ref[d] - 1.0)} | {pct(None if not f39 else f39 / ref[d] - 1.0)} | {pct(x['f'] / 1680.0 - 1.0)} | "
                     f"{'예' if x['at_grid_edge'] else '아니오'} | {'예' if x['ba_at_bound'] else '아니오'} | {long_fmt(x['sharpness'], 4)} |")
        else:
            L.append(f"| {r['video']} | {d:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | {format(f39, '.1f') if f39 else '-'} | - | - | - | - | - | - | - |")
    L += ["", "![비용 곡선](figures/cost_curves.png)", "", "## 한계", "", "- 영상은 거리마다 2개씩이라 기울기 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 기준 f는 실험 35c 때 체커보드 거리 실험의 값(사용자 제공)이며 이번 영상 세션에서 새로 재지 않았다.",
          "- free_template은 템플릿을 f마다 다시 추정하므로 f와 템플릿이 서로 일부 바뀔 수 있다. 실행 전 합성 예비 확인(점이 정확히 평면이고 참 f를 아는 영상 2개, 이 결과와 무관한 개발용 확인)에서 참 f 850 px는 번들 조정 1536 px(격자 상한)로, 참 f 1100 px는 격자 최솟값이 가장자리에 붙고 번들 조정 1123 px로 나왔다.",
          "- 교수님 코드는 그대로 실행되지 않아 여섯 곳을 고쳐서 썼다(방법은 그대로).", ""]
    path.write_text("\n".join(L) + "\n")


def ft_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/40_free_template.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(ft_results_yaml(hf.resolve(a.run), cfg))
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
    log(f"40 free_template 공식 실행: 영상 {len(entries)}개, 추정에 쓰는 점 {cfg['markers']['used_ids']} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    c = {k: cfg[k] for k in FT_KEYS}
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            results = list(ex.map(hf.ft_video, [(e, p, cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    prior = ft_prior(cfg)
    for r in results:
        x = r["marker"]
        log(f"  {r['video']} ({r['distance_cm']:g} cm): " + (f"free_template f {x['f']:.1f} px (격자 최솟값 {x['f_grid']:.1f}{', 격자 가장자리' if x['at_grid_edge'] else ''}{', 번들 조정이 격자 한계에 붙음' if x['ba_at_bound'] else ''}), {x['elapsed_s']:.0f} s" if "f" in x else "제외")
            + (f" [제외: {'; '.join(x['excluded'])}]" if x.get("excluded") else "") + (f" [{x['failed']}]" if x.get("failed") else ""))
    grid = next((np.array(r["marker"]["curve"]["grid"]) for r in results if "f" in r["marker"]), None)
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [(r["distance_cm"], r["marker"]["curve"]["cost"]) for r in results if "f" in r["marker"]]
    ds = sorted({d for d, _ in inc})
    ana = {}
    if not ds or grid is None:
        ana["markers"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        cur = {d: [cc for dd, cc in inc if dd == d] for d in ds}
        sl = hf.pr_slope(cur, grid, bn, bs)
        per = {d: {**hf.ext_pooled(cur[d], grid, {"bootstrap_n": bn, "bootstrap_seed": bs}), "n_total": sum(1 for r in results if r["distance_cm"] == d)} for d in ds}
        nv = {d: len(cur.get(d, [])) for d in ref}
        ana["markers"] = {"slope": sl, "per_distance": per, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        log(f"  free_template: {ana['markers']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else ""))
    ft_figures(run_root / "figures", results, ana, ref, prior)
    ft_report(run_root / "report.md", results, cfg)
    ft_summary(run_root / "summary.md", results, ana, ref, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_40": cfg["code_40"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def ft_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "40_free_template", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": FT_NOTE, "markers": cfg["markers"], "free_template": cfg["free_template"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v["marker"]
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "marker_tilt_valid_ratio": rd(v["marker_stats"].get("tilt_valid_ratio"))}
        if "f" in x:
            e["free_template"] = {"f_ba_px": round(x["f"], 1), "f_grid_px": round(x["f_grid"], 1), "at_grid_edge": x["at_grid_edge"], "ba_at_bound": x["ba_at_bound"], "sharpness": rd(x["sharpness"], 5), "ba_rms_px": rd(x["ba_rms_px"], 4),
                                  "ba_success": x["ba_success"], "n_frames_in_problem": x["n_frames_in_problem"], "elapsed_s": round(x["elapsed_s"])}
        else:
            e["free_template"] = {"excluded": x.get("excluded") or x.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]["markers"]
    if "slope" in a:
        out["by_method"] = {"free_template": {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in a["slope"]["pooled_f"].items()}, "slope": rd(a["slope"]["slope"]),
                                              "slope_interval_95": None if a["slope"]["interval_95"] is None else [round(q, 4) for q in a["slope"]["interval_95"]]}}
    out["verdict"] = {"free_template": {"verdict": a["verdict"]["verdict"], "reason": a["verdict"].get("reason")}}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p



# ====================================================================================================
# s3  (experiment 41: the supervisor's free_shape3d on the marker tracks of 39 / 40; comparison with 40's free_template and 39's fixed_template)
#     python HandCalib/run.py free-shape3d --config HandCalib/configs/41_free_shape3d.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py free-shape3d --config ... results-yaml --run runs/41_free_shape3d/<UTC>
# ====================================================================================================
S3_KEYS = ("exclusion", "markers", "marker_exclusion", "naming", "layout", "free_shape3d")      # the config sections s3_video needs
S3_NOTE = ("실험 39·40의 점 추적 결과(같은 코드와 설정)를 교수님이 새로 보내주신 free_shape3d(평면 free_template 위에 점마다 작은 깊이 변위를 허용하는 3차원 번들 조정)에 넣어 f를 구하고, 실험 40의 free_template·실험 39의 fixed_template과 비교한 실험이다. "
           "호출은 pb, idx = free_problem(trk, (W, H), ecfg, huber_rel, ids) 와 estimate_run(pb, 'free_shape3d', grid, ecfg, sigma2) 이고, 실험 39와 같은 사용 표시 8개를 쓴다. "
           "판정은 영상별 3차원 f의 거리-f 기울기(영상 단위 부트스트랩)로 하고, 판정 규칙과 쓰는 점, 설정은 실행 전에 고정했다. 기준 f와의 비교는 summary.md에만 있다.")


def s3_prior(cfg: dict) -> dict:
    """Per-video f and pooled f of experiments 39 (fixed_template) and 40 (free_template, bundle-adjusted f per video; pooled f of its grid curves); read only."""
    out = {}
    for k in ("39", "40"):
        q = hf.resolve((cfg.get("compare") or {}).get(k, "x"))
        if not q.exists():
            out[k] = {"video": {}, "pooled": {}}
            continue
        r = json.loads(q.read_text())
        sl = (r.get("analysis", {}).get("markers", {}).get("slope") or {}).get("pooled_f") or {}
        out[k] = {"video": {v["video"]: v["marker"]["f"] for v in r["videos"] if "f" in v["marker"]}, "pooled": {float(d): f for d, f in sl.items() if f}}
    return out


def s3_figures(out: Path, results: list, ana: dict, ref: dict, prior: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="체커보드 (같은 날 거리 실험)")
    for k, col, lab in (("39", "#2a6fbb", "실험 39 (fixed_template, 합동)"), ("40", "#3a9a4a", "실험 40 (free_template, 합동)")):
        if prior[k]["pooled"]:
            ax.plot(sorted(prior[k]["pooled"]), [prior[k]["pooled"][d] for d in sorted(prior[k]["pooled"])], "o-", color=col, label=lab)
    sl = ana.get("shape3d", {}).get("slope")
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o--", color="#c47a00", label="free_shape3d (영상별 f의 기하평균)")
    for r in results:
        if "f" in r["marker"]:
            ax.plot([r["distance_cm"]], [r["marker"]["f"]], "x", color="#c47a00", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)
    inc = [r for r in results if "f" in r["marker"]]
    if inc:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for r, col in zip(inc, ("#2a6fbb", "#5aa0e0", "#c47a00", "#e0a540", "#3a9a4a", "#7fcf8a")):
            ax.plot(r["marker"]["joint_ids"], r["marker"]["relief_per_joint"], "o-", color=col, label=f"{r['video'][4:8]} ({r['distance_cm']:g} cm)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel("점 번호")
        ax.set_ylabel("평면에서 벗어난 깊이 / 템플릿 RMS 반지름")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "relief_per_joint.png", dpi=110)
        plt.close(fig)


def s3_report(path: Path, results: list, cfg: dict):
    fs = cfg["free_shape3d"]
    L = ["# 실험 41: free_shape3d (점 전용 추적 위에서)", "", S3_NOTE, "", f"추정에 쓰는 점 {[int(x) for x in cfg['markers']['used_ids']]}, free_shape3d(z_rel {fs['z_rel']}, alt_iters {fs['alt_iters']}, max_frames {fs['max_frames']}), f 격자 {cfg['pipeline']['estimate']['f_grid_n']}점.", "",
         "## 1. 영상별 값", "", "| 영상 | 거리(cm) | 포함 | f 3차원 (px) | f 평면 번들 조정 (px) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 3차원 RMS (px) | 3차원 반복 수 | 사용 프레임 | 걸린 시간(초) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r["marker"]
        if "f" in x:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 포함 | {x['f']:.1f} | {x['f_planar']:.1f} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {long_fmt(x['ba3_rms_px'], 3)} | {x['ba3_nfev']} | {x['n_frames_in_problem']} | {x['elapsed_s']:.0f} |")
        else:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | - | - | - | - | - | - | - |")
    L += ["", "![점마다 깊이 변위](figures/relief_per_joint.png)", "", "## 한계", "",
          "- 영상은 태블릿 화면에 띄운 평면 손 그림이고 거리마다 2개씩이다. 점이 실제로 평면이므로 3차원 깊이 변위는 0에 가까워야 정상이다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 3차원 조정은 f에 대한 비용 곡선이 없어, 판정은 영상별 f의 기울기(영상 단위 부트스트랩)로 한다. 거리마다 영상이 2개라 구간이 거칠다.",
          "- 교수님 코드는 고치지 않고 그대로 썼다.", ""]
    path.write_text("\n".join(L) + "\n")


def s3_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict, prior: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["shape3d"]["verdict"]
    s_ = f"- **판정 (free_shape3d, 영상별 3차원 f 기준): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 거리별 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    sl = ana["shape3d"].get("slope")
    pl = ana.get("planar", {}).get("slope")
    L = ["# 실험 41 summary", "", f"**{S3_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 39·40과 같은 문턱): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 f(영상별 3차원 f의 기하평균)가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 거리에 영상이 없거나 3차원 f가 f 범위 한계에 붙은 영상이 있으면 판단 불가.",
         "- 영상은 태블릿 화면의 평면 손 그림이라 이 판정을 \"원인은 손 자체\" 같은 해석에 쓰지 않는다.", "",
         "## 1. 거리별 f", "", "| 거리(cm) | 영상 | free_shape3d (px) | 실험 40 합동 f (px) | 실험 39 합동 f (px) | 같은 거리 기준 f | shape3d 기준 f 대비 | 실험 40 기준 f 대비 | 실험 39 기준 f 대비 |", "|---|---|---|---|---|---|---|---|---|"]
    for d in (sl or {}).get("distances", []):
        f3, p40, p39 = sl["pooled_f"][d], prior["40"]["pooled"].get(float(d)), prior["39"]["pooled"].get(float(d))
        n_in = sum(1 for r in results if r["distance_cm"] == d and "f" in r["marker"])
        n_all = sum(1 for r in results if r["distance_cm"] == d)
        L.append(f"| {float(d):g} | {n_in} / {n_all} | {f3:.1f} | {format(p40, '.1f') if p40 else '-'} | {format(p39, '.1f') if p39 else '-'} | {ref[float(d)]:.1f} | {pct(f3 / ref[float(d)] - 1.0)} | {pct(None if not p40 else p40 / ref[float(d)] - 1.0)} | {pct(None if not p39 else p39 / ref[float(d)] - 1.0)} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| free_shape3d (영상별 3차원 f) | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |"]
    if pl:
        L.append(f"| 평면 free_template 합동 (실험 40의 판정과 같은 계산) | {long_fmt(pl['slope'], 3)} | {'-' if pl['interval_95'] is None else format(pl['interval_95'][0], '+.3f') + ' ~ ' + format(pl['interval_95'][1], '+.3f')} |")
    for k, name in (("40", "실험 40 (free_template)"), ("39", "실험 39 (fixed_template)")):
        pp = prior[k]["pooled"]
        if len(pp) >= 2:
            L.append(f"| {name} | {long_fmt(hf.ext_loglog_slope(sorted(pp), [pp[d] for d in sorted(pp)]), 3)} | - |")
    L += [f"| 체커보드 | {long_fmt(hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)]), 3)} | - |", "", "![거리 대 f](figures/f_vs_distance_loglog.png)", "",
          "## 2. 영상별 f", "", "| 영상 | 거리(cm) | f 3차원 (px) | f 평면 번들 조정 (px) | 실험 40 f (px) | 실험 39 f (px) | 3차원 ÷ 평면 | 평면 ÷ 실험 40 | 기준 f 대비 (3차원) | 기준 f 대비 (평면) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x = r["distance_cm"], r["marker"]
        f40, f39 = prior["40"]["video"].get(r["video"]), prior["39"]["video"].get(r["video"])
        if "f" in x:
            L.append(f"| {r['video']} | {d:g} | {x['f']:.1f} | {x['f_planar']:.1f} | {format(f40, '.1f') if f40 else '-'} | {format(f39, '.1f') if f39 else '-'} | {x['f'] / x['f_planar']:.3f} | {'-' if not f40 else format(x['f_planar'] / f40, '.4f')} | "
                     f"{pct(x['f'] / ref[d] - 1.0)} | {pct(x['f_planar'] / ref[d] - 1.0)} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} |")
        else:
            L.append(f"| {r['video']} | {d:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | {format(f40, '.1f') if f40 else '-'} | {format(f39, '.1f') if f39 else '-'} | - | - | - | - | - | - |")
    L += ["", "![점마다 깊이 변위](figures/relief_per_joint.png)", "", "## 한계", "", "- 영상은 거리마다 2개씩이라 기울기 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 기준 f는 실험 35c 때 체커보드 거리 실험의 값(사용자 제공)이며 이번 영상 세션에서 새로 재지 않았다.",
          "- 영상 속 점은 실제로 평면(태블릿 화면)이라 깊이 변위가 0에 가까워야 한다. 비평면인 실제 손에서의 동작은 이 실험으로 알 수 없다.", ""]
    path.write_text("\n".join(L) + "\n")


def s3_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/41_free_shape3d.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(s3_results_yaml(hf.resolve(a.run), cfg))
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
    log(f"41 free_shape3d 공식 실행: 영상 {len(entries)}개, 추정에 쓰는 점 {cfg['markers']['used_ids']} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    c = {k: cfg[k] for k in S3_KEYS}
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            results = list(ex.map(hf.s3_video, [(e, p, cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    prior = s3_prior(cfg)
    for r in results:
        x = r["marker"]
        log(f"  {r['video']} ({r['distance_cm']:g} cm): " + (f"free_shape3d f {x['f']:.1f} px (평면 {x['f_planar']:.1f}, 상대 깊이 변위 {x['relief_rel']:.4f}{', f 한계에 붙음' if x['at_bound'] else ''}), {x['elapsed_s']:.0f} s" if "f" in x else "제외")
            + (f" [제외: {'; '.join(x['excluded'])}]" if x.get("excluded") else "") + (f" [{x['failed']}]" if x.get("failed") else ""))
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [r for r in results if "f" in r["marker"]]
    ds = sorted({r["distance_cm"] for r in inc})
    ana = {}
    if not ds:
        ana["shape3d"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        fb = {d: [r["marker"]["f"] for r in inc if r["distance_cm"] == d] for d in ds}
        eb = {d: [r["marker"]["at_bound"] or r["marker"]["at_grid_edge"] for r in inc if r["distance_cm"] == d] for d in ds}
        sl = hf.s3_slope(fb, eb, bn, bs)
        nv = {d: len(fb.get(d, [])) for d in ref}
        ana["shape3d"] = {"slope": sl, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        grid = np.array(inc[0]["marker"]["curve"]["grid"])                                    # the planar curves, the calculation of 40's verdict (a reproducibility check, not the verdict of 41)
        cur = {d: [r["marker"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in ds}
        psl = hf.pr_slope(cur, grid, bn, bs)
        ana["planar"] = {"slope": psl, "verdict": hf.pr_verdict(psl, ref, cfg["verdict"], nv)}
        log(f"  free_shape3d: {ana['shape3d']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else ""))
        log(f"  평면 합동 곡선 (실험 40과 같은 계산): {ana['planar']['verdict']['verdict']}" + (f" (기울기 {psl['slope']:+.3f})" if psl["slope"] is not None else ""))
    s3_figures(run_root / "figures", results, ana, ref, prior)
    s3_report(run_root / "report.md", results, cfg)
    s3_summary(run_root / "summary.md", results, ana, ref, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_41": cfg["code_41"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def s3_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "41_free_shape3d", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": S3_NOTE, "markers": cfg["markers"], "free_shape3d": cfg["free_shape3d"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v["marker"]
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "marker_tilt_valid_ratio": rd(v["marker_stats"].get("tilt_valid_ratio"))}
        if "f" in x:
            e["free_shape3d"] = {"f_3d_px": round(x["f"], 1), "f_planar_px": round(x["f_planar"], 1), "f_grid_px": round(x["f_grid"], 1), "at_grid_edge": x["at_grid_edge"], "at_bound": x["at_bound"], "relief_rel": rd(x["relief_rel"], 5),
                                 "relief_per_joint": [rd(q, 5) for q in x["relief_per_joint"]], "cost_3d": rd(x["cost"], 5), "cost_planar": rd(x["cost_planar"], 5), "ba3_rms_px": rd(x["ba3_rms_px"], 4), "ba3_success": x["ba3_success"],
                                 "ba3_nfev": x["ba3_nfev"], "n_frames_in_problem": x["n_frames_in_problem"], "elapsed_s": round(x["elapsed_s"])}
        else:
            e["free_shape3d"] = {"excluded": x.get("excluded") or x.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]
    if "slope" in a.get("shape3d", {}):
        s = a["shape3d"]["slope"]
        out["by_method"] = {"free_shape3d": {"f_by_distance_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]), "slope_interval_95": None if s["interval_95"] is None else [round(q, 4) for q in s["interval_95"]]}}
        ps = a["planar"]["slope"]
        out["by_method"]["planar_pooled_curves"] = {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in ps["pooled_f"].items()}, "slope": rd(ps["slope"]), "slope_interval_95": None if ps["interval_95"] is None else [round(q, 4) for q in ps["interval_95"]]}
    out["verdict"] = {"free_shape3d": {"verdict": a["shape3d"]["verdict"]["verdict"], "reason": a["shape3d"]["verdict"].get("reason")}}
    if "planar" in a:
        out["verdict"]["planar_pooled_curves"] = {"verdict": a["planar"]["verdict"]["verdict"]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p



# ====================================================================================================
# h3  (experiment 42: free_shape3d of 41 on the real-hand marker videos of 35c; the 35c tracks, the supervisor's call of 41, the verdict rule of 41, comparison with 35c)
#     python HandCalib/run.py free-shape3d-hand --config HandCalib/configs/42_free_shape3d_hand.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py free-shape3d-hand --config ... results-yaml --run runs/42_free_shape3d_hand/<UTC>
# ====================================================================================================
H3_KEYS = ("candidate", "exclusion", "detector_settings", "guide", "markers", "marker_exclusion", "free_shape3d")      # the config sections h3_video needs
H3_NOTE = ("실험 35c의 실제 손 마커 영상 6개(같은 추적 코드와 설정: MediaPipe 손 박스로 점 찾기와 이름 붙이기, 재획득 추적기)에 실험 41과 같은 호출(pb, idx = free_problem(trk, (W, H), ecfg, huber_rel, ids), "
           "estimate_run(pb, 'free_shape3d', grid, ecfg, sigma2))을 적용해 f를 구하고, 실험 35c의 fixed_template 결과와 비교한 실험이다. 판정은 실험 41과 같이 영상별 3차원 f의 거리-f 기울기(영상 단위 부트스트랩)로 하고, "
           "규칙과 쓰는 점, 설정은 실행 전에 고정했다. 기준 f와의 비교는 summary.md에만 있다.")


def h3_prior(cfg: dict) -> dict:
    """Per-video f and pooled f of experiment 35c's marker analysis (fixed_template on the tracked markers) and of its MediaPipe(8) analysis; read only."""
    out = {"markers": {"video": {}, "pooled": {}}, "mediapipe8": {"video": {}, "pooled": {}}}
    q = Path(str((cfg.get("compare") or {}).get("35c", "x"))).expanduser()
    q = q if q.is_absolute() else hf.resolve(q)
    if not q.exists():
        return out
    r = json.loads(q.read_text())
    for key in out:
        sl = (r.get("analysis", {}).get(key, {}).get("slope") or {}).get("pooled_f") or {}
        out[key] = {"video": {v["video"]: v[key]["f"] for v in r["videos"] if isinstance(v.get(key), dict) and "f" in v[key]}, "pooled": {float(d): f for d, f in sl.items() if f}}
    return out


def h3_figures(out: Path, results: list, ana: dict, ref: dict, prior: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="체커보드 (같은 날 거리 실험)")
    for key, col, lab in (("markers", "#2a6fbb", "실험 35c 마커 (fixed_template, 합동)"), ("mediapipe8", "#3a9a4a", "실험 35c MediaPipe 8점 (합동)")):
        if prior[key]["pooled"]:
            ax.plot(sorted(prior[key]["pooled"]), [prior[key]["pooled"][d] for d in sorted(prior[key]["pooled"])], "o-", color=col, label=lab)
    sl = ana.get("shape3d", {}).get("slope")
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o--", color="#c47a00", label="free_shape3d (영상별 f의 기하평균)")
    for r in results:
        if "f" in r["marker"]:
            ax.plot([r["distance_cm"]], [r["marker"]["f"]], "x", color="#c47a00", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)
    inc = [r for r in results if "f" in r["marker"]]
    if inc:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for r, col in zip(inc, ("#2a6fbb", "#5aa0e0", "#c47a00", "#e0a540", "#3a9a4a", "#7fcf8a")):
            ax.plot(r["marker"]["joint_ids"], r["marker"]["relief_per_joint"], "o-", color=col, label=f"{r['video'][4:8]} ({r['distance_cm']:g} cm)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel("점 번호")
        ax.set_ylabel("평면에서 벗어난 깊이 / 템플릿 RMS 반지름")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "relief_per_joint.png", dpi=110)
        plt.close(fig)


def h3_report(path: Path, results: list, cfg: dict):
    fs = cfg["free_shape3d"]
    L = ["# 실험 42: free_shape3d (실제 손 마커 영상 위에서)", "", H3_NOTE, "", f"추정에 쓰는 점 {[int(x) for x in cfg['markers']['used_ids']]}, free_shape3d(z_rel {fs['z_rel']}, alt_iters {fs['alt_iters']}, max_frames {fs['max_frames']}), f 격자 {cfg['pipeline']['estimate']['f_grid_n']}점.", "",
         "## 1. 영상별 값", "", "| 영상 | 거리(cm) | 포함 | f 3차원 (px) | f 평면 번들 조정 (px) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 | 3차원 RMS (px) | 3차원 반복 수 | 사용 프레임 | 걸린 시간(초) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r["marker"]
        if "f" in x:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 포함 | {x['f']:.1f} | {x['f_planar']:.1f} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} | {long_fmt(x['ba3_rms_px'], 3)} | {x['ba3_nfev']} | {x['n_frames_in_problem']} | {x['elapsed_s']:.0f} |")
        else:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | - | - | - | - | - | - | - | - |")
    L += ["", "![점마다 깊이 변위](figures/relief_per_joint.png)", "", "## 한계", "",
          "- 영상은 손 마커를 붙인 실제 손(실험 35c와 같은 영상)이고 거리마다 2개씩이다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 3차원 조정은 f에 대한 비용 곡선이 없어, 판정은 영상별 f의 기울기(영상 단위 부트스트랩)로 한다. 거리마다 영상이 2개라 구간이 거칠다. 3차원 f가 f 범위 한계에 붙었거나 평면 격자 최솟값이 격자 가장자리인 영상이 있으면 판단 불가이다(실험 41과 같은 규칙).",
          "- 점 추적과 사용 점 8개는 실험 35c의 설정을 그대로 썼다(점 찾기에는 MediaPipe 손 박스가 필요하다).", ""]
    path.write_text("\n".join(L) + "\n")


def h3_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict, prior: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["shape3d"]["verdict"]
    s_ = f"- **판정 (free_shape3d, 영상별 3차원 f 기준): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 거리별 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    sl = ana["shape3d"].get("slope")
    pl = ana.get("planar", {}).get("slope")
    L = ["# 실험 42 summary", "", f"**{H3_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 41과 같음): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 f(영상별 3차원 f의 기하평균)가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 거리에 영상이 없거나, 3차원 f가 f 범위 한계에 붙었거나 평면 격자 최솟값이 격자 가장자리인 영상이 있으면 판단 불가.", "",
         "## 1. 거리별 f", "", "| 거리(cm) | 영상 | free_shape3d (px) | 실험 35c 마커 합동 f (px) | 실험 35c MediaPipe 8점 합동 f (px) | 같은 거리 기준 f | shape3d 기준 f 대비 | 35c 마커 기준 f 대비 | 35c MediaPipe 8점 기준 f 대비 |", "|---|---|---|---|---|---|---|---|---|"]
    for d in (sl or {}).get("distances", []):
        f3, pm, pp = sl["pooled_f"][d], prior["markers"]["pooled"].get(float(d)), prior["mediapipe8"]["pooled"].get(float(d))
        n_in = sum(1 for r in results if r["distance_cm"] == d and "f" in r["marker"])
        n_all = sum(1 for r in results if r["distance_cm"] == d)
        L.append(f"| {float(d):g} | {n_in} / {n_all} | {f3:.1f}{' (한계)' if sl['pooled_edge'][d] else ''} | {format(pm, '.1f') if pm else '-'} | {format(pp, '.1f') if pp else '-'} | {ref[float(d)]:.1f} | {pct(f3 / ref[float(d)] - 1.0)} | {pct(None if not pm else pm / ref[float(d)] - 1.0)} | {pct(None if not pp else pp / ref[float(d)] - 1.0)} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| free_shape3d (영상별 3차원 f) | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |"]
    if pl:
        L.append(f"| 평면 free_template 합동 곡선 | {long_fmt(pl['slope'], 3)} | {'-' if pl['interval_95'] is None else format(pl['interval_95'][0], '+.3f') + ' ~ ' + format(pl['interval_95'][1], '+.3f')} |")
    for key, name in (("markers", "실험 35c 마커 (fixed_template)"), ("mediapipe8", "실험 35c MediaPipe 8점")):
        pp = prior[key]["pooled"]
        if len(pp) >= 2:
            L.append(f"| {name} | {long_fmt(hf.ext_loglog_slope(sorted(pp), [pp[d] for d in sorted(pp)]), 3)} | - |")
    L += [f"| 체커보드 | {long_fmt(hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)]), 3)} | - |", "", "![거리 대 f](figures/f_vs_distance_loglog.png)", "",
          "## 2. 영상별 f", "", "| 영상 | 거리(cm) | f 3차원 (px) | f 평면 번들 조정 (px) | 35c 마커 f (px) | 35c MediaPipe 8점 f (px) | 기준 f 대비 (3차원) | 기준 f 대비 (평면) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x = r["distance_cm"], r["marker"]
        fm, fp = prior["markers"]["video"].get(r["video"]), prior["mediapipe8"]["video"].get(r["video"])
        if "f" in x:
            L.append(f"| {r['video']} | {d:g} | {x['f']:.1f} | {x['f_planar']:.1f} | {format(fm, '.1f') if fm else '-'} | {format(fp, '.1f') if fp else '-'} | {pct(x['f'] / ref[d] - 1.0)} | {pct(x['f_planar'] / ref[d] - 1.0)} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} |")
        else:
            L.append(f"| {r['video']} | {d:g} | 제외: {'; '.join(x.get('excluded') or []) or x.get('failed') or '제외'} | - | {format(fm, '.1f') if fm else '-'} | {format(fp, '.1f') if fp else '-'} | - | - | - | - | - |")
    L += ["", "![점마다 깊이 변위](figures/relief_per_joint.png)", "", "## 한계", "", "- 영상은 거리마다 2개씩이라 기울기 구간이 거칠고, 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 기준 f는 실험 35c 때 체커보드 거리 실험의 값(사용자 제공)이다.", "- 점 추적은 실험 35c의 코드와 설정이다(MediaPipe 손 박스를 점 찾기와 이름 붙이기에 쓴다).", ""]
    path.write_text("\n".join(L) + "\n")


def h3_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/42_free_shape3d_hand.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(h3_results_yaml(hf.resolve(a.run), cfg))
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
    log(f"42 free_shape3d 실제 손 영상 공식 실행: 영상 {len(entries)}개, 추정에 쓰는 점 {cfg['markers']['used_ids']} → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    c = {k: cfg[k] for k in H3_KEYS}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = hf.fresp_detect_many([((e["file"], "orig"), p["frames"]) for e, p in zip(entries, preps)], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            results = list(ex.map(hf.h3_video, [(e, p, found[(e["file"], "orig")], cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    prior = h3_prior(cfg)
    for r in results:
        x = r["marker"]
        log(f"  {r['video']} ({r['distance_cm']:g} cm): " + (f"free_shape3d f {x['f']:.1f} px (평면 {x['f_planar']:.1f}, 상대 깊이 변위 {x['relief_rel']:.4f}{', f 한계에 붙음' if x['at_bound'] else ''}{', 평면 격자 가장자리' if x['at_grid_edge'] else ''}), {x['elapsed_s']:.0f} s" if "f" in x else "제외")
            + (f" [제외: {'; '.join(x['excluded'])}]" if x.get("excluded") else "") + (f" [{x['failed']}]" if x.get("failed") else ""))
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [r for r in results if "f" in r["marker"]]
    ds = sorted({r["distance_cm"] for r in inc})
    ana = {}
    if not ds:
        ana["shape3d"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        fb = {d: [r["marker"]["f"] for r in inc if r["distance_cm"] == d] for d in ds}
        eb = {d: [r["marker"]["at_bound"] or r["marker"]["at_grid_edge"] for r in inc if r["distance_cm"] == d] for d in ds}      # as in 41: the 3-D f on a bound, or the planar grid minimum on the grid edge
        sl = hf.s3_slope(fb, eb, bn, bs)
        nv = {d: len(fb.get(d, [])) for d in ref}
        ana["shape3d"] = {"slope": sl, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        grid = np.array(inc[0]["marker"]["curve"]["grid"])
        cur = {d: [r["marker"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in ds}
        psl = hf.pr_slope(cur, grid, bn, bs)
        ana["planar"] = {"slope": psl, "verdict": hf.pr_verdict(psl, ref, cfg["verdict"], nv)}
        log(f"  free_shape3d: {ana['shape3d']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else "") + (f" [{ana['shape3d']['verdict'].get('reason')}]" if ana["shape3d"]["verdict"].get("reason") else ""))
        log(f"  평면 합동 곡선: {ana['planar']['verdict']['verdict']}" + (f" (기울기 {psl['slope']:+.3f})" if psl["slope"] is not None else ""))
    h3_figures(run_root / "figures", results, ana, ref, prior)
    h3_report(run_root / "report.md", results, cfg)
    h3_summary(run_root / "summary.md", results, ana, ref, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_42": cfg["code_42"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def h3_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "42_free_shape3d_hand", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": H3_NOTE, "markers": cfg["markers"], "free_shape3d": cfg["free_shape3d"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v["marker"]
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "marker_tilt_valid_ratio": rd(v["marker_stats"].get("tilt_valid_ratio"))}
        if "f" in x:
            e["free_shape3d"] = {"f_3d_px": round(x["f"], 1), "f_planar_px": round(x["f_planar"], 1), "f_grid_px": round(x["f_grid"], 1), "at_grid_edge": x["at_grid_edge"], "at_bound": x["at_bound"], "relief_rel": rd(x["relief_rel"], 5),
                                 "relief_per_joint": [rd(q, 5) for q in x["relief_per_joint"]], "cost_3d": rd(x["cost"], 5), "cost_planar": rd(x["cost_planar"], 5), "ba3_rms_px": rd(x["ba3_rms_px"], 4), "ba3_success": x["ba3_success"],
                                 "ba3_nfev": x["ba3_nfev"], "n_frames_in_problem": x["n_frames_in_problem"], "elapsed_s": round(x["elapsed_s"])}
        else:
            e["free_shape3d"] = {"excluded": x.get("excluded") or x.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]
    if "slope" in a.get("shape3d", {}):
        s = a["shape3d"]["slope"]
        out["by_method"] = {"free_shape3d": {"f_by_distance_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]), "slope_interval_95": None if s["interval_95"] is None else [round(q, 4) for q in s["interval_95"]]}}
        ps = a["planar"]["slope"]
        out["by_method"]["planar_pooled_curves"] = {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in ps["pooled_f"].items()}, "slope": rd(ps["slope"]), "slope_interval_95": None if ps["interval_95"] is None else [round(q, 4) for q in ps["interval_95"]]}
    out["verdict"] = {"free_shape3d": {"verdict": a["shape3d"]["verdict"]["verdict"], "reason": a["shape3d"]["verdict"].get("reason")}}
    if "planar" in a:
        out["verdict"]["planar_pooled_curves"] = {"verdict": a["planar"]["verdict"]["verdict"]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p



# ====================================================================================================
# e3  (experiment 45: free_shape3d on the real-hand videos of 27 with the MediaPipe joints; the verdict rule of 41 on the right-hand distance videos (G6), the left hand (G7) reported beside it)
#     python HandCalib/run.py free-shape3d-27 --config HandCalib/configs/45_free_shape3d_27.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py free-shape3d-27 --config ... results-yaml --run runs/45_free_shape3d_27/<UTC>
# ====================================================================================================
E3_KEYS = ("candidate", "exclusion", "detector_settings", "free_shape3d")      # the config sections e3_video needs
E3_NOTE = ("실험 27의 실제 손 영상(MediaPipe 관절 16점, 손목 제외, 오른손 거리 영상 G6와 왼손 영상 G7)에 교수님의 free_shape3d를 실험 41과 같은 호출(pb, idx = free_problem(trk, (W, H), ecfg, huber_rel, ids), "
           "estimate_run(pb, 'free_shape3d', grid, ecfg, sigma2))로 적용해 f를 구하고, 실험 27의 fixed_template 값과 비교한 실험이다. 프린트 손(실험 34)과 태블릿 화면 영상은 쓰지 않았다. "
           "판정은 실험 41과 같은 규칙(영상별 3차원 f의 거리-f 기울기, 영상 단위 부트스트랩)을 G6에 적용하고, 규칙과 쓰는 점, 설정은 실행 전에 고정했다. 기준 f와의 비교는 summary.md에만 있다.")


def e3_prior(cfg: dict) -> dict:
    """Per-video f of experiment 27 (fixed_template on the MediaPipe joints; read only)."""
    q = Path(str((cfg.get("compare") or {}).get("27", "x"))).expanduser()
    q = q if q.is_absolute() else hf.resolve(q)
    if not q.exists():
        return {}
    r = json.loads(q.read_text())
    return {v["video"]: v["ft"]["f"] for v in r["videos"] if isinstance(v.get("ft"), dict) and "f" in v["ft"]}


def e3_gm(vals):
    return float(np.exp(np.mean(np.log(np.asarray(vals, float))))) if len(vals) else None


def e3_figures(out: Path, results: list, ana: dict, ref: dict, prior: dict):
    plt = _plt()
    ds = sorted(ref)
    g6 = [r for r in results if r["group"] == "G6"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="체커보드 (같은 날 거리 실험)")
    p27 = {d: e3_gm([prior[r["video"]] for r in g6 if r["distance_cm"] == d and r["video"] in prior]) for d in ds}
    if all(p27.values()):
        ax.plot(ds, [p27[d] for d in ds], "o-", color="#2a6fbb", label="실험 27 (fixed_template, 영상별 f의 기하평균)")
    sl = ana.get("shape3d", {}).get("slope")
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o--", color="#c47a00", label="free_shape3d (영상별 f의 기하평균)")
    for r in g6:
        if "est" in r:
            ax.plot([r["distance_cm"]], [r["est"]["f"]], "x", color="#c47a00", alpha=0.7)
        if r["video"] in prior:
            ax.plot([r["distance_cm"]], [prior[r["video"]]], "+", color="#2a6fbb", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)


def e3_report(path: Path, results: list, cfg: dict):
    fs = cfg["free_shape3d"]
    L = ["# 실험 45: free_shape3d (실험 27의 실제 손 영상, MediaPipe 관절)", "", E3_NOTE, "", f"free_shape3d(z_rel {fs['z_rel']}, alt_iters {fs['alt_iters']}, max_frames {fs['max_frames']}), f 격자 {cfg['pipeline']['estimate']['f_grid_n']}점.", "",
         "## 1. 영상별 값", "", "| 영상 | 그룹 | 거리(cm) | 손 | 포함 | f 3차원 (px) | f 평면 번들 조정 (px) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 | 3차원 RMS (px) | 3차원 반복 수 | 사용 프레임 | 걸린 시간(초) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r.get("est")
        if x:
            L.append(f"| {r['video']} | {r['group']} | {r['distance_cm']:g} | {r['hand']} | 포함 | {x['f']:.1f} | {x['f_planar']:.1f} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} | {long_fmt(x['ba3_rms_px'], 3)} | {x['ba3_nfev']} | {x['n_frames_in_problem']} | {x['elapsed_s']:.0f} |")
        else:
            L.append(f"| {r['video']} | {r['group']} | {r['distance_cm']:g} | {r['hand']} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | - | - | - | - | - | - | - | - |")
    L += ["", "## 한계", "", "- G6(오른손, 50/60/75 cm)은 거리마다 영상 3개(제외된 영상은 빠짐), G7(왼손, 60 cm)은 3개이다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- MediaPipe 관절은 점 위치 오차가 커서 f가 영상마다 크게 달라질 수 있다(실험 27의 f도 영상별로 1298~2231 px였다).",
          "- 3차원 조정은 f에 대한 비용 곡선이 없어, 판정은 영상별 f의 기울기(영상 단위 부트스트랩)로 한다.", ""]
    path.write_text("\n".join(L) + "\n")


def e3_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict, prior: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["shape3d"]["verdict"]
    s_ = f"- **판정 (G6 오른손 거리 영상, free_shape3d 영상별 3차원 f 기준): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 거리별 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    sl = ana["shape3d"].get("slope")
    pl = ana.get("planar", {}).get("slope")
    g6 = [r for r in results if r["group"] == "G6"]
    g7 = [r for r in results if r["group"] == "G7"]
    L = ["# 실험 45 summary", "", f"**{E3_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 41과 같음): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 f(영상별 3차원 f의 기하평균)가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 거리에 영상이 없거나, 3차원 f가 f 범위 한계에 붙었거나 평면 격자 최솟값이 격자 가장자리인 영상이 있으면 판단 불가.", "",
         "## 1. 거리별 f (G6, 오른손)", "", "| 거리(cm) | 영상 | free_shape3d (px) | 실험 27 f의 기하평균 (px) | 같은 거리 기준 f (체커보드) | shape3d 기준 f 대비 | 실험 27 기준 f 대비 | shape3d 1680 대비 |", "|---|---|---|---|---|---|---|---|"]
    for d in (sl or {}).get("distances", []):
        f3 = sl["pooled_f"][d]
        p27 = e3_gm([prior[r["video"]] for r in g6 if r["distance_cm"] == d and r["video"] in prior and "est" in r])
        n_in = sum(1 for r in g6 if r["distance_cm"] == d and "est" in r)
        n_all = sum(1 for r in g6 if r["distance_cm"] == d)
        L.append(f"| {float(d):g} | {n_in} / {n_all} | {f3:.1f}{' (한계)' if sl['pooled_edge'][d] else ''} | {format(p27, '.1f') if p27 else '-'} | {ref[float(d)]:.1f} | {pct(f3 / ref[float(d)] - 1.0)} | {pct(None if not p27 else p27 / ref[float(d)] - 1.0)} | {pct(f3 / 1680.0 - 1.0)} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| free_shape3d (영상별 3차원 f) | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |"]
    if pl:
        L.append(f"| 평면 free_template 합동 곡선 | {long_fmt(pl['slope'], 3)} | {'-' if pl['interval_95'] is None else format(pl['interval_95'][0], '+.3f') + ' ~ ' + format(pl['interval_95'][1], '+.3f')} |")
    ds27 = sorted({r["distance_cm"] for r in g6 if r["video"] in prior and "est" in r})
    if len(ds27) >= 2:
        g27 = [e3_gm([prior[r["video"]] for r in g6 if r["distance_cm"] == d and r["video"] in prior and "est" in r]) for d in ds27]
        L.append(f"| 실험 27 (fixed_template, 같은 영상 f의 기하평균) | {long_fmt(hf.ext_loglog_slope(ds27, g27), 3)} | - |")
    L += [f"| 체커보드 | {long_fmt(hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)]), 3)} | - |", "", "![거리 대 f](figures/f_vs_distance_loglog.png)", "",
          "## 2. 영상별 f", "", "| 영상 | 그룹 | 거리(cm) | f 3차원 (px) | f 평면 번들 조정 (px) | 실험 27 f (px) | 기준 f 대비 (3차원) | 기준 f 대비 (실험 27) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x, f27 = r["distance_cm"], r.get("est"), prior.get(r["video"])
        rf = ref.get(d, 1680.0)
        if x:
            L.append(f"| {r['video']} | {r['group']} | {d:g} | {x['f']:.1f} | {x['f_planar']:.1f} | {format(f27, '.1f') if f27 else '-'} | {pct(x['f'] / rf - 1.0)} | {pct(None if not f27 else f27 / rf - 1.0)} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} |")
        else:
            L.append(f"| {r['video']} | {r['group']} | {d:g} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | {format(f27, '.1f') if f27 else '-'} | - | - | - | - | - |")
    g7e = [r for r in g7 if "est" in r]
    if g7e:
        gm7, gm27 = e3_gm([r["est"]["f"] for r in g7e]), e3_gm([prior[r["video"]] for r in g7e if r["video"] in prior])
        L += ["", f"- 왼손(G7, 60 cm) 영상 {len(g7e)}개: free_shape3d f의 기하평균 {gm7:.1f} px (기준 f {ref[60.0]:.1f} 대비 {pct(gm7 / ref[60.0] - 1.0)}), 실험 27 f의 기하평균 {'-' if not gm27 else format(gm27, '.1f')} px. 판정에는 쓰지 않았다."]
    L += ["", "## 한계", "", "- 기준 f는 실험 35c 때 체커보드 거리 실험의 값(사용자 제공)이며 이 영상 세션에서 새로 재지 않았다. 1680은 실험 27 설정에 적힌 공칭값이다.",
          "- 거리마다 영상이 적어 기울기 구간이 거칠고, MediaPipe 점 오차가 커서 영상별 f의 흩어짐이 크다.", ""]
    path.write_text("\n".join(L) + "\n")


def e3_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/45_free_shape3d_27.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(e3_results_yaml(hf.resolve(a.run), cfg))
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
    log(f"45 free_shape3d 실험 27 영상 공식 실행: 영상 {len(entries)}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    c = {k: cfg[k] for k in E3_KEYS}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            results = list(ex.map(hf.e3_video, [(e, p, found[(p["video"], "orig")], cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    prior = e3_prior(cfg)
    for r in results:
        x = r.get("est")
        log(f"  {r['video']} ({r['group']}, {r['distance_cm']:g} cm, {r['hand']}): " + (f"free_shape3d f {x['f']:.1f} px (평면 {x['f_planar']:.1f}, 상대 깊이 변위 {x['relief_rel']:.4f}{', f 한계에 붙음' if x['at_bound'] else ''}{', 평면 격자 가장자리' if x['at_grid_edge'] else ''}), {x['elapsed_s']:.0f} s" if x else "제외")
            + (f" [제외: {'; '.join(r['excluded'])}]" if r.get("excluded") else "") + (f" [{r['failed']}]" if r.get("failed") else ""))
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [r for r in results if "est" in r and r["group"] == "G6"]
    ds = sorted({r["distance_cm"] for r in inc})
    ana = {}
    if not ds:
        ana["shape3d"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in ds}
        eb = {d: [r["est"]["at_bound"] or r["est"]["at_grid_edge"] for r in inc if r["distance_cm"] == d] for d in ds}
        sl = hf.s3_slope(fb, eb, bn, bs)
        nv = {d: len(fb.get(d, [])) for d in ref}
        ana["shape3d"] = {"slope": sl, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        grid = np.array(inc[0]["est"]["curve"]["grid"])
        cur = {d: [r["est"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in ds}
        psl = hf.pr_slope(cur, grid, bn, bs)
        ana["planar"] = {"slope": psl, "verdict": hf.pr_verdict(psl, ref, cfg["verdict"], nv)}
        log(f"  free_shape3d (G6): {ana['shape3d']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else "") + (f" [{ana['shape3d']['verdict'].get('reason')}]" if ana["shape3d"]["verdict"].get("reason") else ""))
        log(f"  평면 합동 곡선 (G6): {ana['planar']['verdict']['verdict']}" + (f" (기울기 {psl['slope']:+.3f})" if psl["slope"] is not None else ""))
    e3_figures(run_root / "figures", results, ana, ref, prior)
    e3_report(run_root / "report.md", results, cfg)
    e3_summary(run_root / "summary.md", results, ana, ref, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_45": cfg["code_45"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def e3_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": "45_free_shape3d_27", "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": E3_NOTE, "candidate": cfg["candidate"], "free_shape3d": cfg["free_shape3d"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v.get("est")
        e = {"group": v["group"], "distance_cm": v["distance_cm"], "hand": v["hand"], "duration_s": rd(v["duration_s"], 2), "detect_rate": rd(v["detect_rate"]), "edge_rate": rd(v["edge_rate"])}
        if x:
            e["free_shape3d"] = {"f_3d_px": round(x["f"], 1), "f_planar_px": round(x["f_planar"], 1), "f_grid_px": round(x["f_grid"], 1), "at_grid_edge": x["at_grid_edge"], "at_bound": x["at_bound"], "relief_rel": rd(x["relief_rel"], 5),
                                 "relief_per_joint": [rd(q, 5) for q in x["relief_per_joint"]], "cost_3d": rd(x["cost"], 5), "cost_planar": rd(x["cost_planar"], 5), "ba3_rms_px": rd(x["ba3_rms_px"], 4), "ba3_success": x["ba3_success"],
                                 "ba3_nfev": x["ba3_nfev"], "n_frames_in_problem": x["n_frames_in_problem"], "elapsed_s": round(x["elapsed_s"])}
        else:
            e["free_shape3d"] = {"excluded": v.get("excluded") or v.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]
    if "slope" in a.get("shape3d", {}):
        s = a["shape3d"]["slope"]
        out["by_method"] = {"free_shape3d_G6": {"f_by_distance_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]), "slope_interval_95": None if s["interval_95"] is None else [round(q, 4) for q in s["interval_95"]]}}
        ps = a["planar"]["slope"]
        out["by_method"]["planar_pooled_curves_G6"] = {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in ps["pooled_f"].items()}, "slope": rd(ps["slope"]), "slope_interval_95": None if ps["interval_95"] is None else [round(q, 4) for q in ps["interval_95"]]}
    out["verdict"] = {"free_shape3d_G6": {"verdict": a["shape3d"]["verdict"]["verdict"], "reason": a["shape3d"]["verdict"].get("reason")}}
    if "planar" in a:
        out["verdict"]["planar_pooled_curves_G6"] = {"verdict": a["planar"]["verdict"]["verdict"]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# e4  (experiment 46: the same call as 45 (e3_video, free_shape3d on the MediaPipe joints, wrist removed) on 12 newly shot videos, 4 per distance 50 / 60 / 75 cm; one reference f for every distance)
#     python HandCalib/run.py free-shape3d-new --config HandCalib/configs/46_free_shape3d_new.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py free-shape3d-new --config ... results-yaml --run runs/46_free_shape3d_new/<UTC>
# ====================================================================================================
E4_KEYS = E3_KEYS
E4_NOTE = ("새로 촬영한 영상 12개(IMG_7298-7310 중 12개, 50/60/75 cm 거리마다 4개)에 실험 45와 같은 조건을 적용한 실험이다: MediaPipe 관절 16점(손목 제외), 교수님의 free_shape3d를 실험 41·45와 같은 호출로 사용, "
           "제외 규칙(정면 창, 가장자리 접촉 5%, 손 검출 실패 20%)과 판정 규칙(영상별 3차원 f의 거리-f 기울기, 영상 단위 부트스트랩)도 실험 45와 같다. 규칙과 설정은 실행 전에 고정했다. 기준 f와의 비교는 summary.md에만 있다.")


def e4_prior(cfg: dict) -> dict:
    """Pooled 3-D f per distance of experiment 45 (read only; empty when the file is missing)."""
    q = Path(str((cfg.get("compare") or {}).get("45", "x"))).expanduser()
    q = q if q.is_absolute() else hf.resolve(q)
    if not q.exists():
        return {}
    r = json.loads(q.read_text())
    pf = (((r.get("analysis") or {}).get("shape3d") or {}).get("slope") or {}).get("pooled_f") or {}
    return {float(d): float(f) for d, f in pf.items()}


def e4_figures(out: Path, results: list, ana: dict, ref: dict, prior: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="기준 f")
    if prior and all(d in prior for d in ds):
        ax.plot(ds, [prior[d] for d in ds], "o-", color="#2a6fbb", label="실험 45 (free_shape3d, 영상별 f의 기하평균)")
    sl = ana.get("shape3d", {}).get("slope")
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o--", color="#c47a00", label="실험 46 free_shape3d (영상별 f의 기하평균)")
    for r in results:
        if "est" in r:
            ax.plot([r["distance_cm"]], [r["est"]["f"]], "x", color="#c47a00", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)


def e4_report(path: Path, results: list, cfg: dict):
    fs = cfg["free_shape3d"]
    L = ["# 실험 46: free_shape3d (새로 촬영한 영상 12개, MediaPipe 관절)", "", E4_NOTE, "", f"free_shape3d(z_rel {fs['z_rel']}, alt_iters {fs['alt_iters']}, max_frames {fs['max_frames']}), f 격자 {cfg['pipeline']['estimate']['f_grid_n']}점.", "",
         "## 1. 영상별 값", "", "| 영상 | 거리(cm) | 포함 | f 3차원 (px) | f 평면 번들 조정 (px) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 | 3차원 RMS (px) | 3차원 반복 수 | 사용 프레임 | 걸린 시간(초) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r.get("est")
        if x:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 포함 | {x['f']:.1f} | {x['f_planar']:.1f} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} | {long_fmt(x['ba3_rms_px'], 3)} | {x['ba3_nfev']} | {x['n_frames_in_problem']} | {x['elapsed_s']:.0f} |")
        else:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | - | - | - | - | - | - | - | - |")
    L += ["", "## 한계", "", "- 거리마다 영상 4개(제외된 영상은 빠짐)이다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- MediaPipe 관절은 점 위치 오차가 커서 f가 영상마다 크게 달라질 수 있다.",
          "- 3차원 조정은 f에 대한 비용 곡선이 없어, 판정은 영상별 f의 기울기(영상 단위 부트스트랩)로 한다.", ""]
    path.write_text("\n".join(L) + "\n")


def e4_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict, prior: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["shape3d"]["verdict"]
    s_ = f"- **판정 (free_shape3d 영상별 3차원 f 기준): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 거리별 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    sl = ana["shape3d"].get("slope")
    pl = ana.get("planar", {}).get("slope")
    rf0 = ref[sorted(ref)[0]]
    L = ["# 실험 46 summary", "", f"**{E4_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 41·45와 같음): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 f(영상별 3차원 f의 기하평균)가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 한 거리에 영상이 없거나 3차원 f가 격자 한계에 붙거나 평면 격자 최솟값이 가장자리이면 판단 불가.", "",
         f"## 1. 거리별 f (기준 f {rf0:.0f} px)", "", "| 거리(cm) | 영상 | free_shape3d (px) | 기준 f | 기준 f 대비 | 실험 45 (px) |", "|---|---|---|---|---|---|"]
    for d in (sl or {}).get("distances", []):
        f3 = sl["pooled_f"][d]
        n_in = sum(1 for r in results if r["distance_cm"] == d and "est" in r)
        n_all = sum(1 for r in results if r["distance_cm"] == d)
        p45 = prior.get(float(d))
        L.append(f"| {float(d):g} | {n_in} / {n_all} | {f3:.1f}{' (한계)' if sl['pooled_edge'][d] else ''} | {ref[float(d)]:.1f} | {pct(f3 / ref[float(d)] - 1.0)} | {format(p45, '.1f') if p45 else '-'} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| free_shape3d (영상별 3차원 f) | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |"]
    if pl:
        L.append(f"| 평면 free_template 합동 곡선 | {long_fmt(pl['slope'], 3)} | {'-' if pl['interval_95'] is None else format(pl['interval_95'][0], '+.3f') + ' ~ ' + format(pl['interval_95'][1], '+.3f')} |")
    if len(prior) >= 2:
        L.append(f"| 실험 45 (영상이 다름, free_shape3d) | {long_fmt(hf.ext_loglog_slope(sorted(prior), [prior[d] for d in sorted(prior)]), 3)} | - |")
    L += [f"| 기준 f | {long_fmt(hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)]), 3)} | - |", "", "![거리 대 f](figures/f_vs_distance_loglog.png)", "",
          "## 2. 영상별 f", "", "| 영상 | 거리(cm) | f 3차원 (px) | f 평면 번들 조정 (px) | 기준 f 대비 (3차원) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x = r["distance_cm"], r.get("est")
        rf = ref.get(d, rf0)
        if x:
            L.append(f"| {r['video']} | {d:g} | {x['f']:.1f} | {x['f_planar']:.1f} | {pct(x['f'] / rf - 1.0)} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} |")
        else:
            L.append(f"| {r['video']} | {d:g} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | - | - | - | - |")
    L += ["", "## 한계", "", f"- 기준 f {rf0:.0f} px는 사용자가 이번 영상들에 대해 준 값이며 이 코드로 새로 재지 않았다. 실험 45의 기준 f(체커보드 1660.6 / 1653.7 / 1644.9)와 다르다.",
          "- 거리마다 영상이 4개 이하라 기울기 구간이 거칠고, MediaPipe 점 오차가 커서 영상별 f의 흩어짐이 크다. 실험 45와는 영상이 달라 값을 직접 비교하기 어렵다.", ""]
    path.write_text("\n".join(L) + "\n")


def e4_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/46_free_shape3d_new.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(e4_results_yaml(hf.resolve(a.run), cfg))
        return None
    entries = [{"group": "G8", "hand": "unrecorded", **e} for e in cfg["videos"]]
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"46 free_shape3d 새 영상 공식 실행: 영상 {len(entries)}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    c = {k: cfg[k] for k in E4_KEYS}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            results = list(ex.map(hf.e3_video, [(e, p, found[(p["video"], "orig")], cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    prior = e4_prior(cfg)
    for r in results:
        x = r.get("est")
        log(f"  {r['video']} ({r['distance_cm']:g} cm): " + (f"free_shape3d f {x['f']:.1f} px (평면 {x['f_planar']:.1f}, 상대 깊이 변위 {x['relief_rel']:.4f}{', f 한계에 붙음' if x['at_bound'] else ''}{', 평면 격자 최솟값이 가장자리' if x['at_grid_edge'] else ''})" if x else "추정 없음")
            + (f" [제외: {'; '.join(r['excluded'])}]" if r.get("excluded") else "") + (f" [{r['failed']}]" if r.get("failed") else ""))
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [r for r in results if "est" in r]
    ds = sorted({r["distance_cm"] for r in inc})
    ana = {}
    if not ds:
        ana["shape3d"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in ds}
        eb = {d: [r["est"]["at_bound"] or r["est"]["at_grid_edge"] for r in inc if r["distance_cm"] == d] for d in ds}
        sl = hf.s3_slope(fb, eb, bn, bs)
        nv = {d: len(fb.get(d, [])) for d in ref}
        ana["shape3d"] = {"slope": sl, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        grid = np.array(inc[0]["est"]["curve"]["grid"])
        cur = {d: [r["est"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in ds}
        psl = hf.pr_slope(cur, grid, bn, bs)
        ana["planar"] = {"slope": psl, "verdict": hf.pr_verdict(psl, ref, cfg["verdict"], nv)}
        log(f"  free_shape3d: {ana['shape3d']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else ""))
        log(f"  평면 합동 곡선: {ana['planar']['verdict']['verdict']}" + (f" (기울기 {psl['slope']:+.3f})" if psl["slope"] is not None else ""))
    e4_figures(run_root / "figures", results, ana, ref, prior)
    e4_report(run_root / "report.md", results, cfg)
    e4_summary(run_root / "summary.md", results, ana, ref, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_46": cfg["code_46"], "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def e4_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": cfg["experiment"]["name"], "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": E4_NOTE, "candidate": cfg["candidate"], "free_shape3d": cfg["free_shape3d"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v.get("est")
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "detect_rate": rd(v["detect_rate"]), "edge_rate": rd(v["edge_rate"])}
        if x:
            e["free_shape3d"] = {"f_3d_px": round(x["f"], 1), "f_planar_px": round(x["f_planar"], 1), "f_grid_px": round(x["f_grid"], 1), "at_grid_edge": x["at_grid_edge"], "at_bound": x["at_bound"], "relief_rel": rd(x["relief_rel"], 5),
                                 "relief_per_joint": [rd(q, 5) for q in x["relief_per_joint"]], "cost_3d": rd(x["cost"], 5), "cost_planar": rd(x["cost_planar"], 5), "ba3_rms_px": rd(x["ba3_rms_px"], 4), "ba3_success": x["ba3_success"],
                                 "ba3_nfev": x["ba3_nfev"], "n_frames_in_problem": x["n_frames_in_problem"], "elapsed_s": round(x["elapsed_s"])}
        else:
            e["free_shape3d"] = {"excluded": v.get("excluded") or v.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]
    if "slope" in a.get("shape3d", {}):
        s = a["shape3d"]["slope"]
        out["by_method"] = {"free_shape3d": {"f_by_distance_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]), "slope_interval_95": None if s["interval_95"] is None else [round(q, 4) for q in s["interval_95"]]}}
        ps = a["planar"]["slope"]
        out["by_method"]["planar_pooled_curves"] = {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in ps["pooled_f"].items()}, "slope": rd(ps["slope"]), "slope_interval_95": None if ps["interval_95"] is None else [round(q, 4) for q in ps["interval_95"]]}
    out["verdict"] = {"free_shape3d": {"verdict": a["shape3d"]["verdict"]["verdict"], "reason": a["shape3d"]["verdict"].get("reason")}}
    if "planar" in a:
        out["verdict"]["planar_pooled_curves"] = {"verdict": a["planar"]["verdict"]["verdict"]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p



# ====================================================================================================
# mm  (experiment 47: the five MMPose COCO-WholeBody-Hand models as detectors on the boxes of an existing detection cache; verification, then (run_stage2) f on existing videos)
#     python HandCalib/run.py mmpose-hand --config HandCalib/configs/47_mmpose_hand_detectors.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py mmpose-hand --config ... results-yaml --run runs/47_mmpose_hand_detectors/<UTC>
# ====================================================================================================
from detector_worker import MM_HAND_NAMES                             # noqa: E402  (the cache's 21-point names, numpy-only module)
MM_KEYS = ("candidate", "exclusion", "detector_settings", "guide")       # the config sections mm_video needs
MM_NOTE = ("MMPose의 COCO-WholeBody-Hand 학습 손 2D 모델 5종(HRNetv2-w18+DARK, HRNetv2-w18, RTMPose-m, Hourglass52, SCNet-50)을 검출기로 추가하고, 기존 검출 캐시의 손 박스에서 관절을 구해 "
           "기존 FT 코드(후보 설정: 손목 제외, fixed_template)로 f를 기록한 실험이다. 판정은 하지 않는다. 기준 f와의 비교는 summary.md에만 있다.")


def mm_det_settings(cfg: dict, name: str, gpu=None) -> dict:
    s = dict(cfg["detector_settings"][name])
    s["box_source"] = cfg["hand_box_source"]                       # one box source for the five models (yaml hand_box_source)
    if gpu is not None:
        s["device"] = f"cuda:{int(gpu)}"
    return s


def mm_gpu_jobs(jobs: list, cfg: dict, log) -> list:
    """jobs = [callable(gpu) -> result]; len(gpu_ids) x jobs_per_gpu threads, each bound to one GPU, take jobs from one queue; the thermal guard runs before every job."""
    import queue
    import threading
    gpus = [int(g) for g in cfg["gpu_ids"]]
    slots = [g for g in gpus for _ in range(int(cfg.get("jobs_per_gpu", 1)))]
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    out = [None] * len(jobs)

    def worker(g):
        while True:
            try:
                i, j = q.get_nowait()
            except queue.Empty:
                return
            hf.mm_thermal_wait(g, cfg["thermal"], log)
            try:
                out[i] = j(g)
            except Exception as e:                                      # recorded per job
                out[i] = e
    th = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def mm_existing_f(cfg: dict) -> dict:
    """Official mediapipe f, sharpness and FT curve of every video in the existing runs (read only): {video: {f, sharpness, at_grid_edge, curve}}."""
    out = {}
    for src, p in (cfg.get("existing_results") or {}).items():
        q = Path(str(p)).expanduser()
        q = q if q.is_absolute() else hf.resolve(q)
        if not q.exists():
            continue
        r = json.loads(q.read_text())
        for v in r.get("videos") or r.get("new") or []:
            if isinstance(v.get("ft"), dict) and "f" in v["ft"]:
                out[v["video"]] = {"source": str(src), "f": float(v["ft"]["f"]), "sharpness": v["ft"].get("sharpness"), "at_grid_edge": v["ft"].get("at_grid_edge"),
                                   "curve": (v.get("curve") or {}).get("cost")}
    return out


def mm_stage2_slopes(stage2: list, names: list, cfg: dict, existing: dict) -> dict:
    """Experiment 27's distance videos: pooled f per distance and the log f / log distance slope (pr_slope, video-level bootstrap) for every detector and for 27's official values."""
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    out = {}
    g6 = [r for r in stage2 if r["source"] == "27" and r.get("distance_cm") is not None]
    grid = None
    for name in names:
        cur = {}
        for r in g6:
            x = r["detectors"].get(name) or {}
            if "ft" in x and not x.get("excluded"):
                cur.setdefault(float(r["distance_cm"]), []).append(x["curve"]["cost"])
                grid = np.array(x["curve"]["grid"])
        out[name] = hf.pr_slope(cur, grid, bn, bs) if cur and len(cur) >= 2 else None
        if out[name] is not None:
            out[name]["n_videos"] = {d: len(v) for d, v in cur.items()}
    cur = {}
    for r in g6:
        e = existing.get(r["video"])
        x = r["detectors"].get("mediapipe_hands") or {}
        if e and e.get("curve") and not x.get("excluded"):
            cur.setdefault(float(r["distance_cm"]), []).append(e["curve"])
    out["실험 27 공식 (mediapipe_hands)"] = hf.pr_slope(cur, grid, bn, bs) if cur and len(cur) >= 2 and grid is not None else None
    return out


def mm_report(path: Path, cfg: dict, env: dict, checks: dict, agree: dict, order: dict, fail: dict, overlays: list, verified: dict):
    names = list(cfg["detectors"])
    src = cfg["hand_box_source"]
    L = ["# 실험 47: MMPose 손 2D 모델 5종을 검출기로 추가", "", MM_NOTE, "",
         "## 1. 환경과 체크포인트", "",
         f"- 가상환경: {cfg['detector_settings'][names[0]]['python']} (" + ", ".join(f"{k} {v}" for k, v in env.get("versions", {}).items()) + ")",
         f"- 사용 GPU 번호: {', '.join(str(g) for g in cfg['gpu_ids'])} (GPU당 동시 작업 {cfg.get('jobs_per_gpu', 1)}개), 배치 크기 {cfg['detector_settings'][names[0]]['batch_size']}, "
         f"TF32 {'켬' if cfg['detector_settings'][names[0]].get('tf32') else '끔 (완전한 FP32)'}",
         f"- 손 박스 출처: {src} (기존 검출 캐시에서 읽음, 새로 검출하지 않음). 박스가 없는 프레임은 다섯 모델 모두 검출 실패.", "",
         "| 검출기 | 모델 | COCO-WholeBody-Hand EPE | 체크포인트 | SHA256 | 출처 |", "|---|---|---|---|---|---|"]
    for n in names:
        s = cfg["detector_settings"][n]
        L.append(f"| {n} | {s['model']} | {s['epe']} | {Path(s['checkpoint']).name} | {s['checkpoint_sha256']} | {s['checkpoint_url']} |")
    L += ["", "| 검출기 | 설정 파일 | 설정 SHA256 | 데이터 처리 (설정 파일의 test pipeline) | 테스트 설정 |", "|---|---|---|---|---|"]
    for n in names:
        i = env["info"].get(n) or {}
        L.append(f"| {n} | {Path(cfg['detector_settings'][n]['config']).name} | {i.get('config_sha256', '-')} | {' → '.join(i.get('test_pipeline', []))}, 입력 {'×'.join(str(x) for x in i.get('input_size', []))} | "
                 f"{', '.join(f'{k} {v}' for k, v in (i.get('test_cfg') or {}).items())} |")
    L += ["", f"- 기존 rtmpose_hand: {cfg['existing_rtmpose_hand']['checkpoint']} — {cfg['existing_rtmpose_hand']['note']} 기존 검출기와 설정은 바꾸지 않았다.", "",
          "## 2. 관절 순서", "",
          "- 모델이 내는 관절 이름(dataset_meta)을 기존 캐시 순서(0 손목, 1~4 엄지, 5~8 검지, 9~12 중지, 13~16 약지, 17~20 새끼)의 이름과 코드로 대조했다(detector_worker.mm_order).",
          "- 실제 프레임에서도 확인: 같은 박스에서 각 모델 관절과 mediapipe 관절 사이 거리의 중앙값 행렬을 헝가리안 방법으로 짝지었을 때 항등 대응인지 본다.", "",
          "| 검출기 | 이름 대조 | 순서 변환 | 실제 프레임 대응 (헝가리안) | 박스 수 |", "|---|---|---|---|---|"]
    for n in names:
        i = env["info"].get(n) or {}
        o = order.get(n) or {}
        L.append(f"| {n} | {'일치' if i.get('names_ok') else '불일치'} | {'없음 (항등)' if i.get('remap') is None else i.get('remap')} | {'항등' if o.get('identity') else ('불일치: ' + str(o.get('assignment')) if o else '-')} | {o.get('n_boxes', '-')} |")
    L += ["", "## 3. 공식 추론 함수와의 일치", "",
          f"- 같은 프레임·같은 박스를 mmpose.apis.inference_topdown(영상 프레임 PNG 경로 입력, 프레임마다 한 번 호출)과 우리 래퍼(배치 {cfg['detector_settings'][names[0]]['batch_size']}, 여러 프레임의 박스를 함께 처리)로 처리해 관절 좌표의 최대 차이를 본다. 기준 {cfg['verify']['tol_px']} px.",
          "- TF32를 켜면(파이토치 기본값) 배치 구성에 따라 합성곱 결과가 조금 달라져 히트맵 최댓값이 한 칸(시험에서 최대 6.5 px) 옮겨 가는 일이 있었다. 그래서 래퍼와 공식 함수 모두 TF32를 끄고 비교·실행한다.", "",
          "| 검출기 | 영상 | 프레임 | 박스 | 최대 차이 (px) | 신뢰도 최대 차이 | 통과 |", "|---|---|---|---|---|---|---|"]
    for n in names:
        for v, c in (checks.get(n) or {}).items():
            if isinstance(c, dict) and "max_abs_px" in c:
                L.append(f"| {n} | {v} | {len(c['frames'])} | {c['n_boxes']} | {c['max_abs_px']:.4g} | {c['max_abs_conf']:.2g} | {'예' if c['max_abs_px'] <= float(cfg['verify']['tol_px']) else '아니오'} |")
            else:
                L.append(f"| {n} | {v} | - | - | 실패: {c} | - | 아니오 |")
    L += ["", "## 4. 검출기 간 일치도 (mediapipe 대비)", "",
          "- 같은 프레임·같은 박스에서 각 모델 관절과 mediapipe 관절 사이 거리, 손 박스 긴 변 대비 %. 관절 21개 전체.", "",
          "| 검출기 | 영상 | 박스 | 중앙값 (% 박스) | 90 백분위 (% 박스) | 신뢰도 문턱 미만 관절 비율 |", "|---|---|---|---|---|---|"]
    for n in names:
        for v, a in (agree.get(n) or {}).items():
            if a.get("median_pct") is None:
                L.append(f"| {n} | {v} | {a.get('n_boxes', 0)} | - | - | - |")
            else:
                L.append(f"| {n} | {v} | {a['n_boxes']} | {a['median_pct']:.2f} | {a['p90_pct']:.2f} | {100 * a['low_conf_frac']:.1f}% |")
    L += ["", "## 5. 검출 실패율", "", "- 박스 없음 = 박스 출처 캐시에 손이 없는 프레임(다섯 모델 공통). 신뢰도 미만 = 그 모델의 conf_min 미만인 관절의 비율(박스가 있는 프레임).", "",
          "| 영상 | 프레임 | 박스 없음 | " + " | ".join(f"{n} 신뢰도 미만" for n in names) + " |", "|---|---|---|" + "---|" * len(names)]
    for v, fr in fail.items():
        L.append(f"| {v} | {fr['n_frames']} | {100 * fr['no_box_frac']:.1f}% | " + " | ".join("-" if fr["low_conf"].get(n) is None else f"{100 * fr['low_conf'][n]:.1f}%" for n in names) + " |")
    L += ["", "## 6. 오버레이", "", f"- runs 폴더의 overlay/ 에 저장(커밋하지 않음): {', '.join(overlays) if overlays else '없음'}. 칸 순서: " + ", ".join(["mediapipe_hands"] + names) + ". 속이 빈 점 = 신뢰도 문턱 미만.", "",
          "## 7. 검증 결과", "", f"- 공식 추론 일치(모든 모델·영상 ≤ {cfg['verify']['tol_px']} px): {'통과' if verified['official'] else '실패'}",
          f"- 관절 순서(이름 대조와 실제 프레임 대응이 모두 항등): {'통과' if verified['order'] else '실패'}",
          f"- 2단계: {verified['stage2']}", ""]
    path.write_text("\n".join(L) + "\n")


def mm_summary(path: Path, cfg: dict, stage2: list, names: list, existing: dict, slopes: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    rf = cfg["ref_f"]
    ref_of = lambda r: (float(rf["by_distance_27"][int(r["distance_cm"])]) if r["source"] == "27" else float(rf["lens_0.5x"] if r.get("lens") == "0.5x" else rf["lens_1x"]))   # noqa: E731
    dets = ["mediapipe_hands"] + names
    L = ["# 실험 47 summary (2단계: 기존 영상에서 기록, 판정 없음)", "", f"**{MM_NOTE}**", "",
         f"- 대상: 실험 25의 1배 영상 {sum(r['source'] == '25' for r in stage2)}개, 실험 27의 G6 영상 {sum(r['source'] == '27' for r in stage2)}개, 실험 29의 0.5배 영상 {sum(r['source'] == '29' for r in stage2)}개. "
         "후보 설정(손목 제외, fixed_template)과 실험 27의 제외 규칙·처리 설정을 그대로 썼다. mediapipe_hands = 같은 캐시로 이번 실행에서 다시 계산한 값, '공식' = 원래 실험의 결과 파일 값.",
         f"- 기준 f: 실험 25 영상 {rf['lens_1x']} (1배 공칭값), 실험 27 영상 거리별 체커보드 값 " + " / ".join(f"{d} cm {v}" for d, v in rf["by_distance_27"].items()) + f" (실험 45와 같은 값), 실험 29 영상 {rf['lens_0.5x']} (0.5배).", "",
         "## 1. 영상별 f (px)", "", "| 영상 | 실험 | 거리(cm) | 기준 f | 공식 mediapipe | " + " | ".join(dets) + " |", "|---|---|---|---|---|" + "---|" * len(dets)]
    for r in stage2:
        e = existing.get(r["video"])
        cells = []
        for n in dets:
            x = r["detectors"].get(n) or {}
            if "ft" in x and not x.get("excluded"):
                cells.append(f"{x['ft']['f']:.1f}{' (경계)' if x['ft']['at_grid_edge'] else ''}")
            else:
                cells.append("제외: " + ("; ".join(x.get("excluded") or []) or x.get("failed") or "-"))
        L.append(f"| {r['video']} | {r['source']} | {'-' if r.get('distance_cm') is None else format(float(r['distance_cm']), 'g')} | {ref_of(r):g} | {'-' if not e else format(e['f'], '.1f')} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 기준 f 대비 오차", "", "- 영상별 오차 = f ÷ 기준 f − 1. 중앙 절대 오차 = 포함된 영상의 |log(f ÷ 기준 f)| 중앙값을 %로 바꾼 값.", "",
          "| 검출기 | 포함 영상 | 중앙 절대 오차 | 실험 25 오차 중앙값 | 실험 27 오차 중앙값 | 실험 29 오차 중앙값 |", "|---|---|---|---|---|---|"]
    for n in ["공식 mediapipe"] + dets:
        errs = {"25": [], "27": [], "29": []}
        for r in stage2:
            if n == "공식 mediapipe":
                e = existing.get(r["video"])
                x = (r["detectors"].get("mediapipe_hands") or {})
                f = e["f"] if e and "ft" in x and not x.get("excluded") else None
            else:
                x = r["detectors"].get(n) or {}
                f = x["ft"]["f"] if "ft" in x and not x.get("excluded") else None
            if f is not None:
                errs[r["source"]].append(float(np.log(f / ref_of(r))))
        allv = sum(errs.values(), [])
        med = lambda v: None if not v else float(np.exp(np.median(v)) - 1.0)      # noqa: E731
        L.append(f"| {n} | {len(allv)} | {'-' if not allv else format(100 * (np.exp(np.median(np.abs(allv))) - 1.0), '.1f') + '%'} | {pct(med(errs['25']))} | {pct(med(errs['27']))} | {pct(med(errs['29']))} |")
    L += ["", "## 3. 실험 27 거리 영상 (G6): 거리별 합동 f와 기울기", "", "- 합동 f = 포함된 영상의 비용 곡선을 더한 곡선의 최솟값. 기울기 = log f 대 log 거리, 95% 구간은 영상 단위 부트스트랩(실험 39~45와 같은 계산). 체커보드 기준 f의 기울기는 약 -0.02.", "",
          "| 검출기 | 50 cm f | 60 cm f | 75 cm f | 50 cm 기준 대비 | 60 cm 기준 대비 | 75 cm 기준 대비 | 기울기 | 95% 구간 |", "|---|---|---|---|---|---|---|---|---|"]
    for n, s in slopes.items():
        if not s:
            L.append(f"| {n} | - | - | - | - | - | - | - | - |")
            continue
        pf = {float(d): f for d, f in s["pooled_f"].items()}
        fc = [pf.get(float(d)) for d in (50, 60, 75)]
        rc = [pct(None if f is None else f / float(rf["by_distance_27"][d]) - 1.0) for f, d in zip(fc, (50, 60, 75))]
        L.append(f"| {n} | " + " | ".join("-" if f is None else f"{f:.1f}" for f in fc) + " | " + " | ".join(rc) +
                 f" | {'-' if s['slope'] is None else format(s['slope'], '+.3f')} | {'-' if not s['interval_95'] else format(s['interval_95'][0], '+.3f') + ' ~ ' + format(s['interval_95'][1], '+.3f')} |")
    L += ["", "## 4. 날카로움 (FT 비용 곡선)", "", "| 영상 | " + " | ".join(dets) + " |", "|---|" + "---|" * len(dets)]
    for r in stage2:
        L.append(f"| {r['video']} | " + " | ".join("-" if "ft" not in (r["detectors"].get(n) or {}) or (r["detectors"].get(n) or {}).get("excluded") else f"{r['detectors'][n]['ft']['sharpness']:.3f}" for n in dets) + " |")
    L += ["", "## 한계", "", "- 판정은 하지 않는다(기록용). 거리마다 영상이 3개 이하라 기울기 구간이 거칠다.",
          "- 실험 25·29 영상에는 mediapipe 외 기존 검출기(wilor_yolo, rtmpose_hand)의 캐시가 없어 기존 검출기 비교는 mediapipe만 있다.",
          "- 실험 27 영상의 기준 f는 이 영상 세션에서 새로 재지 않은 체커보드 값이다.", ""]
    path.write_text("\n".join(L) + "\n")


def mm_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/47_mmpose_hand_detectors.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(mm_results_yaml(hf.resolve(a.run), cfg))
        return None
    names = [d for d in cfg["detectors"] if d in hf.MM_DETECTORS]
    s2 = list(cfg["stage2_videos"]) if cfg.get("run_stage2") else []
    vids = list(dict.fromkeys(list(cfg["verify"]["videos"]) + [e["file"] for e in s2]))
    if a.videos:
        vids = [v for v in vids if v in a.videos.split(",")]
        s2 = [e for e in s2 if e["file"] in vids]
    missing = [v for v in vids if not (Path(cfg["input_dir"]).expanduser() / v).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    for n in names:                                                    # the checkpoints must be the recorded files
        s = cfg["detector_settings"][n]
        got = hf.sha256_file(Path(s["checkpoint"]).expanduser())
        if got != s["checkpoint_sha256"]:
            raise SystemExit(f"{n}: 체크포인트 SHA256이 다릅니다 ({got})")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    log = Log(run_root / "log.txt")
    log(f"47 MMPose 손 검출기 공식 실행: 검출기 {len(names)}개, 영상 {len(vids)}개 (검증 {len(cfg['verify']['videos'])}, 2단계 {len(s2)}) → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cache_dir = run_root / "detection_cache"
    cache_dir.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = dict(zip(vids, ex.map(hf.fresp_prepare, [(v, cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for v in vids])))
        log(f"  프레임 준비 {time.time() - t0:.0f} s")
        boxes = {}
        for v in vids:
            try:
                boxes[v] = hf.mm_box_cache(preps[v]["frames"], cfg)
                hf.check_alignment(preps[v]["frames"], hf.load_detections(boxes[v]))
            except (hf.SkipDetector, hf.DetectorError) as e:
                log(f"  {v}: 손 박스 없음 — {e}")
        # ---- official-inference check (verification videos) and detection (every video), on the GPUs
        jobs, keys = [], []
        for n in names:
            for v in cfg["verify"]["videos"]:
                if v in boxes:
                    jobs.append(lambda g, n=n, v=v: hf.mm_official_check(preps[v]["frames"], mm_det_settings(cfg, n, g), boxes[v], tmp_root / f"check_{n}_{Path(v).stem}", int(cfg["verify"]["official_check_frames"]), log))
                    keys.append(("check", n, v))
        for n in names:
            for v in vids:
                if v in boxes:
                    jobs.append(lambda g, n=n, v=v: hf.swap_mmpose_hand(preps[v]["frames"], mm_det_settings(cfg, n, g), cache_dir, log, boxes[v], n)[0])
                    keys.append(("det", n, v))
        t1 = time.time()
        res = mm_gpu_jobs(jobs, cfg, log)
        log(f"  공식 일치 확인과 검출 {len(jobs)}개 작업: {time.time() - t1:.0f} s")
        checks, npz = {}, {}
        for (kind, n, v), r in zip(keys, res):
            if isinstance(r, Exception):
                log(f"  {kind} {n} {v}: 실패 — {r}")
            if kind == "check":
                checks.setdefault(n, {})[v] = str(r) if isinstance(r, Exception) else r
            else:
                npz.setdefault(n, {})[v] = None if isinstance(r, Exception) else r
        # ---- verification: order, agreement, failure rates, overlays
        env = {"info": {}, "versions": {}}
        agree, fail, overlays = {}, {}, []
        pooled = {n: ([], []) for n in names}
        for v in vids:
            if v not in boxes:
                continue
            ref = hf.load_detections(boxes[v])
            nf = len(preps[v]["frames"])
            fr = {"n_frames": nf, "no_box_frac": float(np.mean(np.diff(ref["offsets"]) == 0)), "low_conf": {}}
            dets = {"mediapipe_hands": ref}
            for n in names:
                p = (npz.get(n) or {}).get(v)
                if p is None:
                    continue
                d = hf.load_detections(p)
                hf.check_alignment(preps[v]["frames"], d)
                dets[n] = d
                i = dict(d["info"])
                i["names_ok"] = [i["keypoint_names"][k] for k in (i.get("remap") or range(21))] == list(MM_HAND_NAMES)
                env["info"].setdefault(n, i)
                env["versions"] = {k: i[k] for k in ("mmpose", "mmcv", "mmengine", "torch", "torch_cuda", "python") if k in i}
                ag = hf.mm_agreement(ref, d, float(cfg["detector_settings"][n]["conf_min"]))
                fr["low_conf"][n] = ag["low_conf_frac"]
                agree.setdefault(n, {})[v] = {k: ag[k] for k in ("n_boxes", "median_pct", "p90_pct", "per_joint_median_pct", "low_conf_frac")}
                if v in cfg["verify"]["videos"] and len(d["kxy"]):
                    pooled[n][0].append(ref["kxy"])
                    pooled[n][1].append(d["kxy"])
            fail[v] = fr
            if v in cfg["verify"]["videos"]:
                overlays += hf.mm_overlay(preps[v]["frames"], dets, run_root / "overlay", v, int(cfg["verify"]["overlay_frames"]), {k: float(cfg["detector_settings"][k]["conf_min"]) for k in dets})
        order = {}
        for n in names:
            if pooled[n][0]:
                ra, rb = np.concatenate(pooled[n][0]), np.concatenate(pooled[n][1])
                order[n] = {**hf.order_assignment(ra, rb, np.ones((len(ra), 21), bool)), "n_boxes": int(len(ra))}
        tol = float(cfg["verify"]["tol_px"])
        ok_off = all(isinstance(c, dict) and c.get("max_abs_px") is not None and c["max_abs_px"] <= tol for n in names for c in (checks.get(n) or {"-": None}).values())
        ok_order = all((order.get(n) or {}).get("identity") and (env["info"].get(n) or {}).get("names_ok") and (env["info"].get(n) or {}).get("remap") is None for n in names)
        verified = {"official": bool(ok_off), "order": bool(ok_order)}
        log(f"  검증: 공식 일치 {'통과' if ok_off else '실패'}, 관절 순서 {'통과' if ok_order else '실패'}")
        stage2 = []
        if not cfg.get("run_stage2"):
            verified["stage2"] = "꺼짐 (run_stage2: false)"
        elif not (ok_off and ok_order):
            verified["stage2"] = "검증 실패로 실행하지 않음"
        else:
            c = {k: cfg[k] for k in MM_KEYS}
            src_name = hf.MM_BOX_SOURCES[cfg["hand_box_source"]]
            args, owner = [], []                                       # one job per (video, detector); the results are merged per video
            for e in s2:
                by = {src_name: boxes.get(e["file"])} if src_name == "mediapipe_hands" else {}
                by.update({n: (npz.get(n) or {}).get(e["file"]) for n in names})
                for n, p in by.items():
                    args.append((e, preps[e["file"]], {n: p}, cfg["pipeline"], c))
                    owner.append(e["file"])
            t2 = time.time()
            with ProcessPoolExecutor(max_workers=int(cfg["stage2_workers"]), mp_context=mp.get_context("spawn")) as ex:
                parts = list(ex.map(hf.mm_video, args))
            merged = {}
            for v, r in zip(owner, parts):
                merged.setdefault(v, {**r, "detectors": {}})["detectors"].update(r["detectors"])
            stage2 = [merged[e["file"]] for e in s2]
            verified["stage2"] = f"실행함 (영상 {len(stage2)}개, {time.time() - t2:.0f} s)"
            log(f"  2단계 FT: {verified['stage2']}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    existing = mm_existing_f(cfg)
    dets_all = ([hf.MM_BOX_SOURCES[cfg["hand_box_source"]]] if cfg["hand_box_source"] == "mediapipe" else []) + names
    slopes = mm_stage2_slopes(stage2, dets_all, cfg, existing) if stage2 else {}
    for r in stage2:
        log(f"  {r['video']}: " + ", ".join(f"{n} " + (f"{x['ft']['f']:.0f}" if "ft" in x and not x.get("excluded") else "제외") for n, x in r["detectors"].items()))
    mm_report(run_root / "report.md", cfg, env, checks, agree, order, fail, overlays, verified)
    if stage2:
        mm_summary(run_root / "summary.md", cfg, stage2, names, existing, slopes)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"verification": {"checks": checks, "order": order, "agreement": agree, "failure": fail, "verified": verified, "env": env},
                                                                    "stage2": stage2, "slopes": slopes}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_47": cfg["code_47"], "videos": vids, "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, verified


def mm_results_yaml(run_dir: Path, cfg: dict):
    """Values only (no reference f, no comparison with it); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    v = r["verification"]
    out = {"experiment": cfg["experiment"]["name"], "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": MM_NOTE,
           "detectors": {n: {"checkpoint": Path(cfg["detector_settings"][n]["checkpoint"]).name, "checkpoint_sha256": cfg["detector_settings"][n]["checkpoint_sha256"]} for n in cfg["detectors"]},
           "hand_box_source": cfg["hand_box_source"], "verified": v["verified"],
           "official_check_max_px": {n: {vid: (c.get("max_abs_px") if isinstance(c, dict) else None) for vid, c in cs.items()} for n, cs in v["checks"].items()},
           "order_identity": {n: o.get("identity") for n, o in v["order"].items()},
           "agreement_vs_mediapipe_median_pct": {n: {vid: rd(a.get("median_pct"), 3) for vid, a in d.items()} for n, d in v["agreement"].items()}}
    if r["stage2"]:
        out["stage2_f_px"] = {s["video"]: {n: (round(x["ft"]["f"], 1) if "ft" in x and not x.get("excluded") else {"excluded": x.get("excluded") or x.get("failed")}) for n, x in s["detectors"].items()} for s in r["stage2"]}
        out["stage2_sharpness"] = {s["video"]: {n: rd(x["ft"]["sharpness"]) for n, x in s["detectors"].items() if "ft" in x and not x.get("excluded")} for s in r["stage2"]}
        out["distance_27"] = {n: (None if not s else {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]),
                                                       "slope_interval_95": None if not s["interval_95"] else [round(q, 4) for q in s["interval_95"]]}) for n, s in r["slopes"].items()}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p




# ====================================================================================================
# mm2  (experiment 48: experiment 47's five MMPose hand models on the videos of experiment 46, boxes from 46's mediapipe cache; the mm functions of 47 are not changed,
#       mm2_main / mm2_report / mm2_results_yaml are mm_main / mm_report / mm_results_yaml with the summary, the slopes and the comparison of 46 swapped in)
#     python HandCalib/run.py mmpose-hand-46 --config HandCalib/configs/48_mmpose_hand_46_videos.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py mmpose-hand-46 --config ... results-yaml --run runs/48_mmpose_hand_46_videos/<UTC>
# ====================================================================================================
MM2_NOTE = ("실험 47의 MMPose 손 2D 모델 5종(COCO-WholeBody-Hand 학습)을 실험 46의 영상 12개(50/60/75 cm 각 4개)에 적용한 실험이다. 손 박스는 실험 46 실행의 mediapipe 검출 캐시에서 읽었고(새로 검출하지 않음), "
            "f는 기존 FT 코드(후보 설정: 손목 제외, fixed_template)로 구했다. 판정은 하지 않는다. 기준 f와의 비교는 summary.md에만 있다.")


def mm2_existing(cfg: dict) -> dict:
    """Experiment 46 (read only): the per-video free_shape3d f and its distance slope. {"videos": {video: f}, "slope": {...} | None}."""
    q = Path(str(cfg["existing_results"]["46"])).expanduser()
    q = q if q.is_absolute() else hf.resolve(q)
    if not q.exists():
        return {"videos": {}, "slope": None}
    r = json.loads(q.read_text())
    return {"videos": {v["video"]: float(v["est"]["f"]) for v in r["videos"] if isinstance(v.get("est"), dict)}, "slope": ((r.get("analysis") or {}).get("shape3d") or {}).get("slope")}


def mm2_stage2_slopes(stage2: list, names: list, cfg: dict) -> dict:
    """Every video with a distance: pooled f per distance and the log f / log distance slope (pr_slope, video-level bootstrap) for every detector."""
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    out = {}
    for name in names:
        cur, grid = {}, None
        for r in stage2:
            x = r["detectors"].get(name) or {}
            if r.get("distance_cm") is not None and "ft" in x and not x.get("excluded"):
                cur.setdefault(float(r["distance_cm"]), []).append(x["curve"]["cost"])
                grid = np.array(x["curve"]["grid"])
        out[name] = hf.pr_slope(cur, grid, bn, bs) if len(cur) >= 2 else None
        if out[name] is not None:
            out[name]["n_videos"] = {d: len(v) for d, v in cur.items()}
    return out


def mm2_summary(path: Path, cfg: dict, stage2: list, names: list, existing: dict, slopes: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    ref = float(cfg["ref_f"])
    dets = ["mediapipe_hands"] + names
    ds = sorted({float(r["distance_cm"]) for r in stage2})
    ok = lambda x: "ft" in x and not x.get("excluded")      # noqa: E731
    L = [f"# 실험 {cfg['experiment']['name'].split('_')[0]} summary (실험 46 영상, 판정 없음)", "", f"**{MM2_NOTE}**", "",
         f"- 기준 f: {ref:g} px (모든 거리, 사용자 제공). mediapipe_hands = 같은 캐시로 이번 실행에서 계산한 FT 값. 실험 46 = 같은 영상의 free_shape3d 3차원 f(실험 46 결과 파일).", "",
         "## 1. 영상별 f (px)", "", "| 영상 | 거리(cm) | 실험 46 free_shape3d | " + " | ".join(dets) + " |", "|---|---|---|" + "---|" * len(dets)]
    for r in stage2:
        e = existing["videos"].get(r["video"])
        cells = [f"{x['ft']['f']:.1f}{' (경계)' if x['ft']['at_grid_edge'] else ''}" if ok(x) else "제외: " + ("; ".join(x.get("excluded") or []) or x.get("failed") or "-")
                 for x in (r["detectors"].get(n) or {} for n in dets)]
        L.append(f"| {r['video']} | {float(r['distance_cm']):g} | {'-' if e is None else format(e, '.1f')} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 기준 f 대비 오차", "", "- 영상별 오차 = f ÷ 기준 f − 1. 중앙 절대 오차 = |log(f ÷ 기준 f)| 중앙값을 %로 바꾼 값. 거리별 값 = 그 거리 영상 오차의 중앙값.", "",
          "| 검출기 | 포함 영상 | 중앙 절대 오차 | " + " | ".join(f"{d:g} cm 오차 중앙값" for d in ds) + " |", "|---|---|---|" + "---|" * len(ds)]
    for n in ["실험 46 free_shape3d"] + dets:
        lg = {d: [] for d in ds}
        for r in stage2:
            if n == "실험 46 free_shape3d":
                f = existing["videos"].get(r["video"])
            else:
                x = r["detectors"].get(n) or {}
                f = x["ft"]["f"] if ok(x) else None
            if f is not None:
                lg[float(r["distance_cm"])].append(float(np.log(f / ref)))
        allv = sum(lg.values(), [])
        L.append(f"| {n} | {len(allv)} | {'-' if not allv else format(100 * (np.exp(np.median(np.abs(allv))) - 1.0), '.1f') + '%'} | "
                 + " | ".join(pct(None if not lg[d] else float(np.exp(np.median(lg[d])) - 1.0)) for d in ds) + " |")
    L += ["", "## 3. 거리별 합동 f와 기울기", "", "- 합동 f = 포함된 영상의 FT 비용 곡선을 더한 곡선의 최솟값. 기울기 = log f 대 log 거리, 95% 구간은 영상 단위 부트스트랩(실험 47과 같은 계산). "
          "실험 46 행은 영상별 3차원 f의 기하평균과 그 기울기(실험 46의 계산). 기준 f가 거리마다 같으므로 기준 f의 기울기는 0.", "",
          "| 검출기 | " + " | ".join(f"{d:g} cm f" for d in ds) + " | " + " | ".join(f"{d:g} cm 기준 대비" for d in ds) + " | 기울기 | 95% 구간 |", "|---|" + "---|" * (2 * len(ds) + 2)]
    rows = [("실험 46 free_shape3d", existing.get("slope"))] + list(slopes.items())
    for n, s in rows:
        if not s:
            L.append(f"| {n} | " + " | ".join("-" for _ in range(2 * len(ds) + 2)) + " |")
            continue
        pf = {float(d): f for d, f in s["pooled_f"].items()}
        fc = [pf.get(d) for d in ds]
        L.append(f"| {n} | " + " | ".join("-" if f is None else f"{f:.1f}" for f in fc) + " | " + " | ".join(pct(None if f is None else f / ref - 1.0) for f in fc)
                 + f" | {'-' if s['slope'] is None else format(s['slope'], '+.3f')} | {'-' if not s['interval_95'] else format(s['interval_95'][0], '+.3f') + ' ~ ' + format(s['interval_95'][1], '+.3f')} |")
    L += ["", "## 4. 날카로움 (FT 비용 곡선)", "", "| 영상 | " + " | ".join(dets) + " |", "|---|" + "---|" * len(dets)]
    for r in stage2:
        L.append(f"| {r['video']} | " + " | ".join(f"{x['ft']['sharpness']:.3f}" if ok(x) else "-" for x in (r["detectors"].get(n) or {} for n in dets)) + " |")
    L += ["", "## 한계", "", "- 판정은 하지 않는다(기록용). 거리마다 영상이 4개 이하라 기울기 구간이 거칠다.",
          "- 기준 f는 사용자가 이 영상들에 대해 준 값이며 이 코드로 새로 재지 않았다. 실험 46의 값은 free_shape3d(3차원 조정)이고 여기 FT 값은 fixed_template이라 방식이 다르다.",
          "- 실험 46 영상에는 mediapipe 외 기존 검출기의 캐시가 없어 기존 검출기 비교는 mediapipe만 있다.", ""]
    path.write_text("\n".join(L) + "\n")


def mm2_report(path: Path, cfg: dict, env: dict, checks: dict, agree: dict, order: dict, fail: dict, overlays: list, verified: dict):
    names = list(cfg["detectors"])
    src = cfg["hand_box_source"]
    L = [f"# 실험 {cfg['experiment']['name'].split('_')[0]}: MMPose 손 2D 모델 5종 ({cfg['title']})", "", MM2_NOTE, "",
         "## 1. 환경과 체크포인트", "",
         f"- 가상환경: {cfg['detector_settings'][names[0]]['python']} (" + ", ".join(f"{k} {v}" for k, v in env.get("versions", {}).items()) + ")",
         f"- 사용 GPU 번호: {', '.join(str(g) for g in cfg['gpu_ids'])} (GPU당 동시 작업 {cfg.get('jobs_per_gpu', 1)}개), 배치 크기 {cfg['detector_settings'][names[0]]['batch_size']}, "
         f"TF32 {'켬' if cfg['detector_settings'][names[0]].get('tf32') else '끔 (완전한 FP32)'}",
         f"- 손 박스 출처: {src} (기존 검출 캐시에서 읽음, 새로 검출하지 않음). 박스가 없는 프레임은 다섯 모델 모두 검출 실패.", "",
         "| 검출기 | 모델 | COCO-WholeBody-Hand EPE | 체크포인트 | SHA256 | 출처 |", "|---|---|---|---|---|---|"]
    for n in names:
        s = cfg["detector_settings"][n]
        L.append(f"| {n} | {s['model']} | {s['epe']} | {Path(s['checkpoint']).name} | {s['checkpoint_sha256']} | {s['checkpoint_url']} |")
    L += ["", "| 검출기 | 설정 파일 | 설정 SHA256 | 데이터 처리 (설정 파일의 test pipeline) | 테스트 설정 |", "|---|---|---|---|---|"]
    for n in names:
        i = env["info"].get(n) or {}
        L.append(f"| {n} | {Path(cfg['detector_settings'][n]['config']).name} | {i.get('config_sha256', '-')} | {' → '.join(i.get('test_pipeline', []))}, 입력 {'×'.join(str(x) for x in i.get('input_size', []))} | "
                 f"{', '.join(f'{k} {v}' for k, v in (i.get('test_cfg') or {}).items())} |")
    L += ["", f"- 기존 rtmpose_hand: {cfg['existing_rtmpose_hand']['checkpoint']} — {cfg['existing_rtmpose_hand']['note']} 기존 검출기와 설정은 바꾸지 않았다.", "",
          "## 2. 관절 순서", "",
          "- 모델이 내는 관절 이름(dataset_meta)을 기존 캐시 순서(0 손목, 1~4 엄지, 5~8 검지, 9~12 중지, 13~16 약지, 17~20 새끼)의 이름과 코드로 대조했다(detector_worker.mm_order).",
          "- 실제 프레임에서도 확인: 같은 박스에서 각 모델 관절과 mediapipe 관절 사이 거리의 중앙값 행렬을 헝가리안 방법으로 짝지었을 때 항등 대응인지 본다.", "",
          "| 검출기 | 이름 대조 | 순서 변환 | 실제 프레임 대응 (헝가리안) | 박스 수 |", "|---|---|---|---|---|"]
    for n in names:
        i = env["info"].get(n) or {}
        o = order.get(n) or {}
        L.append(f"| {n} | {'일치' if i.get('names_ok') else '불일치'} | {'없음 (항등)' if i.get('remap') is None else i.get('remap')} | {'항등' if o.get('identity') else ('불일치: ' + str(o.get('assignment')) if o else '-')} | {o.get('n_boxes', '-')} |")
    L += ["", "## 3. 공식 추론 함수와의 일치", "",
          f"- 같은 프레임·같은 박스를 mmpose.apis.inference_topdown(영상 프레임 PNG 경로 입력, 프레임마다 한 번 호출)과 우리 래퍼(배치 {cfg['detector_settings'][names[0]]['batch_size']}, 여러 프레임의 박스를 함께 처리)로 처리해 관절 좌표의 최대 차이를 본다. 기준 {cfg['verify']['tol_px']} px.",
          "- TF32를 켜면(파이토치 기본값) 배치 구성에 따라 합성곱 결과가 조금 달라져 히트맵 최댓값이 한 칸(시험에서 최대 6.5 px) 옮겨 가는 일이 있었다. 그래서 래퍼와 공식 함수 모두 TF32를 끄고 비교·실행한다.", "",
          "| 검출기 | 영상 | 프레임 | 박스 | 최대 차이 (px) | 신뢰도 최대 차이 | 통과 |", "|---|---|---|---|---|---|---|"]
    for n in names:
        for v, c in (checks.get(n) or {}).items():
            if isinstance(c, dict) and "max_abs_px" in c:
                L.append(f"| {n} | {v} | {len(c['frames'])} | {c['n_boxes']} | {c['max_abs_px']:.4g} | {c['max_abs_conf']:.2g} | {'예' if c['max_abs_px'] <= float(cfg['verify']['tol_px']) else '아니오'} |")
            else:
                L.append(f"| {n} | {v} | - | - | 실패: {c} | - | 아니오 |")
    L += ["", "## 4. 검출기 간 일치도 (mediapipe 대비)", "",
          "- 같은 프레임·같은 박스에서 각 모델 관절과 mediapipe 관절 사이 거리, 손 박스 긴 변 대비 %. 관절 21개 전체.", "",
          "| 검출기 | 영상 | 박스 | 중앙값 (% 박스) | 90 백분위 (% 박스) | 신뢰도 문턱 미만 관절 비율 |", "|---|---|---|---|---|---|"]
    for n in names:
        for v, a in (agree.get(n) or {}).items():
            if a.get("median_pct") is None:
                L.append(f"| {n} | {v} | {a.get('n_boxes', 0)} | - | - | - |")
            else:
                L.append(f"| {n} | {v} | {a['n_boxes']} | {a['median_pct']:.2f} | {a['p90_pct']:.2f} | {100 * a['low_conf_frac']:.1f}% |")
    L += ["", "## 5. 검출 실패율", "", "- 박스 없음 = 박스 출처 캐시에 손이 없는 프레임(다섯 모델 공통). 신뢰도 미만 = 그 모델의 conf_min 미만인 관절의 비율(박스가 있는 프레임).", "",
          "| 영상 | 프레임 | 박스 없음 | " + " | ".join(f"{n} 신뢰도 미만" for n in names) + " |", "|---|---|---|" + "---|" * len(names)]
    for v, fr in fail.items():
        L.append(f"| {v} | {fr['n_frames']} | {100 * fr['no_box_frac']:.1f}% | " + " | ".join("-" if fr["low_conf"].get(n) is None else f"{100 * fr['low_conf'][n]:.1f}%" for n in names) + " |")
    L += ["", "## 6. 오버레이", "", f"- runs 폴더의 overlay/ 에 저장(커밋하지 않음): {', '.join(overlays) if overlays else '없음'}. 칸 순서: " + ", ".join(["mediapipe_hands"] + names) + ". 속이 빈 점 = 신뢰도 문턱 미만.", "",
          "## 7. 검증 결과", "", f"- 공식 추론 일치(모든 모델·영상 ≤ {cfg['verify']['tol_px']} px): {'통과' if verified['official'] else '실패'}",
          f"- 관절 순서(이름 대조와 실제 프레임 대응이 모두 항등): {'통과' if verified['order'] else '실패'}",
          f"- 2단계: {verified['stage2']}", ""]
    path.write_text("\n".join(L) + "\n")


def mm2_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/48_mmpose_hand_46_videos.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(mm2_results_yaml(hf.resolve(a.run), cfg))
        return None
    names = [d for d in cfg["detectors"] if d in hf.MM_DETECTORS]
    s2 = list(cfg["stage2_videos"]) if cfg.get("run_stage2") else []
    vids = list(dict.fromkeys(list(cfg["verify"]["videos"]) + [e["file"] for e in s2]))
    if a.videos:
        vids = [v for v in vids if v in a.videos.split(",")]
        s2 = [e for e in s2 if e["file"] in vids]
    missing = [v for v in vids if not (Path(cfg["input_dir"]).expanduser() / v).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    for n in names:                                                    # the checkpoints must be the recorded files
        s = cfg["detector_settings"][n]
        got = hf.sha256_file(Path(s["checkpoint"]).expanduser())
        if got != s["checkpoint_sha256"]:
            raise SystemExit(f"{n}: 체크포인트 SHA256이 다릅니다 ({got})")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    log = Log(run_root / "log.txt")
    log(f"{cfg['experiment']['name']} MMPose 손 검출기 공식 실행: 검출기 {len(names)}개, 영상 {len(vids)}개 (검증 {len(cfg['verify']['videos'])}, 2단계 {len(s2)}) → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cache_dir = run_root / "detection_cache"
    cache_dir.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = dict(zip(vids, ex.map(hf.fresp_prepare, [(v, cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for v in vids])))
        log(f"  프레임 준비 {time.time() - t0:.0f} s")
        boxes = {}
        for v in vids:
            try:
                boxes[v] = hf.mm_box_cache(preps[v]["frames"], cfg)
                hf.check_alignment(preps[v]["frames"], hf.load_detections(boxes[v]))
            except (hf.SkipDetector, hf.DetectorError) as e:
                log(f"  {v}: 손 박스 없음 — {e}")
        # ---- official-inference check (verification videos) and detection (every video), on the GPUs
        jobs, keys = [], []
        for n in names:
            for v in cfg["verify"]["videos"]:
                if v in boxes:
                    jobs.append(lambda g, n=n, v=v: hf.mm_official_check(preps[v]["frames"], mm_det_settings(cfg, n, g), boxes[v], tmp_root / f"check_{n}_{Path(v).stem}", int(cfg["verify"]["official_check_frames"]), log))
                    keys.append(("check", n, v))
        for n in names:
            for v in vids:
                if v in boxes:
                    jobs.append(lambda g, n=n, v=v: hf.swap_mmpose_hand(preps[v]["frames"], mm_det_settings(cfg, n, g), cache_dir, log, boxes[v], n)[0])
                    keys.append(("det", n, v))
        t1 = time.time()
        res = mm_gpu_jobs(jobs, cfg, log)
        log(f"  공식 일치 확인과 검출 {len(jobs)}개 작업: {time.time() - t1:.0f} s")
        checks, npz = {}, {}
        for (kind, n, v), r in zip(keys, res):
            if isinstance(r, Exception):
                log(f"  {kind} {n} {v}: 실패 — {r}")
            if kind == "check":
                checks.setdefault(n, {})[v] = str(r) if isinstance(r, Exception) else r
            else:
                npz.setdefault(n, {})[v] = None if isinstance(r, Exception) else r
        # ---- verification: order, agreement, failure rates, overlays
        env = {"info": {}, "versions": {}}
        agree, fail, overlays = {}, {}, []
        pooled = {n: ([], []) for n in names}
        for v in vids:
            if v not in boxes:
                continue
            ref = hf.load_detections(boxes[v])
            nf = len(preps[v]["frames"])
            fr = {"n_frames": nf, "no_box_frac": float(np.mean(np.diff(ref["offsets"]) == 0)), "low_conf": {}}
            dets = {"mediapipe_hands": ref}
            for n in names:
                p = (npz.get(n) or {}).get(v)
                if p is None:
                    continue
                d = hf.load_detections(p)
                hf.check_alignment(preps[v]["frames"], d)
                dets[n] = d
                i = dict(d["info"])
                i["names_ok"] = [i["keypoint_names"][k] for k in (i.get("remap") or range(21))] == list(MM_HAND_NAMES)
                env["info"].setdefault(n, i)
                env["versions"] = {k: i[k] for k in ("mmpose", "mmcv", "mmengine", "torch", "torch_cuda", "python") if k in i}
                ag = hf.mm_agreement(ref, d, float(cfg["detector_settings"][n]["conf_min"]))
                fr["low_conf"][n] = ag["low_conf_frac"]
                agree.setdefault(n, {})[v] = {k: ag[k] for k in ("n_boxes", "median_pct", "p90_pct", "per_joint_median_pct", "low_conf_frac")}
                if v in cfg["verify"]["videos"] and len(d["kxy"]):
                    pooled[n][0].append(ref["kxy"])
                    pooled[n][1].append(d["kxy"])
            fail[v] = fr
            if v in cfg["verify"]["videos"]:
                overlays += hf.mm_overlay(preps[v]["frames"], dets, run_root / "overlay", v, int(cfg["verify"]["overlay_frames"]), {k: float(cfg["detector_settings"][k]["conf_min"]) for k in dets})
        order = {}
        for n in names:
            if pooled[n][0]:
                ra, rb = np.concatenate(pooled[n][0]), np.concatenate(pooled[n][1])
                order[n] = {**hf.order_assignment(ra, rb, np.ones((len(ra), 21), bool)), "n_boxes": int(len(ra))}
        tol = float(cfg["verify"]["tol_px"])
        ok_off = all(isinstance(c, dict) and c.get("max_abs_px") is not None and c["max_abs_px"] <= tol for n in names for c in (checks.get(n) or {"-": None}).values())
        ok_order = all((order.get(n) or {}).get("identity") and (env["info"].get(n) or {}).get("names_ok") and (env["info"].get(n) or {}).get("remap") is None for n in names)
        verified = {"official": bool(ok_off), "order": bool(ok_order)}
        log(f"  검증: 공식 일치 {'통과' if ok_off else '실패'}, 관절 순서 {'통과' if ok_order else '실패'}")
        stage2 = []
        if not cfg.get("run_stage2"):
            verified["stage2"] = "꺼짐 (run_stage2: false)"
        elif not (ok_off and ok_order):
            verified["stage2"] = "검증 실패로 실행하지 않음"
        else:
            c = {k: cfg[k] for k in MM_KEYS}
            src_name = hf.MM_BOX_SOURCES[cfg["hand_box_source"]]
            args, owner = [], []                                       # one job per (video, detector); the results are merged per video
            for e in s2:
                by = {src_name: boxes.get(e["file"])} if src_name == "mediapipe_hands" else {}
                by.update({n: (npz.get(n) or {}).get(e["file"]) for n in names})
                for n, p in by.items():
                    args.append((e, preps[e["file"]], {n: p}, cfg["pipeline"], c))
                    owner.append(e["file"])
            t2 = time.time()
            with ProcessPoolExecutor(max_workers=int(cfg["stage2_workers"]), mp_context=mp.get_context("spawn")) as ex:
                parts = list(ex.map(hf.mm_video, args))
            merged = {}
            for v, r in zip(owner, parts):
                merged.setdefault(v, {**r, "detectors": {}})["detectors"].update(r["detectors"])
            stage2 = [merged[e["file"]] for e in s2]
            verified["stage2"] = f"실행함 (영상 {len(stage2)}개, {time.time() - t2:.0f} s)"
            log(f"  2단계 FT: {verified['stage2']}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    existing = mm2_existing(cfg)
    dets_all = ([hf.MM_BOX_SOURCES[cfg["hand_box_source"]]] if cfg["hand_box_source"] == "mediapipe" else []) + names
    slopes = mm2_stage2_slopes(stage2, dets_all, cfg) if stage2 else {}
    for r in stage2:
        log(f"  {r['video']}: " + ", ".join(f"{n} " + (f"{x['ft']['f']:.0f}" if "ft" in x and not x.get("excluded") else "제외") for n, x in r["detectors"].items()))
    mm2_report(run_root / "report.md", cfg, env, checks, agree, order, fail, overlays, verified)
    if stage2:
        mm2_summary(run_root / "summary.md", cfg, stage2, names, existing, slopes)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"verification": {"checks": checks, "order": order, "agreement": agree, "failure": fail, "verified": verified, "env": env},
                                                                    "stage2": stage2, "slopes": slopes}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {"code_48": cfg["code_48"], "videos": vids, "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, verified


def mm2_results_yaml(run_dir: Path, cfg: dict):
    """Values only (no reference f, no comparison with it); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    v = r["verification"]
    out = {"experiment": cfg["experiment"]["name"], "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": MM2_NOTE,
           "detectors": {n: {"checkpoint": Path(cfg["detector_settings"][n]["checkpoint"]).name, "checkpoint_sha256": cfg["detector_settings"][n]["checkpoint_sha256"]} for n in cfg["detectors"]},
           "hand_box_source": cfg["hand_box_source"], "verified": v["verified"],
           "official_check_max_px": {n: {vid: (c.get("max_abs_px") if isinstance(c, dict) else None) for vid, c in cs.items()} for n, cs in v["checks"].items()},
           "order_identity": {n: o.get("identity") for n, o in v["order"].items()},
           "agreement_vs_mediapipe_median_pct": {n: {vid: rd(a.get("median_pct"), 3) for vid, a in d.items()} for n, d in v["agreement"].items()}}
    if r["stage2"]:
        out["stage2_f_px"] = {s["video"]: {n: (round(x["ft"]["f"], 1) if "ft" in x and not x.get("excluded") else {"excluded": x.get("excluded") or x.get("failed")}) for n, x in s["detectors"].items()} for s in r["stage2"]}
        out["stage2_sharpness"] = {s["video"]: {n: rd(x["ft"]["sharpness"]) for n, x in s["detectors"].items() if "ft" in x and not x.get("excluded")} for s in r["stage2"]}
        out["distance"] = {n: (None if not s else {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]),
                                                       "slope_interval_95": None if not s["interval_95"] else [round(q, 4) for q in s["interval_95"]]}) for n, s in r["slopes"].items()}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p



# ====================================================================================================
# e5  (experiment 49: experiment 46 unchanged (e3_video, free_shape3d; the e4 functions are copied with the experiment number read from the config) on the next newly shot videos; same call as 45 (e3_video, free_shape3d on the MediaPipe joints, wrist removed) on 12 newly shot videos, 4 per distance 50 / 60 / 75 cm; one reference f for every distance)
#     python HandCalib/run.py free-shape3d-new2 --config HandCalib/configs/49_free_shape3d_7312.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py free-shape3d-new2 --config ... results-yaml --run runs/49_free_shape3d_7312/<UTC>
# ====================================================================================================
E5_KEYS = E3_KEYS
E5_NOTE = ("새로 촬영한 영상(설정 파일의 videos, 50/60/75 cm)에 실험 45·46과 같은 조건을 적용한 실험이다: MediaPipe 관절 16점(손목 제외), 교수님의 free_shape3d를 실험 41·45와 같은 호출로 사용, "
           "제외 규칙(정면 창, 가장자리 접촉 5%, 손 검출 실패 20%)과 판정 규칙(영상별 3차원 f의 거리-f 기울기, 영상 단위 부트스트랩)도 실험 45와 같다. 규칙과 설정은 실행 전에 고정했다. 기준 f와의 비교는 summary.md에만 있다.")


def e5_prior(cfg: dict) -> dict:
    """Pooled 3-D f per distance of experiment 45 (read only; empty when the file is missing)."""
    q = Path(str((cfg.get("compare") or {}).get("45", "x"))).expanduser()
    q = q if q.is_absolute() else hf.resolve(q)
    if not q.exists():
        return {}
    r = json.loads(q.read_text())
    pf = (((r.get("analysis") or {}).get("shape3d") or {}).get("slope") or {}).get("pooled_f") or {}
    return {float(d): float(f) for d, f in pf.items()}


def e5_figures(out: Path, results: list, ana: dict, ref: dict, prior: dict):
    plt = _plt()
    ds = sorted(ref)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ds, [ref[d] for d in ds], "s-", color="#555", label="기준 f")
    if prior and all(d in prior for d in ds):
        ax.plot(ds, [prior[d] for d in ds], "o-", color="#2a6fbb", label="실험 45 (free_shape3d, 영상별 f의 기하평균)")
    sl = ana.get("shape3d", {}).get("slope")
    if sl:
        pf = sl["pooled_f"]
        ax.plot(sorted(pf), [pf[d] for d in sorted(pf)], "o--", color="#c47a00", label="free_shape3d, 이번 실험 (영상별 f의 기하평균)")
    for r in results:
        if "est" in r:
            ax.plot([r["distance_cm"]], [r["est"]["f"]], "x", color="#c47a00", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_xlabel("손-카메라 거리 (cm)")
    ax.set_ylabel("f (px)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "f_vs_distance_loglog.png", dpi=110)
    plt.close(fig)


def e5_report(path: Path, results: list, cfg: dict):
    fs = cfg["free_shape3d"]
    L = [f"# 실험 {cfg['experiment']['name'].split('_')[0]}: free_shape3d (새로 촬영한 영상, MediaPipe 관절)", "", E5_NOTE, "", f"free_shape3d(z_rel {fs['z_rel']}, alt_iters {fs['alt_iters']}, max_frames {fs['max_frames']}), f 격자 {cfg['pipeline']['estimate']['f_grid_n']}점.", "",
         "## 1. 영상별 값", "", "| 영상 | 거리(cm) | 포함 | f 3차원 (px) | f 평면 번들 조정 (px) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 | 3차원 RMS (px) | 3차원 반복 수 | 사용 프레임 | 걸린 시간(초) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r.get("est")
        if x:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 포함 | {x['f']:.1f} | {x['f_planar']:.1f} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} | {long_fmt(x['ba3_rms_px'], 3)} | {x['ba3_nfev']} | {x['n_frames_in_problem']} | {x['elapsed_s']:.0f} |")
        else:
            L.append(f"| {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | - | - | - | - | - | - | - | - |")
    L += ["", "## 한계", "", "- 거리마다 영상 4개(제외된 영상은 빠짐)이다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- MediaPipe 관절은 점 위치 오차가 커서 f가 영상마다 크게 달라질 수 있다.",
          "- 3차원 조정은 f에 대한 비용 곡선이 없어, 판정은 영상별 f의 기울기(영상 단위 부트스트랩)로 한다.", ""]
    path.write_text("\n".join(L) + "\n")


def e5_summary(path: Path, results: list, ana: dict, ref: dict, cfg: dict, prior: dict):
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"      # noqa: E731
    vcfg = cfg["verdict"]
    v = ana["shape3d"]["verdict"]
    s_ = f"- **판정 (free_shape3d 영상별 3차원 f 기준): {v['verdict']}**"
    if v.get("slope") is not None and v.get("interval_95"):
        s_ += f" — 기울기 {v['slope']:+.3f}, 95% 구간 {v['interval_95'][0]:+.3f} ~ {v['interval_95'][1]:+.3f}"
        if v.get("within_f_tol"):
            s_ += "; 거리별 f가 기준 f의 ±10% 안: " + ", ".join(f"{d:g} cm {'예' if w else '아니오'}" for d, w in v["within_f_tol"].items())
    elif v.get("reason"):
        s_ += f" — {v['reason']}"
    sl = ana["shape3d"].get("slope")
    pl = ana.get("planar", {}).get("slope")
    rf0 = ref[sorted(ref)[0]]
    L = [f"# 실험 {cfg['experiment']['name'].split('_')[0]} summary", "", f"**{E5_NOTE}**", "", "## 판정", "", s_,
         f"- 판정 규칙(실행 전 고정, 실험 41·45와 같음): 반비례 없음 = 기울기 구간 전체가 {vcfg['slope_none_min']:g}보다 위이고 세 거리의 f(영상별 3차원 f의 기하평균)가 모두 기준 f의 ±{100 * float(vcfg['f_tol']):.0f}% 안; 반비례 있음 = 구간 전체가 {vcfg['slope_yes_max']:g}보다 아래; 그 외 판단 보류; 한 거리에 영상이 없거나 3차원 f가 격자 한계에 붙거나 평면 격자 최솟값이 가장자리이면 판단 불가.", "",
         f"## 1. 거리별 f (기준 f {rf0:.0f} px)", "", "| 거리(cm) | 영상 | free_shape3d (px) | 기준 f | 기준 f 대비 | 실험 45 (px) |", "|---|---|---|---|---|---|"]
    for d in (sl or {}).get("distances", []):
        f3 = sl["pooled_f"][d]
        n_in = sum(1 for r in results if r["distance_cm"] == d and "est" in r)
        n_all = sum(1 for r in results if r["distance_cm"] == d)
        p45 = prior.get(float(d))
        L.append(f"| {float(d):g} | {n_in} / {n_all} | {f3:.1f}{' (한계)' if sl['pooled_edge'][d] else ''} | {ref[float(d)]:.1f} | {pct(f3 / ref[float(d)] - 1.0)} | {format(p45, '.1f') if p45 else '-'} |")
    L += ["", "| 대상 | 기울기 | 95% 구간 |", "|---|---|---|", f"| free_shape3d (영상별 3차원 f) | {long_fmt(sl['slope'] if sl else None, 3)} | {'-' if not sl or sl['interval_95'] is None else format(sl['interval_95'][0], '+.3f') + ' ~ ' + format(sl['interval_95'][1], '+.3f')} |"]
    if pl:
        L.append(f"| 평면 free_template 합동 곡선 | {long_fmt(pl['slope'], 3)} | {'-' if pl['interval_95'] is None else format(pl['interval_95'][0], '+.3f') + ' ~ ' + format(pl['interval_95'][1], '+.3f')} |")
    if len(prior) >= 2:
        L.append(f"| 실험 45 (영상이 다름, free_shape3d) | {long_fmt(hf.ext_loglog_slope(sorted(prior), [prior[d] for d in sorted(prior)]), 3)} | - |")
    L += [f"| 기준 f | {long_fmt(hf.ext_loglog_slope(sorted(ref), [ref[d] for d in sorted(ref)]), 3)} | - |", "", "![거리 대 f](figures/f_vs_distance_loglog.png)", "",
          "## 2. 영상별 f", "", "| 영상 | 거리(cm) | f 3차원 (px) | f 평면 번들 조정 (px) | 기준 f 대비 (3차원) | 상대 깊이 변위 | 3차원 조정이 f 한계에 붙음 | 평면 격자 최솟값이 가장자리 |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, x = r["distance_cm"], r.get("est")
        rf = ref.get(d, rf0)
        if x:
            L.append(f"| {r['video']} | {d:g} | {x['f']:.1f} | {x['f_planar']:.1f} | {pct(x['f'] / rf - 1.0)} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} |")
        else:
            L.append(f"| {r['video']} | {d:g} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | - | - | - | - |")
    L += ["", "## 한계", "", f"- 기준 f {rf0:.0f} px는 사용자가 이번 영상들에 대해 준 값이며 이 코드로 새로 재지 않았다. 실험 45의 기준 f(체커보드 1660.6 / 1653.7 / 1644.9)와 다르다.",
          "- 거리마다 영상이 4개 이하라 기울기 구간이 거칠고, MediaPipe 점 오차가 커서 영상별 f의 흩어짐이 크다. 실험 45와는 영상이 달라 값을 직접 비교하기 어렵다.", ""]
    path.write_text("\n".join(L) + "\n")


def e5_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/49_free_shape3d_7312.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(e5_results_yaml(hf.resolve(a.run), cfg))
        return None
    entries = [{"group": "G8", "hand": "unrecorded", **e} for e in cfg["videos"]]
    if a.videos:
        entries = [e for e in entries if e["file"] in a.videos.split(",")]
    missing = [e["file"] for e in entries if not (Path(cfg["input_dir"]).expanduser() / e["file"]).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    log = Log(run_root / "log.txt")
    log(f"{cfg['experiment']['name']} free_shape3d 새 영상 공식 실행: 영상 {len(entries)}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    c = {k: cfg[k] for k in E5_KEYS}
    dset = cfg["detector_settings"][cfg["candidate"]["detector"]]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for e in entries]))
            found = hf.fresp_detect_many([((p["video"], "orig"), p["frames"]) for p in preps], dset, run_root / "detection_cache", int(cfg["n_workers"]), log)
            results = list(ex.map(hf.e3_video, [(e, p, found[(p["video"], "orig")], cfg["pipeline"], c) for e, p in zip(entries, preps)]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    ref = {float(d): float(f) for d, f in cfg["ref_f_by_distance"].items()}
    prior = e5_prior(cfg)
    for r in results:
        x = r.get("est")
        log(f"  {r['video']} ({r['distance_cm']:g} cm): " + (f"free_shape3d f {x['f']:.1f} px (평면 {x['f_planar']:.1f}, 상대 깊이 변위 {x['relief_rel']:.4f}{', f 한계에 붙음' if x['at_bound'] else ''}{', 평면 격자 최솟값이 가장자리' if x['at_grid_edge'] else ''})" if x else "추정 없음")
            + (f" [제외: {'; '.join(r['excluded'])}]" if r.get("excluded") else "") + (f" [{r['failed']}]" if r.get("failed") else ""))
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    inc = [r for r in results if "est" in r]
    ds = sorted({r["distance_cm"] for r in inc})
    ana = {}
    if not ds:
        ana["shape3d"] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
    else:
        fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in ds}
        eb = {d: [r["est"]["at_bound"] or r["est"]["at_grid_edge"] for r in inc if r["distance_cm"] == d] for d in ds}
        sl = hf.s3_slope(fb, eb, bn, bs)
        nv = {d: len(fb.get(d, [])) for d in ref}
        ana["shape3d"] = {"slope": sl, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
        grid = np.array(inc[0]["est"]["curve"]["grid"])
        cur = {d: [r["est"]["curve"]["cost"] for r in inc if r["distance_cm"] == d] for d in ds}
        psl = hf.pr_slope(cur, grid, bn, bs)
        ana["planar"] = {"slope": psl, "verdict": hf.pr_verdict(psl, ref, cfg["verdict"], nv)}
        log(f"  free_shape3d: {ana['shape3d']['verdict']['verdict']}" + (f" (기울기 {sl['slope']:+.3f}, 구간 {sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})" if sl["slope"] is not None and sl["interval_95"] else ""))
        log(f"  평면 합동 곡선: {ana['planar']['verdict']['verdict']}" + (f" (기울기 {psl['slope']:+.3f})" if psl["slope"] is not None else ""))
    e5_figures(run_root / "figures", results, ana, ref, prior)
    e5_report(run_root / "report.md", results, cfg)
    e5_summary(run_root / "summary.md", results, ana, ref, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {**{k: v for k, v in cfg.items() if k.startswith("code_")}, "videos": [e["file"] for e in entries], "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def e5_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f, no comparison); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": cfg["experiment"]["name"], "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": E5_NOTE, "candidate": cfg["candidate"], "free_shape3d": cfg["free_shape3d"],
           "verdict_rules": cfg["verdict"], "videos": {}}
    for v in r["videos"]:
        x = v.get("est")
        e = {"distance_cm": v["distance_cm"], "duration_s": rd(v["duration_s"], 2), "detect_rate": rd(v["detect_rate"]), "edge_rate": rd(v["edge_rate"])}
        if x:
            e["free_shape3d"] = {"f_3d_px": round(x["f"], 1), "f_planar_px": round(x["f_planar"], 1), "f_grid_px": round(x["f_grid"], 1), "at_grid_edge": x["at_grid_edge"], "at_bound": x["at_bound"], "relief_rel": rd(x["relief_rel"], 5),
                                 "relief_per_joint": [rd(q, 5) for q in x["relief_per_joint"]], "cost_3d": rd(x["cost"], 5), "cost_planar": rd(x["cost_planar"], 5), "ba3_rms_px": rd(x["ba3_rms_px"], 4), "ba3_success": x["ba3_success"],
                                 "ba3_nfev": x["ba3_nfev"], "n_frames_in_problem": x["n_frames_in_problem"], "elapsed_s": round(x["elapsed_s"])}
        else:
            e["free_shape3d"] = {"excluded": v.get("excluded") or v.get("failed") or "제외"}
        out["videos"][v["video"]] = e
    a = r["analysis"]
    if "slope" in a.get("shape3d", {}):
        s = a["shape3d"]["slope"]
        out["by_method"] = {"free_shape3d": {"f_by_distance_px": {f"{float(d):g}": rd(f, 1) for d, f in s["pooled_f"].items()}, "slope": rd(s["slope"]), "slope_interval_95": None if s["interval_95"] is None else [round(q, 4) for q in s["interval_95"]]}}
        ps = a["planar"]["slope"]
        out["by_method"]["planar_pooled_curves"] = {"pooled_f_px": {f"{float(d):g}": rd(f, 1) for d, f in ps["pooled_f"].items()}, "slope": rd(ps["slope"]), "slope_interval_95": None if ps["interval_95"] is None else [round(q, 4) for q in ps["interval_95"]]}
    out["verdict"] = {"free_shape3d": {"verdict": a["shape3d"]["verdict"]["verdict"], "reason": a["shape3d"]["verdict"].get("reason")}}
    if "planar" in a:
        out["verdict"]["planar_pooled_curves"] = {"verdict": a["planar"]["verdict"]["verdict"]}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ====================================================================================================
# mm3  (experiment 50: experiment 48 repeated on the videos of experiment 49 (boxes from 49's mediapipe cache). The mm2 functions write "실험 46" for the source experiment;
#       mm3_* call them unchanged and put the source experiment of the config (source_experiment) in its place; mm3_main = mm2_main with these swapped in)
#     python HandCalib/run.py mmpose-hand-src --config HandCalib/configs/50_mmpose_hand_7312.yaml [--videos A.MOV,B.MOV] [--out DIR]
#     python HandCalib/run.py mmpose-hand-src --config ... results-yaml --run runs/50_mmpose_hand_7312/<UTC>
# ====================================================================================================
def mm3_src(path: Path, cfg: dict):
    path.write_text(path.read_text().replace("실험 46", f"실험 {cfg['source_experiment']}"))


def mm3_existing(cfg: dict) -> dict:
    return mm2_existing({**cfg, "existing_results": {"46": cfg["existing_results"][str(cfg["source_experiment"])]}})


def mm3_report(path: Path, cfg: dict, *a):
    mm2_report(path, cfg, *a)
    mm3_src(path, cfg)


def mm3_summary(path: Path, cfg: dict, *a):
    mm2_summary(path, cfg, *a)
    mm3_src(path, cfg)


def mm3_results_yaml(run_dir: Path, cfg: dict):
    p = mm2_results_yaml(run_dir, cfg)
    mm3_src(p, cfg)
    return p


def mm3_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/50_mmpose_hand_7312.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None, help="comma list of file names narrowing the videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(mm3_results_yaml(hf.resolve(a.run), cfg))
        return None
    names = [d for d in cfg["detectors"] if d in hf.MM_DETECTORS]
    s2 = list(cfg["stage2_videos"]) if cfg.get("run_stage2") else []
    vids = list(dict.fromkeys(list(cfg["verify"]["videos"]) + [e["file"] for e in s2]))
    if a.videos:
        vids = [v for v in vids if v in a.videos.split(",")]
        s2 = [e for e in s2 if e["file"] in vids]
    missing = [v for v in vids if not (Path(cfg["input_dir"]).expanduser() / v).exists()]
    if missing:
        raise SystemExit(f"영상이 없습니다: {missing}")
    for n in names:                                                    # the checkpoints must be the recorded files
        s = cfg["detector_settings"][n]
        got = hf.sha256_file(Path(s["checkpoint"]).expanduser())
        if got != s["checkpoint_sha256"]:
            raise SystemExit(f"{n}: 체크포인트 SHA256이 다릅니다 ({got})")
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    log = Log(run_root / "log.txt")
    log(f"{cfg['experiment']['name']} MMPose 손 검출기 공식 실행: 검출기 {len(names)}개, 영상 {len(vids)}개 (검증 {len(cfg['verify']['videos'])}, 2단계 {len(s2)}) → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    cache_dir = run_root / "detection_cache"
    cache_dir.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = dict(zip(vids, ex.map(hf.fresp_prepare, [(v, cfg["input_dir"], cfg["pipeline"], {}, str(tmp_root)) for v in vids])))
        log(f"  프레임 준비 {time.time() - t0:.0f} s")
        boxes = {}
        for v in vids:
            try:
                boxes[v] = hf.mm_box_cache(preps[v]["frames"], cfg)
                hf.check_alignment(preps[v]["frames"], hf.load_detections(boxes[v]))
            except (hf.SkipDetector, hf.DetectorError) as e:
                log(f"  {v}: 손 박스 없음 — {e}")
        # ---- official-inference check (verification videos) and detection (every video), on the GPUs
        jobs, keys = [], []
        for n in names:
            for v in cfg["verify"]["videos"]:
                if v in boxes:
                    jobs.append(lambda g, n=n, v=v: hf.mm_official_check(preps[v]["frames"], mm_det_settings(cfg, n, g), boxes[v], tmp_root / f"check_{n}_{Path(v).stem}", int(cfg["verify"]["official_check_frames"]), log))
                    keys.append(("check", n, v))
        for n in names:
            for v in vids:
                if v in boxes:
                    jobs.append(lambda g, n=n, v=v: hf.swap_mmpose_hand(preps[v]["frames"], mm_det_settings(cfg, n, g), cache_dir, log, boxes[v], n)[0])
                    keys.append(("det", n, v))
        t1 = time.time()
        res = mm_gpu_jobs(jobs, cfg, log)
        log(f"  공식 일치 확인과 검출 {len(jobs)}개 작업: {time.time() - t1:.0f} s")
        checks, npz = {}, {}
        for (kind, n, v), r in zip(keys, res):
            if isinstance(r, Exception):
                log(f"  {kind} {n} {v}: 실패 — {r}")
            if kind == "check":
                checks.setdefault(n, {})[v] = str(r) if isinstance(r, Exception) else r
            else:
                npz.setdefault(n, {})[v] = None if isinstance(r, Exception) else r
        # ---- verification: order, agreement, failure rates, overlays
        env = {"info": {}, "versions": {}}
        agree, fail, overlays = {}, {}, []
        pooled = {n: ([], []) for n in names}
        for v in vids:
            if v not in boxes:
                continue
            ref = hf.load_detections(boxes[v])
            nf = len(preps[v]["frames"])
            fr = {"n_frames": nf, "no_box_frac": float(np.mean(np.diff(ref["offsets"]) == 0)), "low_conf": {}}
            dets = {"mediapipe_hands": ref}
            for n in names:
                p = (npz.get(n) or {}).get(v)
                if p is None:
                    continue
                d = hf.load_detections(p)
                hf.check_alignment(preps[v]["frames"], d)
                dets[n] = d
                i = dict(d["info"])
                i["names_ok"] = [i["keypoint_names"][k] for k in (i.get("remap") or range(21))] == list(MM_HAND_NAMES)
                env["info"].setdefault(n, i)
                env["versions"] = {k: i[k] for k in ("mmpose", "mmcv", "mmengine", "torch", "torch_cuda", "python") if k in i}
                ag = hf.mm_agreement(ref, d, float(cfg["detector_settings"][n]["conf_min"]))
                fr["low_conf"][n] = ag["low_conf_frac"]
                agree.setdefault(n, {})[v] = {k: ag[k] for k in ("n_boxes", "median_pct", "p90_pct", "per_joint_median_pct", "low_conf_frac")}
                if v in cfg["verify"]["videos"] and len(d["kxy"]):
                    pooled[n][0].append(ref["kxy"])
                    pooled[n][1].append(d["kxy"])
            fail[v] = fr
            if v in cfg["verify"]["videos"]:
                overlays += hf.mm_overlay(preps[v]["frames"], dets, run_root / "overlay", v, int(cfg["verify"]["overlay_frames"]), {k: float(cfg["detector_settings"][k]["conf_min"]) for k in dets})
        order = {}
        for n in names:
            if pooled[n][0]:
                ra, rb = np.concatenate(pooled[n][0]), np.concatenate(pooled[n][1])
                order[n] = {**hf.order_assignment(ra, rb, np.ones((len(ra), 21), bool)), "n_boxes": int(len(ra))}
        tol = float(cfg["verify"]["tol_px"])
        ok_off = all(isinstance(c, dict) and c.get("max_abs_px") is not None and c["max_abs_px"] <= tol for n in names for c in (checks.get(n) or {"-": None}).values())
        ok_order = all((order.get(n) or {}).get("identity") and (env["info"].get(n) or {}).get("names_ok") and (env["info"].get(n) or {}).get("remap") is None for n in names)
        verified = {"official": bool(ok_off), "order": bool(ok_order)}
        log(f"  검증: 공식 일치 {'통과' if ok_off else '실패'}, 관절 순서 {'통과' if ok_order else '실패'}")
        stage2 = []
        if not cfg.get("run_stage2"):
            verified["stage2"] = "꺼짐 (run_stage2: false)"
        elif not (ok_off and ok_order):
            verified["stage2"] = "검증 실패로 실행하지 않음"
        else:
            c = {k: cfg[k] for k in MM_KEYS}
            src_name = hf.MM_BOX_SOURCES[cfg["hand_box_source"]]
            args, owner = [], []                                       # one job per (video, detector); the results are merged per video
            for e in s2:
                by = {src_name: boxes.get(e["file"])} if src_name == "mediapipe_hands" else {}
                by.update({n: (npz.get(n) or {}).get(e["file"]) for n in names})
                for n, p in by.items():
                    args.append((e, preps[e["file"]], {n: p}, cfg["pipeline"], c))
                    owner.append(e["file"])
            t2 = time.time()
            with ProcessPoolExecutor(max_workers=int(cfg["stage2_workers"]), mp_context=mp.get_context("spawn")) as ex:
                parts = list(ex.map(hf.mm_video, args))
            merged = {}
            for v, r in zip(owner, parts):
                merged.setdefault(v, {**r, "detectors": {}})["detectors"].update(r["detectors"])
            stage2 = [merged[e["file"]] for e in s2]
            verified["stage2"] = f"실행함 (영상 {len(stage2)}개, {time.time() - t2:.0f} s)"
            log(f"  2단계 FT: {verified['stage2']}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    existing = mm3_existing(cfg)
    dets_all = ([hf.MM_BOX_SOURCES[cfg["hand_box_source"]]] if cfg["hand_box_source"] == "mediapipe" else []) + names
    slopes = mm2_stage2_slopes(stage2, dets_all, cfg) if stage2 else {}
    for r in stage2:
        log(f"  {r['video']}: " + ", ".join(f"{n} " + (f"{x['ft']['f']:.0f}" if "ft" in x and not x.get("excluded") else "제외") for n, x in r["detectors"].items()))
    mm3_report(run_root / "report.md", cfg, env, checks, agree, order, fail, overlays, verified)
    if stage2:
        mm3_summary(run_root / "summary.md", cfg, stage2, names, existing, slopes)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"verification": {"checks": checks, "order": order, "agreement": agree, "failure": fail, "verified": verified, "env": env},
                                                                    "stage2": stage2, "slopes": slopes}), ensure_ascii=False, indent=1))
    meta = hf.provenance_build(text, a.config, {**{k: v for k, v in cfg.items() if k.startswith("code_")}, "videos": vids, "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, verified


# ====================================================================================================
# fs  (experiment 51: free_shape3d (e3_video of 45, unchanged) on the joints of every detector of 48 / 50: mediapipe from the box-source cache, the five MMPose models from the
#      detection caches of the 48 and 50 runs (never detected anew); 46's settings, verdict rule and reference f)
#     python HandCalib/run.py free-shape3d-detectors --config HandCalib/configs/51_free_shape3d_mmpose.yaml [--out DIR]
#     python HandCalib/run.py free-shape3d-detectors --config ... results-yaml --run runs/51_free_shape3d_mmpose/<UTC>
# ====================================================================================================
FS_NOTE = ("실험 48·50에서 구한 검출기별 관절(mediapipe와 MMPose 손 모델 5종)에 교수님의 free_shape3d를 실험 46·49와 같은 호출·설정(손목 제외 16점, z_rel 0.2, alt_iters 3, max_frames 120, 같은 제외 규칙)으로 "
           "적용해 f를 구한 실험이다. 관절은 기존 검출 캐시에서 읽었고 새로 검출하지 않았다. 판정 규칙은 실험 46과 같다(영상별 3차원 f의 거리-f 기울기, 영상 단위 부트스트랩). 기준 f와의 비교는 summary.md에만 있다.")


def fs_mm_cache(frames, s: dict, box_npz: Path, name: str, cache_dir: Path) -> Path:
    """The cache file swap_mmpose_hand wrote for these frames, settings and boxes (same key); only read."""
    settings = {k: s[k] for k in ("config", "checkpoint", "checkpoint_sha256", "box_source", "batch_size")}
    settings["tf32"] = bool(s.get("tf32", False))
    settings["boxes_sha256"] = hashlib.sha256(Path(box_npz).read_bytes()).hexdigest()
    p = Path(cache_dir) / f"{name}_{hf.frames_key(frames, settings, name)}.npz"
    if not p.exists():
        raise hf.SkipDetector(f"{name} 캐시가 없음 ({p.name}); 새로 검출하지 않음")
    return p


def fs_analyze(results: list, cfg: dict) -> dict:
    """Per set and detector: 46's verdict calculation (s3_slope on the per-video 3-D f, pr_verdict)."""
    ref = {float(d): float(cfg["ref_f"]) for d in (50, 60, 75)}
    bn, bs = int(cfg["bootstrap"]["n"]), int(cfg["bootstrap"]["seed"])
    out = {}
    for st in cfg["sets"]:
        for det in cfg["detectors"]:
            inc = [r for r in results if r["set"] == st["name"] and r["detector"] == det and "est" in r]
            ds = sorted({r["distance_cm"] for r in inc})
            if not ds:
                out.setdefault(st["name"], {})[det] = {"verdict": {"verdict": "판단 불가", "reason": "분석된 영상이 없음"}}
                continue
            fb = {d: [r["est"]["f"] for r in inc if r["distance_cm"] == d] for d in ds}
            eb = {d: [r["est"]["at_bound"] or r["est"]["at_grid_edge"] for r in inc if r["distance_cm"] == d] for d in ds}
            sl = hf.s3_slope(fb, eb, bn, bs)
            nv = {d: len(fb.get(d, [])) for d in ref}
            out.setdefault(st["name"], {})[det] = {"slope": sl, "verdict": hf.pr_verdict(sl, ref, cfg["verdict"], nv)}
    return out


def fs_report(path: Path, results: list, cfg: dict, sources: dict):
    fs = cfg["free_shape3d"]
    L = ["# 실험 51: free_shape3d를 검출기별 관절에 적용 (실험 48·50의 캐시)", "", FS_NOTE, "", f"free_shape3d(z_rel {fs['z_rel']}, alt_iters {fs['alt_iters']}, max_frames {fs['max_frames']}), 검출기별 conf_min: "
         + ", ".join(f"{d} {cfg['detector_settings'][d]['conf_min']}" for d in cfg["detectors"]) + ".", "", "## 1. 관절 출처", "", "| 영상 묶음 | 설정 | 검출 캐시 폴더 |", "|---|---|---|"]
    for st in cfg["sets"]:
        L.append(f"| {st['name']} | {st['config']} | {', '.join(sources.get(st['name'], []))} |")
    L += ["", "## 2. 영상별 값", "", "| 영상 묶음 | 검출기 | 영상 | 거리(cm) | 포함 | f 3차원 (px) | f 평면 (px) | 상대 깊이 변위 | f 한계에 붙음 | 평면 격자 최솟값이 가장자리 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r.get("est")
        if x:
            L.append(f"| {r['set']} | {r['detector']} | {r['video']} | {r['distance_cm']:g} | 포함 | {x['f']:.1f} | {x['f_planar']:.1f} | {x['relief_rel']:.4f} | {'예' if x['at_bound'] else '아니오'} | {'예' if x['at_grid_edge'] else '아니오'} |")
        else:
            L.append(f"| {r['set']} | {r['detector']} | {r['video']} | {r['distance_cm']:g} | 제외: {'; '.join(r.get('excluded') or []) or r.get('failed') or '제외'} | - | - | - | - | - |")
    path.write_text("\n".join(L) + "\n")


def fs_summary(path: Path, results: list, ana: dict, cfg: dict, prior: dict):
    ref = float(cfg["ref_f"])
    pct = lambda x: f"{100 * x:+.1f}%"      # noqa: E731
    fe = lambda f: f"{f:.1f} ({pct(f / ref - 1.0)})"      # noqa: E731
    L = ["# 실험 51 summary", "", f"**{FS_NOTE}**", "", f"- 기준 f: {ref:g} px (모든 거리, 사용자 제공). 표의 값 = 초점 거리 px (기준 f 대비 오차). 거리별 f = 그 거리 영상별 3차원 f의 기하평균(실험 46·49와 같은 계산).", ""]
    for k, st in enumerate(cfg["sets"], 1):
        L += [f"## {k}. {st['name']}", "", "| 검출기 | 50 cm | 60 cm | 75 cm | 기울기 (95% 구간) | 판정 | 영상별 중앙 절대 오차 |", "|---|---|---|---|---|---|---|"]
        for det in cfg["detectors"]:
            a = ana[st["name"]][det]
            sl, v = a.get("slope"), a["verdict"]
            cells = []
            for d in (50.0, 60.0, 75.0):
                n_in = sum(1 for r in results if r["set"] == st["name"] and r["detector"] == det and r["distance_cm"] == d and "est" in r)
                cells.append("-" if not sl or d not in sl["pooled_f"] else fe(sl["pooled_f"][d]) + (" 한계" if sl["pooled_edge"][d] else "") + f", 영상 {n_in}")
            lg = [abs(np.log(r["est"]["f"] / ref)) for r in results if r["set"] == st["name"] and r["detector"] == det and "est" in r]
            slope = "-" if not sl or sl["slope"] is None else f"{sl['slope']:+.3f}" + ("" if not sl["interval_95"] else f" ({sl['interval_95'][0]:+.3f} ~ {sl['interval_95'][1]:+.3f})")
            L.append(f"| {det} | " + " | ".join(cells) + f" | {slope} | {v['verdict']} | {'-' if not lg else format(100 * (np.exp(np.median(lg)) - 1.0), '.1f') + '%'} |")
        pv = prior.get(st["name"]) or {}
        if pv:
            same = [r for r in results if r["set"] == st["name"] and r["detector"] == "mediapipe_hands" and "est" in r]
            dmax = max((abs(r["est"]["f"] - pv[r["video"]]) for r in same if r["video"] in pv), default=None)
            L += ["", f"- mediapipe_hands 재현 확인: {st['prior_label']}의 영상별 3차원 f와의 최대 차이 {'-' if dmax is None else format(dmax, '.4f')} px."]
        L += ["", "| 영상 | 거리(cm) | " + " | ".join(cfg["detectors"]) + " |", "|---|---|" + "---|" * len(cfg["detectors"])]
        vids = [r for r in results if r["set"] == st["name"] and r["detector"] == cfg["detectors"][0]]
        for rv in vids:
            cells = []
            for det in cfg["detectors"]:
                r = next(x for x in results if x["set"] == st["name"] and x["detector"] == det and x["video"] == rv["video"])
                cells.append(fe(r["est"]["f"]) + (" 한계" if r["est"]["at_bound"] else "") if "est" in r else "제외: " + ("; ".join(r.get("excluded") or []) or r.get("failed") or "-"))
            L.append(f"| {rv['video']} | {rv['distance_cm']:g} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 한계", "", "- 거리마다 영상이 4개 이하라 기울기 구간이 거칠다. 판정은 이 세션·이 카메라에 대한 것이다.",
          "- 기준 f는 사용자가 이 영상들에 대해 준 값이며 이 코드로 새로 재지 않았다.",
          "- MMPose 모델은 conf_min 0.3(실험 47 설정), mediapipe는 0.5(실험 46 설정)로 관절을 고른다.", ""]
    path.write_text("\n".join(L) + "\n")


def fs_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(hf.PROJECT_ROOT / "configs/51_free_shape3d_mmpose.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(fs_results_yaml(hf.resolve(a.run), cfg))
        return None
    sets = [(st, yaml.safe_load(hf.resolve(st["config"]).read_text())) for st in cfg["sets"]]
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / utc_stamp()
    run_root.mkdir(parents=True)
    log = Log(run_root / "log.txt")
    log(f"51 free_shape3d 검출기별 공식 실행: 영상 묶음 {len(sets)}개, 검출기 {len(cfg['detectors'])}개 → {run_root}")
    t0 = time.time()
    tmp_root = run_root / "_frames"
    tmp_root.mkdir()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    sources = {}
    try:
        items = [(st, sc, e) for st, sc in sets for e in sc["stage2_videos"]]
        with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
            preps = list(ex.map(hf.fresp_prepare, [(e["file"], sc["input_dir"], sc["pipeline"], {}, str(tmp_root)) for st, sc, e in items]))
        log(f"  프레임 준비 {time.time() - t0:.0f} s")
        jobs, meta = [], []
        for (st, sc, e), prep in zip(items, preps):
            box = hf.mm_box_cache(prep["frames"], sc)
            hf.check_alignment(prep["frames"], hf.load_detections(box))
            sources.setdefault(st["name"], [str(Path(d)) for d in sc["box_caches"]] + [str(hf.resolve(st["detection_cache"]))])
            entry = {"file": e["file"], "group": "G", "distance_cm": e["distance_cm"], "hand": "unrecorded"}
            for det in cfg["detectors"]:
                if det == "mediapipe_hands":
                    npz = box
                else:
                    s = dict(sc["detector_settings"][det])
                    s["box_source"] = sc["hand_box_source"]
                    npz = fs_mm_cache(prep["frames"], s, box, det, hf.resolve(st["detection_cache"]))
                c = {"candidate": {**cfg["candidate"], "detector": det}, "exclusion": cfg["exclusion"], "detector_settings": cfg["detector_settings"], "free_shape3d": cfg["free_shape3d"]}
                jobs.append((entry, prep, str(npz), cfg["pipeline"], c))
                meta.append((st["name"], det))
        t1 = time.time()
        with ProcessPoolExecutor(max_workers=int(cfg["fs_workers"]), mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(hf.e3_video, jobs))
        log(f"  free_shape3d {len(jobs)}개 작업: {time.time() - t1:.0f} s")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    results = []
    for (sname, det), r in zip(meta, res):
        r = {k: v for k, v in r.items() if k not in ("group", "hand")}
        results.append({"set": sname, "detector": det, **r})
    ana = fs_analyze(results, cfg)
    prior = {}
    for st in cfg["sets"]:
        q = hf.resolve(st["prior_results"])
        if q.exists():
            prior[st["name"]] = {v["video"]: float(v["est"]["f"]) for v in json.loads(q.read_text())["videos"] if isinstance(v.get("est"), dict)}
    for st in cfg["sets"]:
        for det in cfg["detectors"]:
            v = ana[st["name"]][det]["verdict"]
            log(f"  {st['name']} {det}: {v['verdict']}" + (f" (기울기 {v['slope']:+.3f})" if v.get("slope") is not None else ""))
    fs_report(run_root / "report.md", results, cfg, sources)
    fs_summary(run_root / "summary.md", results, ana, cfg, prior)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"videos": results, "analysis": ana}), ensure_ascii=False, indent=1))
    meta_ = hf.provenance_build(text, a.config, {**{k: v for k, v in cfg.items() if k.startswith("code_")}, "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta_), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, results, ana


def fs_results_yaml(run_dir: Path, cfg: dict):
    """Values and verdicts only (no reference f); written to output.results_yaml."""
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    try:
        run_name = str(run_dir.resolve().relative_to(hf.PROJECT_ROOT))
    except ValueError:
        run_name = str(run_dir)
    rd = lambda x, n=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(x, n)      # noqa: E731
    out = {"experiment": cfg["experiment"]["name"], "run": run_name, "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"], "note": FS_NOTE,
           "free_shape3d": cfg["free_shape3d"], "verdict_rules": cfg["verdict"], "sets": {}}
    for st in cfg["sets"]:
        o = out["sets"][st["name"]] = {}
        for det in cfg["detectors"]:
            a = r["analysis"][st["name"]][det]
            sl = a.get("slope") or {}
            o[det] = {"f_by_distance_px": {f"{float(d):g}": rd(f, 1) for d, f in (sl.get("pooled_f") or {}).items()}, "slope": rd(sl.get("slope")),
                      "slope_interval_95": None if not sl.get("interval_95") else [round(q, 4) for q in sl["interval_95"]], "verdict": a["verdict"]["verdict"],
                      "videos": {v["video"]: (round(v["est"]["f"], 1) if "est" in v else {"excluded": v.get("excluded") or v.get("failed")}) for v in r["videos"] if v["set"] == st["name"] and v["detector"] == det}}
    p = hf.resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p



COMMANDS = {"protocol": protocol_main, "frame-filter": ff_main, "same-session": session_main, "overlay": overlay_main, "phase-segmentation": pseg_main, "guide-revision": grev_main, "f-response": fresp_main, "error-factors": ferr_main, "long-video-check": long_main, "generalization": gen_main, "main-camera-extension": ext_main, "wide-60cm": wide_main, "jointset-sets": jset_main, "verdict-calibration": cal_main, "printed-hand": pr_main, "hand-markers": mk_main, "markers-only": mo_main, "free-template": ft_main, "free-shape3d": s3_main, "free-shape3d-hand": h3_main, "free-shape3d-27": e3_main, "free-shape3d-new": e4_main, "mmpose-hand": mm_main, "mmpose-hand-46": mm2_main, "free-shape3d-new2": e5_main, "mmpose-hand-src": mm3_main, "free-shape3d-detectors": fs_main}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit("usage: run.py {" + ",".join(COMMANDS) + "} [options]   (see the module docstring)")
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
