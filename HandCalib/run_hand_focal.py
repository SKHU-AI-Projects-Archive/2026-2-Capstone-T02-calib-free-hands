"""Experiment 14: focal length from a hand template (2D YOLO keypoints only), with a separate AnyCalib value on the same frames.

    python HandCalib/run_hand_focal.py --config HandCalib/configs/14_hand_template_focal.yaml                 # estimate (all videos)
    python HandCalib/run_hand_focal.py --config ... estimate --video V.MOV [--estimator joint_delta] [--no-anycalib] [--out DIR]
    python HandCalib/run_hand_focal.py --config ... synthetic-study
    python HandCalib/run_hand_focal.py --config ... normalize V.MOV --out V_norm.mkv

Input videos are only read. Frames go to a temporary directory that is deleted at the end; no frame image is written to the results.
No location or device metadata is read into results, reports or logs (only technical stream fields).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hand_focal import analysis as A                       # noqa: E402
from hand_focal import frames as FR                        # noqa: E402
from hand_focal import normalize as NZ                     # noqa: E402
from hand_focal import provenance as PV                    # noqa: E402
from hand_focal import report as RP                        # noqa: E402
from hand_focal import track as TR                         # noqa: E402
from hand_focal.detectors import adapters as DA            # noqa: E402
from hand_focal.paths import PROJECT_ROOT, resolve         # noqa: E402


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


def load_cfg(path):
    text = Path(path).read_text()
    return yaml.safe_load(text), text


# ---------------------------------------------------------------- estimator resolution
def latest_study(cfg) -> Path | None:
    p = cfg["estimate"].get("synthetic_study_result")
    if p:
        return resolve(p)
    root = resolve(cfg["output"]["root"])
    c = sorted(root.glob("*_synthetic/study.json"))
    return c[-1] if c else None


def resolve_estimator(cfg, override=None):
    sp = latest_study(cfg)
    study = json.loads(sp.read_text()) if sp and sp.exists() else None
    est = override or cfg["estimate"]["estimator"]
    basis = "command line" if override else "yaml"
    if est == "auto":
        if study is None:
            raise SystemExit("estimator: auto 는 합성 연구 결과가 필요합니다. 먼저 synthetic-study 를 실행하세요.")
        est = study["adoption"]["estimator"]
        basis = f"synthetic-study rule ({sp.relative_to(PROJECT_ROOT) if sp.is_relative_to(PROJECT_ROOT) else sp})"
    if est not in A.METHODS:
        raise SystemExit(f"unknown estimator {est}")
    return est, basis, study, (str(sp) if sp else None)


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
        info = NZ.probe(video)
        if info["duration_s"] < float(cfg["min_duration_s"]):
            warnings.append(f"영상 길이 {info['duration_s']:.1f} s 가 {cfg['min_duration_s']} s 보다 짧습니다")
            log(f"  경고: {warnings[-1]}")
        norm = NZ.normalize(video, tmp / "normalized.mkv", cfg)
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        fps = norm["output"]["fps"]
        T["normalize_s"] = time.time() - t
        log(f"  정규화: {info['display_w']}x{info['display_h']} (회전 {info['rotation_deg']}°) -> {W}x{H}, {fps:.3f} fps, 톤매핑 {norm['path']['tonemap']}")

        # 2. frames
        t = time.time()
        frames, k = FR.extract(tmp / "normalized.mkv", tmp / "frames", fps, None, cfg["frames"]["detect_fps"])
        times = np.array([r["time_s"] for r in frames])
        T["frames_s"] = time.time() - t
        log(f"  프레임: {len(frames)}장 (매 {k}번째, 실효 {fps / k:.2f} fps)")

        # 3. detection
        t = time.time()
        det = DA.run(frames, cfg["detector"], resolve(cfg["output"]["root"]) / "detection_cache", log)
        T["detect_s"] = time.time() - t
        shutil.copy2(det["cache_file"], out / "detections.npz")
        log(f"  검출: 후보 {len(det['boxes'])}개, {T['detect_s']:.1f} s")

        # 4. tracking
        tr = TR.track(det, times, (W, H), cfg["tracking"], front_start_s=float(cfg["phases"]["front_skip_s"]))
        order = TR.order_check(tr["kp"], tr["mask"])
        if order["ok"] is False:
            warnings.append("키포인트 순서가 OpenPose 순서와 맞지 않아 보입니다(" + ", ".join(k for k in ("middle_longest", "pinky_shortest", "thumb_base_nearer_index") if not order[k]) + ")")
        if (tr["n_candidates"] > 1).any():
            warnings.append(f"손 후보가 2개 이상인 프레임 {int((tr['n_candidates'] > 1).sum())}장 (IoU 추적으로 대상 손 유지)")
        log(f"  추적: 검출률 {tr['present'].mean():.3f}, 순서 점검 {order['ok']}")

        # 5. analysis
        t = time.time()
        a = A.analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), cfg, methods_full=(estimator,),
                      seed=int(cfg["experiment"]["seed"]))
        T["analysis_s"] = time.time() - t
        warnings += a["warnings"]

        # 6. AnyCalib (separate; never used by the hand method)
        anyc = None
        if use_anycalib and cfg["anycalib"]["enabled"]:
            from hand_focal import anycalib_ref as AC
            t = time.time()
            anyc = AC.run(frames, cfg["anycalib"], a.get("tilt_start_s", float(cfg["phases"]["front_phase_s"])), log)
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
                from hand_focal.synthetic import expected_error
                expected = expected_error(study, a["front_jitter_rel"], estimator)
            dv = RP.derived(pm["f"], W, H, cfg["target_sizes"])
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
            RP.report_md(out / "report.md", ctx)
            RP.per_frame_csv(out / "per_frame.csv", frames, tr, a, estimator, anyc)
            res["figures"] = RP.figures(out, a, times, estimator, anyc)
            log(f"  f_hand ({estimator}) {pm['f']:.1f} px, 판정 {pm['verdict']['verdict']}, 휴리스틱 상대 불확실도 {100 * pm['f_rel_uncertainty']:.1f}%")
            for m_ in A.METHODS:
                if a["solutions"][m_]["at_grid_edge"]:
                    log(f"  경고: {m_} 값은 f 격자 경계({a['solutions'][m_]['f']:.0f} px)입니다(내부 최솟값 없음)")
            log(f"  두 방식: fixed_template {res['f_fixed_template_px']:.1f} px, joint_delta {res['f_joint_delta_px']:.1f} px")
            if pm["verdict"]["verdict"] != "GOOD":
                for x in pm["advice"]:
                    log(f"  안내: {x}")
        RP.write_json(out / "results.json", res)
        small = {k: res.get(k) for k in ("video", "estimator", "verdict", "f_hand_px", "f_hand_at_grid_edge", "f_rel_uncertainty", "f_fixed_template_px", "f_joint_delta_px",
                                         "frontal_assumption_sensitivity")}
        if anyc:
            small["anycalib_median_px"] = {m: d["summary"]["median"] for m, d in anyc["models"].items()}
        small["timings_s"] = T
        (out / "result.yaml").write_text(yaml.safe_dump(RP.jsonable(small), allow_unicode=True, sort_keys=False))
        extra = {"video": video.name, "detector": {"weights": cfg["detector"]["weights"], "weights_sha256": PV.sha256_file(cfg["detector"]["weights"])
                                                   if cfg["detector"]["name"] == "wilor_yolo" else None, "info": det.get("info")},
                 "ffmpeg": norm["tools"], "anycalib": ({m: {"weight": d["weight"], "sha256": d["weight_sha256"]} for m, d in anyc["models"].items()}
                                                       | {"hub_check": anyc["hub_check"]}) if anyc else None,
                 "estimator": estimator, "estimator_basis": basis, "synthetic_study": study_path}
        RP.write_json(out / "metadata.json", PV.build(cfg_text, cfg_path, extra))
        log("  단계 시간(s): " + ", ".join(f"{k} {v:.1f}" for k, v in T.items()))
        return RP.summary_row(video.name, a, estimator, anyc, info["duration_s"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        log.close()


def print_table(rows, log=print):
    log("")
    log("| 영상 | 길이(s) | 판정 | f_hand(px) | 휴리스틱 불확실도 | fixed_template | joint_delta | AnyCalib 중앙값 |")
    log("|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"          # noqa: E731
        unc = "-" if r["unc"] is None else f"{100 * r['unc']:.1f}%"
        log(f"| {r['video']} | {f(r['duration_s'])} | {r['verdict']} | {f(r['f_hand'])} | {unc} | "
            f"{f(r['fixed'])} | {f(r['joint'])} | {f(r['anycalib'])} |")


def cmd_estimate(a, cfg, text):
    est, basis, study, sp = resolve_estimator(cfg, a.estimator)
    if a.video:
        videos = [Path(a.video).expanduser()]
    else:
        d = Path(a.input_dir or cfg["input_dir"]).expanduser()
        exts = {e.lower() for e in cfg["video_extensions"]}
        videos = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not videos:
        raise SystemExit("처리할 영상이 없습니다")
    out_root = Path(a.out).expanduser() if a.out else resolve(cfg["output"]["root"]) / f"{utc_stamp()}_{est}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"추정 방식 {est} ({basis}), 결과 {out_root}")
    rows = []
    for v in videos:
        try:
            rows.append(process_video(v, cfg, text, a.config, out_root, est, basis, study, sp, not a.no_anycalib))
        except DA.DetectorError as e:
            print(f"  검출기 오류: {e}")
            rows.append({"video": v.name, "duration_s": None, "verdict": "ERROR", "f_hand": None, "unc": None, "fixed": None, "joint": None, "anycalib": None})
    log = Log(out_root / "summary.md")
    print_table(rows, log)
    log.close()
    return out_root, rows


def cmd_synthetic(a, cfg, text):
    from hand_focal import synthetic as S
    out = resolve(cfg["output"]["root"]) / f"{utc_stamp()}_synthetic"
    out.mkdir(parents=True)
    print(f"합성 연구 → {out}")
    st = S.run_study(cfg, print)
    RP.write_json(out / "study.json", st)
    RP.write_json(out / "metadata.json", PV.build(text, a.config, {"study": "synthetic", "elapsed_s": st["elapsed_s"]}))
    lines = synthetic_md(st, cfg)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def synthetic_md(st, cfg):
    L = ["# 합성 연구 (실험 14)", "", f"- 조건 {len(st['conditions'])}개 × 반복 {cfg['synthetic']['reps']}, 소요 {st['elapsed_s'] / 60:.1f} 분, "
         f"부트스트랩 {cfg['synthetic']['n_bootstrap']}회(연구용 축소)", f"- f {cfg['synthetic']['f_px']} px, {cfg['synthetic']['image']}, 거리 {cfg['synthetic']['distance_mm']} mm", "",
         "| 조건 | 방식 | |오차| 중앙값 | p90 | 부호 평균 | GOOD/ROUGH/REJECT | GOOD 실패 항목(빈도) | δ 추정 중앙값 | 정면 지터/박스 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in st["summary"].items():
        for m in A.METHODS:
            if m not in s:
                continue
            x = s[m]
            v = x["verdicts"]
            fails = ", ".join(f"{k} {c}" for k, c in sorted(x["failed_good_counts"].items(), key=lambda kv: -kv[1]))
            L.append(f"| {name} | {m} | {100 * x['median_abs']:.2f}% | {100 * x['p90_abs']:.2f}% | {100 * x['signed_mean']:+.2f}% | {v['GOOD']}/{v['ROUGH']}/{v['REJECT']} | "
                     f"{fails} | {x['delta_est_median']:.1f} | {100 * s['front_jitter_rel_median']:.2f}% |")
    L += ["", "## 판정 지표 중앙값 (조건별, 문턱값 점검용)", "| 조건 | 방식 | θ p90 | 축 구간 | RMS/박스 | 집합 범위 | 부트스트랩 | 일관성 | |δ| | 선명도 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in st["summary"].items():
        for m in A.METHODS:
            if m in s:
                q = s[m]["metric_medians"]
                L.append(f"| {name} | {m} | {q['theta_p90']:.1f} | {q['axis_bins6']:.0f} | {100 * q['reproj_rms_rel']:.2f}% | {100 * q['jointset_range']:.1f}% | "
                         f"{100 * q['bootstrap_rel_std']:.2f}% | {100 * q['consistency_range']:.1f}% | {abs(q['delta_deg']):.1f} | {q['sharpness']:.2f} |")
    ad = st["adoption"]
    L += ["", f"## 기본 추정 방식: **{ad['estimator']}** (joint_delta 채택 {ad['adopt_joint_delta']})",
          f"- 규칙: δ ≥ {ad['rule']['delta_min_deg']}° 모든 조건에서 |오차| 중앙값이 {100 * ad['rule']['min_rel_gain']:.0f}% 이상 낮고, δ=0에서 나쁘지 않을 것", ""]
    for r in ad["rows"]:
        L.append(f"- {r['cond']}: fixed {100 * r['fixed']:.2f}%, joint {100 * r['joint']:.2f}%" + (f", 상대 이득 {100 * r['rel_gain']:.0f}%" if "rel_gain" in r else "")
                 + f" → {'통과' if r['pass'] else '실패'}")
    L += ["", "## 회귀 점검 (명세의 기대 범위)"]
    for r in st["regression"]:
        L.append(f"- {r['cond']} / {r['method']}: {100 * r['median_abs']:.2f}% (기대 {100 * r['expected'][0]:.1f}–{100 * r['expected'][1]:.1f}%) → {'범위 안' if r['within'] else '범위 밖'}")
    return L


def cmd_normalize(a, cfg, text):
    dst = Path(a.out).expanduser()
    if dst.exists():
        raise SystemExit(f"출력 파일이 이미 있습니다: {dst}")
    r = NZ.normalize(Path(a.video).expanduser(), dst, cfg)
    print(json.dumps(RP.jsonable({k: r[k] for k in ("output", "path")}), ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/14_hand_template_focal.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("estimate")
    for p, d in ((ap, None), (e, argparse.SUPPRESS)):             # options work before or after the command
        p.add_argument("--input-dir", default=d)
        p.add_argument("--video", default=d)
        p.add_argument("--out", default=d)
        p.add_argument("--estimator", choices=list(A.METHODS), default=d)
        p.add_argument("--no-anycalib", action="store_true", default=False if d is None else d)
    sub.add_parser("synthetic-study")
    n = sub.add_parser("normalize")
    n.add_argument("video")
    n.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg, text = load_cfg(a.config)
    if a.cmd in (None, "estimate"):
        return cmd_estimate(a, cfg, text)
    if a.cmd == "synthetic-study":
        return cmd_synthetic(a, cfg, text)
    if a.cmd == "normalize":
        return cmd_normalize(a, cfg, text)


if __name__ == "__main__":
    main()
