"""Experiment 18: one recording with a hand segment and a checkerboard segment.

    CUDA_VISIBLE_DEVICES=<free gpu> python HandCalib/run_same_session.py --config HandCalib/configs/18_same_session_check.yaml
    python HandCalib/run_same_session.py --config ... results-yaml --run runs/18_same_session_check/<UTC>

Hand segment: 17's process (imported unchanged) on the cut of the normalised video. Checkerboard segment: this recording's own
reference f (models A-D median, plus the pinhole no-distortion model E). The checkerboard f is never an input to the hand method or
AnyCalib; the comparisons are written to summary.md only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_detector_swap as R16                                     # noqa: E402
import run_hand_focal as RUN14                                       # noqa: E402
import run_protocol_rerun as R17                                     # noqa: E402  (17, unchanged)
from hand_focal import frames as FR                                  # noqa: E402
from hand_focal import normalize as NZ                               # noqa: E402
from hand_focal import provenance as PV                              # noqa: E402
from hand_focal import report as RP                                  # noqa: E402
from hand_focal.detectors import adapters as DA                      # noqa: E402
from hand_focal.paths import PROJECT_ROOT, resolve                   # noqa: E402
from same_session import checkerboard as CB                          # noqa: E402
from same_session import cut as CUT                                  # noqa: E402
from same_session import segment as SG                               # noqa: E402

MANDATORY = R16.MANDATORY


def gpu_info():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:                                             # noqa: BLE001
        q = []
    return {"CUDA_VISIBLE_DEVICES": vis, "gpus_at_start": q}


# ---------------------------------------------------------------- segmentation
def scan(norm_video: Path, fps: float, cfg, tmp: Path, cache_dir: Path, log):
    sc = cfg["segmentation"]
    frames, k = FR.extract(norm_video, tmp / "scan", fps, None, sc["scan_fps"])
    times = np.array([r["time_s"] for r in frames])
    board = np.array([SG.board_found(cv2.imread(r["png"], cv2.IMREAD_GRAYSCALE), cfg["checkerboard"]["inner_corners"], sc["board_detect_long_side"])
                      for r in frames])
    s = cfg["detector_settings"]["wilor_yolo"]
    dcfg = {"python": s["python"], "weights": s["weights"], "conf": s["conf"], "iou": s["iou"], "imgsz": s.get("imgsz"), "device": s["device"], "cache": True}
    det = DA.wilor_yolo(frames, dcfg, cache_dir, log)
    hand = np.zeros(len(frames), bool)
    for i in range(len(frames)):
        a, b = int(det["offsets"][i]), int(det["offsets"][i + 1])
        hand[i] = bool(b > a and (det["box_conf"][a:b] >= sc["hand_min_box_conf"]).any())
    shutil.rmtree(tmp / "scan", ignore_errors=True)
    return times, hand, board


def figure_timeline(path: Path, times, hand, board, seg):
    plt = R16._plt()
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
    plt = R16._plt()
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
def board_frames(norm_video: Path, fps, seg, ccfg, tmp: Path):
    rows, k = FR.extract(norm_video, tmp / "board", fps, None, ccfg["sample_fps"])
    out = []
    for r in rows:
        if seg[0] <= r["time_s"] <= seg[1]:
            out.append({"index": r["index"], "time_s": r["time_s"], "gray": cv2.imread(r["png"], cv2.IMREAD_GRAYSCALE)})
    shutil.rmtree(tmp / "board", ignore_errors=True)
    return out


def process(video: Path, cfg, cfg_text, cfg_path, run_root: Path, study, study_path, log):
    pcfg = cfg["pipeline"]
    tmp = Path(tempfile.mkdtemp(prefix="hf18_"))
    T = {}
    try:
        t = time.time()
        norm = NZ.normalize(video, tmp / "normalized.mkv", pcfg)
        W, H, fps = norm["output"]["display_w"], norm["output"]["display_h"], norm["output"]["fps"]
        T["normalize_s"] = time.time() - t
        t = time.time()
        if cfg["segment_mode"] == "manual":
            m = cfg["manual_segments"][video.name]
            seg = {"hand": list(m["hand"]), "board": list(m["board"]), "warnings": [], "mode": "manual"}
            times = hand = board = None
        else:
            times, hand, board = scan(tmp / "normalized.mkv", fps, cfg, tmp, resolve(cfg["output"]["root"]) / "detection_cache", log)
            seg = SG.split(times, hand, board, cfg["segmentation"]["hand_gap_s"])
            seg["mode"] = "auto"
        T["segment_s"] = time.time() - t
        log(f"== {video.name} == 구간: 손 {seg['hand'][0]:.1f}–{seg['hand'][1]:.1f} s, 체커보드 "
            + (f"{seg['board'][0]:.1f}–{seg['board'][1]:.1f} s" if seg["board"] else "없음") + ("; 경고: " + "; ".join(seg["warnings"]) if seg["warnings"] else ""))
        # hand segment: exact cut of the normalised video, named like the original so 17's folders read <video>/<detector>
        t = time.time()
        hdir = tmp / "hand"
        hdir.mkdir()
        hand_cut = hdir / (video.stem + ".mkv")
        cut_info = CUT.cut(tmp / "normalized.mkv", hand_cut, seg["hand"][0], seg["hand"][1], fps)
        v17 = R17.process(hand_cut, cfg, cfg_text, cfg_path, run_root, study, study_path, True, log)
        T["hand_s"] = time.time() - t
        vdir = run_root / video.stem
        if times is not None:
            (vdir / "figures").mkdir(exist_ok=True)
            figure_timeline(vdir / "figures" / "segments_timeline.png", times, hand, board, seg)
        # checkerboard segment
        t = time.time()
        bres = {"failed": "체커보드 구간 없음", "verdict": "REJECT"}
        if seg["board"]:
            fr = board_frames(tmp / "normalized.mkv", fps, seg["board"], cfg["checkerboard"], tmp)
            bres = CB.analyse(fr, cfg["checkerboard"], W, H, log)
        bdir = vdir / "checkerboard"
        bdir.mkdir(exist_ok=True)
        (bdir / "checkerboard.json").write_text(json.dumps(RP.jsonable(bres), ensure_ascii=False))
        figures_board(bdir, bres)
        T["board_s"] = time.time() - t
        res = {"video": video.name, "size": [W, H], "fps": fps, "normalization": {k: norm[k] for k in ("output", "path")}, "segments": seg,
               "scan": None if times is None else {"t": times.tolist(), "hand": hand.tolist(), "board": board.tolist()},
               "hand_cut": cut_info, "hand": v17, "checkerboard": bres, "timings_s": T}
        for ln in block(res):
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


def block(res):
    s = res["segments"]
    L = [f"  [{res['video']}]  구간: 손 {s['hand'][0]:.1f}–{s['hand'][1]:.1f} s, 체커보드 "
         + (f"{s['board'][0]:.1f}–{s['board'][1]:.1f} s" if s["board"] else "없음"), board_line(res["checkerboard"])]
    v = res["hand"]
    for det, d in v["detectors"].items():
        L.append(R17.det_line(det, d, v["ranking"].get(det)))
    ac = v.get("anycalib")
    if ac:
        L.append("    AnyCalib " + ", ".join(f"{m.replace('anycalib_', '')}: 중앙값 {x['summary']['median']:.1f} px (IQR {x['summary']['iqr'][0]:.1f}–{x['summary']['iqr'][1]:.1f})"
                                           for m, x in ac["models"].items()))
    c = v["conditions"]
    L.append(f"    손 구간 촬영 조건: {'가이드 충족' if c['guide'] else '가이드 미충족'} ({R17.CO.short_text(c['metrics'])})")
    if c["reasons"]:
        L.append("      미충족 이유: " + "; ".join(c["reasons"]))
    return L


# ---------------------------------------------------------------- reports
def report(path, vids, cfg, prov, gpu):
    L = ["# 실험 18: 같은 녹화의 체커보드로 기준 f 확인", "", MANDATORY, "",
         "체커보드 기준 f는 비교용이며 손 방식·AnyCalib의 입력·보정·초기값으로 쓰지 않았다. 비교 수치는 summary.md에만 적었다.", "",
         "## 1. 영상별 결과", ""]
    for v in vids:
        L += ["```"] + block(v) + ["```", ""]
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
                cells.append(f"{R16.f_text(d['fixed_template'], d['grid_px'][1])} ({d['verdict']})")
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
        L.append(f"\n시간 구간별 f_reference: " + ", ".join(RP.fmt(x) for x in b["segment_f"]))
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


def rel(x, ref):
    return f"{100 * (x / ref - 1):+.1f}%"


def summary_md(path, vids, cfg, regression):
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
        L += [f"- 체커보드 기준 f = {fr:.1f} px{tag}; 모델 E = {b['f_model_E_px']:.1f} px ({rel(b['f_model_E_px'], fr)})",
              f"- 이전 기준 {prev:.0f} px 대비 이 영상 기준 f: {rel(fr, prev)}", "",
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
                    L.append(f"| {det} {mname} | 값 없음 ({side}) | {'≥' if side == '상한' else '≤'} {rel(s['f'], fr)} | - |")
                else:
                    L.append(f"| {det} {mname} | {s['f']:.1f} px | {rel(s['f'], fr)} | {rel(s['f'], b['f_model_E_px'])} |")
        for m, x in (v["hand"].get("anycalib") or {"models": {}})["models"].items():
            L.append(f"| AnyCalib {m} | {x['summary']['median']:.1f} px | {rel(x['summary']['median'], fr)} | {rel(x['summary']['median'], b['f_model_E_px'])} |")
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
        norm = NZ.normalize(Path(p).expanduser(), tmp / "n.mkv", cfg["pipeline"])
        W, H, fps = norm["output"]["display_w"], norm["output"]["display_h"], norm["output"]["fps"]
        dur = norm["output"]["duration_s"] or 1e9
        fr = board_frames(tmp / "n.mkv", fps, [0.0, float(dur)], cfg["checkerboard"], tmp)
        return CB.analyse(fr, cfg["checkerboard"], W, H, log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def results_yaml(run_dir: Path, cfg):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    out = {"experiment": "18_same_session_check", "run": str(run_dir.relative_to(PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
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
    p = resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(RP.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/18_same_session_check.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.cmd == "results-yaml":
        print(results_yaml(resolve(a.run), cfg))
        return None
    if a.videos:
        cfg["videos"] = a.videos.split(",")
    videos = R17.select_videos(cfg)
    gpu = gpu_info()
    sp = resolve(cfg["reuse_14"]["synthetic_study"])
    study = json.loads(sp.read_text())
    run_root = Path(a.out).expanduser() if a.out else resolve(cfg["output"]["root"]) / RUN14.utc_stamp()
    run_root.mkdir(parents=True)
    log = RUN14.Log(run_root / "log.txt")
    log(f"18 같은 녹화 확인: {[v.name for v in videos]} → {run_root} (CUDA_VISIBLE_DEVICES={gpu['CUDA_VISIBLE_DEVICES']})")
    t0 = time.time()
    vids = [process(v, cfg, text, a.config, run_root, study, str(sp), log) for v in videos]
    reg = regression_check(cfg, log)
    prov = R16.detector_provenance(cfg)
    report(run_root / "report.md", vids, cfg, prov, gpu)
    summary_md(run_root / "summary.md", vids, cfg, reg)
    (run_root / "results.json").write_text(json.dumps(RP.jsonable({"videos": vids, "regression": reg}), ensure_ascii=False, indent=1))
    meta = PV.build(text, a.config, {"code_17": cfg["code_17"], "detectors": prov, "gpu": gpu, "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(RP.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, vids


if __name__ == "__main__":
    main()
