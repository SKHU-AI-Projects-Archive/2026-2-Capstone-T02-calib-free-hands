"""Experiment 17: 16's detector-swap pipeline (imported unchanged) on protocol-compliant videos.

    python HandCalib/run_protocol_rerun.py --config HandCalib/configs/17_protocol_rerun.yaml            # yaml detectors / videos
    python HandCalib/run_protocol_rerun.py --config ... results-yaml --run runs/17_protocol_rerun/<UTC>  # after all official runs

Per video: shooting-condition check (14's indicators, yaml thresholds) first, then each detector's f (values only, no differences or
ratios), the internal quality ranking (not accuracy), and AnyCalib as a separate value.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_detector_swap as R16                                    # noqa: E402  (16, unchanged)
import run_hand_focal as RUN14                                      # noqa: E402
from hand_focal import report as RP                                 # noqa: E402
from hand_focal.paths import PROJECT_ROOT, resolve                  # noqa: E402
from protocol_rerun import conditions as CO                         # noqa: E402
from protocol_rerun import ranking as RK                            # noqa: E402

MANDATORY = R16.MANDATORY
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
    return f"    {det:<16}: f = {s:<16} 판정 {d['verdict']:<7} 날카로움 {R16._n(sh)}   (joint_delta: {sj}){r}"


def video_block(v):
    c = v["conditions"]
    head = "가이드 충족" if c["guide"] else "가이드 미충족"
    L = [f"  [{v['video']}]{' (이전 영상)' if v['previous'] else ''}  촬영 조건: {head} ({CO.short_text(c['metrics'])})"]
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
    plt = R16._plt()
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


def process(video, cfg, cfg_text, cfg_path, run_root, study, study_path, use_anycalib, log):
    v = R16.process(video, cfg, cfg_text, cfg_path, run_root, study, study_path, use_anycalib, log)
    vdir = run_root / video.stem
    pcfg = cfg["pipeline"]
    conds = {}
    for det, d in v["detectors"].items():
        if "skipped" in d:
            continue
        tc = dict(pcfg["tracking"], conf_min=float(cfg["detector_settings"][det]["conf_min"]))
        conds[det] = CO.metrics_from_run(vdir / det, tc, float(pcfg["phases"]["front_skip_s"]))
    ref = cfg["conditions"]["reference_detector"]
    if ref not in conds:
        ref = next(iter(conds), None)
    m = conds.get(ref, {"failed": "검출기 없음"})
    chk = CO.check(m, cfg["conditions"])
    v["conditions"] = {"reference": ref, "metrics": {k: x for k, x in m.items() if k != "series"}, **chk,
                       "by_detector": {d: {"metrics": {k: x for k, x in mm.items() if k != "series"}, **CO.check(mm, cfg["conditions"])}
                                       for d, mm in conds.items()}}
    v["previous"] = video.name in set(cfg.get("previous_videos") or [])
    v["ranking"] = RK.rank(v["detectors"], cfg["ranking"])
    figure_conditions(vdir, conds, cfg["conditions"])
    (vdir / "conditions_series.json").write_text(json.dumps(RP.jsonable({d: mm.get("series") for d, mm in conds.items()})))
    for line in video_block(v):
        log(line)
    return v


def report(path, vids, rsum, cfg, prov):
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
        L.append(f"| {v['video']}{' (이전)' if v['previous'] else ''} | {'충족' if v['conditions']['guide'] else '미충족'} | {CO._f(m['box_short_ratio'], 2)} | "
                 f"{CO._p(m['edge_rate'])} | {'없음' if m['transition_s'] is None else format(m['transition_s'], '.2f')} | {CO._p(m['front_jitter_rel'], 2)} | "
                 f"{CO._f(m['theta_p95_deg'], 0)}° | {CO._f(m['theta_tilt_median_deg'], 0)}° | {CO._f(m['theta_max_deg'], 0)}° | {m['axis_bins']}/6 | "
                 f"{CO._f(m['finger_extension_min'], 3)} | {'; '.join(v['conditions']['reasons']) or '-'} |")
    L += ["", "검출기별 같은 지표:", "", "| 영상 | 검출기 | 충족 | 지표 |", "|---|---|---|---|"]
    for v in vids:
        for d, x in v["conditions"]["by_detector"].items():
            L.append(f"| {v['video']} | {d} | {'충족' if x['guide'] else '미충족'} | {CO.short_text(x['metrics'])} |")
    L += ["", "## 2. 영상별 결과", ""]
    for v in vids:
        L += ["```"] + video_block(v) + ["```", ""]
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
            L.append(f"| {v['video']} | {g} | {det} | {R16.f_text(ft, d['grid_px'][1])} | "
                     f"{'계산 불가' if ft['at_grid_edge'] else R16.spread_text(d['f_rel_uncertainty'], False)} | {d['verdict']} | {R16._n(sh)} | "
                     f"{R16.f_text(jd, d['grid_px'][1])} | {rk.get('rank', '-'):g} |")
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
            L.append(f"| {v['video']} | {det} | {r['rank']:g} | {r['category']} | {R16._n(k['sharpness'])} | "
                     f"{'계산 불가' if k['f_rel_uncertainty'] is None else format(100 * k['f_rel_uncertainty'], '.1f') + '%'} | "
                     f"{'-' if k['reproj_rms_rel'] is None else format(100 * k['reproj_rms_rel'], '.2f') + '%'} | {R16._n(k['detect_rate'])} |")
    L += ["", "## 5. 관절 집합별 비용 프로파일 (진단, f 값 아님)", "", "| 영상 | 검출기 | 관절 집합 | 공식 범위 | 위치(px) | 날카로움 | 연장 격자(0.2–5×) | 위치(px) |",
          "|---|---|---|---|---|---|---|---|"]
    ko = {"interior": "내부", "upper": "상한", "lower": "하한", "none": "-"}
    for v in vids:
        for det, d in v["detectors"].items():
            for k, x in d.get("profiles", {}).items():
                o, e = x["official"], x["extended"]
                L.append(f"| {v['video']} | {det} | {k} | {ko[o['at']]} | {R16._px(o['min_at_px'])} | {R16._n(o['sharpness'])} | {ko[e['at']]} | {R16._px(e['min_at_px'])} |")
    L += ["", "## 6. AnyCalib (검출기와 무관한 별도 값)", "", "| 영상 | 모델 | 중앙값 | IQR | p10–p90 | 정면 구간 중앙값 | 기울기 구간 중앙값 | 변동계수 | 출처 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in vids:
        ac = v.get("anycalib")
        if not ac:
            continue
        for mname, x in ac["models"].items():
            s = x["summary"]
            L.append(f"| {v['video']} | {mname} | {s['median']:.1f} | {s['iqr'][0]:.1f}–{s['iqr'][1]:.1f} | {s['p10_p90'][0]:.1f}–{s['p10_p90'][1]:.1f} | "
                     f"{RP.fmt(s['front_median'])} | {RP.fmt(s['tilt_median'])} | {100 * s['cv']:.1f}% | {'14 재사용' if ac.get('reused_from_14') else '이번 실행'} |")
    L += ["", "## 7. 검출기 정보", ""]
    for det, p in prov.items():
        L.append(f"- **{det}**: {p.get('model_name')}; 가중치 sha256 {str(p.get('weights_sha256'))[:16]}…; 실행 장치 {p.get('device')}; 라이선스 {p.get('license')}; "
                 f"학습 라벨: {p.get('training_labels')}")
    L += ["", "## 8. 한계", "",
          "- 정답 f가 없어 오차를 계산하지 않았다. 품질 순위는 내부 지표 순위이며 정확도 순위가 아니다.",
          "- 촬영 조건은 기준 검출기의 2D 키포인트로 잰 값이다(θ2d는 2D 아핀 지표).",
          "- 박스 정의가 검출기마다 다르다(16과 같음). 파이프라인 설정과 판정 문턱값은 모든 검출기에 같다.", ""]
    path.write_text("\n".join(L) + "\n")


def results_yaml(run_dir: Path, cfg):
    r = json.loads((run_dir / "results.json").read_text())
    m = json.loads((run_dir / "metadata.json").read_text())
    out = {"experiment": "17_protocol_rerun", "run": str(run_dir.relative_to(PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
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
    p = resolve(cfg["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(RP.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/17_protocol_rerun.yaml"))
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
        print(results_yaml(resolve(a.run), cfg))
        return None
    if a.detectors:
        cfg["detectors"] = a.detectors.split(",")
    videos = select_videos(cfg, a.videos.split(",") if a.videos else None)
    sp = resolve(cfg["reuse_14"]["synthetic_study"])
    study = json.loads(sp.read_text())
    run_root = Path(a.out).expanduser() if a.out else resolve(cfg["output"]["root"]) / RUN14.utc_stamp()
    run_root.mkdir(parents=True)
    log = RUN14.Log(run_root / "log.txt")
    log(f"17 프로토콜 재실행: 검출기 {cfg['detectors']}, 영상 {[v.name for v in videos]} → {run_root}")
    t0 = time.time()
    vids = [process(v, cfg, text, a.config, run_root, study, str(sp), not a.no_anycalib, log) for v in videos]
    guide = {v["video"]: v["conditions"]["guide"] for v in vids}
    rsum = RK.summary({v["video"]: v["ranking"] for v in vids}, guide)
    prov = R16.detector_provenance(cfg)
    report(run_root / "report.md", vids, rsum, cfg, prov)
    (run_root / "results.json").write_text(json.dumps(RP.jsonable({"videos": vids, "ranking_summary": rsum, "detectors": cfg["detectors"]}),
                                                      ensure_ascii=False, indent=1))
    from hand_focal import provenance as PV
    meta = PV.build(text, a.config, {"code_16": cfg["code_16"], "detectors": prov, "elapsed_s": time.time() - t0,
                                     "devices": {k: cfg["detector_settings"][k].get("device") for k in cfg["detectors"]}})
    (run_root / "metadata.json").write_text(json.dumps(RP.jsonable(meta), ensure_ascii=False, indent=1))
    log("")
    log("검출기별 요약 (" + RANK_NOTE + ")")
    for det, s in rsum.items():
        log(f"  {det}: 가이드 충족 영상 평균 순위 {'-' if s['mean_rank_guide'] is None else format(s['mean_rank_guide'], '.2f')} "
            f"({s['n_guide_videos']}개), 판정 {s['verdict_counts']}, 경계 {s['n_edge']}회")
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, vids


if __name__ == "__main__":
    main()
