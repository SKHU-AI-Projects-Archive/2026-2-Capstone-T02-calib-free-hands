"""Diagnostic: the 17 pipeline with the tilt-phase frame filter (theta2d <= 45 deg, box off the border), on the 17 runs' detections.

    python HandCalib/run_frame_filter_diag.py --config HandCalib/configs/17_frame_filter_diag.yaml

Not an official estimator. The comparison with the user's checkerboard f (reference_f_px) is made only in this diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hand_focal as RUN14                                   # noqa: E402
from hand_focal import provenance as PV                          # noqa: E402
from hand_focal import report as RP                              # noqa: E402
from hand_focal.paths import resolve                             # noqa: E402
from protocol_rerun import frame_filter as FF                    # noqa: E402

HEAD = "진단 결과이며 공식 f 추정값이 아님. 체커보드 f와의 비교는 이 진단에서만 한다."
KO = {"interior": "내부", "upper": "상한", "lower": "하한"}


def fval(sol, hi):
    if sol is None:
        return "값 없음 (실패)"
    if sol["at_grid_edge"]:
        return "값 없음 (상한)" if sol["f"] >= hi * (1 - 1e-6) else "값 없음 (하한)"
    return f"{sol['f']:.1f} px"


def rel(sol, ref, hi):
    if sol is None:
        return "-"
    if sol["at_grid_edge"]:
        return f"{'≥ +' if sol['f'] >= hi * (1 - 1e-6) else '≤ '}{100 * (sol['f'] / ref - 1):.0f}%"
    return f"{100 * (sol['f'] / ref - 1):+.1f}%"


def official(run_dir: Path):
    r = json.loads((run_dir / "results.json").read_text())
    a = r.get("analysis", {})
    if "solutions" not in a:
        return None, None, r.get("verdict")
    return a["solutions"]["fixed_template"], a["solutions"]["joint_delta"], r["verdict"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", action="append", default=None, help="VIDEO=RUN_DIR; replaces the yaml sources (repeatable)")
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    cfg = yaml.safe_load(text)
    if a.source:
        cfg["sources"] = dict(x.split("=", 1) for x in a.source)
    c17 = yaml.safe_load(resolve(cfg["pipeline_config"]).read_text())
    out = resolve(cfg["output"]["root"]) / f"{RUN14.utc_stamp()}_diag_framefilter"
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    ref = float(cfg["reference_f_px"])
    log = RUN14.Log(out / "log.txt")
    log(f"프레임 거르기 진단 → {out}")
    res = {}
    t0 = time.time()
    for video, src in cfg["sources"].items():
        res[video] = {}
        for det in cfg["detectors"]:
            pc = copy.deepcopy(c17["pipeline"])
            pc["tracking"]["conf_min"] = float(c17["detector_settings"][det]["conf_min"])
            rd = resolve(src) / det
            t = time.time()
            r = FF.run_one(rd, pc, cfg["filter"])
            ft0, jd0, v0 = official(rd)
            r["official"] = {"fixed_template": ft0, "joint_delta": jd0, "verdict": v0}
            res[video][det] = r
            if "failed" in r:
                log(f"  {video} {det}: 실패 ({r['failed']})")
                continue
            hi = r["grid_px"][1]
            pm = r["per_method"]["fixed_template"]
            log(f"  {video} {det}: 기울기 프레임 {r['n_tilt_kept']}/{r['n_tilt_before']} 유지, 템플릿 동일 {r['same_template']}, "
                f"f {fval(r['solutions']['fixed_template'], hi)} (판정 {pm['verdict']}), 거르기 전 {fval(ft0, hi)} — {time.time() - t:.0f} s")
    figure(out, res, ref)
    lines = summary(res, cfg, ref)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "diagnostic.json").write_text(json.dumps(RP.jsonable(res), ensure_ascii=False))
    (out / "metadata.json").write_text(json.dumps(RP.jsonable(PV.build(text, a.config, {"sources": cfg["sources"], "pipeline_config_sha256": PV.sha256_text(resolve(cfg["pipeline_config"]).read_text()),
                                                                                         "elapsed_s": time.time() - t0})), ensure_ascii=False, indent=1))
    for ln in lines:
        log(ln)
    log.close()
    return out, res


def summary(res, cfg, ref):
    f = cfg["filter"]
    L = ["# 17 진단: 기울기 구간 프레임 거르기", "", f"**{HEAD}**", "",
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
            L.append(f"| {v} | {det} | {r['n_tilt_kept']}/{r['n_tilt_before']} | {fval(ft0, hi)} | {rel(ft0, ref, hi)} | **{fval(ft, hi)}** | {rel(ft, ref, hi)} | "
                     f"{pm['verdict']} | {'계산 불가' if edge else format(100 * pm['f_rel_uncertainty'], '.0f') + '%'} | {pm['sharpness']:.3f} | "
                     f"{100 * pm['reproj_rms_rel']:.2f}% | {fval(r['solutions']['joint_delta'], hi)} |")
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


def figure(out, res, ref):
    import run_detector_swap as R16
    plt = R16._plt()
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


if __name__ == "__main__":
    main()
