"""Final benchmark report: the dataset and the calibrator.py performance tables, nothing else.

    python final_report.py [--config ../configs/benchmark.json]

Reads every valid session under data_root (session.json: setting, f_before, f_after,
reference_f, videos[].f) and, for the detected-frame counts, the latest benchmark run
(runs/benchmark/<run>/results.csv) that contains each session.
Writes runs/benchmark_final/<UTC>/final_report.md and final_videos.csv.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from session_rules import check_session, find_sessions

MOTIONS = ("twinkle", "nod", "circle", "mix")


def latest_rows(repo: Path) -> dict:
    rows = {}
    for run in sorted((repo / "runs" / "benchmark").glob("*/results.csv")):     # names are UTC times, so later runs win
        with run.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows[(r["session"], r["file"])] = r
    return rows


def group_row(name, errs, n):
    a = np.abs([e for e in errs if np.isfinite(e)])
    return f"| {name} | {a.mean():.1f}% | {int((a <= 10).sum())} / {n} | {int((a <= 20).sum())} / {n} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "benchmark.json"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parent.parent
    det = latest_rows(repo)

    sessions, videos = [], []
    for s in find_sessions(cfg["data_root"]):
        errors, _, m = check_session(s, probe_videos=False)
        if errors:
            print(f"skip {s.name}: {len(errors)} errors")
            continue
        ref = float(m["reference_f"])
        rotated = any("display_rotation_changed" in (v or {}) for v in (m.get("sources") or {}).values())
        sessions.append((s.name, m, rotated))
        for v in m["videos"]:
            f = v.get("f")
            r = det.get((s.name, v["file"]), {})
            videos.append({"session": s.name, "collector": m["collector"], "file": v["file"], "motion": v["motion"],
                           "f": f, "reference_f": ref,
                           "error_pct": 100 * (f / ref - 1) if isinstance(f, (int, float)) else float("nan"),
                           "detected": r.get("n_detected", ""), "frames": r.get("n_frames", "")})

    out = repo / "runs" / "benchmark_final" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    with (out / "final_videos.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(videos[0]))
        w.writeheader()
        w.writerows(videos)

    L = ["# HandCalib 벤치마크 결과", "",
         f"- 세션 {len(sessions)}개, 손 영상 {len(videos)}개 (세션마다 twinkle, nod, circle, mix 각 2개)",
         "- f 추정: calibrator.py `Calibrator(\"hand\")` (가이드 7장 코드 그대로)",
         "- 참값: 같은 세션 체커보드 영상 2개(checkerboard_before, checkerboard_after)의 `Calibrator(\"checkerboard\", (7, 7))` 값의 평균",
         "- 오차 = (추정 f - 참값) / 참값", "",
         "## 1. 데이터셋", "",
         "| 세션 | 수집자 | 기기 | 렌즈 | 디지털 줌 | 해상도 | fps | f_before | f_after | 참값 f | 비고 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m, rotated in sessions:
        cb = m["checkerboard"]
        L.append(f"| {name} | {m['collector']} | {m['device']} | {m['lens']} | {m['digital_zoom']} | "
                 f"{m['resolution'][0]}x{m['resolution'][1]} | {m['fps']} | {cb['f_before']} | {cb['f_after']} | {m['reference_f']} | "
                 + ("세로로 기록된 표시 회전을 가로로 수정(재인코딩 없음)" if rotated else "") + " |")

    head = ["| 묶음 | 절대오차 평균값 | 10% 이내 | 20% 이내 |", "|---|---|---|---|"]
    L += ["", "## 2. 성능 평가", "", "### 전체", ""] + head
    for mo in MOTIONS:
        sel = [v for v in videos if v["motion"] == mo]
        L.append(group_row(mo, [v["error_pct"] for v in sel], len(sel)))
    L.append(group_row("전체", [v["error_pct"] for v in videos], len(videos)))

    L += ["", "### 세션별", "", "| 세션 | 참값 f | 절대오차 평균값 | 10% 이내 | 20% 이내 |", "|---|---|---|---|---|"]
    for name, m, _ in sessions:
        sel = [v for v in videos if v["session"] == name]
        L.append(f"| {name} | {m['reference_f']} " + group_row("", [v["error_pct"] for v in sel], len(sel))[3:])

    for coll in sorted({v["collector"] for v in videos}):
        L += ["", f"### 수집자 {coll}", ""] + head
        for mo in MOTIONS:
            sel = [v for v in videos if v["collector"] == coll and v["motion"] == mo]
            L.append(group_row(mo, [v["error_pct"] for v in sel], len(sel)))
        sel = [v for v in videos if v["collector"] == coll]
        L.append(group_row("전체", [v["error_pct"] for v in sel], len(sel)))

    L += ["", "### 영상별", ""]
    for name, m, _ in sessions:
        L += [f"#### {name} (참값 {m['reference_f']})", "", "| 영상 | 추정 f | 오차 | 검출 프레임 |", "|---|---|---|---|"]
        for v in [v for v in videos if v["session"] == name]:
            e = v["error_pct"]
            es = f"{e:+.1f}%" if np.isfinite(e) else "-"
            if np.isfinite(e) and abs(e) > 20:
                es = f"**{es}**"
            L.append(f"| {Path(v['file']).stem} | {v['f']} | {es} | {v['detected']} / {v['frames']} |")
        L.append("")
    (out / "final_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out / "final_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
