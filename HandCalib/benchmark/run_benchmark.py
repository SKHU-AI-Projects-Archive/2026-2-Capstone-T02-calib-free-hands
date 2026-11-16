"""Score calibrator.py (hand mode) on every hand video of a HandCalib benchmark.

    python run_benchmark.py --config ../configs/benchmark.json
    python run_benchmark.py --config ../configs/benchmark.json --resume runs/benchmark/20261116T...Z

For each session the guide rules are checked first (session_rules.check_session).
Sessions with errors are skipped unless include_invalid is true. Each hand video
is run through Calibrator("hand").run() exactly as check_session.py does, and the
estimate is compared with session.json reference_f.

Outputs in the run folder:
    run.json      config, code version, environment, per-session validation result
    results.csv   one row per hand video (f estimate, detections, time, status)
    summary.md    per-video errors and aggregate metrics against reference_f
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibrator import Calibrator
from session_rules import HAND_STEMS, check_session, find_sessions, find_video

FIELDS = ("session", "file", "motion", "device", "lens", "resolution", "digital_zoom",
          "status", "f", "n_frames", "n_detected", "seconds", "message")


def gpu_temperature():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return max(int(x) for x in out.split())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def thermal_guard(cfg):
    pause, resume = cfg["thermal"]["pause_c"], cfg["thermal"]["resume_c"]
    t = gpu_temperature()
    if t is None or t < pause:
        return
    print(f"  GPU {t}°C >= {pause}°C, {resume}°C 미만이 될 때까지 대기", flush=True)
    while t is not None and t >= resume:
        time.sleep(cfg["thermal"]["poll_s"])
        t = gpu_temperature()


def git_state(repo):
    def run(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def environment():
    import onnxruntime
    import rtmlib
    return {"python": sys.version.split()[0], "rtmlib": getattr(rtmlib, "__version__", "?"),
            "onnxruntime": onnxruntime.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "providers": onnxruntime.get_available_providers()}


def score_video(hand, path):
    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    hand.reset()
    t0 = time.time()
    row = {"n_frames": n_frames}
    try:
        f = float(hand.run(str(path)))
        row.update(status="ok", f=round(f, 1), message="")
    except Exception as e:  # calibrator.py ends with an error when the hand is barely detected
        row.update(status="error", f="", message=f"{type(e).__name__}: {e}".replace("\n", " ")[:200])
    row["n_detected"] = len(hand.observations)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def metrics(errors):
    e = np.asarray(errors, dtype=float)
    if e.size == 0:
        return None
    a = np.abs(e)
    return {"n": int(e.size), "median_abs": float(np.median(a)), "mean_abs": float(a.mean()),
            "median_signed": float(np.median(e)), "within_5": float((a <= 5).mean()),
            "within_10": float((a <= 10).mean()), "within_20": float((a <= 20).mean())}


def metric_row(name, n_total, m):
    if m is None:
        return f"| {name} | {n_total} | 0 | - | - | - | - | - | - |"
    return (f"| {name} | {n_total} | {m['n']} | {m['median_abs']:.1f} | {m['mean_abs']:.1f} | "
            f"{m['median_signed']:+.1f} | {100 * m['within_5']:.0f} | {100 * m['within_10']:.0f} | "
            f"{100 * m['within_20']:.0f} |")


def write_summary(out, rows, refs, validation, cfg, info):
    lines = ["# HandCalib benchmark", "",
             f"- run: {out.name}",
             f"- code: {info['git']['branch']} {info['git']['commit'][:7]} (dirty: {info['git']['dirty']})",
             f"- data_root: {cfg['data_root']}",
             f"- sessions: {len(validation)} found, "
             f"{sum(v['used'] for v in validation.values())} scored", ""]
    lines += ["## Sessions", "", "| session | reference f | checkerboard diff % | guide check | used |",
              "|---|---|---|---|---|"]
    for name, v in validation.items():
        diff = v.get("checkerboard_diff")
        lines.append(f"| {name} | {v.get('reference_f', '-')} | "
                     f"{'-' if diff is None else f'{100 * diff:.2f}'} | "
                     f"{'OK' if not v['errors'] else str(len(v['errors'])) + ' errors'} | "
                     f"{'yes' if v['used'] else 'no'} |")
    lines.append("")

    scored = []
    lines += ["## Videos", "", "| session | file | f | reference f | error % | detected / frames | status |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        ref = refs.get(r["session"])
        err = ""
        if r["status"] == "ok" and ref:
            err_v = 100 * (float(r["f"]) - ref) / ref
            scored.append((r, err_v))
            err = f"{err_v:+.1f}"
        lines.append(f"| {r['session']} | {r['file']} | {r['f']} | {ref} | {err} | "
                     f"{r['n_detected']} / {r['n_frames']} | {r['status']} |")
    lines.append("")

    head = ["| group | videos | scored | median abs err % | mean abs err % | median signed err % | "
            "within 5 % (%) | within 10 % (%) | within 20 % (%) |",
            "|---|---|---|---|---|---|---|---|---|"]
    lines += ["## Overall", "", "Errors are (f - reference f) / reference f. Videos that end with an "
              "error count in videos but not in scored.", ""] + head
    lines.append(metric_row("all", len(rows), metrics([e for _, e in scored])))
    for key in ("motion", "session", "device", "lens", "resolution"):
        groups = sorted({str(r[key]) for r in rows})
        if len(groups) < 2 and key != "motion":
            continue
        lines += ["", f"## By {key}", ""] + head
        for g in groups:
            n_total = sum(str(r[key]) == g for r in rows)
            lines.append(metric_row(g, n_total, metrics([e for r, e in scored if str(r[key]) == g])))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", help="existing run folder; videos already in results.csv are skipped")
    ap.add_argument("--data-root", help="override data_root in the config")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.data_root:
        cfg["data_root"] = args.data_root
    repo = Path(__file__).resolve().parent.parent
    if args.resume:
        out = Path(args.resume)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(cfg["output_dir"])
        out = (out if out.is_absolute() else repo / out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    info = {"config": cfg, "git": git_state(repo), "environment": environment(),
            "started": datetime.now(timezone.utc).isoformat()}

    validation, refs, todo = {}, {}, []
    sessions = find_sessions(cfg["data_root"])
    for session in sessions:
        errors, warnings, meta = check_session(session, probe_videos=cfg.get("probe_videos", True))
        has_ref = isinstance(meta.get("reference_f"), (int, float)) and meta["reference_f"] > 0
        used = has_ref and (not errors or cfg.get("include_invalid", False))
        validation[session.name] = {"errors": errors, "warnings": warnings, "used": used,
                                    "reference_f": meta.get("reference_f"),
                                    "checkerboard_diff": meta.get("_checkerboard_diff")}
        status = "OK" if not errors else f"{len(errors)} errors"
        print(f"[{status}] {session.name}{'' if used else '  -> skipped'}")
        for e in errors:
            print(f"    {e}")
        if not used:
            continue
        refs[session.name] = float(meta["reference_f"])
        for stem in HAND_STEMS:
            path = find_video(session, stem)
            if path is not None:
                todo.append((session, path, meta))
    info["validation"] = validation
    (out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = out / "results.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    done = {(r["session"], r["file"]) for r in rows}
    todo = [t for t in todo if (t[0].name, t[1].name) not in done]
    print(f"\n{len(todo)} hand videos to score ({len(done)} already in results.csv)", flush=True)

    hand = Calibrator("hand") if todo else None
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, (session, path, meta) in enumerate(todo, 1):
            thermal_guard(cfg)
            row = {"session": session.name, "file": path.name, "motion": path.stem.split("_")[1],
                   "device": meta.get("device", ""), "lens": meta.get("lens", ""),
                   "resolution": "x".join(str(x) for x in meta.get("resolution") or []),
                   "digital_zoom": meta.get("digital_zoom", "")}
            row.update(score_video(hand, path))
            writer.writerow(row)
            fh.flush()
            rows.append({k: str(v) for k, v in row.items()})
            print(f"[{i}/{len(todo)}] {session.name}/{path.name}: f = {row['f']} "
                  f"({row['n_detected']}/{row['n_frames']} frames, {row['seconds']}s) {row['status']}",
                  flush=True)

    info["finished"] = datetime.now(timezone.utc).isoformat()
    (out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(out, rows, refs, validation, cfg, info)
    print(f"\n{out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
