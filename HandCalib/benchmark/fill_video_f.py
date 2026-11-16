"""Write the calibrator.py f of a benchmark run into session.json videos[].f (guide section 6).

    python fill_video_f.py ../runs/benchmark/<run>            # all sessions in the run
    python fill_video_f.py ../runs/benchmark/<run> --dry-run

Only rows with status ok are written; a video that ended with an error keeps f = null
and gets note "calibrator error" if its note is empty. Values come from results.csv of
the run, rounded to 0.1 px. Other fields of session.json are left as they are.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/benchmark/<run> folder (results.csv, run.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    data_root = Path(json.loads((run / "run.json").read_text(encoding="utf-8"))["config"]["data_root"])
    with (run / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_session = {}
    for r in rows:
        by_session.setdefault(r["session"], {})[r["file"]] = r

    for session, files in sorted(by_session.items()):
        path = data_root / session / "session.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        changed = []
        for v in meta.get("videos") or []:
            r = files.get(v.get("file"))
            if r is None:
                continue
            if r["status"] == "ok":
                new = round(float(r["f"]), 1)
            else:
                new = None
                if not v.get("note"):
                    v["note"] = "calibrator error"
            if v.get("f") != new:
                changed.append(f"{v['file']}: {v.get('f')} -> {new}")
                v["f"] = new
        print(f"{session}: {len(changed)} changed")
        for c in changed:
            print(f"  {c}")
        if changed and not args.dry_run:
            path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
