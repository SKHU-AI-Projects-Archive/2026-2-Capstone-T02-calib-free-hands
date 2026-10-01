"""Step 1 - inventory the frame structure, timestamps and index semantics.

Answers Q1-Q5: whether 2D rows, 3D rows and RGB frames share one index, what
chosen_frames means, and whether timestamps are needed for mapping.
"""
from __future__ import annotations

import json
import logging
import re

import numpy as np

from audit_common import (HANDS, RUN_DIR, Sequence, rel, sequences, write_csv)

log = logging.getLogger("cam-exp-001.2")

# rgb_vid/<cam>/<cam>_<ts>.txt lines look like:  frame_1727030430679748_000000000000
TS_LINE = re.compile(r"^frame_(\d+)_(\d+)$")


def video_frame_count(path) -> int:
    import cv2
    if path is None:
        return -1
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n


def parse_timestamps(path):
    """Return (timestamps_us, frame_indices) from a rgb_vid .txt sidecar."""
    if path is None:
        return [], []
    ts, idx = [], []
    for line in open(path):
        m = TS_LINE.match(line.strip())
        if m:
            ts.append(int(m.group(1)))
            idx.append(int(m.group(2)))
    return ts, idx


def main() -> dict:
    inv_rows, ts_rows, sem_rows, map_rows = [], [], [], []
    summary = {}

    for seq_dir in sequences():
        s = Sequence(seq_dir)
        n3 = {h: len(s.kp3(h)) for h in HANDS}
        cams2d = s.cameras_2d("left")
        repro2 = video_frame_count(s.repro_video("2d"))
        repro3 = video_frame_count(s.repro_video("3d"))
        mano = video_frame_count(s.repro_video("mano")) if (s.dir / "repro_mano_vid").exists() else -1
        manov = video_frame_count(sorted((s.dir / "mano_vid").glob("*.mp4"))[0]) \
            if (s.dir / "mano_vid").is_dir() and any((s.dir / "mano_vid").glob("*.mp4")) else -1
        params_len = -1
        pdir = s.dir / "params"
        if pdir.is_dir():
            pf = sorted(pdir.glob("*.json"))
            if pf:
                pj = json.load(open(pf[0]))
                try:
                    params_len = len(pj["left"]["poses"])
                except Exception:
                    params_len = -1

        for cam in cams2d:
            vid = s.video_path(cam)
            nrgb = video_frame_count(vid)
            tsp = s.timestamps_path(cam)
            ts, idx = parse_timestamps(tsp)
            n2 = {h: len(s.kp2(h, cam)) for h in HANDS}
            notes = []
            if n2["left"] != nrgb:
                notes.append("kp2d_left_rows != rgb_frames")
            if len(ts) != nrgb:
                notes.append("timestamp_count != rgb_frames")
            if n3["left"] > nrgb:
                notes.append("kp3d_rows > rgb_frames")
            inv_rows.append({
                "sequence": s.name, "take": s.take, "camera": cam,
                "rgb_frame_count": nrgb, "timestamp_count": len(ts),
                "kp2d_left_rows": n2["left"], "kp2d_right_rows": n2["right"],
                "kp3d_left_rows": n3["left"], "kp3d_right_rows": n3["right"],
                "chosen_left_count": len(s.chosen["left"]),
                "chosen_right_count": len(s.chosen["right"]),
                "chosen_union_count": len(s.union_sorted),
                "chosen_intersect_count": len(s.intersect_sorted),
                "chosen_max_frame_id": max(s.union_sorted) if s.union_sorted else -1,
                "params_entries": params_len,
                "repro2d_frame_count": repro2, "repro3d_frame_count": repro3,
                "mano_vid_frame_count": manov,
                "kp2d_rows_eq_rgb": int(n2["left"] == nrgb),
                "kp3d_rows_eq_maxchosen_plus1": int(
                    n3["left"] == (max(s.union_sorted) + 1 if s.union_sorted else -1)),
                "repro_eq_union": int(repro2 == len(s.union_sorted)),
                "params_eq_union": int(params_len == len(s.union_sorted)),
                "notes": ";".join(notes),
            })

            # timestamp audit on a few cameras per sequence
            if ts and cams2d.index(cam) < 3:
                d = np.diff(ts)
                ts_rows.append({
                    "sequence": s.name, "camera": cam,
                    "n_timestamps": len(ts), "rgb_frame_count": nrgb,
                    "frame_index_is_0_based_contiguous": int(idx == list(range(len(idx)))),
                    "first_timestamp_us": ts[0], "last_timestamp_us": ts[-1],
                    "median_delta_us": int(np.median(d)) if len(d) else "",
                    "implied_fps": round(1e6 / float(np.median(d)), 3) if len(d) else "",
                    "filename_timestamp": int(s.video_path(cam).stem.split("_")[-1]),
                    "filename_ts_minus_first_frame_ts_us": int(
                        s.video_path(cam).stem.split("_")[-1]) - ts[0],
                })

        # --- chosen_frames semantics, tested against the data ---------------
        for h in HANDS:
            rows = s.kp3(h)
            chosen = s.chosen[h]
            in_ch, out_ch = [], []
            for i, row in enumerate(rows):
                a = np.asarray(row, dtype=float)
                nonzero = bool(np.any(a[:, :3] != 0))
                (in_ch if i in chosen else out_ch).append(nonzero)
            sem_rows.append({
                "sequence": s.name, "take": s.take, "hand": h,
                "kp3d_rows": len(rows), "chosen_count": len(chosen),
                "chosen_min": min(chosen) if chosen else "",
                "chosen_max": max(chosen) if chosen else "",
                "chosen_is_contiguous_0_to_max": int(
                    chosen == set(range(max(chosen) + 1)) if chosen else 0),
                "rows_eq_chosen_max_plus_1": int(len(rows) == (max(chosen) + 1 if chosen else -1)),
                "n_rows_in_chosen": len(in_ch), "n_rows_not_in_chosen": len(out_ch),
                "frac_chosen_rows_nonzero": round(float(np.mean(in_ch)), 4) if in_ch else "",
                "frac_unchosen_rows_nonzero": round(float(np.mean(out_ch)), 4) if out_ch else "",
            })

        summary[s.name] = {
            "take": s.take, "union": len(s.union_sorted),
            "intersect": len(s.intersect_sorted),
            "kp3d_rows": n3["left"], "params": params_len,
            "repro2d": repro2, "repro3d": repro3, "mano_vid": manov,
        }

        # --- explicit frame mapping table for a few cameras -----------------
        for cam in cams2d[:3]:
            vid = s.video_path(cam)
            nrgb = video_frame_count(vid)
            ts, idx = parse_timestamps(s.timestamps_path(cam))
            n2 = len(s.kp2("left", cam))
            for f in range(0, nrgb, max(1, nrgb // 25)):
                in3 = f < n3["left"]
                map_rows.append({
                    "sequence": s.name, "camera": cam, "rgb_frame_index": f,
                    "rgb_timestamp_us": ts[f] if f < len(ts) else "",
                    "sidecar_frame_index": idx[f] if f < len(idx) else "",
                    "kp2d_row": f if f < n2 else "",
                    "kp3d_row": f if in3 else "",
                    "union_position": s.union_index(f),
                    "left_chosen": int(f in s.chosen["left"]),
                    "right_chosen": int(f in s.chosen["right"]),
                    "mapping_source": "DIRECT_INDEX" if in3 else "OUT_OF_3D_RANGE",
                    "mapping_confidence": "high" if in3 else "n/a",
                    "notes": ("rgb/2d/3d share one 0-based frame index; "
                              "params and repro videos use union_position"),
                })

    write_csv(RUN_DIR / "results" / "raw" / "frame_structure_inventory.csv", inv_rows)
    write_csv(RUN_DIR / "results" / "raw" / "timestamp_audit.csv", ts_rows)
    write_csv(RUN_DIR / "tables" / "chosen_frames_semantics.csv", sem_rows)
    write_csv(RUN_DIR / "results" / "raw" / "frame_mapping_table.csv", map_rows)

    verdicts = {
        "kp2d_rows_eq_rgb_frames": all(r["kp2d_rows_eq_rgb"] for r in inv_rows),
        "kp3d_rows_eq_max_chosen_plus1": all(r["kp3d_rows_eq_maxchosen_plus1"] for r in inv_rows),
        "repro_videos_eq_union": all(r["repro_eq_union"] for r in inv_rows),
        "params_eq_union": all(r["params_eq_union"] for r in inv_rows if r["params_entries"] > 0),
        "timestamps_eq_rgb_frames": all(r["timestamp_count"] == r["rgb_frame_count"]
                                        for r in inv_rows),
        "sequences": summary,
        "n_cameras_audited": len(inv_rows),
    }
    (RUN_DIR / "results" / "summary" / "_structure_verdicts.json").write_text(
        json.dumps(verdicts, indent=2), encoding="utf-8")
    log.info("structure verdicts: %s",
             {k: v for k, v in verdicts.items() if isinstance(v, bool)})
    return verdicts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
