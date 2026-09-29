"""PHASE B — one WiLoR pass over every scene frame, shared by all three models.

Fairness requirement: `AnyCalib + Hand`, `GeoCalib + Hand` and
`Perspective Fields + Hand` must consume the SAME hand evidence. So the network
runs once per RGB frame and the result is cached; the three `+ Hand` conditions
differ only in the scene term.

A frame with no usable hand is recorded with `hand_available = 0`. **The RGB
frame is not removed from the scene benchmark** — hand annotation quality never
gates scene eligibility.

CAM-EXP-009.4 already cached 6,908 of these frames with identical settings;
those are reused and only the remainder inferred. Each row records whether it
was `REUSED_CACHE` or `NEW_INFERENCE`.

    PYOPENGL_PLATFORM=win32 PYTHONPATH=<repo> \
        .venv-anyhand/Scripts/python src/cache_hand_observations.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (BONES, C0094_WILOR, CACHE, MANIFESTS, N_BONES,  # noqa: E402
                    RAW, REPO, SUM, read_csv, write_csv, write_json)

OUT = CACHE / "wilor"


def load_predictor():
    from rgb_predictor import AnyHandPredictor
    return AnyHandPredictor(
        backend="wilor",
        wilor_ckpt=str(REPO / "models" / "anyhand_wilor.ckpt"),
        wilor_cfg=str(REPO / "models" / "model_config_wilor.yaml"),
        detector_pt=str(REPO / "models" / "detector.pt"),
    )


def bone_unit_directions(kp3):
    X = np.asarray(kp3, float)
    d = np.zeros((N_BONES, 3))
    ok = np.zeros(N_BONES, bool)
    for i, (a, b) in enumerate(BONES):
        v = X[b] - X[a]
        n = np.linalg.norm(v)
        if np.isfinite(n) and n > 1e-9:
            d[i] = v / n
            ok[i] = True
    return d, ok


def save(outp, preds, W, H, src):
    data = {"image_width": np.int32(W), "image_height": np.int32(H),
            "n_hands": np.int32(len(preds)), "source": np.str_(src)}
    for j, p in enumerate(preds):
        data["h%d_bbox" % j] = np.asarray(p.bbox, float).reshape(4)
        data["h%d_cam_t" % j] = np.asarray(p.cam_t, float)
        data["h%d_keypoints_3d" % j] = np.asarray(p.keypoints_3d, float)
        data["h%d_keypoints_2d" % j] = np.asarray(p.keypoints_2d, float)
        data["h%d_is_right" % j] = np.int32(int(p.is_right))
        data["h%d_score" % j] = np.float64(p.score)
    np.savez_compressed(outp, **data)


def summarise(npz_path, seq, cam, fr, source):
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["n_hands"])
    nl = nr = 0
    best = {}
    for j in range(n):
        right = bool(int(d["h%d_is_right" % j]))
        s = float(d["h%d_score" % j])
        if right:
            nr += 1
        else:
            nl += 1
        k = "right" if right else "left"
        if k not in best or s > best[k][1]:
            best[k] = (j, s)
    rec = {"sequence": seq, "camera": cam, "frame": fr,
           "n_hands": n, "n_left_detected": nl, "n_right_detected": nr,
           "duplicate_handedness": int(nl > 1 or nr > 1),
           "source": source}
    for k in ("left", "right"):
        if k in best:
            j = best[k][0]
            kp3 = np.asarray(d["h%d_keypoints_3d" % j], float)
            uv = np.asarray(d["h%d_keypoints_2d" % j], float)
            _, ok = bone_unit_directions(kp3)
            usable = bool(ok.all() and np.isfinite(uv).all())
            rec["%s_available" % k] = int(usable)
            rec["%s_score" % k] = best[k][1]
            rec["%s_selected_index" % k] = j
        else:
            rec["%s_available" % k] = 0
            rec["%s_score" % k] = ""
            rec["%s_selected_index" % k] = ""
    rec["hand_any_available"] = int(rec["left_available"]
                                    or rec["right_available"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import cv2
    from experiments.src.datasets.gigahands import takes

    OUT.mkdir(parents=True, exist_ok=True)
    man = [r for r in read_csv(MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz")
           if r["scene_input"] == "1"]
    by_video = defaultdict(list)
    for r in man:
        by_video[(r["sequence"], r["camera"])].append(int(r["frame"]))

    todo = []
    reused = 0
    for (seq, cam), frames in sorted(by_video.items()):
        for f in sorted(frames):
            name = "%s__%s__%06d.npz" % (seq, cam, f)
            mine = OUT / name
            if mine.exists():
                continue
            old = C0094_WILOR / name
            if old.exists():                      # identical settings
                mine.write_bytes(old.read_bytes())
                reused += 1
                continue
            todo.append((seq, cam, f, mine))
    if args.limit:
        todo = todo[:args.limit]
    print("reused from CAM-EXP-009.4: %d | to infer: %d"
          % (reused, len(todo)), flush=True)

    if todo:
        predictor = load_predictor()
        take_by_name = {t.name: t for t in takes()}
        cur, cap = None, None
        t0, done = time.time(), 0
        by_v = defaultdict(list)
        for seq, cam, f, outp in todo:
            by_v[(seq, cam)].append((f, outp))
        for (seq, cam), items in sorted(by_v.items()):
            want = {f: p for f, p in items}
            vid = take_by_name[seq].video_path(cam)
            cap = cv2.VideoCapture(str(vid))
            idx = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if idx in want:
                    H, W = img.shape[:2]
                    try:
                        preds = predictor.predict(img)
                    except Exception:
                        preds = []
                    save(want[idx], preds, W, H, "NEW_INFERENCE")
                    done += 1
                    if done % 500 == 0:
                        el = time.time() - t0
                        print("  %d/%d  %.1f min  %.2f fps  eta %.1f min"
                              % (done, len(todo), el / 60, done / max(el, 1e-9),
                                 (len(todo) - done) / max(done / max(el, 1e-9),
                                                          1e-9) / 60),
                              flush=True)
                idx += 1
            cap.release()

    # ---- summary table over every scene frame
    rows = []
    for (seq, cam), frames in sorted(by_video.items()):
        for f in sorted(frames):
            p = OUT / ("%s__%s__%06d.npz" % (seq, cam, f))
            if not p.exists():
                rows.append({"sequence": seq, "camera": cam, "frame": f,
                             "n_hands": 0, "n_left_detected": 0,
                             "n_right_detected": 0, "duplicate_handedness": 0,
                             "left_available": 0, "right_available": 0,
                             "hand_any_available": 0,
                             "source": "NOT_INFERRED"})
                continue
            src = "REUSED_CACHE" if (C0094_WILOR /
                                     p.name).exists() else "NEW_INFERENCE"
            rows.append(summarise(p, seq, cam, f, src))
    write_csv(RAW / "hand_observations.csv.gz", rows)

    tot = len(rows)
    write_json(SUM / "hand_coverage_summary.json", {
        "scene_frames": tot,
        "wilor_attempted": sum(1 for r in rows
                               if r["source"] != "NOT_INFERRED"),
        "left_available": sum(r["left_available"] for r in rows),
        "right_available": sum(r["right_available"] for r in rows),
        "both_available": sum(1 for r in rows if r["left_available"]
                              and r["right_available"]),
        "none_available": sum(1 for r in rows if not r["hand_any_available"]),
        "any_available": sum(r["hand_any_available"] for r in rows),
        "duplicate_handedness_frames": sum(r["duplicate_handedness"]
                                           for r in rows),
        "reused_cache": sum(1 for r in rows if r["source"] == "REUSED_CACHE"),
        "new_inference": sum(1 for r in rows
                             if r["source"] == "NEW_INFERENCE"),
        "note": "hand availability never removes an RGB frame from the scene "
                "benchmark",
    })
    print("hand rows %d | left %d | right %d | both %d | none %d"
          % (tot, sum(r["left_available"] for r in rows),
             sum(r["right_available"] for r in rows),
             sum(1 for r in rows if r["left_available"]
                 and r["right_available"]),
             sum(1 for r in rows if not r["hand_any_available"])))


if __name__ == "__main__":
    main()
