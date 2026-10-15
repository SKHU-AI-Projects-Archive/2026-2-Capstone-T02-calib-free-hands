"""Target-hand selection, IoU tracking and point masks."""
from __future__ import annotations

import numpy as np


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ar = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])   # noqa: E731
    return inter / max(ar(a) + ar(b) - inter, 1e-9)


def candidates(det: dict, i: int):
    a, b = int(det["offsets"][i]), int(det["offsets"][i + 1])
    return list(range(a, b))


def track(det: dict, times: np.ndarray, size_wh, tcfg: dict, front_start_s: float = 0.0) -> dict:
    """Target = the candidate with the largest box area x confidence in the first init_window_s of the front phase; then IoU tracking."""
    n = len(times)
    W, H = size_wh
    init = [i for i in range(n) if front_start_s <= times[i] < front_start_s + tcfg["init_window_s"]]
    best, best_i = None, None
    for i in init:
        for k in candidates(det, i):
            b = det["boxes"][k]
            s = (b[2] - b[0]) * (b[3] - b[1]) * det["box_conf"][k]
            if best is None or s > best[0]:
                best, best_i = (s, k), i
    out = {"present": np.zeros(n, bool), "box": np.full((n, 4), np.nan), "cls": np.full(n, -1), "box_conf": np.zeros(n),
           "kp": np.full((n, 21, 2), np.nan), "kconf": np.zeros((n, 21)), "mask": np.zeros((n, 21), bool), "n_candidates": np.zeros(n, int),
           "init_frame": best_i}
    for i in range(n):
        out["n_candidates"][i] = len(candidates(det, i))
    if best is None:
        return out
    ref = det["boxes"][best[1]].copy()
    order = list(range(best_i, n)) + list(range(best_i - 1, -1, -1))     # forward from the init frame, then backward
    last_fwd, last_bwd = ref.copy(), ref.copy()
    for i in order:
        refb = last_fwd if i >= best_i else last_bwd
        cs = candidates(det, i)
        if not cs:
            continue
        ious = [iou(refb, det["boxes"][k]) for k in cs]
        j = int(np.argmax(ious))
        if i == best_i:
            k = best[1]
        elif ious[j] >= tcfg["iou_min"]:
            k = cs[j]
        else:
            continue
        out["present"][i] = True
        out["box"][i], out["cls"][i], out["box_conf"][i] = det["boxes"][k], det["cls"][k], det["box_conf"][k]
        out["kp"][i], out["kconf"][i] = det["kxy"][k], det["kconf"][k]
        if i >= best_i:
            last_fwd = det["boxes"][k].copy()
        else:
            last_bwd = det["boxes"][k].copy()
    kp = out["kp"]
    inside = (kp[..., 0] >= 0) & (kp[..., 0] < W) & (kp[..., 1] >= 0) & (kp[..., 1] < H)
    out["mask"] = out["present"][:, None] & (out["kconf"] >= tcfg["conf_min"]) & inside
    if tcfg.get("smoothing", {}).get("enabled"):
        out["kp"] = smooth(out["kp"], out["mask"], int(tcfg["smoothing"]["window"]))
    return out


def smooth(kp, mask, w):
    """Optional centred moving average over visible points (off by default)."""
    out = kp.copy()
    h = w // 2
    for i in range(len(kp)):
        lo, hi = max(0, i - h), min(len(kp), i + h + 1)
        for j in range(21):
            m = mask[lo:hi, j]
            if mask[i, j] and m.sum() > 1:
                out[i, j] = kp[lo:hi, j][m].mean(0)
    return out


def box_long(box):
    return np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])


def order_check(kp: np.ndarray, mask: np.ndarray) -> dict:
    """OpenPose order on 2D: wrist->MCP distance (middle longest, pinky shortest) and the thumb base nearer the index MCP."""
    ok = mask[:, [0, 1, 5, 9, 13, 17]].all(1)
    if ok.sum() == 0:
        return {"ok": None, "reason": "no frame with the needed points"}
    P = kp[ok]
    d = {n: float(np.median(np.linalg.norm(P[:, j] - P[:, 0], axis=-1))) for n, j in (("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17))}
    ci = float(np.median(np.linalg.norm(P[:, 1] - P[:, 5], axis=-1)))
    cp = float(np.median(np.linalg.norm(P[:, 1] - P[:, 17], axis=-1)))
    res = {"wrist_to_mcp_px": d, "middle_longest": d["middle"] == max(d.values()), "pinky_shortest": d["pinky"] == min(d.values()),
           "thumb_base_nearer_index": ci < cp, "n_frames": int(ok.sum())}
    res["ok"] = bool(res["middle_longest"] and res["pinky_shortest"] and res["thumb_base_nearer_index"])
    return res
