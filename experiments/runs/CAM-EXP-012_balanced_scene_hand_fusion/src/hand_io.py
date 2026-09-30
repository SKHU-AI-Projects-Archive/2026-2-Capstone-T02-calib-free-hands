"""Loading WiLoR hand observations with their frame indices.

CAM-EXP-011's loader discards the frame index, which the temporal-block
cross-fit needs. This returns observations paired with the frame they came
from, so the fold assignment is exact.

Only unit bone DIRECTIONS are taken from the network; its predicted bone
lengths are discarded, as in CAM-EXP-009.4 / CAM-EXP-011.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c012_common import BONES, N_BONES, WILOR  # noqa: E402

MIN_USABLE_JOINTS = 12


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


def load_video_observations(seq, cam, frames, bone_mapping=None):
    """-> ({side: [(dirs, uv, use)]}, {side: [frame_idx]}, W, H).

    `bone_mapping` permutes the RIGHT hand's bone order (the wrong-bone
    control). Correct and control therefore share identical RGB, identical
    observations, identical folds and identical solver.
    """
    obs = {"left": [], "right": []}
    idx = {"left": [], "right": []}
    W = H = None
    for f in frames:
        p = WILOR / ("%s__%s__%06d.npz" % (seq, cam, f))
        if not p.exists():
            continue
        d = np.load(p)
        W, H = int(d["image_width"]), int(d["image_height"])
        for side, want_right in (("left", False), ("right", True)):
            best, bs = None, -1.0
            for j in range(int(d["n_hands"])):
                if bool(int(d["h%d_is_right" % j])) != want_right:
                    continue
                s = float(d["h%d_score" % j])
                if s > bs:
                    best, bs = j, s
            if best is None:
                continue
            kp3 = np.asarray(d["h%d_keypoints_3d" % best], float)
            uv = np.asarray(d["h%d_keypoints_2d" % best], float)
            dirs, ok = bone_unit_directions(kp3)
            use = np.isfinite(uv).all(1)
            if not ok.all() or use.sum() < MIN_USABLE_JOINTS:
                continue
            if bone_mapping is not None and want_right:
                dirs = dirs[np.asarray(bone_mapping, int)]
            obs[side].append((dirs, uv, use))
            idx[side].append(int(f))
    return obs, idx, W, H
