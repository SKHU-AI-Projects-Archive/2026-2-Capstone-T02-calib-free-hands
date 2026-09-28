"""Generic hand anatomy prior from the local MANO model. NO GigaHands geometry.

The prior is the 20 connected bone lengths of the NEUTRAL MANO hand
(betas = 0, rest pose), sum-normalised to proportions. Absolute MANO size is
discarded, so no absolute hand-size assumption enters the method.

Joint construction is audited, not guessed:

  * `smplx.MANO` returns 16 joints in the MANO order
        0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
  * the five fingertips are the vertices named in `smplx.vertex_ids['mano']`
        thumb 744, index 320, middle 443, ring 554, pinky 671
  * the target 21-joint order is the OpenPose order documented in
    `demo/hand_topology.py` (WiLoR `mano_to_openpose`):
        0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky

The resulting mapping is checked empirically against WiLoR's own 21-joint
output before it is written (see `validate_against_wilor`).

Run with the anyhand environment:
    experiments/.venv-anyhand/Scripts/python src/build_mano_prior.py
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
REPO = SRC.parents[3]

BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]

# MANO joint index -> OpenPose slot, for the three non-tip joints of each finger
MANO_TO_OPENPOSE = {
    0: 0,
    13: 1, 14: 2, 15: 3,      # thumb
    1: 5, 2: 6, 3: 7,         # index
    4: 9, 5: 10, 6: 11,       # middle
    10: 13, 11: 14, 12: 15,   # ring
    7: 17, 8: 18, 9: 19,      # pinky
}
TIP_SLOT = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def neutral_hand_21():
    import torch
    import smplx
    from smplx.vertex_ids import vertex_ids

    model = smplx.MANO(model_path=str(REPO / "mano_data" / "MANO_RIGHT.pkl"),
                       is_rhand=True, use_pca=False, flat_hand_mean=True,
                       batch_size=1)
    out = model(betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3),
                hand_pose=torch.zeros(1, 45), return_verts=True)
    J = out.joints[0].detach().numpy()
    V = out.vertices[0].detach().numpy()
    tips = vertex_ids["mano"]

    X = np.zeros((21, 3))
    for mi, slot in MANO_TO_OPENPOSE.items():
        X[slot] = J[mi]
    for finger, slot in TIP_SLOT.items():
        X[slot] = V[tips[finger]]
    return X, tips


def bone_lengths(X):
    return np.array([np.linalg.norm(X[b] - X[a]) for a, b in BONES])


def validate_against_wilor(p_prior):
    """Sanity-check the joint mapping against WiLoR's own 21-joint output.

    This is a CHECK, not a source: the prior is never taken from WiLoR or from
    GigaHands. If the mapping were wrong, the bone-proportion profile would not
    resemble a hand at all, and the rank correlation would collapse.
    """
    import glob
    cache = REPO / "experiments" / "runs" / \
        "CAM-EXP-002_camera_focal_sensitivity" / "cache" / "hand_inference"
    files = sorted(glob.glob(str(cache / "*.npz")))[:200]
    profs = []
    for f in files:
        d = np.load(f)
        n = int(d["n_hands"])
        for j in range(n):
            k = "h%d_keypoints_3d" % j
            if k not in d:
                continue
            l = bone_lengths(np.asarray(d[k], float))
            if np.isfinite(l).all() and l.min() > 1e-6:
                profs.append(l / l.sum())
    if not profs:
        return {"available": False}
    med = np.median(np.vstack(profs), axis=0)

    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[o] = np.arange(len(a))
        return r
    rho = float(np.corrcoef(rank(med), rank(p_prior))[0, 1])
    return {"available": True, "n_hands": len(profs),
            "wilor_median_proportions": [float(x) for x in med],
            "spearman_rank_corr_with_prior": rho,
            "max_abs_proportion_diff": float(np.max(np.abs(med - p_prior))),
            "note": "WiLoR is used only to CHECK the joint mapping. The prior "
                    "is the neutral MANO hand; no WiLoR or GigaHands geometry "
                    "enters it."}


def main():
    X, tips = neutral_hand_21()
    l = bone_lengths(X)
    assert np.isfinite(l).all() and l.min() > 1e-6, "degenerate neutral hand"
    p = l / l.sum()

    check = validate_against_wilor(p)
    out = {
        "source": "mano_data/MANO_RIGHT.pkl, neutral shape (betas=0), rest "
                  "pose, via smplx.MANO(use_pca=False, flat_hand_mean=True)",
        "tips_source": "smplx.vertex_ids['mano']",
        "tip_vertex_ids": {k: int(v) for k, v in tips.items()},
        "mano_to_openpose": {str(k): v for k, v in MANO_TO_OPENPOSE.items()},
        "joint_order": "OpenPose 21, as documented in demo/hand_topology.py",
        "bones": [list(b) for b in BONES],
        "absolute_size_removed": True,
        "neutral_bone_lengths_m": [float(x) for x in l],
        "total_length_m": float(l.sum()),
        "p0_generic_prior": [float(x) for x in p],
        "left_hand_prior": "identical bone PROPORTIONS; the bone set is "
                           "mirror-symmetric, and only proportions are used",
        "gigahands_used": False,
        "wilor_used_as_source": False,
        "mapping_check_against_wilor": check,
    }
    man = REPO / "experiments" / "manifests" / \
        "cam_exp_0094_generic_mano_prior_v1.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (SRC.parents[1] / "results" / "summary").mkdir(parents=True, exist_ok=True)
    (SRC.parents[1] / "results" / "summary" / "generic_prior.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    names = ["thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
             "index_mcp", "index_pip", "index_dip", "index_tip",
             "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
             "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
             "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
    for n, li, pi in zip(names, l, p):
        print("  %-12s %.4f m   p=%.4f" % (n, li, pi))
    print("total %.4f m ; sum p = %.6f" % (l.sum(), p.sum()))
    print("mapping check vs WiLoR:", {k: v for k, v in check.items()
                                      if k in ("n_hands",
                                               "spearman_rank_corr_with_prior",
                                               "max_abs_proportion_diff")})


if __name__ == "__main__":
    main()
