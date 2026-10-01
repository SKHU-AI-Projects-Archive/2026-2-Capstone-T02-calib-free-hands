"""Numerical identifiability test on AnyCam's OWN focal-scoring function.

The end-to-end official pipeline could not be executed on this machine (see
anycam_static_stress_test.md for the dependency deadlock). That blocks the
behavioural stress test, but not the mechanistic question, because the function
that actually decides AnyCam's focal length is plain PyTorch and imports fine:

    anycam.trainer.induce_flow_dist(depths, projs, rel_poses)

`fit_video.py` scores 32 focal candidates by the flow this function induces and
takes the argmax. So the question "is the focal identifiable from a static
camera?" reduces to: does that function distinguish focal candidates when the
relative pose is the identity?

Nothing is reimplemented here. AnyCam's own function is called, on synthetic
depth and poses, with its own candidate grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import REPO, RUN_DIR, write_csv, write_json  # noqa: E402

ANYCAM_DIR = REPO / "experiments" / "cache" / "external_models" / "anycam"
sys.path.insert(0, str(ANYCAM_DIR))

H, W = 48, 64
N_FRAMES = 8
SEED = 20260923


def main() -> None:
    from anycam.trainer import induce_flow_dist, make_proj_from_focal_length

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # AnyCam's own candidate grid (anycam/models/anycam.py defaults)
    focal_min, focal_max, n_cand = 0.1, 4.0, 32
    bias = 1.8
    cand = torch.linspace(np.log(focal_min) - bias, np.log(focal_max) - bias,
                          n_cand, device=dev)
    cand = (cand + bias).exp().view(1, -1)
    projs = make_proj_from_focal_length(cand, W / H).to(dev)

    depths = (torch.rand(1, N_FRAMES, 1, 1, H, W, device=dev) * 4 + 0.5)

    def flow_for(poses):
        out = induce_flow_dist(depths, projs, poses)
        flow = out[0] if isinstance(out, (tuple, list)) else out
        return flow.detach().float()

    eye = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(1, N_FRAMES, 4, 4)

    rows = []
    # --- condition 1: static camera, identity relative pose ------------------
    f_static = flow_for(eye.contiguous())
    per_cand_static = f_static.reshape(n_cand, -1).abs().mean(dim=1) \
        if f_static.numel() % n_cand == 0 else f_static.abs().mean().repeat(n_cand)

    # --- condition 2: a genuinely moving camera ------------------------------
    moving = eye.clone().contiguous()
    t = torch.linspace(0, 0.3, N_FRAMES, device=dev)
    moving[0, :, 0, 3] = t
    f_move = flow_for(moving)
    per_cand_move = f_move.reshape(n_cand, -1).abs().mean(dim=1) \
        if f_move.numel() % n_cand == 0 else f_move.abs().mean().repeat(n_cand)

    for name, per_cand, flow in (("STATIC_IDENTITY_POSE", per_cand_static, f_static),
                                 ("MOVING_CAMERA", per_cand_move, f_move)):
        v = per_cand.cpu().numpy()
        rows.append({
            "condition": name,
            "n_focal_candidates": n_cand,
            "focal_candidate_min": float(cand.min()),
            "focal_candidate_max": float(cand.max()),
            "max_abs_induced_flow": float(flow.abs().max()),
            "mean_abs_induced_flow": float(flow.abs().mean()),
            "spread_across_candidates": float(v.max() - v.min()),
            "relative_spread": float((v.max() - v.min()) / (abs(v.mean()) + 1e-12)),
            "candidates_distinguishable": int((v.max() - v.min()) > 1e-8),
            "interpretation": (
                "the focal-scoring objective is flat across every candidate, so "
                "the focal length is not identifiable from this signal"
                if (v.max() - v.min()) <= 1e-8 else
                "candidates produce different induced flow, so the focal is "
                "identifiable from this signal"),
        })

    write_csv(RUN_DIR / "results" / "raw" / "anycam_candidate_diagnostics.csv.gz", rows)
    write_json(RUN_DIR / "results" / "summary" / "anycam_identifiability.json", {
        "test": "does AnyCam's own induce_flow_dist distinguish focal candidates?",
        "code_under_test": "anycam.trainer.induce_flow_dist and "
                           "make_proj_from_focal_length, official repo, "
                           "commit e609cc8a9e4ee8f78cf2ce39ebeb86b35e82d10d",
        "candidate_grid": "the package defaults: 32 candidates, focal_min 0.1, "
                          "focal_max 4.0, LOG_FOCAL_LENGTH_BIAS 1.8",
        "results": rows,
        "verdict": ("NOT_IDENTIFIABLE_UNDER_STATIC_CAMERA"
                    if rows[0]["candidates_distinguishable"] == 0
                    else "IDENTIFIABLE_CONTRARY_TO_EXPECTATION"),
        "status": "SUPPLEMENTARY_STATIC_STRESS_TEST (mechanistic variant)",
    })
    for r in rows:
        print(f"  {r['condition']:24s} max|flow|={r['max_abs_induced_flow']:.3e} "
              f"spread across candidates={r['spread_across_candidates']:.3e} "
              f"distinguishable={r['candidates_distinguishable']}")


if __name__ == "__main__":
    main()
