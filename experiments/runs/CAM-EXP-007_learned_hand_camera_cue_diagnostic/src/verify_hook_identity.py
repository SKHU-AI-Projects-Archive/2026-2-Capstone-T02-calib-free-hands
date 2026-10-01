"""Verify the latent forward hooks do not change the model's own outputs.

Required by the frozen latent spec. PyTorch forward hooks that return None
cannot alter the output, but this checks it empirically rather than assuming.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, SUM, write_json


def main() -> None:
    sys.path.insert(0, str(REPO))
    import cv2
    from rgb_predictor import AnyHandPredictor
    from experiments.src.datasets.gigahands import takes

    t = takes()[0]
    cam = sorted(t.cameras)[0]
    vc = cv2.VideoCapture(str(t.video_path(cam)))
    frames = []
    for _ in range(40):
        ok, f = vc.read()
        if not ok:
            break
        frames.append(f)
    vc.release()
    frames = frames[::8][:4]

    pred = AnyHandPredictor(backend="wilor")
    _ = pred.predict(np.zeros((256, 256, 3), np.uint8))
    model = pred._wilor_model

    def run():
        out = []
        for f in frames:
            hs = pred.predict(f)
            if isinstance(hs, dict):
                hs = hs.get("wilor", [])
            out.append([(h.keypoints_3d.copy(), h.mano_pose.copy(),
                         np.asarray(h.bbox, float).copy()) for h in hs])
        return out

    before = run()
    cap = []
    h1 = model.backbone.register_forward_hook(
        lambda m, i, o: cap.append(
            (o[3] if isinstance(o, (tuple, list)) else o
             ).detach().float().mean(dim=(-2, -1)).cpu().numpy()))
    h2 = model.register_forward_pre_hook(lambda m, i: cap.append(None))
    hooks = [h1, h2]
    if hasattr(model, "refine_net"):
        hooks.append(model.refine_net.register_forward_hook(
            lambda m, i, o: cap.append(None)))
    after = run()
    for h in hooks:
        h.remove()

    diffs, n = [], 0
    for a, b in zip(before, after):
        if len(a) != len(b):
            diffs.append(float("inf"))
            continue
        for (k1, p1, b1), (k2, p2, b2) in zip(a, b):
            n += 1
            diffs.append(float(max(np.abs(k1 - k2).max(),
                                   np.abs(p1 - p2).max(),
                                   np.abs(b1 - b2).max())))
    mx = max(diffs) if diffs else float("nan")
    res = {
        "frames_tested": len(frames), "hands_compared": n,
        "max_abs_difference": mx,
        "bit_identical": bool(mx == 0.0),
        "verdict": "HOOKS_DO_NOT_ALTER_MODEL_OUTPUT" if mx == 0.0
                   else "HOOKS_ALTER_OUTPUT",
        "note": "forward hooks returning None cannot modify the output by "
                "PyTorch semantics; this confirms it empirically on real "
                "frames rather than assuming it.",
    }
    write_json(SUM / "hook_identity_check.json", res)
    print(res)


if __name__ == "__main__":
    main()
