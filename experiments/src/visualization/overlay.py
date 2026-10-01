"""Overlay GT vs projected 2D joints on a source image or video frame.

Regenerable from the raw CSV plus the dataset images; nothing here is a
source of truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# MANO-style 21-joint chain used by GigaHands / HanCo / AssemblyHands.
EDGES_21 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

GT_COLOR = (0, 255, 0)        # green  = dataset 2D annotation
PROJ_COLOR = (0, 0, 255)      # red    = our projection of GT 3D
LINK_COLOR = (0, 255, 255)    # yellow = GT<->projection residual


def load_frame(path: Path, frame_index: int = 0):
    import cv2
    path = Path(path)
    if path.suffix.lower() in {".mp4", ".avi", ".mov"}:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, im = cap.read()
        cap.release()
        return im if ok else None
    return cv2.imread(str(path))


def draw_overlay(image, uv_proj, uv_gt=None, valid=None, title_lines=(),
                 out_path: Path | None = None, draw_edges: bool = True):
    """Draw projected (red) and GT (green) joints with residual links."""
    import cv2
    im = image.copy()
    n = len(uv_proj)
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, bool)

    def ok(p):
        return p is not None and np.isfinite(p).all()

    if draw_edges and n >= 21:
        for a, b in EDGES_21:
            if valid[a] and valid[b] and ok(uv_proj[a]) and ok(uv_proj[b]):
                cv2.line(im, tuple(np.int32(uv_proj[a])), tuple(np.int32(uv_proj[b])),
                         PROJ_COLOR, 1, cv2.LINE_AA)
    for j in range(n):
        if not valid[j]:
            continue
        p = uv_proj[j]
        if uv_gt is not None and ok(uv_gt[j]) and ok(p):
            cv2.line(im, tuple(np.int32(uv_gt[j])), tuple(np.int32(p)), LINK_COLOR, 1, cv2.LINE_AA)
        if uv_gt is not None and ok(uv_gt[j]):
            cv2.circle(im, tuple(np.int32(uv_gt[j])), 3, GT_COLOR, -1, cv2.LINE_AA)
        if ok(p):
            cv2.circle(im, tuple(np.int32(p)), 2, PROJ_COLOR, -1, cv2.LINE_AA)

    y = 14
    for line in title_lines:
        cv2.putText(im, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        y += 14
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), im)
    return im
