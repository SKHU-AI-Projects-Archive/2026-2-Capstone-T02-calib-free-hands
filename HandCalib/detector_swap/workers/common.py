"""Shared by the standalone workers (numpy only, no repository imports): frame list I/O, hashing, the 14 npz layout."""
import hashlib
import json

import numpy as np


def sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_list(path):
    """14's frame list: index \\t png \\t sha256 per line."""
    rows = [line.rstrip("\n").split("\t") for line in open(path) if line.strip()]
    return [(int(i), p, h) for i, p, h in rows]


def verify(rows):
    for i, p, h in rows:
        if sha256(p) != h:
            raise RuntimeError(f"hash mismatch for frame {i}: {p}")


def to_pixels(xy_norm, width, height):
    """Normalised image coordinates (x / W, y / H, origin at the top-left image corner) -> pixels."""
    xy = np.asarray(xy_norm, dtype=np.float64)
    return np.stack([xy[..., 0] * width, xy[..., 1] * height], axis=-1)


def remap(kxy, kconf, order):
    if order is None:
        return kxy, kconf
    order = list(order)
    return kxy[:, order], kconf[:, order]


def save(out, rows, per_frame, info):
    """per_frame: list (one per row) of lists of dicts {box (4,), box_conf, cls, kxy (21, 2), kconf (21,)}.
    Writes 14's detection npz: frame_index, sha256, offsets, boxes, box_conf, cls, kxy, kconf, info."""
    offsets, boxes, bconf, cls, kxy, kconf = [0], [], [], [], [], []
    for cands in per_frame:
        for c in cands:
            boxes.append(c["box"])
            bconf.append(c["box_conf"])
            cls.append(c["cls"])
            kxy.append(c["kxy"])
            kconf.append(c["kconf"])
        offsets.append(offsets[-1] + len(cands))
    arr = lambda xs, shape, dt: np.asarray(xs, dtype=dt).reshape(shape) if xs else np.zeros((0,) + shape[1:], dtype=dt)   # noqa: E731
    np.savez_compressed(out, frame_index=np.array([r[0] for r in rows], dtype=np.int64), sha256=np.array([r[2] for r in rows]),
                        offsets=np.asarray(offsets, dtype=np.int64), boxes=arr(boxes, (-1, 4), np.float64),
                        box_conf=arr(bconf, (-1,), np.float64), cls=arr(cls, (-1,), np.int64), kxy=arr(kxy, (-1, 21, 2), np.float64),
                        kconf=arr(kconf, (-1, 21), np.float64), info=json.dumps(info))
