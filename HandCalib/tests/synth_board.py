"""Synthetic checkerboard frames rendered with a known pinhole K (no distortion), for 18's tests."""
import cv2
import numpy as np


def board_texture(inner=(7, 10), px=60):
    nx, ny = inner[0] + 1, inner[1] + 1
    img = np.zeros((ny * px + 2 * px, nx * px + 2 * px), np.uint8) + 255
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                img[px + j * px:px + (j + 1) * px, px + i * px:px + (i + 1) * px] = 0
    return img, px


def render(K, W, H, rvec, tvec, inner=(7, 10), square=15.0):
    tex, px = board_texture(inner)
    s = square / px                                       # mm per texture pixel
    # texture pixel (u, v) -> board point (X, Y, 0) mm, board origin at the first inner corner
    A = np.array([[s, 0, -2 * square], [0, s, -2 * square], [0, 0, 1.0]])   # first inner corner = board origin
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    Hb = K @ np.c_[R[:, :2], np.asarray(tvec, float)]     # board (X, Y, 1) -> image
    M = Hb @ A
    img = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=128)
    return img


def poses(n, tilt_range, rng, dist=600.0):
    out = []
    for _ in range(n):
        th = np.radians(rng.uniform(*tilt_range))
        az = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(az), np.sin(az), 0.0])
        r = axis * th + np.array([0, 0, rng.uniform(-0.3, 0.3)])
        R, _ = cv2.Rodrigues(r)
        centre = np.array([5 * 15 / 2, 8 * 15 / 2, 0.0])          # roughly the board centre (mm)
        t = np.array([rng.uniform(-60, 60), rng.uniform(-60, 60), dist * rng.uniform(0.9, 1.1)]) - R @ centre
        out.append((r, t))
    return out
