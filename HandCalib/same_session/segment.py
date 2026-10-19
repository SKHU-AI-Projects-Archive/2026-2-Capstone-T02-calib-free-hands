"""Automatic hand / checkerboard segmentation of one recording."""
from __future__ import annotations

import cv2
import numpy as np


def board_found(gray, inner, long_side):
    """findChessboardCornersSB on a downscaled grey frame (either orientation of the inner-corner grid)."""
    h, w = gray.shape
    s = min(1.0, long_side / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    for ps in (tuple(inner), tuple(inner[::-1])):
        ok, _ = cv2.findChessboardCornersSB(g, ps, cv2.CALIB_CB_EXHAUSTIVE)
        if ok:
            return True
    return False


def split(times, hand, board, gap_s):
    """times (n,), hand / board (n,) bools at the scan rate. Hand segment: [0, first board - gap); board segment: [first board, last board].
    Warnings when a board shows up inside the hand segment or a hand inside the board segment."""
    t = np.asarray(times, float)
    hand, board = np.asarray(hand, bool), np.asarray(board, bool)
    if not board.any():
        return {"hand": [float(t[0]), float(t[-1])], "board": None, "warnings": ["체커보드가 검출되지 않았습니다"]}
    i0, i1 = int(np.flatnonzero(board)[0]), int(np.flatnonzero(board)[-1])
    hand_end = float(t[i0] - gap_s)
    seg = {"hand": [float(t[0]), hand_end], "board": [float(t[i0]), float(t[i1])], "warnings": []}
    if hand_end <= t[0]:
        seg["warnings"].append("손 구간이 없습니다(첫 체커보드 검출이 너무 이릅니다)")
    in_hand = (t < hand_end)
    if (board & in_hand).any():
        seg["warnings"].append(f"손 구간에 체커보드 검출 {int((board & in_hand).sum())}장")
    in_board = (t >= t[i0]) & (t <= t[i1])
    if (hand & in_board).any():
        seg["warnings"].append(f"체커보드 구간에 손 검출 {int((hand & in_board).sum())}장 (보드를 든 손일 수 있음)")
    if not hand[in_hand].any():
        seg["warnings"].append("손 구간에서 손이 검출되지 않았습니다")
    seg["hand_detect_rate_in_hand"] = float(hand[in_hand].mean()) if in_hand.any() else None
    seg["board_detect_rate_in_board"] = float(board[in_board].mean())
    return seg
