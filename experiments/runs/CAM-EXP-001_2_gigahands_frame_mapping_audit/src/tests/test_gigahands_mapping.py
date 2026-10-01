"""Regression tests for the GigaHands mapping rules confirmed in CAM-EXP-001.2.

Run:  PYTHONPATH=<repo root> python src/tests/test_gigahands_mapping.py

These are plain asserts (no pytest dependency) so they can run anywhere the
experiment runs. They read the real dataset; if it is absent they skip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))          # the run's src/
from audit_common import (HAND_POSE_ROOT, HANDS, Sequence, is_zero_pattern,  # noqa: E402
                          sequences)
from experiments.src.datasets import gigahands  # noqa: E402

FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_3d_rows_are_frame_indexed() -> None:
    """keypoints_3d row count must equal max(chosen)+1, i.e. rows are frame ids."""
    for d in sequences():
        s = Sequence(d)
        for h in HANDS:
            rows = len(s.kp3(h))
            check(f"3d rows frame-indexed [{s.name}/{h}]",
                  rows == max(s.union_sorted) + 1,
                  f"rows={rows} max_chosen={max(s.union_sorted)}")


def test_unchosen_3d_rows_are_zero() -> None:
    """Rows outside chosen_frames carry no pose and must never be used."""
    for d in sequences():
        s = Sequence(d)
        for h in HANDS:
            unchosen = [i for i in range(len(s.kp3(h))) if i not in s.chosen[h]]
            if not unchosen:
                continue
            nz = [bool(np.any(np.asarray(s.kp3(h)[i], float)[:, :3] != 0))
                  for i in unchosen]
            check(f"unchosen 3d rows are zero [{s.name}/{h}]", not any(nz),
                  f"{sum(nz)}/{len(unchosen)} non-zero")


def test_chosen_frames_filtering() -> None:
    """The loader must only emit frames listed in chosen_frames for that hand."""
    s = Sequence(sequences()[0])
    emitted = {(x.hand, int(x.frame))
               for x in gigahands.iter_samples(max_sequences=1, max_cameras=2,
                                               max_frames=40)}
    bad = [(h, f) for (h, f) in emitted if f not in s.chosen[h]]
    check("loader emits only chosen frames", not bad, f"{len(bad)} violations")


def test_left_right_separation() -> None:
    """Left and right must come from their own files, not a shared array."""
    s = Sequence(sequences()[0])
    frame = s.intersect_sorted[len(s.intersect_sorted) // 2]
    L = s.joints3d("left", frame)
    R = s.joints3d("right", frame)
    check("left and right 3D differ", float(np.linalg.norm(L.mean(0) - R.mean(0))) > 1e-3,
          f"centroid distance={np.linalg.norm(L.mean(0) - R.mean(0)):.4f} m")


def test_zero_pattern_is_all_or_nothing() -> None:
    """The (0,0) pattern is observed to affect a whole hand, never a few joints."""
    s = Sequence(sequences()[0])
    partial = total = 0
    for cam in s.cameras_2d("left")[:8]:
        for h in HANDS:
            for f in s.union_sorted[::10]:
                g = s.joints2d(h, cam, f)
                if g is None:
                    continue
                z = gigahands.is_zero_2d(g)
                if z.any():
                    total += 1
                    if not z.all():
                        partial += 1
    check("zero pattern is all-or-nothing", partial == 0,
          f"{partial} partial of {total} affected views")


def test_loader_drops_zero_2d() -> None:
    """drop_zero_2d must invalidate (0,0) joints despite their confidence of 1.0."""
    found = False
    for x in gigahands.iter_samples(max_sequences=1, max_cameras=12, max_frames=12):
        if x.extra.get("zero_2d_view"):
            found = True
            check("zero-2d view yields no valid joints", int(x.valid.sum()) == 0,
                  f"{int(x.valid.sum())} valid joints survived")
            check("zero-2d view still reports confidence 1.0",
                  float(np.max(x.extra["conf"])) >= 1.0,
                  "confirms a confidence filter alone cannot remove it")
            break
    if not found:
        print("  [SKIP] no zero-2d view in the sampled subset")
    # and the opt-out still reproduces the old behaviour
    for x in gigahands.iter_samples(max_sequences=1, max_cameras=12, max_frames=12,
                                    drop_zero_2d=False):
        if x.extra.get("zero_2d_view"):
            check("drop_zero_2d=False reproduces old behaviour", int(x.valid.sum()) > 0,
                  f"{int(x.valid.sum())} valid joints")
            break


def test_video_matched_by_exact_stem() -> None:
    """A video is only attached when its filename matches the annotated take."""
    mismatched = 0
    for d in sequences():
        s = Sequence(d)
        for cam in s.cameras_2d("left"):
            v = s.video_path(cam)
            if v is None:
                mismatched += 1
                continue
            check_stem = v.stem.split("_")[-1]
            p2 = s._files2d["left"].get(cam)
            if p2 is not None:
                assert check_stem == p2.stem.split("_")[-1]
    check("cameras whose annotated video is absent are reported, not substituted",
          mismatched > 0, f"{mismatched} cameras without the annotated take "
                          "(expected: p52-instrument-0034)")


def main() -> int:
    if not HAND_POSE_ROOT.exists():
        print("GigaHands demo not present; skipping")
        return 0
    print("CAM-EXP-001.2 mapping regression tests")
    for fn in (test_3d_rows_are_frame_indexed, test_unchosen_3d_rows_are_zero,
               test_chosen_frames_filtering, test_left_right_separation,
               test_zero_pattern_is_all_or_nothing, test_loader_drops_zero_2d,
               test_video_matched_by_exact_stem):
        print(f"\n{fn.__name__}:")
        fn()
    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
