"""Tests that the reconstruction path cannot see dataset-provided 3D.

The scientific claim of CAM-EXP-001.3 rests entirely on non-circularity, so it
is checked mechanically rather than by inspection:

  * static: the reconstruction functions contain no call that reads provided 3D;
  * dynamic: with ``GigaHandsTake.joints3d`` replaced by a tripwire that raises,
    a full reconstruction still succeeds - proving it never consults it;
  * held-out: the excluded camera contributes no observation.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[3]))

import loco  # noqa: E402
from common import TAKES  # noqa: E402

FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_static_reconstruction_never_calls_joints3d():
    """No reconstruction-path function may reference joints3d / provided 3D."""
    banned_calls = {"joints3d", "compare_to_provided"}
    for fn in (loco.gather_observations, loco.reconstruct, loco.reconstruct_all,
               loco.refit_excluding, loco.project_hypothesis):
        tree = ast.parse(inspect.getsource(fn))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        hits = names & banned_calls
        check(f"{fn.__name__} does not touch provided 3D", not hits, f"found {hits}")


def test_dynamic_tripwire():
    """Reconstruction must succeed while reading provided 3D would raise."""
    take = TAKES[0]
    frame = take.intersect_sorted[len(take.intersect_sorted) // 2]
    cam = take.cameras_2d("left")[0]

    original = type(take).joints3d
    calls = []

    def tripwire(self, hand, f):
        calls.append((hand, f))
        raise AssertionError("provided 3D was read during reconstruction")

    type(take).joints3d = tripwire
    try:
        hyp = loco.reconstruct(take, "left", frame, cam, threshold_px=10.0,
                               min_inlier_cameras=3)
        ok = hyp.ok
        err = None
    except AssertionError as e:
        ok, err = False, str(e)
    finally:
        type(take).joints3d = original
    check("reconstruction runs with provided 3D disabled", ok,
          err or f"{int(hyp.ok_joints.sum())} joints reconstructed")
    check("provided 3D was never requested during reconstruction", not calls,
          f"{len(calls)} calls")


def test_heldout_camera_excluded():
    """The held-out camera must not appear among the observations used."""
    take = TAKES[0]
    frame = take.intersect_sorted[len(take.intersect_sorted) // 2]
    held = take.cameras_2d("left")[3]
    cams, uv, valid, names = loco.gather_observations(take, "left", frame, held)
    check("held-out camera absent from observations", held not in names,
          f"{len(names)} cameras used")
    all_cams, *_rest = loco.gather_observations(take, "left", frame, None)
    check("holding out removes exactly one camera",
          len(all_cams) - len(cams) == 1, f"{len(all_cams)} -> {len(cams)}")


def test_provided3d_blocked_for_unchosen_frames():
    """M3: rows outside chosen_frames are placeholders and must not be served."""
    take = TAKES[0]
    bad = max(take.union_sorted) + 3
    check("joints3d refuses an unchosen frame", take.joints3d("left", bad) is None)


def test_zero_pattern_never_used_as_observation():
    """M5: an all-(0,0) record must never enter the observation set."""
    take = TAKES[0]
    from experiments.src.datasets.gigahands import is_zero_2d
    checked = 0
    for frame in take.union_sorted[::40]:
        cams, uv, valid, names = loco.gather_observations(take, "left", frame, None)
        for k, n in enumerate(names):
            g = take.joints2d("left", n, frame)
            if g is not None and is_zero_2d(g).all():
                FAILURES.append("zero pattern used as observation")
            checked += 1
    check("no all-zero record used as an observation", True, f"{checked} observations checked")


def main() -> int:
    print("CAM-EXP-001.3 non-circularity tests")
    for fn in (test_static_reconstruction_never_calls_joints3d,
               test_dynamic_tripwire, test_heldout_camera_excluded,
               test_provided3d_blocked_for_unchosen_frames,
               test_zero_pattern_never_used_as_observation):
        print(f"\n{fn.__name__}:")
        fn()
    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
