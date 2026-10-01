"""PHASE I — the C_WRONG_BONE negative control.

Re-runs the real label-free hand score with the right hand's bone indices
permuted WITHIN each finger, using the permutation frozen in
`cam_exp_0094_controls_spec_v1.json`.

Scope: M3_SCENE_SHARED_GENERIC_FULL only. That was decided from measured
runtime BEFORE any real focal result and is recorded in the method spec: the
control asks whether the FULL method's hand score uses real anatomical bone
correspondence, which M3 alone answers.

The method is never retuned after looking at this control.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def main():
    cmd = [sys.executable, str(SRC / "run_real_label_free.py"),
           "--control-wrong-bone",
           "--methods", "M3_SCENE_SHARED_GENERIC_FULL"]
    print("running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
