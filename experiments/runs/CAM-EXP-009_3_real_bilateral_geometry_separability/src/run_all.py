"""Phase B driver: templates -> distances -> controls -> per-bone -> verdict."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["build_geometry_templates.py", "compute_distances.py",
         "run_controls.py", "per_bone.py", "evaluate.py", "figures.py",
         "self_audit.py"]


def main():
    for s in STEPS:
        print("\n=== %s" % s, flush=True)
        r = subprocess.run([sys.executable, str(SRC / s)])
        if r.returncode != 0:
            raise SystemExit("FAILED: %s" % s)


if __name__ == "__main__":
    main()
