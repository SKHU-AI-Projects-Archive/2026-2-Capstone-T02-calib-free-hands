"""metadata.json: git state, config copy and hash, library / tool versions, weight hashes."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys

from .paths import REPO_ROOT


def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def git_state() -> dict:
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "HandCalib")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_dirty": bool(dirty),
            "dirty_files": dirty.splitlines()[:50], "anycalib_submodule": _submodule()}


def _submodule():
    try:
        return git("-C", str(REPO_ROOT / "HandCalib" / "AnyCalib"), "rev-parse", "HEAD")
    except Exception:
        return None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    import cv2
    import matplotlib
    import numpy
    import scipy
    import yaml
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__, "opencv": cv2.__version__,
           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    return out


def build(cfg_text: str, cfg_path: str, extra: dict) -> dict:
    return {"git": git_state(), "config_path": cfg_path, "config_sha256": sha256_text(cfg_text), "config": cfg_text, "versions": versions(),
            "worktrees": {"used": [], "note": "다른 실험 코드는 쓰지 않음. AnyCalib는 같은 체크아웃의 models/anycalib.py(01 어댑터, main에 있음)를 수정 없이 import"},
            **extra}
