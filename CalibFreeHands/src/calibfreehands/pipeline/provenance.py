from __future__ import annotations

import hashlib
import importlib.util
import platform
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata() -> dict:
    metadata = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy
        metadata["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch
        metadata["torch"] = torch.__version__
        metadata["cuda"] = torch.version.cuda
        metadata["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        metadata["torch"] = None
    spec = importlib.util.find_spec("anycalib")
    if spec is not None:
        metadata["anycalib_module"] = {"name": "anycalib", "origin": spec.origin}
    return metadata
