"""The PC receiver code of ``streaming_test/pc`` (MediaMTX launcher, recorder control, phone link, network info).

It stays the single source of that code, so the stand-alone tool and this application cannot drift apart. Its
modules are plain Python (standard library + PyYAML) and find their files (``mediamtx.exe``, the YAML configs)
relative to their own folder, so they only need the folder on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from calibfreehands.handcalib.runner import repo_root


def pc_dir() -> Path:
    return repo_root() / "streaming_test" / "pc"


def load() -> SimpleNamespace:
    """Import the PC modules once. Raises ImportError with a readable message if the folder is missing."""
    folder = pc_dir()
    if not (folder / "recorder_control.py").is_file():
        raise ImportError(f"The PC receiver folder was not found: {folder}")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import manual_mode
    import netinfo
    import phone_link
    import recorder_control
    import run_pc
    return SimpleNamespace(manual_mode=manual_mode, netinfo=netinfo, phone_link=phone_link,
                           recorder_control=recorder_control, run_pc=run_pc, folder=folder)


def publish_url(settings: dict, ip: str) -> str:
    """The address the phone app streams to (what the QR code holds)."""
    return f"srt://{ip}:{settings['srt_port']}?streamid=publish:{settings['stream_path']}"
