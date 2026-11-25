"""Manual recording mode glue for run_pc.py: settings, MediaMTX env overrides, GUI process.

Settings = run_config_manual.yaml (record_mode, api_*, ...) + stream_path / verify_hook from
run_config.yaml.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never through a proxy

BASE = Path(__file__).resolve().parent

MANUAL_KEYS = (
    "record_mode", "api_host", "api_port", "api_timeout_s", "poll_interval_s",
    "mediamtx_config_manual", "recordings_dir",
    "start_wait_timeout_s", "segment_settle_s", "segment_wait_timeout_s", "timestamp_format",
)
RUN_KEYS = ("stream_path", "verify_hook", "srt_port", "rtsp_port", "lan_ip", "lan_probe_address")
# srt_port, lan_ip, lan_probe_address: Phone QR; rtsp_port: live view
# Optional keys of run_config_manual.yaml (old copies without them keep working).
OPTIONAL_DEFAULTS = {
    "lock_on_record": True,   # lock the phone's exposure/focus while recording (phone_link.py)
    "phone_port": 8898,       # the phone app's remote control port (RemoteControl.REMOTE_PORT in the app)
    "phone_address": "",      # empty: the phone's address is read from the receiver
    "phone_timeout_s": 2.0,
    "live_view": True,        # live camera picture in the recorder window (needs ffmpeg)
    "output_format": "mp4",   # mp4: a finished take is copied into an .mp4 (no re-encoding); ts: keep the .ts
}
RECORD_MODES = ("auto", "manual")


def _read(path, keys):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in keys if k not in cfg]
    if missing:
        sys.exit(f"config {path}: missing keys: {', '.join(missing)}")
    return cfg


def load_settings(run_config, manual_config):
    """Merged settings for recorder_control / recorder_gui / run_pc."""
    manual = _read(manual_config, MANUAL_KEYS)
    if manual["record_mode"] not in RECORD_MODES:
        sys.exit(f"config {manual_config}: record_mode must be one of {RECORD_MODES}")
    run = _read(run_config, RUN_KEYS)
    optional = {k: manual.get(k, default) for k, default in OPTIONAL_DEFAULTS.items()}
    return {**{k: run[k] for k in RUN_KEYS}, **{k: manual[k] for k in MANUAL_KEYS}, **optional}


def load(run_config, manual_config, force=False):
    """Settings if record_mode is manual (or force=True, e.g. run_recorder.bat), else None (= auto: nothing changes)."""
    if not Path(manual_config).exists():
        return None
    settings = load_settings(run_config, manual_config)
    if force:
        settings["record_mode"] = "manual"
    return settings if settings["record_mode"] == "manual" else None


def update_env(env, settings):
    """MediaMTX env overrides for manual mode (the API listens on the configured localhost port)."""
    env["MTX_APIADDRESS"] = f"{settings['api_host']}:{settings['api_port']}"


def api_alive(settings):
    """True if a MediaMTX Control API already answers on the configured address, i.e. a receiver is running."""
    url = f"http://{settings['api_host']}:{settings['api_port']}/v3/paths/list"
    try:
        with _OPENER.open(url, timeout=2):
            return True
    except urllib.error.HTTPError:
        return True  # something answers, even if it refuses us
    except (OSError, ValueError):
        return False


def stream_online(settings):
    """True if a phone is publishing right now (some path is ready), False if not, None if the API is unreachable."""
    url = f"http://{settings['api_host']}:{settings['api_port']}/v3/paths/list"
    try:
        with _OPENER.open(url, timeout=2) as resp:
            items = json.load(resp)["items"]
    except (OSError, ValueError, KeyError):
        return None
    return any(item.get("ready") for item in items)


def describe(settings):
    return (f"Record mode: MANUAL - start/stop recording in the recorder window "
            f"(Control API http://{settings['api_host']}:{settings['api_port']}, localhost only)")


def launch_gui(run_config, manual_config, verbose=False):
    cmd = [sys.executable, str(BASE / "recorder_gui.py"),
           "--config", str(run_config), "--manual-config", str(manual_config)]
    if verbose:
        cmd.append("--verbose")
    return subprocess.Popen(cmd, cwd=BASE)


def stop_gui(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
