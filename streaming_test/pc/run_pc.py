"""One-command PC receiver: (download and) start MediaMTX, print phone URLs, auto-verify recordings.

Usage: python run_pc.py [run_config.yaml]      (or double-click run_pc.bat)
Settings come from run_config.yaml. Requires: pyyaml, and ffprobe (ffmpeg) for auto_verify.
"""

import argparse
import errno
import hashlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

import manual_mode
import netinfo

BASE = Path(__file__).resolve().parent

REQUIRED_KEYS = (
    "mediamtx_version", "mediamtx_download_url", "mediamtx_checksum_url", "download_timeout_s",
    "mediamtx_exe", "mediamtx_config", "verify_hook",
    "stream_path", "srt_port", "rtsp_port", "auto_verify", "stop_timeout_s",
    "lan_ip", "lan_probe_address", "emulator_host",
)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"config {path}: missing keys: {', '.join(missing)}")
    cfg["mediamtx_version"] = str(cfg["mediamtx_version"]).lstrip("v")
    return cfg


def fetch(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def ensure_mediamtx(cfg):
    """Return the path of mediamtx.exe, downloading and checking it first if it is missing."""
    exe = BASE / cfg["mediamtx_exe"]
    if exe.exists():
        return exe
    version = cfg["mediamtx_version"]
    url = cfg["mediamtx_download_url"].format(version=version)
    zip_name = url.rsplit("/", 1)[-1]
    timeout = cfg["download_timeout_s"]
    print(f"{exe.name} not found, downloading MediaMTX v{version}\n  {url}", flush=True)
    try:
        data = fetch(url, timeout)
        checksums = fetch(cfg["mediamtx_checksum_url"].format(version=version), timeout).decode()
    except OSError as e:
        sys.exit(f"download failed: {e}")

    expected = None
    for line in checksums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == zip_name:
            expected = parts[0].lower()
    actual = hashlib.sha256(data).hexdigest()
    if expected is None:
        sys.exit(f"no checksum for {zip_name} in the release checksum list")
    if actual != expected:
        sys.exit(f"checksum mismatch for {zip_name}: expected {expected}, got {actual}")
    print("  checksum OK", flush=True)

    # Extract only the executable, by exact name (never trust paths inside the archive).
    with tempfile.TemporaryDirectory(dir=BASE) as tmp:
        zpath = Path(tmp) / zip_name
        zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            if exe.name not in z.namelist():
                sys.exit(f"{exe.name} not found inside {zip_name}")
            exe.write_bytes(z.read(exe.name))
    print(f"  extracted {exe}", flush=True)
    return exe


def lan_ip(cfg):
    """Address in the printed phone URL: lan_ip, or (auto) the Wi-Fi address, else the default route's."""
    return netinfo.lan_ip(cfg)


def build_env(cfg, verbose=False):
    """MediaMTX reads MTX_<SETTING> variables over the yml, so the yml stays untouched."""
    env = os.environ.copy()
    if not verbose:
        env["MTX_LOGLEVEL"] = "warn"  # only warnings and errors in the console
    env["MTX_SRTADDRESS"] = f":{cfg['srt_port']}"
    env["MTX_RTSPADDRESS"] = f":{cfg['rtsp_port']}"
    if cfg["auto_verify"]:
        hook = BASE / cfg["verify_hook"]
        env["MTX_PATHDEFAULTS_RUNONRECORDSEGMENTCOMPLETE"] = f'"{sys.executable}" "{hook}" "$MTX_SEGMENT_PATH"'
    else:
        env["MTX_PATHDEFAULTS_RUNONRECORDSEGMENTCOMPLETE"] = ""
    return env


def print_info(cfg, verbose=False, manual=False):
    if not verbose:  # no addresses or ports on screen; --verbose shows them
        print("\nCalib-free-hands - PC receiver")
        print("Phone: scan the QR code that opens with the Recorder window (Phone QR button) with the app." if manual
              else "Phone: scan the QR code from qr_window.py with the app. Recording starts when the phone connects.")
        print("Press Ctrl+C to stop. (--verbose shows addresses and the receiver's log.)\n", flush=True)
        return
    ip, path = lan_ip(cfg), cfg["stream_path"]
    print()
    print("Phone publish URL (same Wi-Fi/LAN as this PC):")
    print(f"  srt://{ip}:{cfg['srt_port']}?streamid=publish:{path}")
    print("Android emulator publish URL:")
    print(f"  srt://{cfg['emulator_host']}:{cfg['srt_port']}?streamid=publish:{path}")
    print("Live preview (another terminal, while the phone is streaming):")
    print(f"  ffplay rtsp://localhost:{cfg['rtsp_port']}/{path}")
    print(f"Recordings: {BASE / 'recordings' / path}")
    print("Auto-verify each finished recording: " + ("ON" if cfg["auto_verify"] else "OFF"))
    print("Press Ctrl+C to stop.")
    print(flush=True)


def stop(proc, timeout):
    """Ask MediaMTX to shut down like Ctrl+C would (so it closes the recording), then wait."""
    if proc.poll() is None:
        # MediaMTX runs in its own process group (see main), so it does not get the user's Ctrl+C.
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        print("MediaMTX did not exit in time, terminating", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def can_bind(kind, port):
    """Whether the port is free. MediaMTX listens on the IPv6 dual-stack wildcard, an IPv4-only check misses it."""
    for family, address in ((socket.AF_INET6, "::"), (socket.AF_INET, "")):
        try:
            with socket.socket(family, kind) as s:
                if family == socket.AF_INET6:
                    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                s.bind((address, port))
        except OSError as e:
            if family == socket.AF_INET6 and e.errno in (errno.EAFNOSUPPORT, 10047):  # no IPv6 on this PC
                continue
            return False
    return True


def busy_ports(cfg, api_port=None):
    """Names of the receiver ports that something else already uses (the receiver could not start)."""
    checks = [("SRT", socket.SOCK_DGRAM, cfg["srt_port"]), ("RTSP", socket.SOCK_STREAM, cfg["rtsp_port"])]
    if api_port:
        checks.append(("Control API", socket.SOCK_STREAM, api_port))
    return [f"{name} ({'UDP' if kind == socket.SOCK_DGRAM else 'TCP'} {port})"
            for name, kind, port in checks if not can_bind(kind, port)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=str(BASE / "run_config.yaml"),
                        help="YAML config path")
    parser.add_argument("--manual-config", default=str(BASE / "run_config_manual.yaml"),
                        help="record_mode (auto|manual) settings")
    parser.add_argument("--manual", action="store_true",
                        help="manual recording (Recorder window) whatever record_mode says")
    parser.add_argument("--verbose", action="store_true",
                        help="print addresses, ports and the receiver's full log")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manual = manual_mode.load(args.config, args.manual_config, force=args.manual)  # None = record_mode auto
    if manual:
        cfg["auto_verify"] = False  # manual mode verifies on stop, not through a MediaMTX hook

    exe = ensure_mediamtx(cfg)
    config_file = BASE / (manual["mediamtx_config_manual"] if manual else cfg["mediamtx_config"])
    if not config_file.exists():
        sys.exit(f"MediaMTX config not found: {config_file}")

    print_info(cfg, args.verbose, bool(manual))
    if manual and manual_mode.api_alive(manual):
        # A receiver from an earlier start is still running (its Recorder window may be gone): reuse it.
        print("A receiver is already running. Not starting another one: opening only the Recorder window.\n"
              "Closing the window (or Ctrl+C here) ends this; the running receiver keeps running\n"
              "(stop it with Ctrl+C in its own window).", flush=True)
        gui = manual_mode.launch_gui(args.config, args.manual_config, args.verbose)
        try:
            gui.wait()
        except KeyboardInterrupt:
            pass
        finally:
            manual_mode.stop_gui(gui)
        return 0
    busy = busy_ports(cfg, manual["api_port"] if manual else None)
    if busy:
        sys.exit("Cannot start the receiver, already in use: " + ", ".join(busy) + "\n"
                 "Another receiver (an earlier run_pc.bat / run_recorder.bat window) is probably still running:\n"
                 "close it with Ctrl+C, then start again.")
    env = build_env(cfg, args.verbose)
    if manual:
        if args.verbose:
            print(manual_mode.describe(manual), flush=True)
        manual_mode.update_env(env, manual)
    # Own process group: only this launcher handles Ctrl+C and stops MediaMTX in a controlled way.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen([str(exe), str(config_file)], cwd=BASE, env=env,
                            creationflags=flags)
    gui = manual_mode.launch_gui(args.config, args.manual_config, args.verbose) if manual else None
    try:
        while True:
            try:
                code = proc.wait(0.5)
            except subprocess.TimeoutExpired:
                if gui is not None and gui.poll() is not None:
                    # The Recorder window was closed: end the receiver too, so the phone's stream ends with it.
                    print("Recorder window closed. Stopping the receiver ...", flush=True)
                    stop(proc, cfg["stop_timeout_s"])
                    print("Stopped.", flush=True)
                    return 0
                continue
            print(f"MediaMTX exited with code {code}", flush=True)
            return code
    except KeyboardInterrupt:
        print("\nStopping MediaMTX ...", flush=True)
        stop(proc, cfg["stop_timeout_s"])
        print("Stopped.", flush=True)
        return 0
    finally:
        manual_mode.stop_gui(gui)


if __name__ == "__main__":
    sys.exit(main())
