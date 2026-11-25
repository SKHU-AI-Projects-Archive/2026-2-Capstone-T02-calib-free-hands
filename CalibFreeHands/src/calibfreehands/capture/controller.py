"""Runs the PC receiver (MediaMTX) and records takes; everything slow happens in worker threads.

The controller owns no widgets: it emits Qt signals (queued to the GUI thread) and the capture page reacts.
It is the Qt counterpart of ``streaming_test/pc/run_pc.py`` + ``recorder_gui.py``'s logic, built on the same modules.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from . import pc as pc_loader

log = logging.getLogger("calibfreehands.capture")
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never through a proxy


class CaptureController(QObject):
    status = Signal(dict)              # {"api": bool, "stream": bool, "record": bool, "recording_take": str|None}
    receiver_state = Signal(str, str)  # state: starting | running | reused | busy | failed, plain-language detail
    take_started = Signal(str, dict)   # role, {"locked": bool|None, "start": datetime|None, "latency": float|None}
    take_saved = Signal(str, list, str)  # role, final file paths, verify report
    take_failed = Signal(str, str)     # role, plain-language reason

    def __init__(self, parent: QObject | None = None, pc=None) -> None:
        super().__init__(parent)
        self.pc = pc or pc_loader.load()
        self.run_config = self.pc.folder / "run_config.yaml"
        self.manual_config = self.pc.folder / "run_config_manual.yaml"
        try:
            self.settings = self.pc.manual_mode.load_settings(self.run_config, self.manual_config)
        except SystemExit as exc:  # the PC modules report bad config files with sys.exit
            raise RuntimeError(str(exc)) from None
        self.recorder = self.pc.recorder_control.Recorder(self.settings)
        self._proc: subprocess.Popen | None = None
        self._owns_receiver = False
        self._polling = False
        self._timer = QTimer(self)
        self._timer.setInterval(max(200, int(float(self.settings["poll_interval_s"]) * 1000)))
        self._timer.timeout.connect(self._poll)

    # ---- receiver --------------------------------------------------------------------------------------------
    def begin(self) -> None:
        """Start polling the receiver and make sure one is running."""
        self._timer.start()
        threading.Thread(target=self._start_receiver, name="receiver-start", daemon=True).start()

    def _start_receiver(self) -> None:
        pc, s = self.pc, self.settings
        try:
            if pc.manual_mode.api_alive(s):
                self.receiver_state.emit("reused", "A receiver is already running; it is used as it is.")
                return
            self.receiver_state.emit("starting", "Starting the receiver ...")
            cfg = pc.run_pc.load_config(self.run_config)
            cfg["auto_verify"] = False  # manual mode verifies when a take stops, not through a MediaMTX hook
            exe = pc.run_pc.ensure_mediamtx(cfg)
            busy = pc.run_pc.busy_ports(cfg, s["api_port"])
            if busy:
                self.receiver_state.emit("busy", "The receiver ports are used by another program: " + ", ".join(busy))
                return
            env = pc.run_pc.build_env(cfg)
            pc.manual_mode.update_env(env, s)
            config = pc.folder / s["mediamtx_config_manual"]
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self._proc = subprocess.Popen([str(exe), str(config)], cwd=pc.folder, env=env, creationflags=flags,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._owns_receiver = True
            self.receiver_state.emit("running", "The receiver is running.")
        except SystemExit as exc:  # run_pc reports download / config problems with sys.exit
            log.error("receiver could not be started: %s", exc)
            self.receiver_state.emit("failed", "The receiver could not be started (see the log).")
        except Exception as exc:  # noqa: BLE001 - shown to the user in plain words, details in the log
            log.exception("receiver could not be started")
            self.receiver_state.emit("failed", f"The receiver could not be started ({type(exc).__name__}).")

    def shutdown(self) -> None:
        """Stop polling; end the phone's stream and stop the receiver if this application started it."""
        self._timer.stop()
        if not self._owns_receiver or self._proc is None or self._proc.poll() is not None:
            return
        self._kick_publishers()  # the phone learns at once that the PC is gone and ends its stream
        try:
            self.pc.run_pc.stop(self._proc, float(self.pc.run_pc.load_config(self.run_config)["stop_timeout_s"]))
        except Exception:  # noqa: BLE001
            log.exception("graceful receiver stop failed")
            self._proc.kill()

    def _api(self, method: str, endpoint: str):
        s = self.settings
        request = urllib.request.Request(f"http://{s['api_host']}:{s['api_port']}{endpoint}", method=method)
        with _NO_PROXY.open(request, timeout=float(s["api_timeout_s"])) as response:
            body = response.read()
        return json.loads(body) if body else None

    def _kick_publishers(self) -> None:
        try:
            for connection in (self._api("GET", "/v3/srtconns/list") or {}).get("items", []):
                if connection.get("state") == "publish":
                    self._api("POST", f"/v3/srtconns/kick/{connection['id']}")
        except (OSError, ValueError, KeyError):
            log.warning("could not disconnect the phone before stopping the receiver", exc_info=True)

    # ---- status ----------------------------------------------------------------------------------------------
    def _poll(self) -> None:
        if self._polling:
            return
        self._polling = True

        def work() -> None:
            try:
                self.status.emit(self.recorder.status())
            finally:
                self._polling = False

        threading.Thread(target=work, name="receiver-poll", daemon=True).start()

    def network_choices(self) -> list[tuple[str, str]]:
        """(label, address) of the PC networks the phone could be in; the Wi-Fi one first."""
        return list(self.pc.netinfo.choices(self.settings))

    def phone_url(self, ip: str) -> str:
        return pc_loader.publish_url(self.settings, ip)

    # ---- takes -----------------------------------------------------------------------------------------------
    def start_take(self, role: str, take_name: str) -> None:
        """Lock the phone's exposure/focus, then start recording (worker thread)."""
        def work() -> None:
            RecorderError = self.pc.recorder_control.RecorderError
            try:
                locked = self.pc.phone_link.set_hold(self.settings, True)  # first frame is already locked
                try:
                    result = self.recorder.start(take_name)
                except RecorderError:
                    if locked:
                        self.pc.phone_link.set_hold(self.settings, False)
                    raise
                self.take_started.emit(role, {"locked": locked, "start": result["start_time"],
                                              "latency": result["latency_s"]})
            except RecorderError as exc:
                log.error("start of the %s take failed: %s", role, exc)
                self.take_failed.emit(role, "The recording could not be started. Is the phone connected?")
            except Exception as exc:  # noqa: BLE001
                log.exception("start of the %s take failed", role)
                self.take_failed.emit(role, f"The recording could not be started ({type(exc).__name__}).")

        threading.Thread(target=work, name="take-start", daemon=True).start()

    def stop_take(self, role: str, session_dir: Path, final_stem: str) -> None:
        """Release the lock, stop and check the take, move it into *session_dir* as ``<final_stem>.mp4``."""
        def work() -> None:
            RecorderError = self.pc.recorder_control.RecorderError
            threading.Thread(target=self.pc.phone_link.set_hold, args=(self.settings, False), daemon=True).start()
            try:
                result = self.recorder.stop()
                files = self._move_into(Path(session_dir), result["files"], final_stem)
                self.take_saved.emit(role, [str(f) for f in files], result["summary"])
            except RecorderError as exc:
                log.error("stop of the %s take failed: %s", role, exc)
                self.take_failed.emit(role, "The recording could not be finished cleanly.")
            except Exception as exc:  # noqa: BLE001
                log.exception("stop of the %s take failed", role)
                self.take_failed.emit(role, f"The recording could not be saved ({type(exc).__name__}).")

        threading.Thread(target=work, name="take-stop", daemon=True).start()

    @staticmethod
    def _move_into(directory: Path, files: list[Path], stem: str) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        moved = []
        for position, source in enumerate(files):
            name = stem if position == 0 else f"{stem}_part{position + 1}"
            target = directory / f"{name}{source.suffix}"
            counter = 1
            while target.exists():  # never overwrite an earlier take
                counter += 1
                target = directory / f"{name}_{counter}{source.suffix}"
            shutil.move(str(source), str(target))
            report = source.with_suffix(".intervals.csv")
            if report.is_file():
                shutil.move(str(report), str(target.with_suffix(".intervals.csv")))
            moved.append(target)
        return moved

    @property
    def recording_dir(self) -> Path:
        return Path(self.recorder.rec_dir)
