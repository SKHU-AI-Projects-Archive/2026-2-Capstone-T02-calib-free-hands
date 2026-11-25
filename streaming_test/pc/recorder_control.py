"""Start/stop recording of a MediaMTX path through the Control API (localhost only).

Endpoints (MediaMTX v1.21.4):
  GET   /v3/paths/get/{name}             stream online?
  GET   /v3/config/paths/get/{name}      record flag
  PATCH /v3/config/paths/patch/{name}    {"record": true|false}
  POST  /v3/config/paths/add/{name}      create the path config if it does not exist
  GET   /v3/recordings/get/{name}        segment start times

Usage:  rec = Recorder(manual_mode.load_settings(run_config, manual_config))
        rec.status(); rec.start("take1"); rec.stop()
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
log = logging.getLogger("recorder")


class RecorderError(RuntimeError):
    pass


class ApiError(RecorderError):
    def __init__(self, code, message):
        super().__init__(f"API error {code}: {message}")
        self.code = code


class Recorder:
    def __init__(self, settings):
        self.s = settings
        self.path = settings["stream_path"]
        self.base_url = f"http://{settings['api_host']}:{settings['api_port']}"
        self.rec_dir = BASE / settings["recordings_dir"] / self.path
        self._take = None  # set while recording: name, files seen before start, start time

    # ---- HTTP ----
    def _api(self, method, endpoint, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + endpoint, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.s["api_timeout_s"]) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode(errors="replace")) from None
        except (urllib.error.URLError, OSError) as e:
            raise RecorderError(f"MediaMTX API not reachable at {self.base_url}: {e}") from None

    def _name(self):
        return urllib.parse.quote(self.path, safe="")

    # ---- status ----
    def status(self):
        """{'api': bool, 'stream': bool, 'record': bool, 'recording_take': str|None}"""
        out = {"api": True, "stream": False, "record": False,
               "recording_take": self._take["name"] if self._take else None}
        try:
            try:
                p = self._api("GET", f"/v3/paths/get/{self._name()}")
                out["stream"] = bool(p.get("online", p.get("ready")))
            except ApiError as e:
                if e.code != 404:
                    raise
            try:
                out["record"] = bool(self._api("GET", f"/v3/config/paths/get/{self._name()}").get("record"))
            except ApiError as e:
                if e.code != 404:
                    raise
        except RecorderError:
            out["api"] = False
        return out

    # ---- record on/off ----
    def _set_record(self, on):
        try:
            self._api("PATCH", f"/v3/config/paths/patch/{self._name()}", {"record": on})
        except ApiError as e:
            if e.code != 404:
                raise
            # Path has no config entry yet (stream_path differs from the one in the yml): create it.
            self._api("POST", f"/v3/config/paths/add/{self._name()}", {"record": on})

    def _files(self):
        if not self.rec_dir.is_dir():
            return {}
        return {p: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in self.rec_dir.iterdir() if p.is_file() and p.suffix != ".csv"}

    def _segment_start(self):
        """Start time (local, aware) of the newest segment according to the API, or None."""
        try:
            segs = self._api("GET", f"/v3/recordings/get/{self._name()}").get("segments") or []
            return datetime.fromisoformat(segs[-1]["start"].replace("Z", "+00:00")).astimezone()
        except (RecorderError, IndexError, KeyError, ValueError):
            return None

    @staticmethod
    def _safe(name):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
        if not safe:
            raise RecorderError("take name is empty")
        return safe

    def start(self, take_name):
        """Switch recording on. Returns {'take', 'start_time', 'latency_s'}."""
        if self._take:
            raise RecorderError(f"already recording take '{self._take['name']}'")
        take = self._safe(take_name)
        st = self.status()
        if not st["api"]:
            raise RecorderError(f"MediaMTX API not reachable at {self.base_url}")
        if not st["stream"]:
            raise RecorderError("no stream connected")
        before = set(self._files())
        t_req = time.time()
        self._set_record(True)
        self._take = {"name": take, "before": before, "start": None, "t_req": t_req}
        log.info("recording requested for take '%s' (starts at the next keyframe)", take)

        deadline = time.monotonic() + self.s["start_wait_timeout_s"]
        while time.monotonic() < deadline:
            if set(self._files()) - before:
                break
            time.sleep(0.2)
        else:
            log.warning("no recording file after %s s; recording stays enabled, start time unknown",
                        self.s["start_wait_timeout_s"])
            return {"take": take, "start_time": None, "latency_s": None}
        start = self._segment_start() or datetime.now().astimezone()
        self._take["start"] = start
        latency = start.timestamp() - t_req
        log.info("recording started: take '%s', actual start %s (%.2f s after the request)",
                 take, start.isoformat(timespec="milliseconds"), latency)
        return {"take": take, "start_time": start, "latency_s": latency}

    def _wait_closed(self, before):
        """New segment files, once they have stopped changing (= recorder closed them)."""
        deadline = time.monotonic() + self.s["segment_wait_timeout_s"]
        last, stable_since = None, None
        while time.monotonic() < deadline:
            cur = {p: v for p, v in self._files().items() if p not in before}
            now = time.monotonic()
            if cur and cur == last:
                stable_since = stable_since or now
                if now - stable_since >= self.s["segment_settle_s"]:
                    return sorted(cur)
            else:
                last, stable_since = cur, None
            time.sleep(0.25)
        raise RecorderError("recording file did not settle in time")

    def _rename(self, files, take, start):
        stamp = (start or datetime.fromtimestamp(self._take["t_req"])).strftime(self.s["timestamp_format"])
        out = []
        for i, f in enumerate(files, 1):
            suffix = "" if i == 1 else f"_part{i}"
            target = f.with_name(f"{take}_{stamp}{suffix}{f.suffix}")
            n = 1
            while target.exists():
                n += 1
                target = f.with_name(f"{take}_{stamp}{suffix}_{n}{f.suffix}")
            f.rename(target)
            out.append(target)
        return out

    def _to_mp4(self, path):
        """MediaMTX records MPEG-TS (safe if the PC dies mid-take). The finished take is copied into an .mp4 without
        re-encoding (about a second, no quality loss; frame times are kept) and the .ts is removed. If that fails
        the .ts stays and is used as it is."""
        if self.s.get("output_format", "mp4") != "mp4" or path.suffix.lower() != ".ts":
            return path
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            log.warning("ffmpeg not found: the take stays an .ts file")
            return path
        target = path.with_suffix(".mp4")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-c", "copy",
               "-movflags", "+faststart", str(target)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("mp4 conversion failed (%s): the take stays an .ts file", e)
            target.unlink(missing_ok=True)
            return path
        if proc.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            log.warning("mp4 conversion failed: %s", proc.stderr.strip()[-300:])
            target.unlink(missing_ok=True)
            return path
        path.unlink()
        return target

    def _verify(self, path):
        hook = BASE / self.s["verify_hook"]
        proc = subprocess.run([sys.executable, str(hook), str(path.resolve())],
                              capture_output=True, text=True, cwd=BASE)
        return (proc.stdout + proc.stderr).strip()

    def stop(self):
        """Switch recording off, rename the closed take(s), verify. Returns {'files', 'summary'}."""
        if not self._take:
            raise RecorderError("not recording")
        take = self._take
        self._set_record(False)
        log.info("recording stop requested for take '%s'", take["name"])
        try:
            files = self._wait_closed(take["before"])
        finally:
            self._take = None
        files = self._rename(files, take["name"], take["start"])
        files = [self._to_mp4(f) for f in files]
        summary = "\n\n".join(self._verify(f) for f in files)
        log.info("take '%s' saved: %s", take["name"], ", ".join(str(f) for f in files))
        return {"files": files, "summary": summary}
