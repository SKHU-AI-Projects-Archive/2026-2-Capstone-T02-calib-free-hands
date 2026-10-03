"""Benchmark 실행 시간과 GPU telemetry를 기록하는 작은 보조 함수 모음."""

import csv
import gzip
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def physical_gpu_token():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [] if value is None else [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU token for benchmark mode.")
    return tokens[0]


def query_nvidia_smi(token):
    query = ",".join(("uuid", "name", "driver_version", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw", "power.limit"))
    command = ["nvidia-smi", "-i", token, f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if not output:
        return None
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 9:
        return None

    def number(value):
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "gpu_uuid": values[0], "gpu_name": values[1], "nvidia_driver_version": values[2],
        "gpu_util_percent": number(values[3]), "gpu_memory_used_mib": number(values[4]),
        "gpu_memory_total_mib": number(values[5]), "gpu_temperature_c": number(values[6]),
        "gpu_power_w": number(values[7]), "gpu_power_limit_w": number(values[8]),
    }


def write_gzip_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


class TelemetrySampler:
    """Collect nvidia-smi and PyTorch memory snapshots without blocking inference."""

    def __init__(self, token, state_provider, interval=1.0):
        self.token = token
        self.state_provider = state_provider
        self.interval = interval
        self.rows = []
        self._started = None
        self._thread = None
        self._stop = threading.Event()

    def _sample(self):
        row = {"elapsed_seconds": time.perf_counter() - self._started, **self.state_provider()}
        gpu = query_nvidia_smi(self.token)
        if gpu is not None:
            row.update(gpu)
            row["telemetry_available"] = True
        else:
            row["telemetry_available"] = False
        self.rows.append(row)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._started = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def finite_mean(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def finite_max(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return max(values) if values else None
