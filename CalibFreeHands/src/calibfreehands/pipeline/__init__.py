from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CanonicalResult": ".result",
    "save_result": ".result",
    "JobState": ".worker",
    "JobUpdate": ".worker",
    "ProcessingJob": ".worker",
    "JobThread": ".qt_worker",
    "runtime_metadata": ".provenance",
    "sha256_file": ".provenance",
    "process_and_save": ".service",
    "FrameHandMetrics": ".metrics",
    "MetricDefinition": ".metrics",
    "MetricStatus": ".metrics",
    "compare_results": ".metrics",
    "metric_registry": ".metrics",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
