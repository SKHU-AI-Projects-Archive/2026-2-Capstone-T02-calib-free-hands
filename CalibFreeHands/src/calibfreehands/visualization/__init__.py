from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ResultViewer": ".viewer",
    "load_npz_result": ".viewer",
    "ComparisonViewer": ".comparison",
    "calculate_reference_metrics": ".comparison",
    "export_metrics_csv": ".comparison",
    "export_metrics_json": ".comparison",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
