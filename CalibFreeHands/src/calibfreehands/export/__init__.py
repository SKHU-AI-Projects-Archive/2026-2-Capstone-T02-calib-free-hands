from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "export_csv": ".tabular",
    "export_json": ".tabular",
    "encode_preview": ".preview",
    "choose_output_directory": ".paths",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
