"""Lazy framework adapters for Hugging Face, Ultralytics, and Keras."""

import importlib


_ALIASES = {
    "huggingface": "huggingface",
    "hf": "huggingface",
    "ultralytics": "ultralytics",
    "ul": "ultralytics",
    "keras": "keras",
    "kerashub": "keras",
    "keras_hub": "keras",
    "common": "common",
}


def __getattr__(name: str):
    """Import a framework adapter only when callers select it."""
    module_name = _ALIASES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(_ALIASES))


__all__ = list(_ALIASES)
