"""Lazy object-detection model classes."""

import importlib


_CLASS_MODULES = {
    "Detector": "base",
    "GroundingDinoDetect": "grounding_dino",
    "YOLODetect": "yolo",
    "LocateAnythingDetect": "locate_anything",
}


def __getattr__(name: str):
    module_name = _CLASS_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_CLASS_MODULES))


__all__ = list(_CLASS_MODULES)
