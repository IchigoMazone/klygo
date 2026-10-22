"""Structured object-detection results through :mod:`klygo.outputs.detect`."""

from . import detect
from .detect import build_detections

__all__ = [
    "detect",
    "build_detections",
]
