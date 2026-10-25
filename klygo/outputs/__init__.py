"""Structured object-detection results through :mod:`klygo.outputs.detect`."""

from . import detect
from .detect import Box, Detection, Detections, build_detections

__all__ = [
    "detect",
    "Box",
    "Detection",
    "Detections",
    "build_detections",
]
