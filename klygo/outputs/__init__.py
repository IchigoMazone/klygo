"""Structured object-detection results through :mod:`klygo.outputs.detect`."""

from . import detect
from .detect import Box, Detection, Detections, build_detections
from .group import DetectionGroup, group

__all__ = [
    "detect",
    "Box",
    "Detection",
    "Detections",
    "build_detections",
    "DetectionGroup",
    "group",
]
