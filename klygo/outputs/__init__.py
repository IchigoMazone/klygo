"""Structured object-detection result containers and group operations.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/outputs/Document_Outputs_Overview.ipynb

Public API (1 function, 4 classes, 1 submodule):
    Detection containers (`detect.py`):
        1. detect              Submodule containing core detection container types.
        2. Box                 Single bounding box with label, score, class_id, and metadata.
        3. Detection           Single-frame detection container with box/label/device helpers.
        4. Detections          Multi-frame collection supporting streaming, packing, and export.
        5. build_detections(...) Pack normalized raw predictions into a Detections collection.

    Aligned multi-collection group (`detection_group.py`):
        6. DetectionGroup      Coordinate shared operations across aligned Detections sets.
"""

from . import detect
from .detect import Box, Detection, Detections, build_detections
from .detection_group import DetectionGroup

__all__ = [
    "detect",
    "Box",
    "Detection",
    "Detections",
    "build_detections",
    "DetectionGroup",
]
