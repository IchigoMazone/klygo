"""
Giao diện chung & Lớp kết quả đầu ra chuẩn hóa (`klygo.models.interfaces`).
"""

from .base import DetectorModel
from .outputs import (
    CropResult,
    CropResults,
    DetectionResult,
    DetectionResults,
    DetectedObject,
    CroppedObject,
    PreviewResult,
)

__all__ = [
    "DetectorModel",
    "CropResult",
    "CropResults",
    "DetectionResult",
    "DetectionResults",
    "DetectedObject",
    "CroppedObject",
    "PreviewResult",
]
