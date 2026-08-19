"""
Giao diện chung & Lớp kết quả đầu ra chuẩn hóa (`klygo.models.interfaces`).
"""

from .base import DetectorModel
from .outputs import DetectedObject, DetectionResult, CroppedObject, CropResult

__all__ = [
    "DetectorModel",
    "DetectedObject",
    "DetectionResult",
    "CroppedObject",
    "CropResult",
]
