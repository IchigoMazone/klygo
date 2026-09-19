"""
Các lớp mô hình nhận diện đối tượng cụ thể (`klygo.models.detection`).
"""

from .base import Detector
from .grounding_dino import GroundingDinoDetect
from .yolo import YOLODetect
from .locate_anything import LocateAnythingDetect

__all__ = ["Detector", "GroundingDinoDetect", "YOLODetect", "LocateAnythingDetect"]
