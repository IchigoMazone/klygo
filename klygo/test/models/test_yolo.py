"""YOLO closed-set and open-vocabulary prompt behavior."""

import unittest

from PIL import Image

from klygo import models
from klygo.models.detection.yolo import YOLODetect
from klygo.models.load import _resolve


class _OpenVocabularyModel:
    def __init__(self):
        self.class_updates = []

    def set_classes(self, classes):
        self.class_updates.append(classes)

    def __call__(self, images, **kwargs):
        return []


class _ClosedSetModel:
    def __call__(self, images, **kwargs):
        return []


class TestYOLOPromptDetection(unittest.TestCase):
    def _detector(self, wrapped_model):
        metadata = models.configure(
            "yolo-audit",
            metadata={
                "class": "klygo.models.detection.YOLODetect",
                "details": {
                    "name": "YOLO",
                    "task": "Object-Detection",
                    "backend": "Ultralytics",
                    "library": "ultralytics",
                },
                "flags": {"model": False, "post": True},
                "model": {},
                "post": {"threshold": 0.25, "iou": 0.7},
                "priority": {"post": ("threshold", "iou")},
            },
        )
        detector = YOLODetect(metadata)
        detector.model = wrapped_model
        detector.format_results = lambda results: [
            {"boxes": [], "scores": [], "labels": []}
        ]
        return detector

    def test_registry_resolves_yolo_world(self):
        entry = _resolve("yolov8s-worldv2.pt")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["details"]["name"], "YOLO-World")

        entry = _resolve("yoloe-11s-seg.pt")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["details"]["name"], "YOLOE")

    def test_prompt_updates_open_vocabulary_only_when_changed(self):
        wrapped_model = _OpenVocabularyModel()
        detector = self._detector(wrapped_model)
        image = Image.new("RGB", (8, 8))

        first = detector.forward([image], prompt=["helmet", "forklift"])
        detector.forward([image], prompt=["helmet", "forklift"])
        detector.forward([image], prompt="hard hat")

        self.assertEqual(
            wrapped_model.class_updates,
            [["helmet", "forklift"], ["hard hat"]],
        )
        self.assertEqual(first[0].prompt, ["helmet", "forklift"])

    def test_prompt_rejects_closed_set_checkpoint(self):
        detector = self._detector(_ClosedSetModel())

        with self.assertRaisesRegex(ValueError, "closed-set"):
            detector.forward([Image.new("RGB", (8, 8))], prompt=["helmet"])


if __name__ == "__main__":
    unittest.main()
