"""Class identity contracts shared by model backend adapters."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from klygo.models.backend import keras, ultralytics
from klygo.models import detection
from klygo.models.detection.yolo_world import YOLOWorldDetect
from klygo.models.detection.yolo import YOLODetect


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Coordinates:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _UltralyticsBox:
    def __init__(self, class_id, score, coordinates):
        self.cls = [_Scalar(class_id)]
        self.conf = [_Scalar(score)]
        self.xyxy = [_Coordinates(coordinates)]


class _UltralyticsResult:
    names = {4: "cat"}
    boxes = [_UltralyticsBox(4, 0.75, [1, 2, 3, 4])]


class _YOLOWorld:
    def __init__(self, model_id):
        self.model_id = model_id
        self.names = {0: "person", 1: "car"}
        self.last_options = None

    def set_classes(self, classes):
        self.names = {index: name for index, name in enumerate(classes)}

    def __call__(self, images, **options):
        self.last_options = options
        return [SimpleNamespace(names=self.names, boxes=[]) for _ in images]


class _YOLO:
    def __init__(self, model_id):
        self.model_id = model_id
        self.names = {0: "person", 1: "car"}
        self.last_options = None

    def __call__(self, images, **options):
        self.last_options = options
        return [SimpleNamespace(names=self.names, boxes=[]) for _ in images]


class TestClassRegistryBackends(unittest.TestCase):
    def test_yolo_world_is_a_lazy_detection_export(self):
        self.assertIn("YOLOWorldDetect", detection.__all__)
        self.assertIs(detection.YOLOWorldDetect, YOLOWorldDetect)

    def test_ultralytics_keeps_native_class_ids(self):
        result = ultralytics.format_results([_UltralyticsResult()])[0]

        self.assertEqual(result["class_ids"], [4])
        self.assertEqual(result["labels"], ["cat"])
        self.assertEqual(result["boxes"], [[1, 2, 3, 4]])

    def test_keras_keeps_numeric_classes_separate_from_labels(self):
        fake_keras = SimpleNamespace(
            ops=SimpleNamespace(convert_to_numpy=lambda value: np.asarray(value))
        )
        with patch.dict("sys.modules", {"keras": fake_keras}):
            result = keras.format_results({
                "boxes": np.array([[[1, 2, 3, 4]]], dtype=np.float32),
                "scores": np.array([[0.75]], dtype=np.float32),
                "classes": np.array([[6]], dtype=np.int64),
            })[0]

        self.assertEqual(result["class_ids"], [6])
        self.assertEqual(result["labels"], ["6"])

    def test_keras_vectorized_format_filters_padding_and_converts_yxyx(self):
        fake_keras = SimpleNamespace(
            ops=SimpleNamespace(convert_to_numpy=lambda value: np.asarray(value))
        )
        with patch.dict("sys.modules", {"keras": fake_keras}):
            result = keras.format_results({
                "boxes": np.array([[[2, 1, 8, 7], [0, 0, 0, 0]]], dtype=np.float32),
                "confidence": np.array([[0.75, -1.0]], dtype=np.float32),
                "classes": np.array([[6, -1]], dtype=np.int64),
                "num_detections": np.array([2]),
                "_bounding_box_format": "yxyx",
            })[0]

        self.assertEqual(result["boxes"], [[1.0, 2.0, 7.0, 8.0]])
        self.assertEqual(result["scores"], [0.75])
        self.assertEqual(result["class_ids"], [6])
        self.assertEqual(result["labels"], ["6"])

    def test_yolo_world_keeps_only_init_and_forward_wrappers(self):
        fake_ultralytics = SimpleNamespace(YOLOWorld=_YOLOWorld)
        metadata = {
            "model_id": "fake-world.pt",
            "details": {
                "name": "Fake YOLO-World",
                "task": "Object-Detection",
                "backend": "Ultralytics",
                "library": "ultralytics",
            },
            "flags": {"model": False, "post": True},
            "priority": {},
            "model": {"imgsz": 320},
            "post": {
                "threshold": 0.25,
                "iou": 0.7,
                "agnostic_nms": True,
            },
        }
        with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}):
            model = YOLOWorldDetect(metadata)

        own_methods = {
            name
            for name, value in YOLOWorldDetect.__dict__.items()
            if callable(value)
        }
        self.assertTrue({"__init__", "forward"}.issubset(own_methods))

        model.forward([Image.new("RGB", (8, 8))], prompt=["cat", "dog"])
        self.assertEqual(model.original_names, {0: "person", 1: "car"})
        self.assertEqual(model.names, {0: "cat", 1: "dog"})
        self.assertEqual(
            model.model.last_options,
            {
                "conf": 0.25,
                "iou": 0.7,
                "verbose": False,
                "imgsz": 320,
                "agnostic_nms": True,
            },
        )

        model.model.set_classes(list(model.original_names.values()))
        model.forward([Image.new("RGB", (8, 8))])
        self.assertEqual(model.names, model.original_names)

    def test_fixed_yolo_uses_current_wrapper_and_native_class_ids(self):
        fake_ultralytics = SimpleNamespace(YOLO=_YOLO)
        metadata = {
            "model_id": "fake-yolo.pt",
            "details": {
                "name": "Fake YOLO",
                "task": "Object-Detection",
                "backend": "Ultralytics",
                "library": "ultralytics",
            },
            "flags": {"model": False, "post": True},
            "priority": {},
            "model": {"imgsz": 320},
            "post": {
                "threshold": 0.25,
                "iou": 0.7,
                "agnostic_nms": True,
            },
        }
        with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}):
            model = YOLODetect(metadata)

        own_methods = {
            name
            for name, value in YOLODetect.__dict__.items()
            if callable(value)
        }
        self.assertTrue({"__init__", "forward"}.issubset(own_methods))

        model.forward([Image.new("RGB", (8, 8))], prompt=["car"])
        self.assertEqual(model.names, {0: "person", 1: "car"})
        self.assertEqual(
            model.model.last_options,
            {
                "conf": 0.25,
                "iou": 0.7,
                "classes": [1],
                "verbose": False,
                "imgsz": 320,
                "agnostic_nms": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
