"""Behavioral tests for the public detection result hierarchy."""

import unittest

from PIL import Image

from klygo import outputs


class TestDetectOutputs(unittest.TestCase):
    def test_box_crop_and_immutable_transform(self):
        source = Image.new("RGB", (40, 30), "white")
        box = outputs.detect.Box(0, "cat", 0.9, [5, 6, 20, 16], source)
        self.assertEqual(box.image.size, (15, 10))
        moved = box.translate(dx=3)
        self.assertEqual(box.box, [5.0, 6.0, 20.0, 16.0])
        self.assertEqual(moved.box, [8.0, 6.0, 23.0, 16.0])
        with self.assertRaises(ValueError):
            box.with_box([1, 2, 3])

    def test_detection_hierarchy(self):
        source = Image.new("RGB", (40, 30), "white")
        box = outputs.detect.Box(0, "cat", 0.9, [5, 6, 20, 16])
        frame = outputs.detect.Detection(source, [box])
        results = outputs.detect.Detections([frame], source_type="image")
        self.assertIs(results[0], frame)
        self.assertIs(frame[0], box)
        self.assertEqual(results.total_frames, 1)

    def test_aliases_resolve_to_canonical_classes(self):
        self.assertIs(outputs.detect.Boxes, outputs.detect.Crops)
        self.assertIs(outputs.detect.DetectionResult, outputs.detect.Detection)
