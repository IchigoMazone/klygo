"""Registered Torchvision Faster R-CNN behavior without downloading weights."""

import unittest
from unittest.mock import patch

from PIL import Image
import torch

from klygo import models, outputs
from klygo.models import detection


class FakeFasterRCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 12.0, 14.0], [2.0, 3.0, 8.0, 9.0]]),
                "scores": torch.tensor([0.9, 0.2]),
                "labels": torch.tensor([1, 2]),
            }
            for _image in images
        ]


class TestFasterRCNN(unittest.TestCase):
    def test_lazy_detection_export(self):
        self.assertIn("FasterRCNNDetect", detection.__all__)
        self.assertIs(detection.FasterRCNNDetect, detection.FasterRCNNDetect)

    def test_registered_model_runs_through_torchvision_backend_on_cpu(self):
        native = FakeFasterRCNN()
        with patch(
            "klygo.models.detection.faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn",
            return_value=native,
        ) as constructor:
            detector = models.load(
                "torchvision/fasterrcnn_mobilenet_v3_large_fpn",
                device="cpu",
                model_weights=None,
                threshold=0.5,
            )
            result = detector(
                Image.new("RGB", (20, 20)),
                prompt=["person"],
                verbose=False,
            )

        constructor.assert_called_once_with(weights=None, weights_backbone=None)
        self.assertIsInstance(result, outputs.Detections)
        self.assertEqual(result[0].labels, ["person"])
        self.assertEqual(result[0].class_ids, [1])
        self.assertEqual(str(detector.device), "cpu")

    def test_short_and_provider_names_resolve_to_the_same_model(self):
        from klygo.models.load import _resolve

        short = _resolve("fasterrcnn_mobilenet_v3_large_fpn")
        qualified = _resolve("torchvision/fasterrcnn_mobilenet_v3_large_fpn")
        self.assertEqual(short["model_id"], "fasterrcnn_mobilenet_v3_large_fpn")
        self.assertEqual(qualified, short)


if __name__ == "__main__":
    unittest.main()
