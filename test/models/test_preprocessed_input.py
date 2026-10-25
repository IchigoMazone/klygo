"""Regression coverage for preprocessed images passed to detection models."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from klygo import media, preprocessing
from klygo.models.detection.base import Detector


class StrictImageDetector(Detector):
    def forward(self, images, prompt=None):
        for image in images:
            if type(image) is not Image.Image or image.mode != "RGB":
                raise TypeError("The model backend requires a materialized RGB PIL image")
            if image.size != (40, 40):
                raise ValueError("The model received an image before preprocessing")
        return [
            {"boxes": [[2, 2, 8, 8]], "scores": [0.9], "labels": ["object"]}
            for _ in images
        ]


class TestPreprocessedInput(unittest.TestCase):
    def test_predict_materializes_pipeline_and_preserves_box_restore(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.png"
            Image.new("RGB", (100, 80), "red").save(path)
            pipeline = preprocessing.compose(
                preprocessing.crop((10, 10, 50, 50)),
                preprocessing.grayscale(),
            )
            processed = pipeline(media.load(path)[0])
            model = StrictImageDetector(metadata={"backend": "Custom"})

            sources = (
                ("single", processed, False),
                ("list", [processed], False),
                ("frames", media.load(path).transform(pipeline), False),
                ("stream", media.load(path, stream=True).transform(pipeline), True),
            )
            for name, source, stream in sources:
                with self.subTest(name=name):
                    results = model.predict(source, stream=stream, verbose=False)
                    self.assertEqual(
                        results[0].boxes,
                        [(12.0, 12.0, 18.0, 18.0)],
                    )
                    self.assertIsInstance(results[0]._resolve_source(), media.LazyImage)
