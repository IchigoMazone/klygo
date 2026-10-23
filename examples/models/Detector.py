"""Demonstrate models.Detector."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from PIL import Image

from klygo import models, outputs


class DemoDetector(models.Detector):
    """Small deterministic example; no external weights are required."""

    def __init__(self):
        super().__init__(
            backend="Custom",
            metadata={
                "model_id": "demo",
                "flags": {"post": True},
                "priority": {"post": ["threshold"]},
                "post": {"threshold": 0.25},
            },
        )

    def forward(self, images, prompt=None):
        labels = prompt or ["object"]
        self.names = dict(enumerate(labels))
        predictions = [
            {
                "boxes": [[1, 2, 10, 12]],
                "scores": [0.9],
                "labels": [labels[0]],
                "class_ids": [0],
            }
            for _ in images
        ]
        return outputs.build_detections(images, predictions, names=self.names)


detector = DemoDetector()
image = Image.new("RGB", (24, 24), "white")
first = detector(image, prompt=["person"], threshold=0.5, verbose=False)
second = detector(image, verbose=False)
print("Remembered configuration:", detector.predict_config)
print("Original result context:", first[0].config)
print("Second result context:", second[0].config)
print("Device / dtype:", detector.device, detector.dtype)
report = detector.benchmark(
    image,
    threshold=0.8,
    iterations=2,
    warmup=1,
    verbose=False,
)
print("Benchmark configuration:", report["predict_config"])
print("Configuration after benchmark:", detector.predict_config)
detector.reset()
print("Default configuration:", detector.predict_config)
