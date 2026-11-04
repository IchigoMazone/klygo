"""Demonstrate models.export."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models

output = models.export(
    "ultralytics/yolo11n.pt",
    "artifacts/models/yolo11n.json",
    overwrite=True,
    device="cpu",
    threshold=0.4,
)
print(output)
