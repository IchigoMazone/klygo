"""Demonstrate models.export."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models

output = "artifacts/models/yolo11n.json"
models.export(
    "ultralytics/yolo11n.pt",
    output,
    overwrite=True,
    device="cpu",
    threshold=0.4,
)
print(output)
