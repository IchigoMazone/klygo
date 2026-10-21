"""Demonstrate models.Detector."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
detector = models.Detector()
print(detector.task, isinstance(detector, models.BaseModel))