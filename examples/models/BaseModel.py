"""Demonstrate models.BaseModel."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(issubclass(models.Detector, models.BaseModel))