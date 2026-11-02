"""Demonstrate models.load."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
model = models.Detector()
print(models.load(model) is model)