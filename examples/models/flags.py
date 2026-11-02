"""Demonstrate models.flags."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(models.flags(model=False, post=True))