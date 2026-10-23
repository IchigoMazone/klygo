"""Demonstrate models.priority."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
groups = models.flags("model", "post")
print(models.priority(groups, model=("device",), post=("threshold",)))