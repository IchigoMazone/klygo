"""Demonstrate models.utils."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(models.utils.normalize_flags(("model", "post")))