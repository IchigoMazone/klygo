"""Demonstrate models.details."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
info = models.details(name="Demo", task="Detection", backend="Custom", library="local", version="1")
print(info.name, info.version)