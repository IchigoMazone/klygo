"""Demonstrate models.metadata."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
definition = models.metadata(models.flags("model"), details=models.details(name="Demo", task="Detection", backend="Custom", library="local"))
print(definition.flags, definition.details.name)