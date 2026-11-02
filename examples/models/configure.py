"""Demonstrate models.configure."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
definition = models.configure("demo", flags=("model", "post"), model_device="cpu", post_threshold=0.5)
print(definition.model.device, definition.post.threshold)