"""Demonstrate models.suppress_warnings."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
import warnings
with models.suppress_warnings():
    warnings.warn("hidden", UserWarning)
print("warning scope exited")