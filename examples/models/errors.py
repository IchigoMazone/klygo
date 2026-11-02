"""Demonstrate models.errors."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
try:
    raise models.errors.InvalidConfigError("bad setting")
except models.errors.ModelFrameworkError as error:
    print(type(error).__name__)