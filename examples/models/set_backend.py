"""Demonstrate models.set_backend."""

import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
import os
from unittest.mock import patch
with patch.dict(os.environ, {}, clear=False):
    models.set_backend("custom")
    print(models.get_backend())