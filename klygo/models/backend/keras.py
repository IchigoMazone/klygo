"""Compatibility import for ``klygo.models.runtime.keras``."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("klygo.models.runtime.keras")
