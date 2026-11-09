"""Compatibility import for ``klygo.models.detection._ultralytics``."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("klygo.models.detection._ultralytics")
