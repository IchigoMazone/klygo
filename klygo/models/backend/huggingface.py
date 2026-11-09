"""Compatibility import for ``klygo.models.detection._huggingface``."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("klygo.models.detection._huggingface")
