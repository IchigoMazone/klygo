"""Compatibility import for ``klygo.models.detection._torchvision``."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("klygo.models.detection._torchvision")
