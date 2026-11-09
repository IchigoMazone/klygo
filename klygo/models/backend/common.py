"""Compatibility import for ``klygo.models._compat``."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("klygo.models._compat")
