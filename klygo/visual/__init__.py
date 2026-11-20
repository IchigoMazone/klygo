"""Deprecated compatibility alias for :mod:`klygo.graphics`."""

import warnings

from klygo.graphics import *  # noqa: F401,F403
from klygo.graphics import __all__  # noqa: F401
from klygo.graphics.draw_bboxes import draw_box  # noqa: F401

warnings.warn(
    "klygo.visual is deprecated and will be removed in a future release; use klygo.graphics instead.",
    DeprecationWarning,
    stacklevel=2,
)
