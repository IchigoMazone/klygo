"""Post-process normalized Klygo results without coupling to model backends.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/postprocessing/Document_Postprocessing_Overview.ipynb

Public API (9 functions, 2 classes across 2 submodules):
    Pipeline composition (`core.py`):
        1. Operation           Lazy postprocessing step wrapper.
        2. Pipeline            Composable sequence of postprocessing operations.
        3. compose(...)        Chain operations into a reusable Pipeline.
        4. custom(...)         Wrap a user function into an Operation.

    Detection box operations (`detect.py`):
        5. map_boxes(...)      Transform each Box with a custom mapper.
        6. box_iou(...)        Compute pairwise IoU matrices between box sets.
        7. threshold(...)      Filter boxes below a minimum confidence score.
        8. clip(...)           Clip box coordinates to frame boundaries.
        9. remove_invalid(...) Drop degenerate or zero-area boxes.
       10. nms(...)            Suppress overlapping boxes via Non-Maximum Suppression.
       11. top(...)            Retain the highest-scoring boxes per frame.
"""

from . import core as _core
from . import detect as _detect
from .core import *
from .detect import *

__all__ = list(dict.fromkeys(_core.__all__ + _detect.__all__))
