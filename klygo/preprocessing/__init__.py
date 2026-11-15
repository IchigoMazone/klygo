"""Lazy, backend-aware image preprocessing for :class:`klygo.media.LazyImage`.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/preprocessing/Document_Preprocessing_Overview.ipynb

Public API (130 functions, 4 classes across 4 submodules):
    Core pipeline & composition (`core.py`):
        - Operation, Compose, Metric, Predicate
        - compose(...), apply(...), materialize(...), inspect(...), optimize(...)
        - custom(...), register(...), execute_pipeline(...), all_of(...), any_of(...), not_(...), between(...)

    Transforms (`transforms.py`):
        - Geometry, color, filtering, thresholding, morphology, edge detection, tensor prep,
          blending, and seeded random augmentations.

    Image quality metrics (`metrics.py`):
        - brightness_score, contrast_score, sharpness_score, blur_score, entropy_score,
          noise_score, exposure_score, motion_score, similarity_score, perceptual_hash,
          histogram, mean, std, min_max, percentile

    Coordinate trace restoration (`trace.py`):
        - get_trace(...), original_size(...), processed_size(...), restore_boxes(...), clear_trace(...)
"""

from . import core as _core
from . import metrics as _metrics
from . import trace as _trace
from . import transforms as _transforms
from .core import *
from .metrics import *
from .trace import *
from .transforms import *

__all__ = list(dict.fromkeys(
    _core.__all__ + _transforms.__all__ + _metrics.__all__ + _trace.__all__
))
