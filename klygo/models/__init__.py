"""Configure, load, and run multi-backend AI model wrappers.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/models/Document_Models_Overview.ipynb

Public API (10 functions, 2 base classes, 3 submodules):
    Core hierarchy & detection base (`base.py`, `detection/base.py`):
        1. base                Submodule containing BaseModel and model state helpers.
        2. BaseModel           Shared lifecycle, metadata, flags, and info contract.
        3. Detector            Backend-neutral object-detection base with predict/warmup/benchmark.

    Loading, export & backend selection (`load.py`):
        4. load(...)           Instantiate a ready model from a path, id, or configured spec.
        5. export(...)         Export a loaded model to a target format (e.g. ONNX/TorchScript).
        6. configure(...)      Bind custom implementation and runtime overrides to a model spec.
        7. set_backend(...)    Configure backend/engine preferences (e.g. Keras torch/jax/tf).
        8. get_backend(...)    Inspect the active backend configuration.

    Schema & metadata (`metadata.py`):
        9. metadata(...)       Build a structured model parameter schema.
       10. priority(...)       Define parameter override priority across groups.
       11. details(...)        Build descriptive model metadata (name, task, backend, library).
       12. flags(...)          Declare supported parameter groups for a model.

    Errors & utilities (`errors.py`, `utils.py`):
       13. errors              Submodule containing model loading and runtime exceptions.
       14. utils               Submodule containing warning and input resolution helpers.
       15. suppress_warnings() Silence framework warnings as a context manager or decorator.
"""

from . import utils

utils.suppress_ai_warnings()

from . import base
from . import errors
from .base import BaseModel
from .detection.base import Detector
from .load import load, export, configure, set_backend, get_backend
from .metadata import details, flags, metadata, priority
from .utils import suppress_warnings

__all__ = [
    "base", "BaseModel", "Detector",
    "load", "export", "configure", "set_backend", "get_backend",
    "metadata", "priority", "details", "flags",
    "errors", "utils", "suppress_warnings",
]
