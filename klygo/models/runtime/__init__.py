"""Lazy accelerator lifecycle helpers; no model inference or result dispatch."""

import importlib
from contextlib import nullcontext
from klygo.models.utils import backend_family

KERAS_BACKENDS = {"Keras", "KerasHub", "TensorFlow"}
_TORCH_EXTRAS = {
    "Hugging Face": "transformers", "Ultralytics": "ultralytics",
    "Torchvision": "torchvision",
}


def _engine(backend):
    family = backend_family(backend)
    if family in KERAS_BACKENDS:
        return importlib.import_module(".keras", __name__)
    if family in _TORCH_EXTRAS:
        try:
            return importlib.import_module(".torch", __name__)
        except ImportError as error:
            extra = _TORCH_EXTRAS[family]
            raise ImportError(f'Install the model extra with pip install "klygo[{extra}]"') from error
    return None


def current_device(model, backend=None):
    """Return the native device of the selected runtime."""
    engine = _engine(backend)
    return engine.current_device(model) if engine else "cpu"


def current_dtype(model, backend=None):
    """Return the parameter precision of the selected runtime."""
    engine = _engine(backend)
    return engine.current_dtype(model) if engine else "float32"


def inference_context(backend):
    """Return an inference context without loading unrelated engines."""
    engine = _engine(backend)
    return engine.inference_context() if engine else nullcontext()


def is_accelerator_available(backend, device=None):
    """Check accelerator availability through the selected engine."""
    engine = _engine(backend)
    return engine.is_accelerator_available(device) if engine else False


def synchronize(backend, device=None, value=None):
    """Wait for the selected accelerator before measuring elapsed time."""
    engine = _engine(backend)
    if engine:
        engine.synchronize(device=device, value=value)


def clear_cache(backend):
    """Release unused runtime caches without unloading active models."""
    engine = _engine(backend)
    if engine:
        engine.clear_cache()
