"""Keras 3 and KerasHub backend adapter (`klygo.models.backend.keras`).

Supports PyTorch (`torch`), JAX (`jax`), and TensorFlow (`tensorflow`) computation engines
with lazy imports so environments without `keras` / `keras_hub` remain unaffected.
"""

import importlib
import os
from contextlib import nullcontext
from typing import Any, List, Dict, Optional

__all__ = [
    "ensure_backend", "get_active_backend", "cast_inputs", "run_inference",
    "format_results", "current_device", "current_dtype", "inference_context",
    "is_accelerator_available", "synchronize", "clear_cache",
]


def _torch_runtime():
    """Load PyTorch support only when Keras actually uses that engine."""
    try:
        return importlib.import_module(".torch", __package__)
    except ImportError as exc:
        raise ImportError(
            "The Keras Torch engine requires PyTorch. "
            "Install it using 'pip install \"klygo[torch]\"'."
        ) from exc


def ensure_backend() -> str:
    """Configure `KERAS_BACKEND` automatically if not already set (priority: torch -> jax -> tensorflow)."""
    if "KERAS_BACKEND" not in os.environ:
        try:
            import torch
            os.environ["KERAS_BACKEND"] = "torch"
        except ImportError:
            try:
                import jax
                os.environ["KERAS_BACKEND"] = "jax"
            except ImportError:
                try:
                    import tensorflow
                    os.environ["KERAS_BACKEND"] = "tensorflow"
                except ImportError:
                    os.environ["KERAS_BACKEND"] = "torch"
    return os.environ.get("KERAS_BACKEND", "torch")


def _require_keras():
    """Import Keras with an actionable message for an omitted model extra."""
    try:
        return importlib.import_module("keras")
    except ImportError as exc:
        raise ImportError(
            "KerasHub models require the 'keras-hub' extra. "
            "Install it using 'pip install \"klygo[keras-hub]\"'."
        ) from exc


def get_active_backend() -> str:
    """Return the active Keras computation backend (`torch`, `jax`, or `tensorflow`)."""
    ensure_backend()
    keras = _require_keras()
    return keras.backend.backend()


def current_device(model: Any) -> Any:
    """Detect the active device for a Keras 3 model across PyTorch, TensorFlow, or JAX."""
    active_backend = get_active_backend()
    if active_backend == "torch":
        try:
            import torch
            if model is not None and hasattr(model, "parameters"):
                try:
                    return next(model.parameters()).device
                except Exception:
                    pass
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"
    elif active_backend == "tensorflow":
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "/GPU:0"
            tpus = tf.config.list_physical_devices("TPU")
            return "/TPU:0" if tpus else "/CPU:0"
        except Exception:
            return "cpu"
    elif active_backend == "jax":
        try:
            import jax
            devs = jax.devices()
            return devs[0] if devs else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def current_dtype(model: Any) -> str:
    """Return the active dtype of a Keras 3 / KerasHub model."""
    if model is not None:
        if hasattr(model, "compute_dtype") and model.compute_dtype is not None:
            return str(getattr(model.compute_dtype, "name", model.compute_dtype))
        if hasattr(model, "dtype") and model.dtype is not None:
            return str(getattr(model.dtype, "name", model.dtype))
    return "float32"


def inference_context():
    """Return a safe context for the active Keras execution engine."""
    if get_active_backend() == "torch":
        return _torch_runtime().inference_context()
    return nullcontext()


def is_accelerator_available(device: Any = None) -> bool:
    """Check accelerator availability through the active Keras engine."""
    active_backend = get_active_backend()
    if active_backend == "torch":
        return _torch_runtime().is_accelerator_available(device)
    if active_backend == "tensorflow":
        try:
            import tensorflow as tf

            if device is not None and not any(
                marker in str(device).lower() for marker in ("gpu", "tpu")
            ):
                return False
            return bool(
                tf.config.list_physical_devices("GPU")
                or tf.config.list_physical_devices("TPU")
            )
        except Exception:
            return False
    if active_backend == "jax":
        try:
            import jax

            devices = [device] if device is not None else jax.devices()
            return any(
                str(getattr(item, "platform", item)).lower().split(":", 1)[0]
                in {"gpu", "tpu", "cuda", "rocm"}
                for item in devices
            )
        except Exception:
            return False
    return False


def _block_until_ready(value: Any) -> bool:
    """Block on a nested TensorFlow/JAX value when it exposes a sync method."""
    if isinstance(value, dict):
        blocked = False
        for item in value.values():
            blocked = _block_until_ready(item) or blocked
        return blocked
    if isinstance(value, (list, tuple)):
        blocked = False
        for item in value:
            blocked = _block_until_ready(item) or blocked
        return blocked
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
        return True
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        to_numpy()
        return True
    return False


def synchronize(device: Any = None, value: Any = None) -> None:
    """Synchronize work through Torch, TensorFlow, or JAX as configured."""
    active_backend = get_active_backend()
    if active_backend == "torch":
        _torch_runtime().synchronize(device=device, value=value)
        return
    if active_backend == "tensorflow":
        try:
            if value is not None:
                _block_until_ready(value)
        except Exception:
            pass
        return
    if active_backend == "jax":
        try:
            import jax

            if value is not None:
                block = getattr(jax, "block_until_ready", None)
                if callable(block):
                    block(value)
                    return
                if _block_until_ready(value):
                    return
            barrier = getattr(jax, "effects_barrier", None)
            if callable(barrier):
                barrier()
        except Exception:
            pass


def clear_cache() -> None:
    """Clear safe runtime caches for the active Keras engine.

    TensorFlow has no public equivalent to PyTorch's allocator cache operation,
    so active models are left untouched.
    """
    active_backend = get_active_backend()
    if active_backend == "torch":
        _torch_runtime().clear_cache()
    elif active_backend == "jax":
        try:
            import jax

            clear = getattr(jax, "clear_caches", None)
            if callable(clear):
                clear()
        except Exception:
            pass


def __getattr__(name):
    if name in {"cast_inputs", "run_inference", "format_results", "_to_numpy_array"}:
        return getattr(importlib.import_module("klygo.models.detection._keras"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
