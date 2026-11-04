"""Common Backend Logic & Unified Dispatchers (`klygo.models.backend.common`).

Provides shared device/dtype detection, GPU cache management, and lifecycle
dispatchers across all supported model backends (Hugging Face, Ultralytics,
Torchvision, and Keras).
"""

import importlib
from typing import Any, Optional, Dict, List

KERAS_BACKENDS = {"Keras", "KerasHub", "TensorFlow"}

__all__ = [
    "KERAS_BACKENDS",
    "backend_family",
    "current_device",
    "current_dtype",
    "is_accelerator_available",
    "synchronize",
    "clear_cache",
    "inference_context",
    "cast_inputs",
    "run_inference",
    "get_output_device",
    "sync_device",
    "format_results",
]


def _import_adapter(name: str):
    """Import one framework adapter without loading unrelated dependencies."""
    return importlib.import_module(f".{name}", __package__)


def backend_family(backend: Optional[str]) -> str:
    """Normalize decorated backend names such as ``'Hugging Face (Offline)'`` to ``'Hugging Face'``."""
    return (backend or "").split(" (", 1)[0]


def _runtime_adapter(backend: Optional[str]):
    """Return the framework adapter responsible for accelerator operations."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface")
    if family == "Ultralytics":
        return _import_adapter("ultralytics")
    if family == "Torchvision":
        return _import_adapter("torchvision")
    if family in KERAS_BACKENDS:
        return _import_adapter("keras")
    return None


def current_device(model: Any, backend: Optional[str] = None) -> Any:
    """Detect the active device hosting the wrapped model across supported backends."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        return adapter.current_device(model)
    return "cpu"


def current_dtype(model: Any, backend: Optional[str] = None) -> Any:
    """Detect the numerical precision (dtype) of the wrapped model across supported backends."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        return adapter.current_dtype(model)
    return "float32"


def is_accelerator_available(backend: str, device: Any = None) -> bool:
    """Return whether the selected backend's hardware accelerator (e.g., CUDA) is available."""
    adapter = _runtime_adapter(backend)
    if adapter is None:
        return False
    return adapter.is_accelerator_available(device)


def synchronize(
    backend: str,
    device: Any = None,
    value: Any = None,
) -> None:
    """Synchronize asynchronous accelerator queues through the selected backend adapter."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        adapter.synchronize(device=device, value=value)


def clear_cache(backend: str) -> None:
    """Release unused accelerator memory caches held by the selected backend framework."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        adapter.clear_cache()


def inference_context(backend: str):
    """Return the no-gradient inference context manager provided by the selected backend."""
    adapter = _runtime_adapter(backend)
    if adapter is None:
        from contextlib import nullcontext

        return nullcontext()
    return adapter.inference_context()


def cast_inputs(backend: str, inputs: Any, dev: Any, dtype: Any) -> Any:
    """Dispatch input tensor casting to ``(dev, dtype)`` via the selected backend adapter."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface").cast_inputs(inputs, dev=dev, dtype=dtype)
    if family == "Torchvision":
        return _import_adapter("torchvision").cast_inputs(inputs, dev=dev, dtype=dtype)
    if family in KERAS_BACKENDS:
        return _import_adapter("keras").cast_inputs(inputs, dev=dev, dtype=dtype)
    return inputs


def _optimize_model(backend: str, model: Any) -> Any:
    """Apply a backend's transparent runtime optimization once."""
    if backend_family(backend) == "Ultralytics":
        return _import_adapter("ultralytics").optimize_model(model)
    return model


def _native_inputs(source: Any) -> tuple[list[Any], bool]:
    """Normalize in-memory native inputs without invoking Klygo media decoding."""
    from os import PathLike

    if isinstance(source, (str, bytes, PathLike)) or getattr(source, "is_stream", False):
        raise TypeError(
            "This backend requires in-memory native inputs; use native=False for paths, videos, or MediaFrames"
        )
    if isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("native input must contain at least one item")
        return list(source), False
    return [source], True


def run_inference(backend: str, model: Any, inputs: Any, cur_dtype: Any, **model_kwargs) -> Any:
    """Dispatch model forward execution (including AMP/preprocessing where applicable) to the selected backend adapter."""
    family = backend_family(backend)
    if family == "Ultralytics":
        adapter = _import_adapter("ultralytics")
        reuse_predictor = model_kwargs.pop("_reuse_predictor", False)
        if reuse_predictor:
            outputs = adapter.predict_prepared(model, inputs, options=model_kwargs)
            if outputs is not None:
                return outputs
        return model(inputs, **model_kwargs)
    if family == "Hugging Face":
        return _import_adapter("huggingface").run_inference(
            model, inputs, cur_dtype=cur_dtype, **model_kwargs
        )
    if family == "Torchvision":
        return _import_adapter("torchvision").run_inference(model, inputs, **model_kwargs)
    if family in KERAS_BACKENDS:
        return _import_adapter("keras").run_inference(model, inputs, **model_kwargs)
    if model is not None and callable(model):
        return model(inputs, **model_kwargs)
    return inputs


def get_output_device(backend: str, outputs: Any, default_device: Any) -> Any:
    """Detect the device of raw output tensors produced by the selected backend."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface").get_output_device(
            outputs, default_device=default_device
        )
    if family in KERAS_BACKENDS:
        return default_device
    return default_device


def sync_device(backend: str, tensor: Any, target_device: Any) -> Any:
    """Move ``tensor`` to ``target_device`` when required by multi-GPU backend execution."""
    if backend_family(backend) == "Hugging Face":
        return _import_adapter("huggingface").sync_device(tensor, target_device)
    return tensor


def format_results(backend: str, raw_outputs: Any) -> List[Dict[str, Any]]:
    """Normalize backend-specific raw predictions into ``[{'boxes', 'scores', 'labels', 'class_ids'}]`` dicts."""
    family = backend_family(backend)
    if family == "Ultralytics":
        return _import_adapter("ultralytics").format_results(raw_outputs)
    if family == "Torchvision":
        return _import_adapter("torchvision").format_results(raw_outputs)
    if family in KERAS_BACKENDS:
        return _import_adapter("keras").format_results(raw_outputs)
    return raw_outputs
