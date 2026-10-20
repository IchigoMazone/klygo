"""PyTorch-specific accelerator operations shared by Torch model adapters."""

from contextlib import nullcontext
from typing import Any, Optional

import torch


def _device_type(device: Any) -> Optional[str]:
    """Return a normalized PyTorch device type without constructing a device."""
    if device is None:
        return None
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type).lower()
    return str(device).split(":", 1)[0].lower()


def current_device(model: Any) -> torch.device:
    """Return the device owning a PyTorch model's parameters or state."""
    if model is not None:
        if hasattr(model, "parameters"):
            try:
                return next(model.parameters()).device
            except (StopIteration, Exception):
                pass
        if hasattr(model, "device"):
            try:
                return torch.device(model.device)
            except Exception:
                pass
    return torch.device("cpu")


def current_dtype(model: Any) -> torch.dtype:
    """Return the dtype of a PyTorch model's parameters or state."""
    if model is not None:
        if hasattr(model, "parameters"):
            try:
                return next(model.parameters()).dtype
            except (StopIteration, Exception):
                pass
        if hasattr(model, "dtype"):
            return model.dtype
    return torch.float32


def is_accelerator_available(device: Any = None) -> bool:
    """Return whether CUDA is available and ``device`` is CUDA-compatible."""
    try:
        if not torch.cuda.is_available():
            return False
        device_type = _device_type(device)
        return device_type in (None, "cuda")
    except Exception:
        return False


def synchronize(device: Any = None, value: Any = None) -> None:
    """Synchronize queued CUDA work for the selected device when available."""
    del value  # PyTorch synchronization is device-wide, not value-based.
    if not is_accelerator_available(device):
        return
    try:
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device=device)
    except Exception:
        pass


def clear_cache() -> None:
    """Release unused blocks held by PyTorch's CUDA caching allocator."""
    if not is_accelerator_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def autocast(
    use_half: bool = False,
    dtype: Optional[str] = None,
    device_type: Optional[str] = None,
):
    """Return the appropriate PyTorch AMP context for an inference call."""
    selected_device = device_type or (
        "cuda" if is_accelerator_available() else "cpu"
    )
    normalized_dtype = (dtype or "").lower()

    if normalized_dtype in ("bfloat16", "bf16"):
        return torch.amp.autocast(
            device_type=selected_device,
            dtype=torch.bfloat16,
        )
    if (
        use_half or normalized_dtype in ("float16", "fp16", "half")
    ) and selected_device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def inference_context():
    """Return PyTorch's no-gradient inference context."""
    return torch.inference_mode()


__all__ = [
    "current_device",
    "current_dtype",
    "is_accelerator_available",
    "synchronize",
    "clear_cache",
    "autocast",
    "inference_context",
]
