"""Native huggingface detection helpers shared by compatible wrappers."""

from typing import Any, Optional

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "Hugging Face models require the 'transformers' extra. "
        "Install it using 'pip install \"klygo[transformers]\"'."
    ) from exc

from klygo.models.runtime import torch as torch_runtime


def cast_inputs(inputs: Any, dev: torch.device, dtype: torch.dtype) -> Any:
    """Cast floating-point tensors in Hugging Face ``BatchFeature`` or dict inputs to ``(dev, dtype)``.

    Preserves integer token tensors (such as ``input_ids`` and ``attention_mask``)
    while moving them to ``dev``.
    """
    is_cpu = (dev.type == "cpu")
    is_cuda = (dev.type == "cuda")

    if hasattr(inputs, "keys"):
        for k in list(inputs.keys()):
            v = inputs[k]
            if not isinstance(v, torch.Tensor):
                continue
            if v.is_floating_point():
                if is_cpu:
                    inputs[k] = v.to(device=dev, dtype=torch.float32)
                elif dtype in (torch.float16, torch.bfloat16):
                    inputs[k] = v.to(device=dev, dtype=dtype, non_blocking=is_cuda)
                else:
                    inputs[k] = v.to(device=dev, dtype=torch.float32, non_blocking=is_cuda)
            else:
                inputs[k] = v.to(device=dev, non_blocking=is_cuda)
    elif isinstance(inputs, (list, tuple)):
        casted = []
        for v in inputs:
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    if is_cpu:
                        casted.append(v.to(device=dev, dtype=torch.float32))
                    elif dtype in (torch.float16, torch.bfloat16):
                        casted.append(v.to(device=dev, dtype=dtype, non_blocking=is_cuda))
                    else:
                        casted.append(v.to(device=dev, dtype=torch.float32, non_blocking=is_cuda))
                else:
                    casted.append(v.to(device=dev, non_blocking=is_cuda))
            else:
                casted.append(v)
        inputs = type(inputs)(casted)
    elif isinstance(inputs, torch.Tensor):
        if inputs.is_floating_point():
            if is_cpu:
                inputs = inputs.to(device=dev, dtype=torch.float32)
            elif dtype in (torch.float16, torch.bfloat16):
                inputs = inputs.to(device=dev, dtype=dtype, non_blocking=is_cuda)
            else:
                inputs = inputs.to(device=dev, dtype=torch.float32, non_blocking=is_cuda)
        else:
            inputs = inputs.to(device=dev, non_blocking=is_cuda)

    return inputs


def run_inference(model: Any, inputs: Any, cur_dtype: torch.dtype, **model_kwargs) -> Any:
    """Execute a Hugging Face model forward pass inside PyTorch AMP ``autocast``."""
    use_half = (cur_dtype == torch.float16)
    if cur_dtype == torch.bfloat16:
        eff_dtype = "bfloat16"
    elif use_half:
        eff_dtype = "float16"
    else:
        eff_dtype = "float32"

    with torch_runtime.autocast(use_half=use_half, dtype=eff_dtype):
        if hasattr(inputs, "items") or isinstance(inputs, dict):
            outputs = model(**inputs, **model_kwargs)
        elif isinstance(inputs, (list, tuple)):
            outputs = model(*inputs, **model_kwargs)
        else:
            outputs = model(inputs, **model_kwargs)

    return outputs


def get_output_device(outputs: Any, default_device: Optional[torch.device] = None) -> torch.device:
    """Detect the actual device of Hugging Face ``ModelOutput`` tensors (``logits``, ``pred_boxes``)."""
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits.device
    if hasattr(outputs, "pred_boxes") and isinstance(outputs.pred_boxes, torch.Tensor):
        return outputs.pred_boxes.device
    if isinstance(outputs, torch.Tensor):
        return outputs.device
    return default_device or torch.device("cpu")


def sync_device(tensor: Any, target_device: Any) -> Any:
    """Move ``tensor`` (such as ``input_ids``) to ``target_device`` for multi-GPU setups."""
    if isinstance(tensor, torch.Tensor) and hasattr(target_device, "type"):
        if tensor.device != target_device:
            return tensor.to(target_device, non_blocking=(target_device.type == "cuda"))
    return tensor

# Historical adapter exports share the same runtime implementations.
from klygo.models.runtime.torch import (
    current_device, current_dtype, inference_context, is_accelerator_available,
    synchronize, clear_cache,
)
