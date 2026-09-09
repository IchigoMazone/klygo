"""
Hugging Face Backend Logic (klygo.models.backend.hf).
Chứa toàn bộ logic xử lý đặc thù cho các mô hình Hugging Face Transformers.
"""

import os
from typing import Any, Optional
import torch

from klygo.models import utils


def cast_inputs(inputs: Any, dev: torch.device, dtype: torch.dtype) -> Any:
    """
    Cast an toàn các floating tensors trong inputs (Hugging Face BatchFeature / Dict)
    lên đúng device + dtype của model, bảo vệ nguyên vẹn các tensor số nguyên (input_ids).
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
    """
    Thực thi forward của mô hình Hugging Face với AMP autocast và đồng bộ GPU.
    """
    use_half = (cur_dtype == torch.float16)
    if cur_dtype == torch.bfloat16:
        eff_dtype = "bfloat16"
    elif use_half:
        eff_dtype = "float16"
    else:
        eff_dtype = "float32"

    with utils.amp_autocast_if_needed(use_half=use_half, dtype=eff_dtype):
        if hasattr(inputs, "items") or isinstance(inputs, dict):
            outputs = model(**inputs, **model_kwargs)
        elif isinstance(inputs, (list, tuple)):
            outputs = model(*inputs, **model_kwargs)
        else:
            outputs = model(inputs, **model_kwargs)

    utils.cuda_sync()
    return outputs


def get_output_device(outputs: Any, default_device: Optional[torch.device] = None) -> torch.device:
    """
    Dò tìm thiết bị thực tế của tensor đầu ra Hugging Face ModelOutput (logits, pred_boxes).
    """
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits.device
    if hasattr(outputs, "pred_boxes") and isinstance(outputs.pred_boxes, torch.Tensor):
        return outputs.pred_boxes.device
    if isinstance(outputs, torch.Tensor):
        return outputs.device
    return default_device or torch.device("cpu")


def save(model: Any, processor: Optional[Any], output_dir: str) -> None:
    """Lưu model và processor theo chuẩn Hugging Face save_pretrained()."""
    abs_out = os.path.abspath(output_dir)
    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(abs_out)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(abs_out)
