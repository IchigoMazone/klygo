"""
Common Backend Logic & Unified Dispatchers (klygo.models.backend.common).
Chứa toàn bộ logic dùng chung cho việc dò tìm device/dtype, dọn dẹp cache GPU,
và bộ điều phối (dispatchers) vòng đời cho các backend framework khác nhau.
"""

import os
import shutil
import sys
from typing import Any, Optional, Dict, List, Tuple
import torch

from . import huggingface
from . import ultralytics
from . import tensorflow


def current_device(model: Any, backend: Optional[str] = None) -> Any:
    """Dò tìm device thực tế của model."""
    if backend == "TensorFlow":
        return tensorflow.current_device(model)

    if model is not None:
        try:
            if hasattr(model, "parameters"):
                return next(model.parameters()).device
        except (StopIteration, Exception):
            pass
        if hasattr(model, "device"):
            try:
                return torch.device(model.device)
            except Exception:
                pass
    return torch.device("cpu")


def current_dtype(model: Any, backend: Optional[str] = None) -> Any:
    """Dò tìm dtype thực tế của model."""
    if backend == "TensorFlow":
        return tensorflow.current_dtype(model)

    if model is not None:
        try:
            if hasattr(model, "parameters"):
                return next(model.parameters()).dtype
        except (StopIteration, Exception):
            pass
        if hasattr(model, "dtype"):
            return model.dtype
    return torch.float32


def clear_cache() -> None:
    """Xóa bộ nhớ cache GPU an toàn nếu CUDA khả dụng."""
    try:
        from klygo import cuda
        if cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def cast_inputs(backend: str, inputs: Any, dev: Any, dtype: Any) -> Any:
    """Điều phối ép kiểu inputs theo backend."""
    if backend == "Hugging Face":
        return huggingface.cast_inputs(inputs, dev=dev, dtype=dtype)
    if backend == "TensorFlow":
        return tensorflow.cast_inputs(inputs, dev=dev, dtype=dtype)
    return inputs


def run_inference(backend: str, model: Any, inputs: Any, cur_dtype: Any, **model_kwargs) -> Any:
    """Điều phối suy luận với autocast theo backend."""
    if backend == "Hugging Face":
        return huggingface.run_inference(model, inputs, cur_dtype=cur_dtype, **model_kwargs)
    if backend == "TensorFlow":
        return tensorflow.run_inference(model, inputs, **model_kwargs)
    if model is not None and callable(model):
        return model(inputs, **model_kwargs)
    return inputs


def get_output_device(backend: str, outputs: Any, default_device: Any) -> Any:
    """Điều phối dò tìm device đầu ra theo backend."""
    if backend == "Hugging Face":
        return huggingface.get_output_device(outputs, default_device=default_device)
    if backend == "TensorFlow":
        return default_device
    return default_device


def format_results(backend: str, raw_outputs: Any) -> List[Dict[str, Any]]:
    """Điều phối chuẩn hóa kết quả thô theo backend."""
    if backend == "Ultralytics":
        return ultralytics.format_results(raw_outputs)
    if backend == "TensorFlow":
        return tensorflow.format_results(raw_outputs)
    return raw_outputs


def reset(backend: str, model: Any, processor: Optional[Any] = None) -> None:
    """Điều phối reset mô hình về CPU theo backend."""
    if backend == "Hugging Face":
        huggingface.reset(model, processor=processor)
    elif backend == "Ultralytics":
        ultralytics.reset(model)
    elif backend == "TensorFlow":
        tensorflow.reset(model)
    elif model is not None and hasattr(model, "cpu"):
        model.cpu()


def unload(backend: str, model: Any, processor: Optional[Any] = None) -> None:
    """Điều phối giải phóng tài nguyên mô hình theo backend."""
    clear_cache()
    if backend == "Hugging Face":
        huggingface.unload(model, processor=processor)
    elif backend == "Ultralytics":
        ultralytics.unload(model)
    elif backend == "TensorFlow":
        tensorflow.unload(model)
    elif model is not None and hasattr(model, "cpu"):
        model.cpu()


def save(
    backend: str,
    model: Any,
    output_dir: str,
    metadata: Dict[str, Any],
    class_module: str,
    processor: Optional[Any] = None,
) -> None:
    """
    Lưu toàn bộ mô hình (Metadata klygo.json + Custom class model.py + Trọng số backend).
    Được dùng chung cho mọi task trong Klygo.
    """
    from klygo import files

    abs_out = os.path.abspath(output_dir)
    files.mkdir(abs_out)

    # 1. Ghi klygo.json
    meta = dict(metadata)
    meta.pop("num_params", None)
    files.save(os.path.join(abs_out, "klygo.json"), meta, verbose=False)

    # 2. Xử lý Custom Class (copy model.py nếu không phải built-in)
    if not class_module.startswith("klygo.models."):
        mod = sys.modules.get(class_module)
        if mod and hasattr(mod, "__file__") and mod.__file__:
            source_file = mod.__file__
            if os.path.exists(source_file):
                shutil.copy2(source_file, os.path.join(abs_out, "model.py"))

    # 3. Trọng số & Artifacts ủy thác theo backend
    if backend == "Hugging Face" or processor is not None:
        huggingface.save(model, processor, abs_out)
    elif backend == "Ultralytics":
        ultralytics.save(model, abs_out)
    elif backend == "TensorFlow":
        tensorflow.save(model, abs_out)
    elif model is not None:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(abs_out)
        elif hasattr(model, "save"):
            model.save(abs_out)
