"""
Common Backend Logic & Unified Dispatchers (klygo.models.backend.common).
Chứa toàn bộ logic dùng chung cho việc dò tìm device/dtype, dọn dẹp cache GPU,
và bộ điều phối (dispatchers) vòng đời cho các backend framework khác nhau.
"""

import importlib
import sys
from typing import Any, Optional, Dict, List

KERAS_BACKENDS = {"Keras", "KerasHub", "TensorFlow"}


def _import_adapter(name: str):
    """Import one framework adapter without loading unrelated dependencies."""
    return importlib.import_module(f".{name}", __package__)


def backend_family(backend: Optional[str]) -> str:
    """Normalize decorated names such as ``Hugging Face (Offline)``."""
    return (backend or "").split(" (", 1)[0]


def _runtime_adapter(backend: Optional[str]):
    """Return the framework adapter responsible for accelerator operations."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface")
    if family == "Ultralytics":
        return _import_adapter("ultralytics")
    if family in KERAS_BACKENDS:
        return _import_adapter("keras")
    return None


def current_device(model: Any, backend: Optional[str] = None) -> Any:
    """Dò tìm device thực tế của model."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        return adapter.current_device(model)
    return "cpu"


def current_dtype(model: Any, backend: Optional[str] = None) -> Any:
    """Dò tìm dtype thực tế của model."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        return adapter.current_dtype(model)
    return "float32"


def is_accelerator_available(backend: str, device: Any = None) -> bool:
    """Ask the selected framework whether its accelerator is available."""
    adapter = _runtime_adapter(backend)
    if adapter is None:
        return False
    return adapter.is_accelerator_available(device)


def synchronize(
    backend: str,
    device: Any = None,
    value: Any = None,
) -> None:
    """Synchronize work through the selected framework adapter."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        adapter.synchronize(device=device, value=value)


def clear_cache(backend: str) -> None:
    """Clear only caches that the selected framework can safely release."""
    adapter = _runtime_adapter(backend)
    if adapter is not None:
        adapter.clear_cache()


def inference_context(backend: str):
    """Return the inference context provided by the selected framework."""
    adapter = _runtime_adapter(backend)
    if adapter is None:
        from contextlib import nullcontext

        return nullcontext()
    return adapter.inference_context()


def cast_inputs(backend: str, inputs: Any, dev: Any, dtype: Any) -> Any:
    """Điều phối ép kiểu inputs theo backend."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface").cast_inputs(inputs, dev=dev, dtype=dtype)
    if family in KERAS_BACKENDS:
        return _import_adapter("keras").cast_inputs(inputs, dev=dev, dtype=dtype)
    return inputs


def run_inference(backend: str, model: Any, inputs: Any, cur_dtype: Any, **model_kwargs) -> Any:
    """Điều phối suy luận với autocast theo backend."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface").run_inference(
            model, inputs, cur_dtype=cur_dtype, **model_kwargs
        )
    if family in KERAS_BACKENDS:
        return _import_adapter("keras").run_inference(model, inputs, **model_kwargs)
    if model is not None and callable(model):
        return model(inputs, **model_kwargs)
    return inputs


def get_output_device(backend: str, outputs: Any, default_device: Any) -> Any:
    """Điều phối dò tìm device đầu ra theo backend."""
    family = backend_family(backend)
    if family == "Hugging Face":
        return _import_adapter("huggingface").get_output_device(
            outputs, default_device=default_device
        )
    if family in KERAS_BACKENDS:
        return default_device
    return default_device


def sync_device(backend: str, tensor: Any, target_device: Any) -> Any:
    """Điều phối đồng bộ device của tensor."""
    if backend_family(backend) == "Hugging Face":
        return _import_adapter("huggingface").sync_device(tensor, target_device)
    return tensor


def format_results(backend: str, raw_outputs: Any) -> List[Dict[str, Any]]:
    """Điều phối chuẩn hóa kết quả thô theo backend."""
    family = backend_family(backend)
    if family == "Ultralytics":
        return _import_adapter("ultralytics").format_results(raw_outputs)
    if family in KERAS_BACKENDS:
        return _import_adapter("keras").format_results(raw_outputs)
    return raw_outputs


def reset(backend: str, model: Any, processor: Optional[Any] = None) -> None:
    """Điều phối reset mô hình về CPU theo backend."""
    family = backend_family(backend)
    if family == "Hugging Face":
        _import_adapter("huggingface").reset(model, processor=processor)
    elif family == "Ultralytics":
        _import_adapter("ultralytics").reset(model)
    elif family in KERAS_BACKENDS:
        _import_adapter("keras").reset(model)
    elif model is not None and hasattr(model, "cpu"):
        model.cpu()
    clear_cache(backend)


def unload(backend: str, model: Any, processor: Optional[Any] = None) -> None:
    """Điều phối giải phóng tài nguyên mô hình theo backend."""
    family = backend_family(backend)
    if family == "Hugging Face":
        _import_adapter("huggingface").unload(model, processor=processor)
    elif family == "Ultralytics":
        _import_adapter("ultralytics").unload(model)
    elif family in KERAS_BACKENDS:
        _import_adapter("keras").unload(model)
    elif model is not None and hasattr(model, "cpu"):
        model.cpu()
    clear_cache(backend)


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

    abs_out = files.resolve(output_dir)
    files.mkdir(abs_out)

    # 1. Ghi klygo.json
    meta = dict(metadata)
    meta.pop("num_params", None)
    files.save(files.join(abs_out, "klygo.json"), meta, verbose=False)

    # 2. Xử lý Custom Class (copy model.py nếu không phải built-in)
    if not class_module.startswith("klygo.models."):
        mod = sys.modules.get(class_module)
        if mod and hasattr(mod, "__file__") and mod.__file__:
            source_file = mod.__file__
            if files.exists(source_file):
                files.copy(source_file, files.join(abs_out, "model.py"), overwrite=True)

    # 3. Trọng số & Artifacts ủy thác theo backend
    family = backend_family(backend)
    if family == "Hugging Face" or processor is not None:
        _import_adapter("huggingface").save(model, processor, abs_out)
    elif family == "Ultralytics":
        _import_adapter("ultralytics").save(model, abs_out)
    elif family in KERAS_BACKENDS:
        _import_adapter("keras").save(model, abs_out)
    elif model is not None:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(abs_out)
        elif hasattr(model, "save"):
            model.save(abs_out)
