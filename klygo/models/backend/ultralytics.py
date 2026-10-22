"""Ultralytics Backend Logic (`klygo.models.backend.ultralytics`).

Provides runtime adaptation, result formatting, and lifecycle helpers for
Ultralytics YOLO and YOLO-World detection models.
"""

from typing import Any, List, Dict

try:
    from . import torch_runtime
except ImportError as exc:
    raise ImportError(
        "Ultralytics models require the 'ultralytics' extra. "
        "Install it using 'pip install \"klygo[ultralytics]\"'."
    ) from exc


def current_device(model: Any):
    """Return the device owning an Ultralytics PyTorch model."""
    return torch_runtime.current_device(model)


def current_dtype(model: Any):
    """Return the dtype used by an Ultralytics PyTorch model."""
    return torch_runtime.current_dtype(model)


def inference_context():
    """Return the PyTorch inference context used by Ultralytics models."""
    return torch_runtime.inference_context()


def is_accelerator_available(device: Any = None) -> bool:
    """Return whether Ultralytics can use its PyTorch CUDA device."""
    return torch_runtime.is_accelerator_available(device)


def synchronize(device: Any = None, value: Any = None) -> None:
    """Synchronize Ultralytics inference through the PyTorch runtime."""
    torch_runtime.synchronize(device=device, value=value)


def clear_cache() -> None:
    """Release unused PyTorch CUDA cache blocks."""
    torch_runtime.clear_cache()


def format_results(ultra_results: Any) -> List[Dict[str, Any]]:
    """Extract normalized ``boxes``, ``scores``, ``labels``, and ``class_ids`` from Ultralytics Results."""
    raw_list = []
    if ultra_results is not None:
        for res in ultra_results:
            boxes_data, scores_data, labels_data, class_ids = [], [], [], []
            if getattr(res, "boxes", None) is not None:
                for b in res.boxes:
                    boxes_data.append(b.xyxy[0].tolist())
                    scores_data.append(float(b.conf[0].item()))
                    cls_id = int(b.cls[0].item())
                    class_ids.append(cls_id)
                    labels_data.append(res.names.get(cls_id, str(cls_id)))
            raw_list.append(
                {
                    "boxes": boxes_data,
                    "scores": scores_data,
                    "labels": labels_data,
                    "class_ids": class_ids,
                }
            )
    return raw_list


def save(model: Any, output_dir: str) -> None:
    """Save weights using the Ultralytics model ``save()`` method."""
    from klygo import files
    abs_out = files.resolve(output_dir)
    if model is not None and hasattr(model, "save"):
        model.save(abs_out)


def reset(model: Any) -> None:
    """Move an Ultralytics model back to CPU."""
    if model is not None and hasattr(model, "cpu"):
        model.cpu()


def unload(model: Any) -> None:
    """Release Ultralytics model accelerator resources."""
    if model is not None and hasattr(model, "cpu"):
        model.cpu()
