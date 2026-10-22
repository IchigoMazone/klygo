"""Runtime adaptation for caller-owned Torchvision detection models."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torchvision  # noqa: F401 - validate the selected optional backend.
except (ImportError, RuntimeError) as exc:
    raise ImportError(
        "Torchvision detection requires the 'torchvision' extra. "
        "Install it using 'pip install \"klygo[torchvision]\"'."
    ) from exc

from . import torch_runtime


def current_device(model: Any):
    return torch_runtime.current_device(model)


def current_dtype(model: Any):
    return torch_runtime.current_dtype(model)


def is_accelerator_available(device: Any = None) -> bool:
    return torch_runtime.is_accelerator_available(device)


def synchronize(device: Any = None, value: Any = None) -> None:
    torch_runtime.synchronize(device=device, value=value)


def clear_cache() -> None:
    torch_runtime.clear_cache()


def inference_context():
    return torch_runtime.inference_context()


def cast_inputs(inputs: list[torch.Tensor], dev: torch.device, dtype: torch.dtype):
    return [image.to(device=dev, dtype=dtype) for image in inputs]


def run_inference(model: Any, inputs: list[torch.Tensor], **kwargs):
    return model(inputs, **kwargs)


def format_results(raw_outputs: Any) -> list[dict[str, Any]]:
    results = []
    for item in raw_outputs:
        class_ids = [int(value) for value in item["labels"].detach().cpu().tolist()]
        results.append(
            {
                "boxes": item["boxes"].detach().cpu().tolist(),
                "scores": item["scores"].detach().cpu().tolist(),
                "labels": list(class_ids),
                "class_ids": class_ids,
            }
        )
    return results


def reset(model: Any) -> None:
    if model is not None:
        model.cpu()


def unload(model: Any) -> None:
    reset(model)


def save(model: Any, output_dir: str) -> None:
    """Save a caller-owned Torchvision state dict without choosing its class."""
    from klygo import files

    if model is not None:
        torch.save(model.state_dict(), files.join(output_dir, "model.pt"))


