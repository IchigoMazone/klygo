"""Native torchvision detection helpers shared by compatible wrappers."""

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

from klygo.models.runtime import torch as torch_runtime


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

# Historical adapter exports share the same runtime implementations.
from klygo.models.runtime.torch import (
    current_device, current_dtype, inference_context, is_accelerator_available,
    synchronize, clear_cache,
)
