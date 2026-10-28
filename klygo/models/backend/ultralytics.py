"""Ultralytics Backend Logic (`klygo.models.backend.ultralytics`).

Provides runtime adaptation, result formatting, and lifecycle helpers for
Ultralytics YOLO and YOLO-World detection models.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

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
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes):
                if hasattr(boxes, "xyxy"):
                    boxes_data = boxes.xyxy.detach().cpu().numpy()
                    scores_data = boxes.conf.detach().cpu().numpy()
                    class_ids = boxes.cls.detach().to(dtype=torch_runtime.torch.int32).cpu().numpy()
                    labels_data = [res.names.get(int(class_id), str(int(class_id))) for class_id in class_ids]
                else:
                    for box in boxes:
                        boxes_data.append(box.xyxy[0].tolist())
                        scores_data.append(float(box.conf[0].item()))
                        class_id = int(box.cls[0].item())
                        class_ids.append(class_id)
                        labels_data.append(res.names.get(class_id, str(class_id)))
            raw_list.append(
                {
                    "boxes": boxes_data,
                    "scores": scores_data,
                    "labels": labels_data,
                    "class_ids": class_ids,
                }
            )
    return raw_list


def _option_matches(current: Any, expected: Any) -> bool:
    """Compare predictor options without array truth-value errors."""
    try:
        comparison = current == expected
        return bool(comparison if isinstance(comparison, bool) else comparison.all())
    except (AttributeError, TypeError, ValueError):
        return False


def predict_prepared(
    model: Any,
    images: List[Any],
    options: Optional[Dict[str, Any]] = None,
) -> Optional[List[Any]]:
    """Run an initialized predictor on decoded BGR arrays without rebuilding its source loader.

    ``None`` is returned when the installed Ultralytics predictor does not expose
    the required stable operations, allowing callers to use the public prediction
    API as a compatibility fallback.
    """
    predictor = getattr(model, "predictor", None)
    required = (
        "_lock", "model", "preprocess", "inference", "postprocess", "run_callbacks"
    )
    if (
        predictor is None
        or not images
        or not all(isinstance(image, np.ndarray) for image in images)
        or any(not hasattr(predictor, name) for name in required)
    ):
        return None

    args = getattr(predictor, "args", None)
    if args is None or any(
        bool(getattr(args, name, False))
        for name in ("save", "save_txt", "save_crop", "show", "visualize", "embed")
    ):
        return None
    if options and any(
        hasattr(args, name) and not _option_matches(getattr(args, name), value)
        for name, value in options.items()
        if name != "verbose"
    ):
        return None

    paths = [f"image{index}.jpg" for index in range(len(images))]
    messages = [""] * len(images)
    with predictor._lock:
        predictor.seen = 0
        predictor.batch = (paths, images, messages)
        predictor.run_callbacks("on_predict_start")
        predictor.run_callbacks("on_predict_batch_start")

        started = time.perf_counter()
        inputs = predictor.preprocess(images)
        preprocess_ms = (time.perf_counter() - started) * 1000.0 / len(images)
        if not predictor.done_warmup:
            predictor.model.warmup(im=inputs)
            predictor.done_warmup = True

        started = time.perf_counter()
        predictions = predictor.inference(inputs)
        inference_ms = (time.perf_counter() - started) * 1000.0 / len(images)

        started = time.perf_counter()
        results = predictor.postprocess(predictions, inputs, images)
        postprocess_ms = (time.perf_counter() - started) * 1000.0 / len(images)
        predictor.results = results
        predictor.run_callbacks("on_predict_postprocess_end")
        for result in results:
            predictor.seen += 1
            result.speed = {
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
            }
        predictor.run_callbacks("on_predict_batch_end")
        predictor.run_callbacks("on_predict_end")
        return results


def save(model: Any, output_dir: str) -> None:
    """Save weights using the Ultralytics model ``save()`` method."""
    from klygo import files
    abs_out = files.resolve(output_dir)
    target = files.join(abs_out, "model.pt") if files.is_dir(abs_out) else abs_out
    if model is not None and hasattr(model, "save"):
        model.save(target)


def reset(model: Any) -> None:
    """Move an Ultralytics model back to CPU."""
    if model is not None and hasattr(model, "cpu"):
        model.cpu()


def unload(model: Any) -> None:
    """Release Ultralytics model accelerator resources."""
    if model is not None and hasattr(model, "cpu"):
        model.cpu()
