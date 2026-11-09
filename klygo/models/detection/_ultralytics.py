"""Native ultralytics detection helpers shared by compatible wrappers."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from klygo.models.runtime import torch as torch_runtime
except ImportError as exc:
    raise ImportError(
        "Ultralytics models require the 'ultralytics' extra. "
        "Install it using 'pip install \"klygo[ultralytics]\"'."
    ) from exc


def format_results(ultra_results: Any) -> List[Dict[str, Any]]:
    """Extract normalized ``boxes``, ``scores``, ``labels``, and ``class_ids`` from Ultralytics Results."""
    raw_list = []
    if ultra_results is not None:
        for res in ultra_results:
            boxes_data, scores_data, labels_data, class_ids = [], [], [], []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes):
                if hasattr(boxes, "data") and boxes.data.shape[-1] >= 6:
                    # Copy the contiguous detection payload to the host once.
                    # Separate ``xyxy``, ``conf``, and ``cls`` transfers force
                    # three accelerator synchronizations for every frame.
                    packed = boxes.data[:, :6].detach().to(device="cpu").numpy()
                    boxes_data = packed[:, :4]
                    scores_data = packed[:, 4]
                    class_ids = packed[:, 5].astype(np.int32, copy=False)
                    labels_data = [res.names.get(int(class_id), str(int(class_id))) for class_id in class_ids]
                elif hasattr(boxes, "xyxy"):
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


def optimize_model(model: Any) -> Any:
    """Reuse an initialized predictor for decoded image batches transparently.

    The concrete detector continues to call the public Ultralytics model API.
    Unsupported sources, streaming calls, changed options, and custom predictors
    fall back to the original ``predict`` implementation.
    """
    if model is None or getattr(model, "_klygo_predict_optimized", False):
        return model

    original_predict = getattr(model, "predict", None)
    if not callable(original_predict):
        return model

    def optimized_predict(source=None, stream=False, predictor=None, **kwargs):
        if predictor is None and not stream:
            prepared = predict_prepared(model, source, options=kwargs)
            if prepared is not None:
                return prepared
        call_options = dict(kwargs)
        if predictor is not None:
            call_options["predictor"] = predictor
        return original_predict(source=source, stream=stream, **call_options)

    model.predict = optimized_predict
    model._klygo_predict_optimized = True
    return model

# Historical adapter exports share the same runtime implementations.
from klygo.models.runtime.torch import (
    current_device, current_dtype, inference_context, is_accelerator_available,
    synchronize, clear_cache,
)
