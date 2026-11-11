"""Optional Keras detection input conversion, inference and output packing."""

from typing import Any, List, Dict, Optional
from klygo.models.runtime.keras import ensure_backend, _require_keras

def cast_inputs(inputs: Any, dev: Any = None, dtype: Any = None) -> Any:
    """Convert image or NumPy inputs into Keras 3 tensors for the active compute backend."""
    ensure_backend()
    keras = _require_keras()
    import numpy as np
    from klygo import media

    if hasattr(inputs, "shape") and hasattr(inputs, "dtype") and not isinstance(inputs, np.ndarray):
        return inputs

    if isinstance(inputs, (list, tuple)):
        np_arrs = []
        for item in inputs:
            if isinstance(item, np.ndarray):
                np_arrs.append(item)
            else:
                try:
                    np_arrs.append(media.to_array(item))
                except Exception:
                    if hasattr(item, "__array__"):
                        np_arrs.append(np.array(item))
                    else:
                        np_arrs.append(item)

        if np_arrs and isinstance(np_arrs[0], np.ndarray):
            stacked = np.stack(np_arrs, axis=0)
            if hasattr(keras, "ops") and hasattr(keras.ops, "convert_to_tensor"):
                return keras.ops.convert_to_tensor(stacked, dtype="float32")
            return stacked

    if hasattr(keras, "ops") and hasattr(keras.ops, "convert_to_tensor"):
        return keras.ops.convert_to_tensor(inputs, dtype="float32")
    return inputs


def run_inference(model: Any, inputs: Any, **model_kwargs) -> Any:
    """Execute a forward pass on a Keras 3 or KerasHub model."""
    if model is None:
        return inputs

    bbox_fmt = getattr(
        getattr(model, "prediction_decoder", model),
        "bounding_box_format",
        getattr(model, "bounding_box_format", None),
    )

    # 1. KerasHub ObjectDetector with preprocessor + decode_predictions
    if hasattr(model, "decode_predictions") and callable(model.decode_predictions):
        try:
            keras = _require_keras()
            preprocessor = getattr(model, "preprocessor", None)
            if callable(preprocessor):
                img_conv = getattr(preprocessor, "image_converter", None)
                target_size = getattr(img_conv, "image_size", None) if img_conv is not None else None
                prep_inputs = (
                    keras.ops.image.resize(inputs, target_size)
                    if target_size is not None and hasattr(keras.ops, "image")
                    else inputs
                )
                prep = preprocessor(prep_inputs)
            else:
                prep = inputs
            raw_preds = model(prep, training=False, **model_kwargs)
            decoded = model.decode_predictions(raw_preds, inputs)
            if isinstance(decoded, dict) and bbox_fmt:
                decoded = dict(decoded)
                decoded["_bounding_box_format"] = str(bbox_fmt)
            return decoded
        except Exception:
            pass

    # 2. Standard Keras model.predict
    if hasattr(model, "predict") and callable(model.predict):
        try:
            out = model.predict(inputs, verbose=0, **model_kwargs)
            if isinstance(out, dict) and bbox_fmt:
                out = dict(out)
                out["_bounding_box_format"] = str(bbox_fmt)
            return out
        except (TypeError, ImportError):
            pass

    # 3. Direct callable
    if callable(model):
        try:
            return model(inputs, training=False, **model_kwargs)
        except TypeError:
            return model(inputs, **model_kwargs)

    return inputs


def _to_numpy_array(tensor: Any) -> Optional[Any]:
    """Convert a Keras 3 tensor (torch/jax/tf) to a NumPy ndarray."""
    if tensor is None:
        return None
    try:
        import keras
        if hasattr(keras, "ops") and hasattr(keras.ops, "convert_to_numpy"):
            return keras.ops.convert_to_numpy(tensor)
    except Exception:
        pass

    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    import numpy as np
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.asarray(tensor)


def format_results(raw_outputs: Any) -> List[Dict[str, Any]]:
    """Normalize KerasHub / KerasCV detection outputs into `[{boxes, scores, labels, class_ids}]`."""
    raw_list: List[Dict[str, Any]] = []
    if raw_outputs is None:
        return raw_list

    if isinstance(raw_outputs, dict):
        boxes_tensor = raw_outputs.get("boxes", raw_outputs.get("detection_boxes"))
        scores_tensor = raw_outputs.get("confidence", raw_outputs.get("scores", raw_outputs.get("detection_scores")))
        classes_tensor = raw_outputs.get("classes", raw_outputs.get("labels", raw_outputs.get("detection_classes")))
        num_det_tensor = raw_outputs.get("num_detections")
        bbox_fmt = str(raw_outputs.get("_bounding_box_format", "")).lower()

        b_np = _to_numpy_array(boxes_tensor)
        s_np = _to_numpy_array(scores_tensor)
        c_np = _to_numpy_array(classes_tensor)
        n_np = _to_numpy_array(num_det_tensor)

        if b_np is not None and s_np is not None:
            import numpy as np

            batch_size = len(b_np)
            for b_idx in range(batch_size):
                sample_b = np.asarray(b_np[b_idx])
                sample_s = np.asarray(s_np[b_idx])
                sample_c = (
                    np.asarray(c_np[b_idx])
                    if c_np is not None
                    else np.zeros(len(sample_s), dtype=np.int32)
                )
                limit = min(
                    int(n_np[b_idx]) if n_np is not None else len(sample_s),
                    len(sample_b),
                    len(sample_s),
                    len(sample_c),
                )
                valid = (sample_s[:limit] >= 0.0) & (sample_c[:limit] >= 0)
                boxes_data = sample_b[:limit][valid]
                if bbox_fmt == "yxyx" and boxes_data.ndim == 2 and boxes_data.shape[1] == 4:
                    boxes_data = boxes_data[:, [1, 0, 3, 2]]
                scores_data = sample_s[:limit][valid]
                class_ids = sample_c[:limit][valid].astype(np.int32, copy=False)
                labels_data = [str(int(class_id)) for class_id in class_ids]

                raw_list.append(
                    {
                        "boxes": boxes_data.tolist(),
                        "scores": scores_data.tolist(),
                        "labels": labels_data,
                        "class_ids": class_ids.tolist(),
                    }
                )
            return raw_list

    return raw_list



from klygo.models.runtime.keras import (
    current_device, current_dtype, inference_context, is_accelerator_available,
    synchronize, clear_cache, get_active_backend,
)
