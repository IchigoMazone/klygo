"""Keras 3 and KerasHub backend adapter (`klygo.models.backend.keras`).

Supports PyTorch (`torch`), JAX (`jax`), and TensorFlow (`tensorflow`) computation engines
with lazy imports so environments without `keras` / `keras_hub` remain unaffected.
"""

import importlib
import os
from contextlib import nullcontext
from typing import Any, List, Dict, Optional


def _torch_runtime():
    """Load PyTorch support only when Keras actually uses that engine."""
    try:
        return importlib.import_module(".torch_runtime", __package__)
    except ImportError as exc:
        raise ImportError(
            "The Keras Torch engine requires PyTorch. "
            "Install it using 'pip install \"klygo[torch]\"'."
        ) from exc


def ensure_backend() -> str:
    """Configure `KERAS_BACKEND` automatically if not already set (priority: torch -> jax -> tensorflow)."""
    if "KERAS_BACKEND" not in os.environ:
        try:
            import torch
            os.environ["KERAS_BACKEND"] = "torch"
        except ImportError:
            try:
                import jax
                os.environ["KERAS_BACKEND"] = "jax"
            except ImportError:
                try:
                    import tensorflow
                    os.environ["KERAS_BACKEND"] = "tensorflow"
                except ImportError:
                    os.environ["KERAS_BACKEND"] = "torch"
    return os.environ.get("KERAS_BACKEND", "torch")


def _require_keras():
    """Import Keras with an actionable message for an omitted model extra."""
    try:
        return importlib.import_module("keras")
    except ImportError as exc:
        raise ImportError(
            "KerasHub models require the 'keras-hub' extra. "
            "Install it using 'pip install \"klygo[keras-hub]\"'."
        ) from exc


def get_active_backend() -> str:
    """Return the active Keras computation backend (`torch`, `jax`, or `tensorflow`)."""
    ensure_backend()
    keras = _require_keras()
    return keras.backend.backend()


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


def current_device(model: Any) -> Any:
    """Detect the active device for a Keras 3 model across PyTorch, TensorFlow, or JAX."""
    active_backend = get_active_backend()
    if active_backend == "torch":
        try:
            import torch
            if model is not None and hasattr(model, "parameters"):
                try:
                    return next(model.parameters()).device
                except Exception:
                    pass
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"
    elif active_backend == "tensorflow":
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "/GPU:0"
            tpus = tf.config.list_physical_devices("TPU")
            return "/TPU:0" if tpus else "/CPU:0"
        except Exception:
            return "cpu"
    elif active_backend == "jax":
        try:
            import jax
            devs = jax.devices()
            return devs[0] if devs else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def current_dtype(model: Any) -> str:
    """Return the active dtype of a Keras 3 / KerasHub model."""
    if model is not None:
        if hasattr(model, "compute_dtype") and model.compute_dtype is not None:
            return str(getattr(model.compute_dtype, "name", model.compute_dtype))
        if hasattr(model, "dtype") and model.dtype is not None:
            return str(getattr(model.dtype, "name", model.dtype))
    return "float32"


def inference_context():
    """Return a safe context for the active Keras execution engine."""
    if get_active_backend() == "torch":
        return _torch_runtime().inference_context()
    return nullcontext()


def is_accelerator_available(device: Any = None) -> bool:
    """Check accelerator availability through the active Keras engine."""
    active_backend = get_active_backend()
    if active_backend == "torch":
        return _torch_runtime().is_accelerator_available(device)
    if active_backend == "tensorflow":
        try:
            import tensorflow as tf

            if device is not None and not any(
                marker in str(device).lower() for marker in ("gpu", "tpu")
            ):
                return False
            return bool(
                tf.config.list_physical_devices("GPU")
                or tf.config.list_physical_devices("TPU")
            )
        except Exception:
            return False
    if active_backend == "jax":
        try:
            import jax

            devices = [device] if device is not None else jax.devices()
            return any(
                str(getattr(item, "platform", item)).lower().split(":", 1)[0]
                in {"gpu", "tpu", "cuda", "rocm"}
                for item in devices
            )
        except Exception:
            return False
    return False


def _block_until_ready(value: Any) -> bool:
    """Block on a nested TensorFlow/JAX value when it exposes a sync method."""
    if isinstance(value, dict):
        blocked = False
        for item in value.values():
            blocked = _block_until_ready(item) or blocked
        return blocked
    if isinstance(value, (list, tuple)):
        blocked = False
        for item in value:
            blocked = _block_until_ready(item) or blocked
        return blocked
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
        return True
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        to_numpy()
        return True
    return False


def synchronize(device: Any = None, value: Any = None) -> None:
    """Synchronize work through Torch, TensorFlow, or JAX as configured."""
    active_backend = get_active_backend()
    if active_backend == "torch":
        _torch_runtime().synchronize(device=device, value=value)
        return
    if active_backend == "tensorflow":
        try:
            if value is not None:
                _block_until_ready(value)
        except Exception:
            pass
        return
    if active_backend == "jax":
        try:
            import jax

            if value is not None:
                block = getattr(jax, "block_until_ready", None)
                if callable(block):
                    block(value)
                    return
                if _block_until_ready(value):
                    return
            barrier = getattr(jax, "effects_barrier", None)
            if callable(barrier):
                barrier()
        except Exception:
            pass


def clear_cache() -> None:
    """Clear safe runtime caches for the active Keras engine.

    TensorFlow has no public equivalent to PyTorch's allocator cache operation,
    so active models are left untouched. Keras session cleanup remains part of
    ``reset`` and ``unload`` where invalidating framework state is expected.
    """
    active_backend = get_active_backend()
    if active_backend == "torch":
        _torch_runtime().clear_cache()
    elif active_backend == "jax":
        try:
            import jax

            clear = getattr(jax, "clear_caches", None)
            if callable(clear):
                clear()
        except Exception:
            pass


def save(model: Any, output_dir: str) -> None:
    """Save a Keras 3 or KerasHub model to `output_dir`."""
    from klygo import files
    abs_out = str(files.resolve(output_dir))
    files.mkdir(abs_out)

    if model is not None:
        # 1. KerasHub preset export
        if hasattr(model, "save_to_preset") and callable(model.save_to_preset):
            try:
                model.save_to_preset(abs_out)
                return
            except Exception:
                pass

        # 2. Standard .keras format
        if hasattr(model, "save") and callable(model.save):
            try:
                keras_file = str(files.join(abs_out, "model.keras"))
                model.save(keras_file)
                return
            except Exception:
                pass


def reset(model: Any) -> None:
    """Clear the Keras session when resetting model state."""
    try:
        import keras
        if hasattr(keras.backend, "clear_session"):
            keras.backend.clear_session()
    except Exception:
        pass


def unload(model: Any) -> None:
    """Release Keras model resources."""
    try:
        import keras
        if hasattr(keras.backend, "clear_session"):
            keras.backend.clear_session()
    except Exception:
        pass
