"""
Keras 3 & KerasHub Backend Logic (klygo.models.backend.keras).
Chứa toàn bộ logic xử lý đặc thù cho hệ sinh thái Keras 3 và kho mô hình KerasHub.
Tự động hỗ trợ cả 3 computation engines: PyTorch (torch), JAX (jax), và TensorFlow (tensorflow).
Lazy-import để không gây crash nếu môi trường chưa cài đặt keras / keras_hub.
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
    """
    Tự động dò tìm và cấu hình KERAS_BACKEND tối ưu nếu chưa được thiết lập.
    Thứ tự ưu tiên: PyTorch (torch) -> JAX (jax) -> TensorFlow (tensorflow).
    """
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
    """Trả về backend tính toán hiện hành của Keras (torch, jax, hoặc tensorflow)."""
    ensure_backend()
    keras = _require_keras()
    return keras.backend.backend()


def cast_inputs(inputs: Any, dev: Any = None, dtype: Any = None) -> Any:
    """
    Chuyển đổi dữ liệu ảnh/mảng NumPy sang dạng Tensor đa nền tảng của Keras 3.
    Tự động hỗ trợ đồng bộ theo backend tính toán (PyTorch, JAX, TensorFlow).
    """
    ensure_backend()
    keras = _require_keras()
    import numpy as np
    from klygo import media

    # Nếu đã là Keras/Backend tensor
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
    """
    Thực thi forward pass của mô hình Keras 3 / KerasHub.
    Ưu tiên gọi model.predict(inputs) hoặc model(inputs, training=False).
    """
    if model is None:
        return inputs

    # 1. KerasHub / Keras model.predict
    if hasattr(model, "predict") and callable(model.predict):
        try:
            return model.predict(inputs, verbose=0, **model_kwargs)
        except TypeError:
            pass

    # 2. Direct callable
    if callable(model):
        try:
            return model(inputs, training=False, **model_kwargs)
        except TypeError:
            return model(inputs, **model_kwargs)

    return inputs


def _to_numpy_array(tensor: Any) -> Optional[Any]:
    """Chuyển đổi linh hoạt tensor từ Keras 3 (torch/jax/tf) sang NumPy ndarray."""
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
    """
    Chuẩn hóa kết quả nhận diện từ output của KerasHub / KerasCV.
    Hỗ trợ định dạng dict chuẩn của KerasHub:
    {'boxes': ..., 'confidence': ... (hoặc 'scores'), 'classes': ... (hoặc 'labels')}.
    """
    raw_list: List[Dict[str, Any]] = []
    if raw_outputs is None:
        return raw_list

    if isinstance(raw_outputs, dict):
        boxes_tensor = raw_outputs.get("boxes", raw_outputs.get("detection_boxes"))
        scores_tensor = raw_outputs.get("confidence", raw_outputs.get("scores", raw_outputs.get("detection_scores")))
        classes_tensor = raw_outputs.get("classes", raw_outputs.get("labels", raw_outputs.get("detection_classes")))

        b_np = _to_numpy_array(boxes_tensor)
        s_np = _to_numpy_array(scores_tensor)
        c_np = _to_numpy_array(classes_tensor)

        if b_np is not None and s_np is not None:
            batch_size = len(b_np)
            for b_idx in range(batch_size):
                boxes_data, scores_data, labels_data = [], [], []
                sample_b = b_np[b_idx]
                sample_s = s_np[b_idx]
                sample_c = c_np[b_idx] if c_np is not None else [0] * len(sample_s)

                for box, score, cls_id in zip(sample_b, sample_s, sample_c):
                    boxes_data.append([float(x) for x in box])
                    scores_data.append(float(score))
                    labels_data.append(str(int(cls_id)))

                raw_list.append({"boxes": boxes_data, "scores": scores_data, "labels": labels_data})
            return raw_list

    return raw_list


def current_device(model: Any) -> Any:
    """
    Dò tìm thiết bị thực tế của model Keras 3 theo backend đang chạy
    (PyTorch CUDA/CPU, JAX device, hoặc TensorFlow GPU/CPU).
    """
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
    """Dò tìm dtype hiện tại của model Keras 3 / KerasHub."""
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
    """
    Lưu model Keras 3 / KerasHub:
    1. Nếu là KerasHub preset -> model.save_to_preset(abs_out)
    2. Nếu là Keras model chuẩn -> model.save(abs_out / "model.keras")
    """
    from klygo import files
    abs_out = files.resolve(output_dir)
    files.mkdir(abs_out)

    if model is not None:
        # 1. KerasHub preset export
        if hasattr(model, "save_to_preset") and callable(model.save_to_preset):
            try:
                model.save_to_preset(abs_out)
                return
            except Exception:
                pass

        # 2. File định dạng chuẩn .keras
        if hasattr(model, "save") and callable(model.save):
            try:
                keras_file = files.join(abs_out, "model.keras")
                model.save(keras_file)
                return
            except Exception:
                pass


def reset(model: Any) -> None:
    """Dọn dẹp session Keras và giải phóng bộ nhớ khi reset trạng thái."""
    try:
        import keras
        if hasattr(keras.backend, "clear_session"):
            keras.backend.clear_session()
    except Exception:
        pass


def unload(model: Any) -> None:
    """Thu hồi tài nguyên Keras model."""
    try:
        import keras
        if hasattr(keras.backend, "clear_session"):
            keras.backend.clear_session()
    except Exception:
        pass
