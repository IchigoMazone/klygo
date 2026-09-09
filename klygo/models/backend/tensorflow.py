"""
TensorFlow Backend Logic (klygo.models.backend.tensorflow).
Chứa toàn bộ logic xử lý đặc thù cho các mô hình TensorFlow, Keras, và TF Hub.
Lazy-import tensorflow để không gây crash nếu môi trường chưa cài đặt TensorFlow.
"""

import os
from typing import Any, List, Dict, Optional


def cast_inputs(inputs: Any, dev: Any = None, dtype: Any = None) -> Any:
    """
    Chuyển đổi dữ liệu ảnh/mảng NumPy sang dạng Tensor của TensorFlow/Keras.
    """
    try:
        import tensorflow as tf
        if isinstance(inputs, tf.Tensor):
            return inputs
        if isinstance(inputs, (list, tuple)):
            import numpy as np
            from klygo import media
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
                return tf.convert_to_tensor(stacked, dtype=tf.float32)
        return tf.convert_to_tensor(inputs, dtype=tf.float32)
    except ImportError:
        return inputs


def run_inference(model: Any, inputs: Any, **model_kwargs) -> Any:
    """Thực thi forward của mô hình TensorFlow/Keras."""
    if model is not None and callable(model):
        try:
            return model(inputs, training=False, **model_kwargs)
        except TypeError:
            return model(inputs, **model_kwargs)
    return inputs


def format_results(tf_outputs: Any) -> List[Dict[str, Any]]:
    """
    Bóc tách kết quả nhận diện từ output của mô hình TensorFlow / TF-OD API.
    Hỗ trợ cả format dict ({'detection_boxes', 'detection_scores', 'detection_classes'})
    lẫn tuple/list thông thường.
    """
    raw_list = []
    if tf_outputs is None:
        return raw_list

    # 1. Định dạng chuẩn của TensorFlow Object Detection API / TF Hub
    if isinstance(tf_outputs, dict):
        boxes_tensor = tf_outputs.get("detection_boxes")
        scores_tensor = tf_outputs.get("detection_scores")
        classes_tensor = tf_outputs.get("detection_classes")

        # Chuyển sang NumPy
        def _to_np(t):
            if hasattr(t, "numpy"):
                return t.numpy()
            return t

        b_np = _to_np(boxes_tensor)
        s_np = _to_np(scores_tensor)
        c_np = _to_np(classes_tensor)

        if b_np is not None and s_np is not None:
            # Batch size
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


def current_device(model: Any) -> str:
    """Dò tìm device hiện tại của TensorFlow graph/model."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return "cuda:0" if gpus else "cpu"
    except Exception:
        return "cpu"


def current_dtype(model: Any) -> str:
    """Dò tìm dtype hiện tại của model TensorFlow/Keras."""
    if hasattr(model, "compute_dtype") and model.compute_dtype is not None:
        return str(model.compute_dtype.name if hasattr(model.compute_dtype, "name") else model.compute_dtype)
    if hasattr(model, "dtype") and model.dtype is not None:
        return str(model.dtype.name if hasattr(model.dtype, "name") else model.dtype)
    return "float32"


def save(model: Any, output_dir: str) -> None:
    """Lưu model theo định dạng chuẩn TensorFlow / Keras (SavedModel hoặc .keras)."""
    abs_out = os.path.abspath(output_dir)
    if model is not None:
        if hasattr(model, "save"):
            model.save(abs_out)
        else:
            try:
                import tensorflow as tf
                tf.saved_model.save(model, abs_out)
            except Exception:
                pass


def reset(model: Any) -> None:
    """Xóa session Keras khi reset trạng thái."""
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass


def unload(model: Any) -> None:
    """Thu hồi session và dọn dẹp bộ nhớ TensorFlow."""
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
