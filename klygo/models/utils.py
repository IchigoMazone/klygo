"""
Các hàm tiện ích và tiền xử lý dùng chung cho các mô hình AI (`klygo.models.utils`).
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Any, List, Dict, Union, Tuple, Optional
import PIL.Image


def suppress_ai_warnings() -> None:
    """
    Tắt toàn bộ các cảnh báo không cần thiết từ Hugging Face Hub, Transformers, PyTorch và Tokenizers.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    try:
        import huggingface_hub.utils.logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        import transformers.utils.logging as tf_logging
        tf_logging.set_verbosity_error()
    except Exception:
        pass

    for logger_name in ["huggingface_hub", "huggingface_hub.utils._http", "transformers"]:
        try:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        except Exception:
            pass


def resolve_images(
    source: Any,
    step: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[List[PIL.Image.Image], bool]:
    """
    Tự động phân giải nguồn dữ liệu đầu vào (1 ảnh đơn lẻ hoặc video / folder / list ảnh),
    hỗ trợ bỏ bớt frame (step/stride) và giới hạn số frame (max_frames).

    Đầu vào:
    - source: 1 ảnh PIL.Image, numpy array, hoặc danh sách/generator video từ media.load.
    - step [int]: Bước nhảy frame khi duyệt video/folder (Mặc định: 1 - lấy đủ mọi frame). Ví dụ step=5 lấy frame 0, 5, 10...
    - max_frames [Optional[int]]: Giới hạn số lượng frame tối đa cần xử lý.

    Đầu ra:
    - (List[PIL.Image.Image], is_single: bool):
      + is_single = True nếu đầu vào là 1 ảnh duy nhất.
      + is_single = False nếu đầu vào là video, folder hoặc list ảnh.
    """
    if isinstance(source, (str, Path)):
        raise TypeError(
            f"Đầu vào dạng đường dẫn chuỗi '{source}' chưa được nạp. "
            "Vui lòng nạp qua klygo.media.load() trước khi truyền vào model.predict() "
            "(Ví dụ: img = media.load('image.jpg'); model.predict(img, ...))."
        )

    # 1. Trường hợp 1 ảnh PIL.Image đơn lẻ
    if isinstance(source, PIL.Image.Image):
        return [source.convert("RGB")], True

    # 2. Trường hợp numpy array
    if hasattr(source, "shape"):
        if getattr(source, "ndim", 0) == 3:
            return [PIL.Image.fromarray(source).convert("RGB")], True
        elif getattr(source, "ndim", 0) == 4:
            # Batch numpy array (B, H, W, C)
            batch_list = [PIL.Image.fromarray(img).convert("RGB") for img in source]
            step = max(1, int(step))
            if step > 1:
                batch_list = batch_list[::step]
            if max_frames is not None and max_frames > 0:
                batch_list = batch_list[:max_frames]
            return batch_list, False

    # 3. Trường hợp danh sách / tuple / generator
    if isinstance(source, (list, tuple)):
        raw_list = list(source)
    elif hasattr(source, "__iter__"):
        raw_list = list(source)
    else:
        raise TypeError(
            f"Đầu vào '{type(source).__name__}' không hợp lệ. "
            "model.predict() nhận ảnh từ klygo.media.load() (PIL.Image hoặc List[PIL.Image])."
        )

    if not raw_list:
        return [], False

    # Áp dụng bước nhảy frame (step) và giới hạn số frame (max_frames)
    step = max(1, int(step))
    if step > 1:
        raw_list = raw_list[::step]
    if max_frames is not None and max_frames > 0:
        raw_list = raw_list[:max_frames]

    # Chuyển đổi toàn bộ phần tử trong list sang RGB PIL.Image
    cleaned_images = []
    for item in raw_list:
        if isinstance(item, PIL.Image.Image):
            cleaned_images.append(item.convert("RGB"))
        elif hasattr(item, "shape") and getattr(item, "ndim", 0) == 3:
            cleaned_images.append(PIL.Image.fromarray(item).convert("RGB"))
        else:
            raise TypeError(
                f"Phần tử trong danh sách có kiểu '{type(item).__name__}' không phải ảnh hợp lệ."
            )

    return cleaned_images, False


def prepare_image(image: Any) -> PIL.Image.Image:
    """
    Chuẩn hóa 1 bức ảnh đầu vào.
    """
    images, _ = resolve_images(image)
    if not images:
        raise ValueError("Không có dữ liệu ảnh hợp lệ.")
    return images[0]


def normalize_prompt(text_prompt: Union[str, List[str]]) -> List[str]:
    """
    Chuẩn hóa prompt nhãn từ khóa thành danh sách List[str].
    """
    if isinstance(text_prompt, str):
        return [text_prompt]
    elif isinstance(text_prompt, (list, tuple)):
        return list(text_prompt)
    else:
        raise TypeError("text_prompt phải là chuỗi ký tự (str) hoặc danh sách chuỗi ký tự (list[str]).")


def calculate_speed(t_start: float, t_pre: float, t_infer: float, t_post: float, count: int = 1) -> Dict[str, float]:
    """
    Tính toán bảng tốc độ suy luận (ms/ảnh).
    """
    n = max(1, count)
    return {
        "preprocess": round(((t_pre - t_start) * 1000) / n, 2),
        "inference": round(((t_infer - t_pre) * 1000) / n, 2),
        "postprocess": round(((t_post - t_infer) * 1000) / n, 2),
        "total": round(((t_post - t_start) * 1000) / n, 2),
    }
