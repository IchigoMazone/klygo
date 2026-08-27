"""
Các hàm tiện ích và tiền xử lý dùng chung cho các mô hình AI (`klygo.models.utils`).
"""

import os
import time
import logging
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Dict, Union, Tuple, Optional
import PIL.Image


def suppress_ai_warnings() -> None:
    """
    Tắt toàn bộ các cảnh báo không cần thiết từ Hugging Face Hub, Transformers, PyTorch và Tokenizers.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
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

    for logger_name in [
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "transformers",
        "urllib3",
        "torch",
    ]:
        try:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        except Exception:
            pass

    # Tự động đồng bộ dtype trong grid_sample (sửa lỗi deformable attention của Transformers trên CPU/Multi-GPU)
    try:
        import torch
        import torch.nn.functional as F
        if not getattr(F, "_klygo_grid_sample_patched", False):
            _orig_grid_sample = F.grid_sample

            def _safe_grid_sample(input, grid, *args, **kwargs):
                if hasattr(grid, "dtype") and hasattr(input, "dtype") and grid.dtype != input.dtype:
                    grid = grid.to(input.dtype)
                return _orig_grid_sample(input, grid, *args, **kwargs)

            F.grid_sample = _safe_grid_sample
            F._klygo_grid_sample_patched = True
    except Exception:
        pass


def resolve_sub_kwargs(
    kwargs: Dict[str, Any],
    json_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Phân giải và chia tách thành 3 sub-kwargs độc lập:
    - model_kwargs
    - processor_kwargs
    - post_kwargs

    Hỗ trợ đồng thời cả 3 dạng truyền cấu hình:
    1. Dạng 1: Tên chuẩn nguyên bản có sẵn trong config.json -> Tự động định tuyến
    2. Dạng 2: Cú pháp kẹp tiền/hậu tố: model_<param>_, processor_<param>_, post_<param>_ -> 100% chống trùng
    3. Dạng 3: Khối Dictionary tường minh model={...}, processor={...}, post={...}
    """
    json_cfg = dict(json_config or {})

    # 1. Khởi tạo 3 nhóm từ config.json gốc
    model_kwargs = dict(json_cfg.get("model", {}))
    processor_kwargs = dict(json_cfg.get("processor", {}))
    post_kwargs = dict(json_cfg.get("post", {}))

    # Mặc định các tham số post cốt lõi nếu chưa có
    post_kwargs.setdefault("threshold", 0.25)
    post_kwargs.setdefault("text_threshold", 0.3)

    # 2. Xử lý Dạng 3: Khối Dictionary tường minh
    if "model" in kwargs and isinstance(kwargs["model"], dict):
        model_kwargs.update(kwargs["model"])
    if "processor" in kwargs and isinstance(kwargs["processor"], dict):
        processor_kwargs.update(kwargs["processor"])
    if "post" in kwargs and isinstance(kwargs["post"], dict):
        post_kwargs.update(kwargs["post"])

    # 3. Duyệt qua tất cả các tham số phẳng còn lại
    for key, value in kwargs.items():
        if key in ("model", "processor", "post"):
            continue

        # Alias tiện ích: conf -> threshold
        if key == "conf":
            post_kwargs["threshold"] = value
            continue

        # Điều hướng các tham số Model phổ biến: torch_dtype, dtype, device_map
        if key in ("torch_dtype", "dtype"):
            model_kwargs["torch_dtype"] = value
            continue
        if key in ("device_map", "low_cpu_mem_usage", "load_in_8bit", "load_in_4bit", "quantization_config"):
            model_kwargs[key] = value
            continue

        # Dạng 1: Đã có sẵn trong config.json -> Ghi đè tự nhiên
        if key in model_kwargs:
            model_kwargs[key] = value
        elif key in processor_kwargs:
            processor_kwargs[key] = value
        elif key in post_kwargs:
            post_kwargs[key] = value

        # Dạng 2: Cú pháp tiền tố model_<param>, processor_<param>, post_<param> (hỗ trợ cả có hoặc không có dấu gạch dưới cuối)
        elif key.startswith("model_"):
            clean_key = key[6:-1] if key.endswith("_") else key[6:]
            model_kwargs[clean_key] = value
        elif key.startswith("processor_"):
            clean_key = key[10:-1] if key.endswith("_") else key[10:]
            processor_kwargs[clean_key] = value
        elif key.startswith("post_"):
            clean_key = key[5:-1] if key.endswith("_") else key[5:]
            post_kwargs[clean_key] = value

    return model_kwargs, processor_kwargs, post_kwargs


def resolve_images(
    source: Any,
    step: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[List[PIL.Image.Image], bool]:
    """
    Tự động phân giải nguồn dữ liệu đầu vào thông qua klygo.media.
    Đảm bảo 100% đầu ra chuyển về danh sách đối tượng PIL.Image (RGB).
    """
    from klygo import media

    # 1. Nếu là đường dẫn chuỗi hoặc Path -> Giao 100% cho klygo.media.load (backend='pil')
    if isinstance(source, (str, Path)):
        loaded = media.load(source, verbose=False)
        raw_list = loaded if isinstance(loaded, list) else [loaded]
        is_single = len(raw_list) == 1 and not str(source).lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'))
    # 2. Nếu đã là 1 ảnh PIL.Image đơn lẻ
    elif isinstance(source, PIL.Image.Image):
        return [source.convert("RGB")], True
    # 3. Nếu là numpy array hoặc Tensor (tự động chuyển sang PIL)
    elif hasattr(source, "shape"):
        if getattr(source, "ndim", 0) in (2, 3):
            return [media.to_pil(source).convert("RGB")], True
        elif getattr(source, "ndim", 0) == 4:
            batch_list = [media.to_pil(img).convert("RGB") for img in source]
            step = max(1, int(step))
            if step > 1:
                batch_list = batch_list[::step]
            if max_frames is not None and max_frames > 0:
                batch_list = batch_list[:max_frames]
            return batch_list, False
        raw_list = list(source)
        is_single = False
    # 4. Nếu là danh sách ảnh hoặc Iterator
    elif isinstance(source, (list, tuple)):
        raw_list = list(source)
        is_single = len(raw_list) == 1
    elif hasattr(source, "__iter__"):
        raw_list = list(source)
        is_single = len(raw_list) == 1
    else:
        raise TypeError(
            f"Đầu vào '{type(source).__name__}' không hợp lệ. Vui lòng truyền đường dẫn ảnh/video hoặc dữ liệu media."
        )

    if not raw_list:
        return [], is_single

    step = max(1, int(step))
    if step > 1:
        raw_list = raw_list[::step]
    if max_frames is not None and max_frames > 0:
        raw_list = raw_list[:max_frames]

    cleaned_images = []
    for item in raw_list:
        if isinstance(item, PIL.Image.Image):
            cleaned_images.append(item.convert("RGB"))
        else:
            try:
                cleaned_images.append(media.to_pil(item).convert("RGB"))
            except Exception as err:
                raise TypeError(f"Không thể chuyển đổi phần tử kiểu '{type(item).__name__}' sang PIL Image: {err}")

    return cleaned_images, is_single


def normalize_prompt(text_prompt: Union[str, List[str]]) -> List[str]:
    """Chuẩn hóa prompt nhãn từ khóa thành danh sách List[str]."""
    from klygo.validators import validate_type
    validate_type(text_prompt, (str, list, tuple), "prompt")
    if isinstance(text_prompt, str):
        return [text_prompt.strip()]
    return [str(p).strip() for p in text_prompt if str(p).strip()]


def amp_autocast_if_needed(
    use_half: bool = False,
    dtype: Optional[str] = None,
    device_type: Optional[str] = None,
):
    """Context manager bọc torch.amp.autocast khi chạy FP16 hoặc BFLOAT16."""
    from klygo import cuda
    try:
        import torch
        dev_type = device_type or ("cuda" if cuda.is_available() else "cpu")
        dt = (dtype or "").lower()

        if dt in ("bfloat16", "bf16"):
            return torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16)
        elif (use_half or dt in ("float16", "fp16", "half")) and dev_type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        pass
    return nullcontext()


def cuda_sync() -> None:
    """Đồng bộ dòng tính toán trên GPU để đo đạc thời gian chính xác."""
    from klygo import cuda
    if cuda.is_available():
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
