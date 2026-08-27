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


def resolve_sub_kwargs_dict(
    kwargs: Dict[str, Any],
    json_config: Optional[Dict[str, Any]] = None,
    groups: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Phân giải và chia tách thành dictionary các nhóm cấu hình động.
    Mặc định hỗ trợ chuẩn Hugging Face ('model', 'processor', 'post') nhưng tự động
    nhận diện bất kỳ nhóm cờ nào được khai báo trong json_config hoặc groups.
    """
    json_cfg = dict(json_config or {})

    # Xác định danh sách các nhóm cờ
    if groups:
        group_list = tuple(groups)
    elif any(isinstance(v, dict) for v in json_cfg.values()):
        group_list = tuple(k for k, v in json_cfg.items() if isinstance(v, dict))
    else:
        group_list = ("model", "processor", "post")

    # Khởi tạo các nhóm từ json_config
    buckets: Dict[str, Dict[str, Any]] = {g: dict(json_cfg.get(g, {})) for g in group_list}

    # Mặc định tham số post nếu có nhóm 'post'
    if "post" in buckets:
        buckets["post"].setdefault("threshold", 0.25)
        buckets["post"].setdefault("text_threshold", 0.3)

    # 1. Xử lý dạng Dictionary tường minh: group={...}
    for g in group_list:
        if g in kwargs and isinstance(kwargs[g], dict):
            buckets[g].update(kwargs[g])

    # 2. Xử lý các tham số phẳng còn lại
    for key, value in kwargs.items():
        if key in group_list:
            continue

        # Alias tiện ích: conf -> threshold
        if key == "conf" and "post" in buckets:
            buckets["post"]["threshold"] = value
            continue

        # Điều hướng các tham số Model phổ biến
        if key in ("torch_dtype", "dtype") and "model" in buckets:
            buckets["model"]["torch_dtype"] = value
            continue
        if key in ("device_map", "low_cpu_mem_usage", "load_in_8bit", "load_in_4bit", "quantization_config") and "model" in buckets:
            buckets["model"][key] = value
            continue

        # Dạng 1: Khớp trực tiếp key có sẵn trong nhóm nào đó
        matched = False
        for g in group_list:
            if key in buckets[g]:
                buckets[g][key] = value
                matched = True
                break
        if matched:
            continue

        # Dạng 2: Cú pháp tiền tố động: <group>_<param>_ hoặc <group>_<param>
        for g in group_list:
            prefix = f"{g}_"
            if key.startswith(prefix):
                clean_key = key[len(prefix):-1] if key.endswith("_") else key[len(prefix):]
                buckets[g][clean_key] = value
                matched = True
                break

        # Dạng 3: Bất kỳ tham số tùy biến / metadata nào chưa khớp -> Cho vào nhóm post (hoặc nhóm cuối cùng)
        if not matched:
            target_group = "post" if "post" in buckets else group_list[-1]
            buckets[target_group][key] = value

    return buckets


def resolve_sub_kwargs(
    kwargs: Dict[str, Any],
    json_config: Optional[Dict[str, Any]] = None,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """
    Phân giải và trả về tuple các nhóm cấu hình (mặc định tuple 3 phần tử: model_kw, proc_kw, post_kw).
    """
    buckets = resolve_sub_kwargs_dict(kwargs=kwargs, json_config=json_config, groups=groups)
    if groups:
        return tuple(buckets.get(g, {}) for g in groups)
    if "model" in buckets and "processor" in buckets and "post" in buckets:
        return buckets["model"], buckets["processor"], buckets["post"]
    return tuple(buckets.values())


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
