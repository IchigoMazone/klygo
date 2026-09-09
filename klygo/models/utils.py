"""
Các hàm tiện ích và tiền xử lý dùng chung cho các mô hình AI (`klygo.models.utils`).
"""

import os
import time
import logging
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Dict, Union, Tuple, Optional, Sequence, Set
import PIL.Image


import functools

class KlygoKwargWarning(UserWarning):
    """Cảnh báo khi người dùng truyền tham số lạ không thuộc nhóm cấu hình nào."""
    pass


def suppress_warnings(func=None):
    """
    Decorator hoặc Context Manager tắt mọi warning (Python warnings + Hugging Face/PyTorch loggers),
    nhưng vẫn giữ lại cảnh báo KlygoKwargWarning của Klygo.
    """
    class SuppressContext:
        def __enter__(self):
            suppress_ai_warnings()
            self._ctx = warnings.catch_warnings()
            self._ctx.__enter__()
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("always", category=KlygoKwargWarning)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)

        def __call__(self, fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self:
                    return fn(*args, **kwargs)
            return wrapper

    if func is not None:
        return SuppressContext()(func)
    return SuppressContext()


def suppress_ai_warnings() -> None:
    """
    Tắt toàn bộ các cảnh báo không cần thiết từ Hugging Face Hub, Transformers, PyTorch và Tokenizers.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    try:
        import accelerate
    except Exception:
        pass

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
    groups: Optional[Sequence[str]] = None,
    warn_unmatched: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    """
    Phân giải và chia tách 2 tầng tham số:
    - Tầng 1: Cấu hình mặc định (từ model.json / self.settings)
    - Tầng 2: Tham số runtime ghi đè (kwargs truyền vào khi gọi hàm)

    Áp dụng 3 quy tắc:
    - Quy tắc 1 (Tự động): Key đã có trong default config -> gom vào nhóm đó mà không cần tiền tố.
    - Quy tắc 2 (Tường minh): Truyền dict {group}={...} hoặc tiền tố {group}_{key} -> thêm vào nhóm đó.
    - Quy tắc 3 (Tham số ma): Key lạ không qua được 1 và 2 -> phát cảnh báo KlygoKwargWarning và bỏ qua.
    """
    json_cfg = dict(json_config or {})

    # Xác định danh sách nhóm theo flags
    if groups:
        group_list = tuple(groups)
    elif any(isinstance(v, dict) for v in json_cfg.values()):
        group_list = tuple(k for k, v in json_cfg.items() if isinstance(v, dict))
    else:
        group_list = ("model", "processor", "post")

    # Tầng 1: Khởi tạo buckets từ default json_config
    buckets: Dict[str, Dict[str, Any]] = {g: dict(json_cfg.get(g, {})) for g in group_list}

    unmatched_keys = []

    # Tầng 2: Phân giải runtime kwargs
    for key, value in kwargs.items():
        # Quy tắc 2A: Dict tường minh theo nhóm — post={"threshold": 0.5, "custom": 1}
        if key in group_list and isinstance(value, dict):
            buckets[key].update(value)
            continue

        # Quy tắc 2B: Tiền tố nhóm tường minh — post_threshold=0.5, processor_max_length=256
        matched_prefix = False
        for g in group_list:
            if key.startswith(f"{g}_"):
                clean_key = key[len(g) + 1:]
                buckets[g][clean_key] = value
                matched_prefix = True
                break
        if matched_prefix:
            continue

        # Quy tắc 1: Tham số phẳng ĐÃ CÓ trong cấu hình mặc định -> tự động map vào nhóm đó
        matched_flat = False
        for g in group_list:
            if key in json_cfg.get(g, {}):
                buckets[g][key] = value
                matched_flat = True
                break
        if matched_flat:
            continue

        # Quy tắc 3: Tham số lạ không thuộc nhóm nào và không có tiền tố -> ghi nhận để cảnh báo
        unmatched_keys.append(key)

    if warn_unmatched and unmatched_keys:
        available_groups = ", ".join(repr(g) for g in group_list)
        for uk in unmatched_keys:
            warnings.warn(
                f"[Klygo Warning] Tham số lạ '{uk}' không thuộc bất kỳ nhóm cấu hình mặc định nào ({available_groups}) "
                f"và không có tiền tố nhóm hợp lệ. Tham số này sẽ bị bỏ qua!",
                KlygoKwargWarning,
                stacklevel=3,
            )

    if groups:
        return tuple(buckets.get(g, {}) for g in groups)
    return tuple(buckets.values())


def resolve_sub_kwargs_dict(
    kwargs: Dict[str, Any],
    json_config: Optional[Dict[str, Any]] = None,
    groups: Optional[Sequence[str]] = None,
    warn_unmatched: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Phân giải cấu hình và trả về dict các nhóm thay vì tuple."""
    json_cfg = dict(json_config or {})
    if groups:
        group_list = tuple(groups)
    elif any(isinstance(v, dict) for v in json_cfg.values()):
        group_list = tuple(k for k, v in json_cfg.items() if isinstance(v, dict))
    else:
        group_list = ("model", "processor", "post")
    result = resolve_sub_kwargs(kwargs=kwargs, json_config=json_config, groups=group_list, warn_unmatched=warn_unmatched)
    return dict(zip(group_list, result))

def resolve_images(
    source: Any,
    step: int = 1,
    max_frames: Optional[int] = None,
    stream: bool = False,
) -> Tuple[Union[List[PIL.Image.Image], Any], bool]:
    """
    Tự động phân giải nguồn dữ liệu đầu vào thông qua klygo.media.
    Hỗ trợ stream=True để trả về Generator (chống văng RAM khi đọc video lớn).
    """
    from klygo import media

    step = max(1, int(step))

    # Xử lý luồng Generator/Stream
    if stream:
        def _build_generator():
            if isinstance(source, (str, Path)):
                raw_stream = media.load(source, stream=True, verbose=False)
            elif hasattr(source, "__iter__") and not isinstance(source, (list, tuple)):
                raw_stream = source
            else:
                raw_stream = source if isinstance(source, (list, tuple)) else [source]
            
            count = 0
            for idx, item in enumerate(raw_stream):
                if max_frames is not None and count >= max_frames:
                    break
                if idx % step == 0:
                    count += 1
                    if isinstance(item, PIL.Image.Image):
                        yield item.convert("RGB")
                    else:
                        yield media.to_pil(item).convert("RGB")

        is_single = False
        if isinstance(source, (str, Path)):
            is_single = Path(str(source)).suffix.lower() not in media.VIDEO_SUFFIXES
        elif isinstance(source, PIL.Image.Image):
            is_single = True
        return _build_generator(), is_single

    # Xử lý luồng List (RAM tiêu chuẩn) - Giữ nguyên logic cũ
    if isinstance(source, (str, Path)):
        loaded = media.load(source, stream=False, verbose=False)
        raw_list = loaded if isinstance(loaded, list) else [loaded]
        is_single = len(raw_list) == 1 and Path(str(source)).suffix.lower() not in media.VIDEO_SUFFIXES
    elif isinstance(source, PIL.Image.Image):
        return [source.convert("RGB")], True
    elif hasattr(source, "shape"):
        if getattr(source, "ndim", 0) in (2, 3):
            return [media.to_pil(source).convert("RGB")], True
        elif getattr(source, "ndim", 0) == 4:
            batch_list = [media.to_pil(img).convert("RGB") for img in source]
            if step > 1:
                batch_list = batch_list[::step]
            if max_frames is not None and max_frames > 0:
                batch_list = batch_list[:max_frames]
            return batch_list, False
        raw_list = list(source)
        is_single = False
    elif isinstance(source, (list, tuple)):
        raw_list = list(source)
        is_single = len(raw_list) == 1
    elif hasattr(source, "__iter__"):
        raw_list = list(source)
        is_single = len(raw_list) == 1
    else:
        raise TypeError(f"Đầu vào '{type(source).__name__}' không hợp lệ.")

    if not raw_list:
        return [], is_single

    if step > 1:
        raw_list = raw_list[::step]
    if max_frames is not None and max_frames > 0:
        raw_list = raw_list[:max_frames]

    cleaned_images = []
    for item in raw_list:
        if isinstance(item, PIL.Image.Image):
            cleaned_images.append(item.convert("RGB"))
        else:
            cleaned_images.append(media.to_pil(item).convert("RGB"))

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
