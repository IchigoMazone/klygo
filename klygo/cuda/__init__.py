import platform
import sys
from typing import Optional
import torch


def is_available() -> bool:
    """Kiểm tra xem GPU CUDA của NVIDIA có khả dụng hay không.

    Trả về False trên macOS (Darwin), hệ thống AMD ROCm/HIP,
    và các môi trường không hỗ trợ NVIDIA CUDA.
    """
    if sys.platform == "darwin" or platform.system() == "Darwin":
        return False
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "cuda", None) is None:
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    try:
        if torch.cuda.device_count() > 0:
            device_name = torch.cuda.get_device_name(0).upper()
            if any(amd_kw in device_name for amd_kw in ["AMD", "RADEON", "ROCM"]):
                return False
    except Exception:
        return False
    return True


def device_count() -> int:
    """Trả về số lượng GPU NVIDIA CUDA khả dụng."""
    if is_available():
        try:
            return torch.cuda.device_count()
        except Exception:
            return 0
    return 0


def get_device_name(device: Optional[int] = None) -> Optional[str]:
    """Trả về tên card màn hình NVIDIA CUDA khả dụng."""
    if is_available():
        try:
            return torch.cuda.get_device_name(device if device is not None else 0)
        except Exception:
            return None
    return None


__all__ = [
    "is_available",
    "device_count",
    "get_device_name",
]
