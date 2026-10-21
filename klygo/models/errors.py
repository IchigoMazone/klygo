"""Exceptions raised by model configuration and runtime operations."""

__all__ = [
    "ModelFrameworkError", "InvalidStateError", "InvalidConfigError",
    "InvalidDeviceError", "InvalidDtypeError",
]


class ModelFrameworkError(Exception):
    """Ngoại lệ cơ sở cho toàn bộ hệ thống mô hình AI trong Klygo."""
    pass


class InvalidStateError(ModelFrameworkError):
    """Ném ra khi gọi phương thức ở trạng thái không hợp lệ (ví dụ: gọi predict sau khi model đã unload)."""
    pass


class InvalidConfigError(ModelFrameworkError, ValueError):
    """Ném ra khi cấu hình truyền vào không hợp lệ."""
    pass


class InvalidDeviceError(ModelFrameworkError, ValueError):
    """Ném ra khi thiết bị phần cứng được chỉ định không hợp lệ."""
    pass


class InvalidDtypeError(ModelFrameworkError, ValueError):
    """Ném ra khi kiểu dữ liệu độ chính xác (dtype) không được hỗ trợ."""
    pass
