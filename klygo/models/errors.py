"""
Hệ thống ngoại lệ chuẩn hóa cho Klygo AI Models Framework (klygo.models.errors).
"""


class ModelFrameworkError(Exception):
    """Ngoại lệ cơ sở cho toàn bộ hệ thống mô hình AI trong Klygo."""
    pass


class UnsupportedOperationError(ModelFrameworkError):
    """Ném ra khi gọi một phương thức/tính năng bị vô hiệu hóa trong __UNSUPPORTED__."""
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
