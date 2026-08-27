"""
Lớp nền tảng trừu tượng thuần túy cho MỌI mô hình AI trong Klygo (klygo.models.base).
TẦNG 1: Pure Abstract Interface - Chỉ định nghĩa hợp đồng, quản lý Blacklist __UNSUPPORTED__,
vòng đời và soi chiếu mô hình (Introspection).
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Sequence, Set, List

from .errors import UnsupportedOperationError, InvalidStateError


def override(func):
    """Decorator đánh dấu phương thức ghi đè từ lớp cha (tùy chọn)."""
    func.__is_override__ = True
    return func


class BaseModel(ABC):
    """
    TẦNG 1: Universal Abstract Interface cho mọi mô hình AI trong Klygo.
    Chỉ quản lý Metadata, Config, Blacklist __UNSUPPORTED__ và các hợp đồng trừu tượng.
    """

    __UNSUPPORTED__: Sequence[str] = ()

    def __init__(
        self,
        metadata: Dict[str, Any],
        unsupported: Optional[Union[Sequence[str], Set[str]]] = None,
        **kwargs,
    ) -> None:
        self.state: str = "LOADING"

        # 1. Metadata & Định danh mô hình
        self.metadata: Dict[str, Any] = dict(metadata)
        self.model_id: str = str(self.metadata.get("model_id", "custom-model"))
        self.backend: str = str(self.metadata.get("backend", "PyTorch"))
        self.task: str = str(self.metadata.get("task", "Universal"))
        self.class_name: str = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        # 2. Cấu hình 2 tầng (default_config gốc & config runtime)
        self.default_config: Dict[str, Any] = dict(self.metadata.get("config", {}))
        self.config: Dict[str, Any] = dict(self.default_config)

        # 3. Quản lý tập Unsupported
        self._unsupported: Set[str] = set(unsupported or ())
        if hasattr(self, "__UNSUPPORTED__"):
            self._unsupported.update(getattr(self, "__UNSUPPORTED__"))

        self.state = "READY"

    # =========================================================================
    # QUẢN LÝ UNSUPPORTED & INTROSPECTION
    # =========================================================================
    def unsupport(self, *operations: Union[str, Sequence[str]]) -> "BaseModel":
        """Vô hiệu hóa thêm các tính năng lúc runtime."""
        for item in operations:
            if isinstance(item, (list, tuple, set)):
                self._unsupported.update(str(x) for x in item)
            else:
                self._unsupported.add(str(item))
        return self

    def _check_supported(self, op_name: str) -> None:
        """Kiểm tra quyền trước khi thực thi. Nếu bị unsupport -> Chặn ngay lập tức!"""
        if self.state == "UNLOADED":
            raise InvalidStateError(f"Mô hình '{self.model_id}' đã bị UNLOADED. Không thể gọi '{op_name}()'.")
        if op_name in self._unsupported:
            raise UnsupportedOperationError(
                f"Mô hình '{self.model_id}' ({self.class_name}) không hỗ trợ thao tác '{op_name}()'."
            )

    def supports(self, op_name: str) -> bool:
        """Kiểm tra tính năng có được hỗ trợ thực tế hay không."""
        return hasattr(self, op_name) and (op_name not in self._unsupported)

    def methods(self) -> Dict[str, List[str]]:
        """Báo cáo danh sách method theo nhóm (Supported vs Unsupported)."""
        public_methods = [m for m in dir(self) if not m.startswith("_") and callable(getattr(self, m))]
        return {
            "supported": [m for m in public_methods if m not in self._unsupported],
            "unsupported": sorted(list(self._unsupported)),
        }

    def info(self) -> None:
        """In bảng tóm tắt trực quan trạng thái và cấu hình mô hình."""
        print(f"{self.class_name}")
        print("=" * 60)
        print(f"Model ID    : {self.model_id}")
        print(f"Backend/Task: {self.backend} / {self.task}")
        print(f"State       : {self.state}")
        print(f"Device/Dtype: {getattr(self, 'device', 'cpu')} / {getattr(self, 'dtype', 'float32')}")
        print(f"Config      : {self.config}")
        print(f"Unsupported : {sorted(list(self._unsupported))}")
        print("=" * 60)

    # =========================================================================
    # HỢP ĐỒNG PHẦN CỨNG & ĐỘ CHÍNH XÁC (Pure Abstract Interface)
    # =========================================================================
    @property
    @abstractmethod
    def device(self) -> str:
        """Trả về tên thiết bị tính toán hiện tại (ví dụ: 'cpu', 'cuda:0')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> str:
        """Trả về kiểu dữ liệu độ chính xác hiện tại ('float32', 'float16')."""
        raise NotImplementedError

    @abstractmethod
    def to(self, device_name: str) -> "BaseModel":
        """Chuyển mô hình lên thiết bị tính toán được chỉ định."""
        raise NotImplementedError

    @abstractmethod
    def cpu(self) -> "BaseModel":
        """Chuyển mô hình về CPU."""
        raise NotImplementedError

    @abstractmethod
    def cuda(self) -> "BaseModel":
        """Chuyển mô hình lên GPU CUDA mặc định."""
        raise NotImplementedError

    @abstractmethod
    def half(self) -> "BaseModel":
        """Chuyển mô hình sang nửa độ chính xác FP16 (Half-Precision)."""
        raise NotImplementedError

    @abstractmethod
    def bfloat16(self) -> "BaseModel":
        """Chuyển mô hình sang độ chính xác Brain Floating Point 16 (BFLOAT16)."""
        raise NotImplementedError

    @abstractmethod
    def bfloat(self) -> "BaseModel":
        """Alias của bfloat16."""
        raise NotImplementedError

    @abstractmethod
    def float(self) -> "BaseModel":
        """Chuyển mô hình về độ chính xác chuẩn FP32 (Single-Precision)."""
        raise NotImplementedError

    # =========================================================================
    # HỢP ĐỒNG VÒNG ĐỜI & BỘ NHỚ (Pure Abstract Interface)
    # =========================================================================
    @abstractmethod
    def reset(self) -> "BaseModel":
        """Khôi phục cấu hình và trạng thái runtime về mặc định ban đầu."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        """Khởi động mô hình với dữ liệu giả lập."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Giải phóng hoàn toàn mô hình khỏi GPU/RAM."""
        raise NotImplementedError

    @abstractmethod
    def clear_cache(self) -> None:
        """Dọn dẹp cache GPU tạm thời."""
        raise NotImplementedError

    # =========================================================================
    # HỢP ĐỒNG VÒNG ĐỜI AI TOÀN DIỆN (Pure Abstract Interface)
    # =========================================================================
    @abstractmethod
    def train(self, *args, **kwargs):
        """Hợp đồng huấn luyện / fine-tuning."""
        raise NotImplementedError

    @abstractmethod
    def val(self, *args, **kwargs):
        """Hợp đồng đánh giá / kiểm định."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Hợp đồng suy luận chính của mô hình."""
        raise NotImplementedError

    @abstractmethod
    def benchmark(self, *args, **kwargs):
        """Hợp đồng đo đạc tốc độ suy luận (Latency ms / FPS)."""
        raise NotImplementedError

    @abstractmethod
    def export(self, output_dir: str) -> str:
        """Hợp đồng xuất mô hình thành thư mục Offline độc lập."""
        raise NotImplementedError

    @abstractmethod
    def help(self) -> None:
        """In hướng dẫn sử dụng của mô hình."""
        raise NotImplementedError


# Alias tương thích ngược
DetectorModel = BaseModel
