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

        # 2. Cấu hình runtime Klygo (settings) & cấu hình mặc định
        self._default_settings: Dict[str, Any] = dict(self.metadata.get("config", {}))
        self._settings: Dict[str, Any] = dict(self._default_settings)

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

    @property
    def config(self) -> Any:
        """Trả về PretrainedConfig (nếu là HF model) hoặc runtime settings dict."""
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "config") and model.config is not None:
            return model.config
        return getattr(self, "_settings", {})

    @config.setter
    def config(self, value: Any) -> None:
        if isinstance(value, dict):
            self._settings = dict(value)
        else:
            model = getattr(self, "model", None)
            if model is not None and hasattr(model, "config"):
                model.config = value
            else:
                self._settings = value

    @property
    def hf_config(self) -> Any:
        """Truy cập trực tiếp PretrainedConfig của Hugging Face."""
        model = getattr(self, "model", None)
        return getattr(model, "config", None) if model is not None else None

    @property
    def settings(self) -> Dict[str, Any]:
        """Cấu hình tham số runtime của Klygo."""
        return getattr(self, "_settings", {})

    @settings.setter
    def settings(self, value: Dict[str, Any]) -> None:
        self._settings = dict(value)

    @property
    def default_config(self) -> Dict[str, Any]:
        """Cấu hình mặc định gốc của Klygo."""
        return getattr(self, "_default_settings", {})

    @property
    def runtime_config(self) -> Dict[str, Any]:
        """Alias của settings."""
        return self.settings

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
    def to(self, device_name: Union[str, int]) -> "BaseModel":
        """Chuyển mô hình lên thiết bị tính toán được chỉ định (ví dụ: 'cpu', 'cuda:0', 'cuda:1', 1)."""
        raise NotImplementedError

    @abstractmethod
    def cpu(self) -> "BaseModel":
        """Chuyển mô hình về CPU."""
        raise NotImplementedError

    @abstractmethod
    def cuda(self, device: Optional[Union[int, str]] = None) -> "BaseModel":
        """Chuyển mô hình lên GPU CUDA chỉ định (ví dụ: 0, 1, 'cuda:1', hoặc None cho CUDA mặc định)."""
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
    # HỢP ĐỒNG PYTORCH MODULE & THAM SỐ (PyTorch Module Proxy Interface)
    # =========================================================================
    def parameters(self, recurse: bool = True):
        """Trả về iterator các tham số (torch.nn.Parameter) của mô hình."""
        self._check_supported("parameters")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "parameters"):
                return self.model.parameters(recurse=recurse)
            if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
                return self.model.model.parameters(recurse=recurse)
        return iter(())

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Trả về iterator các cặp (tên, tham số) của mô hình."""
        self._check_supported("named_parameters")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "named_parameters"):
                try:
                    return self.model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
                except TypeError:
                    return self.model.named_parameters(prefix=prefix, recurse=recurse)
            if hasattr(self.model, "model") and hasattr(self.model.model, "named_parameters"):
                try:
                    return self.model.model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
                except TypeError:
                    return self.model.model.named_parameters(prefix=prefix, recurse=recurse)
        return iter(())

    def buffers(self, recurse: bool = True):
        """Trả về iterator các bộ đệm (buffers) của mô hình."""
        self._check_supported("buffers")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "buffers"):
                return self.model.buffers(recurse=recurse)
            if hasattr(self.model, "model") and hasattr(self.model.model, "buffers"):
                return self.model.model.buffers(recurse=recurse)
        return iter(())

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Trả về iterator các cặp (tên, buffer) của mô hình."""
        self._check_supported("named_buffers")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "named_buffers"):
                try:
                    return self.model.named_buffers(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
                except TypeError:
                    return self.model.named_buffers(prefix=prefix, recurse=recurse)
            if hasattr(self.model, "model") and hasattr(self.model.model, "named_buffers"):
                try:
                    return self.model.model.named_buffers(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
                except TypeError:
                    return self.model.model.named_buffers(prefix=prefix, recurse=recurse)
        return iter(())

    def modules(self):
        """Trả về iterator qua toàn bộ các module trong mô hình."""
        self._check_supported("modules")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "modules"):
                return self.model.modules()
            if hasattr(self.model, "model") and hasattr(self.model.model, "modules"):
                return self.model.model.modules()
        return iter(())

    def named_modules(self, memo: Optional[Set[Any]] = None, prefix: str = "", remove_duplicate: bool = True):
        """Trả về iterator các cặp (tên, module) của mô hình."""
        self._check_supported("named_modules")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "named_modules"):
                try:
                    return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
                except TypeError:
                    return self.model.named_modules(memo=memo, prefix=prefix)
            if hasattr(self.model, "model") and hasattr(self.model.model, "named_modules"):
                try:
                    return self.model.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
                except TypeError:
                    return self.model.model.named_modules(memo=memo, prefix=prefix)
        return iter(())

    def children(self):
        """Trả về iterator các module con trực tiếp."""
        self._check_supported("children")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "children"):
                return self.model.children()
            if hasattr(self.model, "model") and hasattr(self.model.model, "children"):
                return self.model.model.children()
        return iter(())

    def named_children(self):
        """Trả về iterator các cặp (tên, module con trực tiếp)."""
        self._check_supported("named_children")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "named_children"):
                return self.model.named_children()
            if hasattr(self.model, "model") and hasattr(self.model.model, "named_children"):
                return self.model.model.named_children()
        return iter(())

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Trả về state_dict (weights & buffers) của mô hình PyTorch."""
        self._check_supported("state_dict")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "state_dict"):
                return self.model.state_dict(*args, **kwargs)
            if hasattr(self.model, "model") and hasattr(self.model.model, "state_dict"):
                return self.model.model.state_dict(*args, **kwargs)
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Nạp trọng số từ state_dict vào mô hình."""
        self._check_supported("load_state_dict")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "load_state_dict"):
                return self.model.load_state_dict(state_dict, strict=strict)
            if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                return self.model.model.load_state_dict(state_dict, strict=strict)
        raise AttributeError(f"Mô hình '{self.model_id}' không hỗ trợ 'load_state_dict'.")

    def eval(self) -> "BaseModel":
        """Chuyển mô hình sang chế độ Evaluation (eval)."""
        self._check_supported("eval")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "eval"):
                self.model.eval()
            elif hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
                self.model.model.eval()
        return self

    def train(self, mode: bool = True, *args, **kwargs) -> Any:
        """Chuyển mô hình sang chế độ Huấn luyện (train mode) hoặc thực hiện huấn luyện."""
        if args or (kwargs and not set(kwargs.keys()).issubset({"mode"})):
            self._check_supported("train")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "train"):
                self.model.train(mode)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "train"):
                self.model.model.train(mode)
    def zero_grad(self, set_to_none: bool = True) -> Any:
        """Xóa gradients của tất cả tham số mô hình."""
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "zero_grad"):
                return self.model.zero_grad(set_to_none=set_to_none)
            if hasattr(self.model, "model") and hasattr(self.model.model, "zero_grad"):
                return self.model.model.zero_grad(set_to_none=set_to_none)

    def requires_grad_(self, requires_grad: bool = True) -> "BaseModel":
        """Thay đổi thuộc tính requires_grad của toàn bộ tham số."""
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "requires_grad_"):
                self.model.requires_grad_(requires_grad)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "requires_grad_"):
                self.model.model.requires_grad_(requires_grad)
        return self

    def __getattr__(self, name: str) -> Any:
        """Ủy quyền truy xuất thuộc tính sang mô hình PyTorch bên dưới nếu không tìm thấy trên wrapper."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        model = self.__dict__.get("model", None)
        if model is not None and hasattr(model, name):
            return getattr(model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # =========================================================================
    # HỢP ĐỒNG VÒNG ĐỜI AI TOÀN DIỆN (Pure Abstract Interface)
    # =========================================================================
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
