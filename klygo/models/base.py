"""
Lop nen tang truu tuong cho MOI mo hinh AI trong Klygo (klygo.models.base).
TANG 1: PyTorch-Core Interface - Ke thua nn.Module lam goc, bo sung Klygo lifecycle
va co che khoa method (_unsupported), properties transparent sang Hugging Face.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Sequence, Set, List, Tuple

import torch
import torch.nn as nn

from .errors import UnsupportedOperationError, InvalidStateError




class BaseModel(ABC):
    """
    TANG 1: Universal Abstract Interface cho mọi mô hình AI trong Klygo.
    Độc lập hoàn toàn với PyTorch ở tầng Base, cho phép load ONNX, OpenVINO, hoặc API models.
    """

    __UNSUPPORTED__: Sequence[str] = ()

    def __init__(
        self,
        metadata: Dict[str, Any],
        flags: Sequence[str],
        unsupported: Optional[Union[Sequence[str], Set[str]]] = None,
        **kwargs,
    ) -> None:

        from . import utils
        utils.suppress_ai_warnings()

        self.state: str = "LOADING"
        self.metadata: Dict[str, Any] = dict(metadata)
        self.model_id: str = str(self.metadata.get("model_id", "custom-model"))
        self.backend: str = str(self.metadata.get("backend", "PyTorch"))
        self.task: str = str(self.metadata.get("task", "Universal"))
        self.class_name: str = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self._default_settings: Dict[str, Any] = dict(self.metadata.get("config", {}))
        self._settings: Dict[str, Any] = dict(self._default_settings)
        self._flags: Tuple[str, ...] = tuple(flags)
        self._unsupported: Set[str] = set(unsupported or ())
        if hasattr(self, "__UNSUPPORTED__"):
            self._unsupported.update(getattr(self, "__UNSUPPORTED__"))
        self.state = "READY"

    def suppress_warnings(self):
        """Context manager / Helper tắt mọi cảnh báo cho các tác vụ xử lý custom."""
        from . import utils
        return utils.suppress_warnings()

    def _inner_model(self) -> Optional[Any]:
        """Tra ve inner model (self.model) an toan qua _modules hoac __dict__."""
        inst_dict = object.__getattribute__(self, "__dict__")
        mod = inst_dict.get("_modules", {}).get("model", None)
        if mod is not None:
            return mod
        return inst_dict.get("model", None)

    # =========================================================================
    # QUAN LY UNSUPPORTED & INTROSPECTION
    # =========================================================================
    def unsupport(self, *operations: Union[str, Sequence[str]]) -> "BaseModel":
        for item in operations:
            if isinstance(item, (list, tuple, set)):
                self._unsupported.update(str(x) for x in item)
            else:
                self._unsupported.add(str(item))
        return self


    def __getattribute__(self, name: str) -> Any:
        """
        Universal guard: tu dong chan TẤT CA cac public method bi unsupported tai diem truy cap.
        Khong can them _check_supported() vao tung method nua — them method moi la tu dong duoc bao ve.
        """
        attr = super().__getattribute__(name)
        # Chi guard cac public callable method (khong phai dunder, khong phai attribute thuong)
        if name.startswith('_') or not callable(attr) or isinstance(attr, type):
            return attr
        # Doc internal state bang object.__getattribute__ de tranh goi de quy vao chinh no
        try:
            d = object.__getattribute__(self, '__dict__')
            state = d.get('state', 'READY')
            unsupported = d.get('_unsupported', set())
            model_id = d.get('model_id', 'model')
            class_name = d.get('class_name', '')
        except AttributeError:
            return attr
        if state == 'UNLOADED' and name not in ('reset', 'unload', 'info', 'help', 'supports', 'methods'):
            raise InvalidStateError(
                "Mo hinh '{}' da bi UNLOADED. Khong the goi '{}'.".format(model_id, name)
            )
        if name in unsupported:
            raise UnsupportedOperationError(
                "Mo hinh '{}' ({}) khong ho tro thao tac '{}'.".format(model_id, class_name, name)
            )
        return attr

    def supports(self, op_name: str) -> bool:
        d = object.__getattribute__(self, '__dict__')
        unsupported = d.get('_unsupported', set())
        if op_name in unsupported:
            return False
        return hasattr(type(self), op_name) or (op_name in d)

    def methods(self) -> Dict[str, List[str]]:
        d = object.__getattribute__(self, '__dict__')
        unsupported = d.get('_unsupported', set())
        # Bypass __getattribute__ guard khi lay list method (tranh raise luc introspect)
        cls_attrs = [m for m in dir(type(self)) if not m.startswith('_')]
        public = [m for m in cls_attrs if callable(getattr(type(self), m, None))]
        return {
            "supported": [m for m in public if m not in unsupported],
            "unsupported": sorted(list(unsupported)),
        }

    def info(self) -> None:
        print(self.class_name)
        print("=" * 60)
        print("Model ID    : " + self.model_id)
        print("Backend/Task: " + self.backend + " / " + self.task)
        print("State       : " + self.state)
        dev = getattr(self, "device", "cpu")
        dt = getattr(self, "dtype", "float32")
        print("Device/Dtype: " + str(dev) + " / " + str(dt))
        print("Flags       : " + str(list(self.flags)))
        print("Settings    : " + str(self.settings))
        if self.params:
            print("Params      : " + ", ".join(f"{k}={v}" for k, v in self.params.items()))
        print("Unsupported : " + str(sorted(list(self._unsupported))))
        print("=" * 60)

    # =========================================================================
    # PROPERTIES: flags, config, hf_config, settings, params
    # =========================================================================
    @property
    def flags(self) -> Tuple[str, ...]:
        """Danh sach cac co/nhom module cua mo hinh (vi du: ('model', 'processor', 'post') hoac ('backbone', 'head', 'nms'))."""
        return getattr(self, "_flags", ("model", "processor", "post"))

    def has_flag(self, flag: str) -> bool:
        """Kiem tra xem mo hinh co ho tro co/nhom module nay khong."""
        return str(flag) in self.flags

    @property
    def params(self) -> Dict[str, Any]:
        """
        Tong hop toan bo tham so dac thu co the tuy chinh khi goi predict().
        Tu dong quet moi nhom cau hinh trong settings.
        """
        s = getattr(self, "_settings", {})
        combined: Dict[str, Any] = {}
        for k, v in s.items():
            if isinstance(v, dict):
                combined.update(v)
            else:
                combined[k] = v
        return combined

    @property
    def config(self) -> Any:
        """PyTorch-Core-First: HF PretrainedConfig neu co, fallback ve Klygo settings."""
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "config") and inner.config is not None:
            return inner.config
        return self._settings

    @config.setter
    def config(self, value: Any) -> None:
        if isinstance(value, dict):
            self._settings = dict(value)
        else:
            inner = self._inner_model()
            if inner is not None and hasattr(inner, "config"):
                inner.config = value
            else:
                self._settings = value

    @property
    def settings(self) -> Dict[str, Any]:
        """Cau hinh tham so runtime cua Klygo."""
        return getattr(self, "_settings", {})

    @settings.setter
    def settings(self, value: Dict[str, Any]) -> None:
        self._settings = dict(value)

    @property
    def devices(self) -> List[str]:
        """Danh sách tất cả các devices mà tham số mô hình đang nằm trên đó."""
        dev_set = set()
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "parameters"):
            try:
                for p in inner.parameters():
                    dev_set.add(str(p.device))
            except Exception:
                pass
        return sorted(list(dev_set)) if dev_set else [str(getattr(self, "device", "cpu"))]

    def eval(self) -> "BaseModel":
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()
        return self

    def train(self, mode: bool = True, *args, **kwargs) -> Any:
        inner = self._inner_model()
        if args or (kwargs and not set(kwargs.keys()).issubset({"mode"})):
            if inner is not None and hasattr(inner, "train") and callable(inner.train):
                return inner.train(*args, **kwargs)
            raise NotImplementedError(
                f"Model '{self.model_id}' chưa hỗ trợ pipeline train() với tham số này."
            )
        if inner is not None and hasattr(inner, "train"):
            inner.train(mode)
        return self

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "state_dict"):
            return inner.state_dict(*args, **kwargs)
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "load_state_dict"):
            return inner.load_state_dict(state_dict, strict=strict)
        return None

    def to(self, *args, **kwargs) -> "BaseModel":
        """
        [ĐƯỜNG ỐNG TRONG SUỐT] Delegate hàm to() cho framework gốc.
        """
        if self._is_multi_gpu():
            return self

        inner = self._inner_model()
        if inner is not None and hasattr(inner, "to"):
            inner.to(*args, **kwargs)
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError("'{}' has no attribute '{}'".format(type(self).__name__, name))
        inner = self._inner_model()
        if inner is not None:
            try:
                return getattr(inner, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # =========================================================================
    # HOP DONG PHAN CUNG (Abstract)
    # =========================================================================
    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def benchmark(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def help(self) -> None:
        raise NotImplementedError

