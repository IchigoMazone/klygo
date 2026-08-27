"""
Lop nen tang truu tuong cho MOI mo hinh AI trong Klygo (klygo.models.base).
TANG 1: PyTorch-Core Interface - Ke thua nn.Module lam goc, bo sung Klygo lifecycle
va co che khoa method (_unsupported), properties transparent sang Hugging Face.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Sequence, Set, List

import torch
import torch.nn as nn

from .errors import UnsupportedOperationError, InvalidStateError


def override(func):
    func.__is_override__ = True
    return func


class BaseModel(ABC, nn.Module):
    """
    TANG 1: Universal Abstract Interface cho moi mo hinh AI trong Klygo.
    Ke thua truc tiep tu PyTorch nn.Module (Core) -- moi method PyTorch deu la cong
    dan hang nhat, khong can viet proxy thu cong.
    Bo sung Klygo: Metadata, Blacklist __UNSUPPORTED__, Properties transparent sang HF.
    """

    __UNSUPPORTED__: Sequence[str] = ()

    def __init__(
        self,
        metadata: Dict[str, Any],
        unsupported: Optional[Union[Sequence[str], Set[str]]] = None,
        **kwargs,
    ) -> None:
        # nn.Module PHAI duoc khoi tao TRUOC MOI assignment
        nn.Module.__init__(self)

        self.state: str = "LOADING"
        self.metadata: Dict[str, Any] = dict(metadata)
        self.model_id: str = str(self.metadata.get("model_id", "custom-model"))
        self.backend: str = str(self.metadata.get("backend", "PyTorch"))
        self.task: str = str(self.metadata.get("task", "Universal"))
        self.class_name: str = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self._default_settings: Dict[str, Any] = dict(self.metadata.get("config", {}))
        self._settings: Dict[str, Any] = dict(self._default_settings)
        self._unsupported: Set[str] = set(unsupported or ())
        if hasattr(self, "__UNSUPPORTED__"):
            self._unsupported.update(getattr(self, "__UNSUPPORTED__"))
        self.state = "READY"

    def _inner_model(self) -> Optional[Any]:
        """Tra ve inner nn.Module (self.model) an toan qua _modules."""
        return self.__dict__.get("_modules", {}).get("model", None)

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

    def _check_supported(self, op_name: str) -> None:
        if self.state == "UNLOADED":
            raise InvalidStateError(
                "Mo hinh '{}' da bi UNLOADED. Khong the goi '{}'.".format(self.model_id, op_name)
            )
        if op_name in self._unsupported:
            raise UnsupportedOperationError(
                "Mo hinh '{}' ({}) khong ho tro thao tac '{}'.".format(
                    self.model_id, self.class_name, op_name
                )
            )

    def supports(self, op_name: str) -> bool:
        return hasattr(self, op_name) and (op_name not in self._unsupported)

    def methods(self) -> Dict[str, List[str]]:
        public = [m for m in dir(self) if not m.startswith("_") and callable(getattr(self, m))]
        return {
            "supported": [m for m in public if m not in self._unsupported],
            "unsupported": sorted(list(self._unsupported)),
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
        print("Settings    : " + str(self.settings))
        print("Unsupported : " + str(sorted(list(self._unsupported))))
        print("=" * 60)

    # =========================================================================
    # PROPERTIES: config, hf_config, settings
    # =========================================================================
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
    def hf_config(self) -> Any:
        """Truy cap truc tiep PretrainedConfig cua Hugging Face."""
        inner = self._inner_model()
        return getattr(inner, "config", None) if inner is not None else None

    @property
    def settings(self) -> Dict[str, Any]:
        """Cau hinh tham so runtime cua Klygo."""
        return getattr(self, "_settings", {})

    @settings.setter
    def settings(self, value: Dict[str, Any]) -> None:
        self._settings = dict(value)

    @property
    def default_config(self) -> Dict[str, Any]:
        return getattr(self, "_default_settings", {})

    @property
    def runtime_config(self) -> Dict[str, Any]:
        return self.settings

    # =========================================================================
    # OVERRIDE nn.Module METHODS DE THEM _check_supported GUARD
    # =========================================================================
    def eval(self) -> "BaseModel":
        """Che do Evaluation voi Klygo guard."""
        self._check_supported("eval")
        nn.Module.eval(self)
        return self

    def train(self, mode: bool = True, *args, **kwargs) -> Any:
        """
        Chuyen che do train PyTorch OR thuc thi pipeline huan luyen.
        train(True/False) -> chuyen che do nn.Module, KHONG can guard.
        train(dataloader, ...) -> pipeline huan luyen, CO guard unsupported.
        """
        if args or (kwargs and not set(kwargs.keys()).issubset({"mode"})):
            self._check_supported("train")
            raise NotImplementedError(
                "Model '{}' chua ho tro pipeline train() voi tham so nay.".format(self.model_id)
            )
        nn.Module.train(self, mode)
        return self

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Backward compat: state_dict cua inner model (khong co prefix 'model.')."""
        self._check_supported("state_dict")
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "state_dict"):
            return inner.state_dict(*args, **kwargs)
        return nn.Module.state_dict(self, *args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Nap trong so tu state_dict vao mo hinh."""
        self._check_supported("load_state_dict")
        inner = self._inner_model()
        if inner is not None and hasattr(inner, "load_state_dict"):
            return inner.load_state_dict(state_dict, strict=strict)
        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    # =========================================================================
    # __getattr__: Delegate sang self.model neu khong tim thay tren wrapper.
    # Thu tu: nn.Module.__getattr__ -> inner model (HF/PyTorch)
    # =========================================================================
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError("'{}' has no attribute '{}'".format(type(self).__name__, name))
        # 1. Thu nn.Module.__getattr__ truoc
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # 2. Delegate sang inner model
        inner = self.__dict__.get("_modules", {}).get("model", None)
        if inner is not None:
            try:
                return getattr(inner, name)
            except AttributeError:
                pass
        raise AttributeError("'{}' has no attribute '{}'".format(type(self).__name__, name))

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
    def val(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def benchmark(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def export(self, output_dir: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def help(self) -> None:
        raise NotImplementedError


# Alias tuong thich nguoc
DetectorModel = BaseModel
