"""
Lop nen tang truu tuong cho MOI mo hinh AI trong Klygo (klygo.models.base).
TANG 1: High-level API Interface — predict, benchmark, help, settings, flags.
Khong quan ly phan cung. Hardware -> dung model.model truc tiep.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Sequence, Set, List, Tuple

from .errors import UnsupportedOperationError, InvalidStateError


class BaseModel(ABC):
    """
    TANG 1: High-level API wrapper cho mọi mô hình AI trong Klygo.
    Chỉ cung cấp: predict, benchmark, help, settings, flags, guard.
    Mọi thứ phần cứng → dùng model.model trực tiếp.
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
        """Context manager / Helper tắt mọi cảnh báo."""
        from . import utils
        return utils.suppress_warnings()

    # =========================================================================
    # GUARD: Chặn unsupported + UNLOADED state
    # =========================================================================
    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if name.startswith('_') or not callable(attr) or isinstance(attr, type):
            return attr
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

    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    def unsupport(self, *operations: Union[str, Sequence[str]]) -> "BaseModel":
        for item in operations:
            if isinstance(item, (list, tuple, set)):
                self._unsupported.update(str(x) for x in item)
            else:
                self._unsupported.add(str(item))
        return self

    def supports(self, op_name: str) -> bool:
        d = object.__getattribute__(self, '__dict__')
        unsupported = d.get('_unsupported', set())
        if op_name in unsupported:
            return False
        return hasattr(type(self), op_name) or (op_name in d)

    def methods(self) -> Dict[str, List[str]]:
        d = object.__getattribute__(self, '__dict__')
        unsupported = d.get('_unsupported', set())
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
        print("Flags       : " + str(list(self.flags)))
        print("Settings    : " + str(self.settings))
        if self.params:
            print("Params      : " + ", ".join(f"{k}={v}" for k, v in self.params.items()))
        print("Unsupported : " + str(sorted(list(self._unsupported))))
        print("=" * 60)

    # =========================================================================
    # PROPERTIES: flags, config, settings, params
    # =========================================================================
    @property
    def flags(self) -> Tuple[str, ...]:
        return getattr(self, "_flags", ())

    def has_flag(self, flag: str) -> bool:
        return str(flag) in self.flags

    @property
    def params(self) -> Dict[str, Any]:
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
        """Trả về config của inner model nếu có, fallback về Klygo settings."""
        model = object.__getattribute__(self, "__dict__").get("model")
        if model is not None and hasattr(model, "config") and model.config is not None:
            return model.config
        return self._settings

    @config.setter
    def config(self, value: Any) -> None:
        if isinstance(value, dict):
            self._settings = dict(value)
        else:
            model = object.__getattribute__(self, "__dict__").get("model")
            if model is not None and hasattr(model, "config"):
                model.config = value
            else:
                self._settings = value

    @property
    def settings(self) -> Dict[str, Any]:
        return getattr(self, "_settings", {})

    @settings.setter
    def settings(self, value: Dict[str, Any]) -> None:
        self._settings = dict(value)

    # =========================================================================
    # HOP DONG (Abstract)
    # =========================================================================
    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def benchmark(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def help(self) -> None:
        raise NotImplementedError




