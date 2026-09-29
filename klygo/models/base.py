"""
Lop nen tang truu tuong cho MOI mo hinh AI trong Klygo (klygo.models.base).
TANG 1: High-level API Interface — predict, benchmark, help, metadata, settings.
Khong quan ly phan cung. Hardware -> dung model.model truc tiep.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Any, Optional, List, Mapping

from box import Box

from .errors import InvalidStateError


class BaseModel(ABC):
    """
    TANG 1: High-level API wrapper cho mọi mô hình AI trong Klygo.
    Chỉ cung cấp: predict, benchmark, help, metadata, settings, guard.
    Mọi thứ phần cứng → dùng model.model trực tiếp.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        model_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> None:

        from . import utils
        utils.suppress_ai_warnings()

        self.state: str = "LOADING"
        self._runtime_metadata: ContextVar[Optional[Box]] = ContextVar(
            f"klygo_runtime_metadata_{id(self)}",
            default=None,
        )
        self._metadata = Box(utils.normalize_metadata(metadata))
        self.details = Box(self.metadata.get("details") or {})
        self.flags = tuple(utils.parameter_groups(self.metadata))
        self.model_id: str = str(model_id or self.metadata.get("model_id", "custom-model"))
        self._backend: Optional[str] = (
            backend or self.metadata.get("backend") or self.details.get("backend")
        )
        self.task: str = str(
            task or self.metadata.get("task") or self.details.get("task") or "Universal"
        )
        self.class_name: str = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self._default_settings: Dict[str, Any] = utils.parameter_groups(self.metadata)
        self._settings: Dict[str, Any] = utils.parameter_groups(self.metadata)
        self.state = "READY"

    @property
    def backend(self) -> str:
        """Dò tìm backend tự động từ self._backend, os.environ, self.model hoặc metadata."""
        if getattr(self, "_backend", None):
            return self._backend
        import os
        env_backend = os.environ.get("KLYGO_BACKEND")
        if env_backend:
            return env_backend
        model = getattr(self, "model", None)
        if model is not None:
            mod_cls = getattr(getattr(model, "__class__", None), "__module__", "")
            if "keras" in mod_cls or hasattr(model, "save_to_preset"):
                return "Keras"
            if "ultralytics" in mod_cls or hasattr(model, "predictor"):
                return "Ultralytics"
            if "transformers" in mod_cls or hasattr(model, "save_pretrained"):
                return "Hugging Face"
            if "torch" in mod_cls:
                return "PyTorch"
        return self.metadata.get("backend", "PyTorch")

    @backend.setter
    def backend(self, value: str) -> None:
        self._backend = value
        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            self.metadata["backend"] = value

    def suppress_warnings(self):
        """Context manager / Helper tắt mọi cảnh báo."""
        from . import utils
        return utils.suppress_warnings()

    # =========================================================================
    # GUARD: Chặn thao tác khi model đã UNLOADED
    # =========================================================================
    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if name.startswith('_') or not callable(attr) or isinstance(attr, type):
            return attr
        try:
            d = object.__getattribute__(self, '__dict__')
            state = d.get('state', 'READY')
            model_id = d.get('model_id', 'model')
        except AttributeError:
            return attr
        if state == 'UNLOADED' and name not in ('reset', 'unload', 'info', 'help', 'supports', 'methods'):
            raise InvalidStateError(
                "Mo hinh '{}' da bi UNLOADED. Khong the goi '{}'.".format(model_id, name)
            )
        return attr

    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    def supports(self, op_name: str) -> bool:
        d = object.__getattribute__(self, '__dict__')
        class_member = getattr(type(self), op_name, None)
        instance_member = d.get(op_name)
        return callable(class_member) or callable(instance_member)

    def has_flag(self, name: str) -> bool:
        """Return whether the model declares a parameter-group flag."""
        return str(name) in self.flags

    def methods(self) -> List[str]:
        cls_attrs = [m for m in dir(type(self)) if not m.startswith('_')]
        return sorted(m for m in cls_attrs if callable(getattr(type(self), m, None)))

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
        print("=" * 60)

    # =========================================================================
    # PROPERTIES: config, settings, params
    # =========================================================================
    @property
    def metadata(self) -> Box:
        """Return call-local runtime metadata or the model defaults."""
        runtime = self._runtime_metadata.get()
        return runtime if runtime is not None else self._metadata

    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None:
        from . import utils
        self._metadata = Box(utils.normalize_metadata(value))

    @contextmanager
    def _use_metadata(self, value: Mapping[str, Any]):
        """Expose runtime metadata through ``self.metadata`` for one call."""
        token = self._runtime_metadata.set(Box(value))
        try:
            yield
        finally:
            self._runtime_metadata.reset(token)

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




