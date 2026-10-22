"""Abstract base class for all AI models in Klygo (`klygo.models.base`).

LAYER 1: High-level API interface providing predict, benchmark, help, metadata, and settings.
Hardware management is deferred to the underlying backend model.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Any, Optional, List, Mapping

from box import Box

from .errors import InvalidStateError

__all__ = ["BaseModel"]


class BaseModel(ABC):
    """Abstract model wrapper with metadata, settings, and lifecycle guards.

    Subclasses implement ``predict``, ``benchmark``, and ``help``. Backend
    inference remains the responsibility of the concrete implementation.
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
        """Auto-detect backend from ``self._backend``, ``KLYGO_BACKEND`` env var, ``self.model``, or metadata."""
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
        """Delegate to :func:`klygo.models.utils.suppress_warnings` to silence noisy framework warnings."""
        from . import utils
        return utils.suppress_warnings()

    # =========================================================================
    # GUARD: Block public method calls after the model has been UNLOADED
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
                "Model '{}' is UNLOADED. Cannot call '{}'.".format(model_id, name)
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
        """Print model architecture, backend, state, and active settings in a Rich table."""
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Model Information: {self.model_id}")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_row("Class Name", str(self.class_name))
        table.add_row("Model ID", str(self.model_id))
        table.add_row("Backend", str(self.backend))
        table.add_row("Task", str(self.task))
        table.add_row("State", str(self.state))
        table.add_row("Flags", ", ".join(self.flags) if self.flags else "(None)")
        table.add_row("Settings", str(self.settings))
        if self.params:
            table.add_row("Parameters", ", ".join(f"{k}={v}" for k, v in self.params.items()))
        Console().print(table)

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
        """Return the inner model's ``config`` if available, otherwise fall back to Klygo ``_settings``."""
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
    # ABSTRACT CONTRACT
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




