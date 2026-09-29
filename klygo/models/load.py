"""
Bộ phân giải và nạp mô hình AI Klygo (klygo.models.load).
"""

import os
import fnmatch
import importlib
import importlib.util
from typing import Dict, Any, Optional, Union, Mapping, Sequence
from functools import lru_cache
from box import Box

from klygo import files
from .base import BaseModel
from . import utils
from .detection.base import Detector

CLASS_MAPPING = {
    "Detector": Detector,
    "GroundingDinoDetect": "klygo.models.detection.grounding_dino.GroundingDinoDetect",
    "YOLODetect": "klygo.models.detection.yolo.YOLODetect",
    "LocateAnythingDetect": "klygo.models.detection.locate_anything.LocateAnythingDetect",
    "klygo.models.detection.Detector": Detector,
    "klygo.models.detection.base.Detector": Detector,
    "klygo.models.detection.GroundingDinoDetect": "klygo.models.detection.grounding_dino.GroundingDinoDetect",
    "klygo.models.detection.YOLODetect": "klygo.models.detection.yolo.YOLODetect",
    "klygo.models.detection.LocateAnythingDetect": "klygo.models.detection.locate_anything.LocateAnythingDetect",
    "klygo.models.detection.grounding_dino.GroundingDinoDetect": "klygo.models.detection.grounding_dino.GroundingDinoDetect",
    "klygo.models.detection.yolo.YOLODetect": "klygo.models.detection.yolo.YOLODetect",
    "klygo.models.detection.locate_anything.LocateAnythingDetect": "klygo.models.detection.locate_anything.LocateAnythingDetect",
}

_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def _get_registry() -> Dict[str, Any]:
    """Tải và lưu vào bộ nhớ đệm models.json để truy xuất tức thì O(1)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        current_dir = files.parent(files.resolve(__file__))
        models_json_path = files.join(current_dir, "models.json")
        _REGISTRY_CACHE = files.load(models_json_path, verbose=False)
    return _REGISTRY_CACHE


def _resolve_class(class_path: str, search_dir: Optional[str] = None) -> Any:
    """Nạp động lớp mô hình từ đường dẫn module hoặc từ file .py cục bộ."""
    if not isinstance(class_path, str) or not class_path.strip():
        raise ValueError(
            "Model metadata must define a non-empty 'class' import path."
        )
    if class_path in CLASS_MAPPING:
        target = CLASS_MAPPING[class_path]
        if not isinstance(target, str):
            return target
        module_path, class_name = target.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)

    # 1. Nạp từ file model.py cục bộ nếu nằm trong thư mục model custom
    if search_dir and files.is_dir(search_dir):
        py_files = [files.join(search_dir, "model.py")]
        if "." in class_path:
            mod_part = class_path.rsplit(".", 1)[0]
            py_files.append(files.join(search_dir, f"{mod_part}.py"))

        for py_path in py_files:
            if files.exists(py_path):
                try:
                    spec = importlib.util.spec_from_file_location("custom_model_module", py_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        class_name = class_path.rsplit(".", 1)[-1]
                        if hasattr(mod, class_name):
                            return getattr(mod, class_name)
                except Exception:
                    pass

    # 2. Nạp từ package/module chuẩn
    if "." in class_path:
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except (ImportError, AttributeError):
            pass

    raise KeyError(
        f"Lớp mô hình '{class_path}' không tồn tại hoặc không thể nạp. Các lớp hỗ trợ: {list(CLASS_MAPPING.keys())}"
    )


@lru_cache(maxsize=128)
def _resolve(name: str) -> Optional[Dict[str, Any]]:
    """Phân giải định danh mô hình với LRU Cache."""
    registry = _get_registry()
    entry = registry.get(name)
    if entry and "*" not in name:
        return dict(entry)

    for pattern, base_entry in registry.items():
        if fnmatch.fnmatch(name, pattern):
            entry = dict(base_entry)
            option = entry.pop("option", {})
            for key, override_cfg in option.items():
                if key in name:
                    override = dict(override_cfg)
                    if "details" in override:
                        merged_details = dict(entry.get("details", {}))
                        merged_details.update(dict(override.pop("details")))
                        entry["details"] = merged_details
                    entry.update(override)
                    break

            num_params = entry.get(
                "num_params",
                dict(entry.get("details", {})).get("num_params"),
            )
            if num_params is None:
                return None

            return entry
    return None


def _instantiate(
    entry: Mapping[str, Any],
    *,
    source_id: Any,
    overrides: Mapping[str, Any],
    search_dir: Optional[str] = None,
) -> BaseModel:
    """Configure one metadata entry and instantiate its declared class."""
    model_id = entry.get("model_id") or source_id
    configured = configure(
        str(model_id),
        metadata=entry,
        **dict(overrides),
    )
    cls = _resolve_class(configured.get("class"), search_dir=search_dir)
    with utils.suppress_warnings():
        return cls(metadata=configured)


def load(model: Union[str, Any], **kwargs) -> BaseModel:
    """Load a model from metadata, a configuration file, or a model source.

    ``model`` may be an existing :class:`BaseModel`, a metadata mapping, a
    :class:`klygo.config.Config`, a JSON/YAML/TOML metadata file, an offline
    model directory, a registered model name, a YOLO ``.pt`` file, or an
    in-memory module exposing ``parameters()``.

    Keyword arguments override parameter groups declared by the resolved
    metadata. Priority parameters keep their original names; all other
    parameters use ``<group>_<name>`` or an explicit group mapping.

    Parameters
    ----------
    model : object
        Model source or metadata definition.
    **kwargs : object
        Parameter overrides resolved according to the metadata's ``priority``
        declaration.

    Returns
    -------
    BaseModel
        The loaded model wrapper.

    Raises
    ------
    ValueError
        If the source cannot be resolved, its class is missing, or an override
        does not map to a declared parameter group.
    """
    # 0. Nhận trực tiếp BaseModel hoặc PyTorch nn.Module instance
    if isinstance(model, BaseModel):
        return model

    # 0.1. Nhận trực tiếp Box / Config object hoặc mapping metadata.
    if isinstance(model, Box):
        model = model.to_dict()

    from klygo.config import Config

    if isinstance(model, Config):
        data = model.to_dict()
        if not data:
            data = model.read(verbose=False).to_dict()
        model = data
    if isinstance(model, Mapping):
        entry = dict(model)
        return _instantiate(
            entry,
            source_id=entry.get("model_id") or "custom-model",
            overrides=kwargs,
        )

    try:
        import torch.nn as nn
        is_torch_module = isinstance(model, nn.Module)
    except Exception:
        is_torch_module = False

    if is_torch_module or (not isinstance(model, (str, os.PathLike)) and hasattr(model, "parameters")):
        model_cls_name = f"{model.__class__.__module__}.{model.__class__.__qualname__}" if hasattr(model, "__class__") else "CustomModule"
        num_params = "Custom"
        if hasattr(model, "parameters"):
            try:
                num_params = sum(p.numel() for p in model.parameters())
            except Exception:
                pass

        final_metadata = {
            "class": "klygo.models.detection.Detector",
            "model_id": getattr(model, "name", getattr(model, "__name__", model_cls_name)),
            "details": {
                "name": model_cls_name,
                "task": getattr(model, "task", "Object-Detection"),
                "backend": "PyTorch (In-Memory)",
                "library": "torch",
                "num_params": num_params,
            },
            "flags": {"model": False, "post": True},
            "model": dict(kwargs),
            "post": {},
            "priority": {},
        }
        inst = Detector(metadata=final_metadata)
        inst.model = model
        inst.state = "READY"
        return inst

    entry = None
    search_dir = None

    # 1. Nạp từ thư mục Offline hoặc Thư mục Export
    if files.is_dir(model):
        abs_model_path = str(files.resolve(model))
        search_dir = abs_model_path
        klygo_path = files.join(abs_model_path, "klygo.json")
        config_path = files.join(abs_model_path, "config.json")

        if files.exists(klygo_path):
            entry = files.load(klygo_path, verbose=False)
            if isinstance(entry, dict) and not files.is_absolute(str(entry.get("model_id", ""))):
                entry["model_id"] = abs_model_path
        elif files.exists(config_path):
            cfg = files.load(config_path, verbose=False)
            if isinstance(cfg, dict):
                if "class" in cfg:
                    entry = cfg
                elif "grounding_dino" in cfg.get("model_type", "") or "GroundingDino" in str(
                    cfg.get("architectures", [])
                ):
                    entry = {
                        "class": "klygo.models.detection.GroundingDinoDetect",
                        "model_id": abs_model_path,
                        "details": {
                            "name": "Grounding DINO",
                            "task": "Object-Detection",
                            "backend": "Hugging Face (Offline)",
                            "library": "transformers",
                            "num_params": "Offline",
                        },
                        "flags": {"model": False, "processor": True, "post": True},
                        "model": {},
                        "processor": {},
                        "post": {"threshold": 0.25, "text_threshold": 0.3},
                        "priority": {
                            "post": ("threshold", "text_threshold"),
                        },
                    }

    # 2. Nạp từ file config .json trực tiếp
    elif files.is_file(model) and files.extension(model).lower() in {
        ".json", ".yaml", ".yml", ".toml"
    }:
        entry = files.load(model, verbose=False)
        search_dir = str(files.parent(files.resolve(model)))

    # 3. Nạp từ file trọng số YOLO .pt
    elif files.is_file(model) and files.extension(model).lower() == ".pt":
        entry = {
            "class": "klygo.models.detection.YOLODetect",
            "model_id": str(files.resolve(model)),
            "details": {
                "name": "YOLO",
                "task": "Object-Detection",
                "backend": "Ultralytics (Offline)",
                "library": "ultralytics",
                "num_params": "Offline",
            },
            "flags": {"model": False, "post": True},
            "model": {},
            "post": {"threshold": 0.25, "iou": 0.7},
            "priority": {
                "post": ("threshold", "iou"),
            },
        }

    # 4. Tra cứu từ Registry Trực Tuyến
    if entry is None:
        entry = _resolve(model)

    if entry is None:
        raise ValueError(
            f"Mô hình '{model}' không tồn tại trong registry và không phải thư mục/file mô hình offline hợp lệ."
        )

    # 5. Mọi nguồn đều đi qua cùng pipeline configure -> instantiate.
    return _instantiate(
        entry,
        source_id=model,
        overrides=kwargs,
        search_dir=search_dir,
    )
def configure(
    model_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
    flags: Optional[Union[Mapping[str, bool], Sequence[str]]] = None,
    priority: Optional[Mapping[str, Any]] = None,
    details: Optional[Mapping[str, Any]] = None,
    implementation: Optional[Union[type, str]] = None,
    **kwargs,
) -> Box:
    """Bind a model identifier and parameter overrides to model metadata.

    Parameters
    ----------
    model_id : str
        Local path, hub identifier, or other identifier consumed by the model
        implementation.
    metadata : mapping, optional
        Definition created by :func:`klygo.models.metadata`. Legacy metadata
        containing a ``config`` wrapper is normalized automatically.
    flags : mapping of str to bool or sequence of str, optional
        Explicit parameter-group schema. Boolean values control predict-time
        overrides; configuration and loading may initialize every group.
    priority : mapping, optional
        Unprefixed parameter names grouped by their destination flag.
    details : mapping, optional
        Descriptive model information. The required fields are ``name``,
        ``task``, ``backend``, and ``library``.
    implementation : type, str, or None, optional
        Model implementation class or import path. It is serialized under the
        ``class`` key so the configured metadata can be passed to
        :func:`models.load`.
    **kwargs : object
        Parameter overrides. Priority names are unprefixed; other names use
        ``<group>_<parameter>`` or an explicit group mapping.

    Returns
    -------
    box.Box
        Independent, serializable metadata supporting both attribute and
        mapping access.

    Examples
    --------
    >>> group_flags = models.flags("model", "post")
    >>> definition = models.metadata(
    ...     details=models.details(
    ...         name="My detector",
    ...         task="Object-Detection",
    ...         backend="Custom",
    ...         library="custom",
    ...     ),
    ...     flags=group_flags,
    ...     priority=models.priority(
    ...         group_flags,
    ...         post=("threshold",),
    ...     ),
    ... )
    >>> result = models.configure(
    ...     "weights.pt", metadata=definition, threshold=0.4
    ... )
    >>> result.post.threshold
    0.4
    """
    base = utils.normalize_metadata(metadata)
    if flags is not None:
        base["flags"] = utils.normalize_flags(flags)
    if not utils.parameter_groups(base):
        # Backward-compatible default for callers that have not supplied a
        # metadata definition yet. New code should declare its actual groups.
        base.update({"model": {}, "processor": {}, "post": {}})
    if "flags" in base:
        base["flags"] = utils.normalize_flags(base["flags"])
    else:
        base["flags"] = {
            name: True for name in utils.parameter_groups(base)
        }
    base["model_id"] = str(model_id)
    if priority is not None:
        base["priority"] = utils.normalize_priority(
            utils.parameter_groups(base),
            priority,
        )
    if details is not None:
        from .metadata import metadata as build_metadata

        validated = build_metadata(
            base["flags"],
            base.get("priority"),
            details,
        )
        base["details"] = validated["details"]
    if implementation is not None:
        base["class"] = utils.normalize_implementation(implementation)
    else:
        base.setdefault("class", None)
    base.setdefault("details", None)
    base.setdefault("priority", None)
    return utils.resolve_metadata(base, kwargs)
def set_backend(backend: str, engine: Optional[str] = None) -> None:
    """
    Thiết lập backend mặc định cho Klygo (và engine tính toán cho Keras 3 nếu có).
    
    Ví dụ:
        models.set_backend("keras", engine="torch")
        models.set_backend("ultralytics")
        models.set_backend("huggingface")
    """
    os.environ["KLYGO_BACKEND"] = str(backend)
    if engine is not None:
        os.environ["KERAS_BACKEND"] = str(engine)


def get_backend() -> str:
    """Trả về backend hiện tại đang được cấu hình qua os.environ."""
    return os.environ.get("KLYGO_BACKEND", "auto")
