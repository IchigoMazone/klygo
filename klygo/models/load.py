"""
Bộ phân giải và nạp mô hình AI Klygo (klygo.models.load).
"""

import os
import fnmatch
import importlib
import importlib.util
from typing import Dict, Any, Optional, Union
from functools import lru_cache

from klygo import files
from .base import BaseModel
from . import utils
from .detection.base import Detector
from .detection.grounding_dino import GroundingDinoDetect
from .detection.yolo import YOLODetect
from .detection.locate_anything import LocateAnythingDetect

CLASS_MAPPING = {
    "Detector": Detector,
    "GroundingDinoDetect": GroundingDinoDetect,
    "YOLODetect": YOLODetect,
    "LocateAnythingDetect": LocateAnythingDetect,
    "klygo.models.detection.Detector": Detector,
    "klygo.models.detection.base.Detector": Detector,
    "klygo.models.detection.GroundingDinoDetect": GroundingDinoDetect,
    "klygo.models.detection.YOLODetect": YOLODetect,
    "klygo.models.detection.LocateAnythingDetect": LocateAnythingDetect,
    "klygo.models.detection.grounding_dino.GroundingDinoDetect": GroundingDinoDetect,
    "klygo.models.detection.yolo.YOLODetect": YOLODetect,
    "klygo.models.detection.locate_anything.LocateAnythingDetect": LocateAnythingDetect,
}

_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def _get_registry() -> Dict[str, Any]:
    """Tải và lưu vào bộ nhớ đệm models.json để truy xuất tức thì O(1)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_json_path = os.path.join(current_dir, "models.json")
        _REGISTRY_CACHE = files.load(models_json_path, verbose=False)
    return _REGISTRY_CACHE


def register(name: str, cls: Any, metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Đăng ký một lớp mô hình hoặc metadata tùy chỉnh (Custom Model) vào hệ thống Klygo lúc Runtime trong RAM.
    """
    if isinstance(cls, dict):
        meta = dict(cls)
        meta.setdefault("model_id", name)
        resolved_cls_path = meta.get("class", "klygo.models.detection.Detector")
        actual_cls = _resolve_class(resolved_cls_path) if isinstance(resolved_cls_path, str) else resolved_cls_path
        if actual_cls:
            CLASS_MAPPING[name] = actual_cls
        registry = _get_registry()
        registry[name] = meta
        return

    class_full_name = f"{cls.__module__}.{cls.__qualname__}" if hasattr(cls, "__module__") else (cls.__name__ if hasattr(cls, "__name__") else str(cls))
    CLASS_MAPPING[name] = cls
    CLASS_MAPPING[class_full_name] = cls
    if hasattr(cls, "__name__"):
        CLASS_MAPPING[cls.__name__] = cls

    registry = _get_registry()
    meta = dict(metadata or {})
    meta.setdefault("class", class_full_name)
    meta.setdefault("task", getattr(cls, "task", "Object-Detection"))
    meta.setdefault("backend", "Custom")
    meta.setdefault("num_params", "Custom")
    meta.setdefault("model_id", name)
    if config:
        meta["config"] = config
    registry[name] = meta


def _resolve_class(class_path: str, search_dir: Optional[str] = None) -> Any:
    """Nạp động lớp mô hình từ đường dẫn module hoặc từ file .py cục bộ."""
    if class_path in CLASS_MAPPING:
        return CLASS_MAPPING[class_path]

    # 1. Nạp từ file model.py cục bộ nếu nằm trong thư mục model custom
    if search_dir and os.path.isdir(search_dir):
        py_files = [os.path.join(search_dir, "model.py")]
        if "." in class_path:
            mod_part = class_path.rsplit(".", 1)[0]
            py_files.append(os.path.join(search_dir, f"{mod_part}.py"))

        for py_path in py_files:
            if os.path.exists(py_path):
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
                    entry.update(override_cfg)
                    break

            if entry.get("num_params") is None:
                return None

            return entry
    return None


def load(model: Union[str, Any], **kwargs) -> BaseModel:
    """
    Nạp mô hình AI từ Online Hub Registry, Thư mục / File Offline cục bộ,
    hoặc nhận trực tiếp một instance mô hình PyTorch (torch.nn.Module).
    Tự động phân giải cấu hình 3 nhóm (model, processor, post).
    """
    # 0. Nhận trực tiếp BaseModel hoặc PyTorch nn.Module instance
    if isinstance(model, BaseModel):
        return model

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
            "task": getattr(model, "task", "Object-Detection"),
            "backend": "PyTorch (In-Memory)",
            "num_params": num_params,
            "model_id": getattr(model, "name", getattr(model, "__name__", model_cls_name)),
            "config": kwargs,
        }
        inst = Detector(metadata=final_metadata)
        inst.model = model
        inst.state = "READY"
        return inst

    entry = None
    search_dir = None

    # 1. Nạp từ thư mục Offline hoặc Thư mục Export
    if files.is_dir(model):
        abs_model_path = os.path.abspath(model)
        search_dir = abs_model_path
        klygo_path = os.path.join(abs_model_path, "klygo.json")
        config_path = os.path.join(abs_model_path, "config.json")

        if files.exists(klygo_path):
            entry = files.load(klygo_path, verbose=False)
            if isinstance(entry, dict) and not os.path.isabs(str(entry.get("model_id", ""))):
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
                        "task": "Object-Detection",
                        "backend": "Hugging Face (Offline)",
                        "num_params": "Offline",
                        "model_id": abs_model_path,
                    }

    # 2. Nạp từ file config .json trực tiếp
    elif files.is_file(model) and model.endswith(".json"):
        entry = files.load(model, verbose=False)
        search_dir = os.path.dirname(os.path.abspath(model))

    # 3. Nạp từ file trọng số YOLO .pt
    elif files.is_file(model) and model.endswith(".pt"):
        entry = {
            "class": "klygo.models.detection.YOLODetect",
            "task": "Object-Detection",
            "backend": "Ultralytics (Offline)",
            "num_params": "Offline",
            "model_id": os.path.abspath(model),
        }

    # 4. Tra cứu từ Registry Trực Tuyến
    if entry is None:
        entry = _resolve(model)

    if entry is None:
        raise ValueError(
            f"Mô hình '{model}' không tồn tại trong registry và không phải thư mục/file mô hình offline hợp lệ."
        )

    # 5. Phân giải và hợp nhất 3 nhóm cấu hình
    model_kwargs, processor_kwargs, post_kwargs = utils.resolve_sub_kwargs(
        kwargs=kwargs,
        json_config=entry.get("config"),
    )

    final_metadata = dict(entry)
    final_metadata["config"] = {
        "model": model_kwargs,
        "processor": processor_kwargs,
        "post": post_kwargs,
    }

    class_path = final_metadata.get("class")
    cls = _resolve_class(class_path, search_dir=search_dir)
    return cls(metadata=final_metadata)
