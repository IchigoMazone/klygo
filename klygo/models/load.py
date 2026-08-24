import os
import fnmatch
import importlib
import importlib.util
from typing import Dict, Any, Optional
from functools import lru_cache

from klygo import files
from . import base
from .detection.grounding_dino import GroundingDinoDetect
from .detection.yolo import YOLODetect
from .detection.locate_anything import LocateAnythingDetect

CLASS_MAPPING = {
    "GroundingDinoDetect": GroundingDinoDetect,
    "YOLODetect": YOLODetect,
    "LocateAnythingDetect": LocateAnythingDetect,
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


def register(name: str, cls: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Tác dụng:
    - Đăng ký một lớp mô hình nhận diện tùy chỉnh (Custom Model) vào hệ thống Klygo lúc Runtime.

    Đầu vào:
    - name [str]: Tên định danh gọi model (VD: 'my-custom-model').
    - cls [type]: Lớp mô hình kế thừa từ base.Detector.
    - metadata [Optional[dict]]: Cấu hình metadata tùy chọn.
    """
    class_full_name = f"{cls.__module__}.{cls.__qualname__}" if hasattr(cls, "__module__") else cls.__name__
    CLASS_MAPPING[name] = cls
    CLASS_MAPPING[class_full_name] = cls
    CLASS_MAPPING[cls.__name__] = cls

    registry = _get_registry()
    meta = dict(metadata or {})
    meta.setdefault("class", class_full_name)
    meta.setdefault("task", "Object-Detection")
    meta.setdefault("backend", "Custom")
    meta.setdefault("num_params", "Custom")
    meta.setdefault("model_id", name)
    registry[name] = meta


def _resolve_class(class_path: str, search_dir: Optional[str] = None) -> Any:
    """
    Nạp động lớp mô hình từ đường dẫn module hoặc từ file .py cục bộ trong thư mục model.
    """
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
    """Phân giải định danh mô hình với LRU Cache để tăng tốc độ phân giải tên lặp lại."""
    registry = _get_registry()
    entry = registry.get(name)
    if entry and "*" not in name:
        return dict(entry)

    for pattern, base_entry in registry.items():
        if fnmatch.fnmatch(name, pattern):
            entry = dict(base_entry)
            option = entry.pop("option", {})
            for key, override in option.items():
                if key in name:
                    entry.update(override)
                    break

            if entry.get("num_params") is None:
                return None

            return entry
    return None


def load(model: str, **kwargs) -> base.Detector:
    """
    Tác dụng:
    - Nạp mô hình nhận diện đối tượng từ Registry trực tuyến (Online) hoặc từ Thư mục / File trọng số cục bộ (Offline).

    Đầu vào:
    - model [str]: Tên định danh mô hình (VD: 'grounding-dino-tiny') hoặc đường dẫn thư mục offline/exported, file .pt.

    Đầu ra:
    - [Detector]: Đối tượng mô hình nhận diện kế thừa từ lớp cơ sở Detector (mặc định nạp trên CPU).
    """
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

    class_path = entry.get("class")
    cls = _resolve_class(class_path, search_dir=search_dir)
    return cls(metadata=entry, **kwargs)
