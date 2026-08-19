import os
import fnmatch
from typing import Dict, Any, Optional
from functools import lru_cache

from klygo import files
from .interfaces import DetectorModel
from .detection.grounding_dino import GroundingDinoDetect
from .detection.yolo import YOLODetect
from .detection.locate_anything import LocateAnythingDetect

CLASS_MAPPING = {
    "GroundingDinoDetect": GroundingDinoDetect,
    "YOLODetect": YOLODetect,
    "LocateAnythingDetect": LocateAnythingDetect,
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


@lru_cache(maxsize=128)
def _resolve(name: str) -> Optional[Dict[str, Any]]:
    """Phân giải định danh mô hình với LRU Cache để tăng tốc độ phân giải tên lặp lại."""
    registry = _get_registry()
    entry = registry.get(name)
    if entry and "*" not in name:
        return entry

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


def load(model: str) -> DetectorModel:
    """
    Tác dụng:
    - Nạp mô hình nhận diện đối tượng từ Registry trực tuyến hoặc từ thư mục Grounding DINO offline đã tải về.

    Đầu vào:
    - model [str]: Tên định danh mô hình (VD: 'grounding-dino-tiny') hoặc đường dẫn thư mục offline/exported của Grounding DINO.

    Đầu ra:
    - [DetectorModel]: Đối tượng mô hình nhận diện kế thừa từ lớp cơ sở DetectorModel.
    """
    entry = None

    # 1. Nạp từ thư mục Grounding DINO offline hoặc thư mục export
    if files.is_dir(model):
        abs_model_path = os.path.abspath(model)
        klygo_path = os.path.join(abs_model_path, "klygo.json")
        config_path = os.path.join(abs_model_path, "config.json")

        if files.exists(klygo_path):
            entry = files.load(klygo_path, verbose=False)
        elif files.exists(config_path):
            cfg = files.load(config_path, verbose=False)
            if isinstance(cfg, dict):
                # Thư mục xuất từ Klygo (legacy)
                if "class" in cfg:
                    entry = cfg
                # Thư mục mô hình Grounding DINO tải về từ Hugging Face
                elif "grounding_dino" in cfg.get("model_type", "") or "GroundingDino" in str(
                    cfg.get("architectures", [])
                ):
                    entry = {
                        "class": "GroundingDinoDetect",
                        "task": "Object-Detection",
                        "backend": "Hugging Face (Offline)",
                        "num_params": "Offline",
                        "model_id": abs_model_path,
                    }

    # 2. Nạp từ file config/klygo .json trực tiếp
    elif files.is_file(model) and model.endswith(".json"):
        entry = files.load(model, verbose=False)

    # 3. Tra cứu từ Registry
    if entry is None:
        entry = _resolve(model)

    if entry is None:
        raise ValueError(
            f"Mô hình '{model}' không tồn tại trong registry và không phải thư mục Grounding DINO offline hợp lệ."
        )

    entry_copy = dict(entry)
    class_name = entry_copy.pop("class")
    if class_name not in CLASS_MAPPING:
        raise KeyError(
            f"Lớp mô hình '{class_name}' không được hỗ trợ. Các lớp hỗ trợ: {list(CLASS_MAPPING.keys())}"
        )

    cls = CLASS_MAPPING[class_name]
    return cls(**entry_copy)
