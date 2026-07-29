import builtins
import copy
from functools import reduce
from operator import getitem
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

from box import Box

from klygo.files import load as _file_load, save as _file_save
from klygo.validators import validate_type
from klygo.validators.config import ConfigSource, ExportFile

_SUPPORTED_SUFFIXES = {".yaml", ".yml", ".json", ".jsonl", ".toml", ".csv", ".txt", ".log"}


# =========================================================================
# Helper Functions
# =========================================================================

def _resolve_key_path(key_path: str) -> List[str]:
    if not isinstance(key_path, str) or not key_path.strip():
        raise ValueError("key_path must be a non-empty string")
    return [k for k in key_path.replace("/", ".").split(".") if k]


def _traverse_expand_root(cfg: dict) -> None:
    keys_stack: list = []

    def _get_val(ks: list, tree: dict) -> Any:
        return reduce(getitem, ks, tree)

    def _assign_val(ks: list, tree: dict, val: Any) -> None:
        cur = reduce(getitem, ks[:-1], tree)
        cur[ks[-1]] = val

    def _visit(tree: dict) -> None:
        for key, value in tree.items():
            if key == "default":
                continue

            keys_stack.append(key)

            if len(keys_stack) > 1:
                current = _get_val(keys_stack, cfg)
                if isinstance(current, str) and current.startswith("."):
                    root = cfg["default"]["root"]
                    value = root + current[1:]
                    _assign_val(keys_stack, cfg, value)

            if isinstance(value, dict):
                _visit(value)

            keys_stack.pop()

    if "default" in cfg and isinstance(cfg["default"], dict) and "root" in cfg["default"]:
        _visit(cfg)


# =========================================================================
# 1. Load / Save / Convert
# =========================================================================

def load(path: Union[str, Path], verbose: bool = True) -> Box:
    """Đọc file cấu hình và trả về đối tượng Box hỗ trợ dot-notation."""
    params = ConfigSource(config_path=path)
    cfg_dict = dict(_file_load(params.config_path, verbose=verbose))
    _traverse_expand_root(cfg_dict)
    return Box(cfg_dict)


def save(
    path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Ghi dữ liệu cấu hình ra file theo định dạng đuôi file."""
    p = Path(path)
    _file_save(p, data, overwrite=overwrite, verbose=verbose)
    return p


def convert(
    source: Union[str, Path],
    target: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Chuyển đổi file cấu hình từ định dạng này sang định dạng khác."""
    cfg = load(source, verbose=verbose)
    return save(target, cfg.to_dict(), overwrite=overwrite, verbose=verbose)


# =========================================================================
# 2. Creation & Defaults
# =========================================================================

def defaults(default_data: Optional[Dict[str, Any]] = None) -> dict:
    """Lấy dictionary cấu hình mặc định."""
    base = {
        "default": {
            "root": "./data",
        },
        "model": {
            "name": "yolov8n",
            "epochs": 100,
            "batch": 16,
            "lr": 0.01,
        }
    }
    if default_data is None:
        return copy.deepcopy(base)

    return update(base, default_data, deep=True).to_dict()


def create(
    path: Union[str, Path],
    default_data: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Box:
    """Tạo file cấu hình mặc định."""
    data = defaults(default_data)
    save(path, data, overwrite=overwrite, verbose=verbose)
    return load(path, verbose=verbose)


# =========================================================================
# 3. Merging & Updating
# =========================================================================

def merge(*configs: dict, deep: bool = True) -> Box:
    """Gộp nhiều dictionary cấu hình thành một."""
    result: dict = {}
    for cfg in configs:
        if isinstance(cfg, Box):
            cfg = cfg.to_dict()
        result = update(result, cfg, deep=deep).to_dict()
    return Box(result)


def update(config_data: dict, updates: dict, deep: bool = True) -> Box:
    """Cập nhật dictionary cấu hình với các thông số mới."""
    if isinstance(config_data, Box):
        res = config_data.to_dict()
    else:
        res = copy.deepcopy(dict(config_data))

    if isinstance(updates, Box):
        upd = updates.to_dict()
    else:
        upd = dict(updates)

    for k, v in upd.items():
        if deep and k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = update(res[k], v, deep=True).to_dict()
        else:
            res[k] = copy.deepcopy(v)

    return Box(res)


# =========================================================================
# 4. Accessing & Modifying
# =========================================================================

def get(config_data: dict, key_path: str, default: Any = None) -> Any:
    """Truy xuất giá trị cấu hình theo đường dẫn dot-notation (vd: 'model.name')."""
    parts = _resolve_key_path(key_path)
    cur = config_data
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, Box) and hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return default
    return cur


def set(config_data: dict, key_path: str, value: Any) -> Box:
    """Gán giá trị cấu hình theo đường dẫn dot-notation (vd: 'model.batch', 32)."""
    parts = _resolve_key_path(key_path)
    if isinstance(config_data, Box):
        res = config_data.to_dict()
    else:
        res = dict(config_data)

    cur = res
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return Box(res)


def has(config_data: dict, key_path: str) -> bool:
    """Kiểm tra đường dẫn dot-notation có tồn tại trong cấu hình không."""
    parts = _resolve_key_path(key_path)
    cur = config_data
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, Box) and hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return False
    return True


def delete(config_data: dict, key_path: str) -> bool:
    """Xóa đường dẫn dot-notation khỏi cấu hình."""
    parts = _resolve_key_path(key_path)
    cur = config_data
    for part in parts[:-1]:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False

    if isinstance(cur, dict) and parts[-1] in cur:
        del cur[parts[-1]]
        return True
    return False


# =========================================================================
# 5. Keys / Values / Items
# =========================================================================

def _flatten(d: dict, parent_key: str = "") -> List[Tuple[str, Any]]:
    items_list: list = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict) and v:
            items_list.extend(_flatten(v, new_key))
        else:
            items_list.append((new_key, v))
    return items_list


def keys(config_data: dict, flat: bool = False) -> List[str]:
    """Danh sách các key trong cấu hình."""
    d = config_data.to_dict() if isinstance(config_data, Box) else dict(config_data)
    if not flat:
        return builtins.list(d.keys())
    return [k for k, _ in _flatten(d)]


def values(config_data: dict, flat: bool = False) -> List[Any]:
    """Danh sách các giá trị trong cấu hình."""
    d = config_data.to_dict() if isinstance(config_data, Box) else dict(config_data)
    if not flat:
        return builtins.list(d.values())
    return [v for _, v in _flatten(d)]


def items(config_data: dict, flat: bool = False) -> List[Tuple[str, Any]]:
    """Danh sách cặp (key, value) trong cấu hình."""
    d = config_data.to_dict() if isinstance(config_data, Box) else dict(config_data)
    if not flat:
        return builtins.list(d.items())
    return _flatten(d)


# =========================================================================
# 6. Validation & Export
# =========================================================================

def validate(config_data: dict, required_keys: Optional[Union[List[str], Dict[str, Any]]] = None) -> bool:
    """Kiểm tra cấu hình có chứa đầy đủ các key yêu cầu không."""
    if required_keys is None:
        return True

    if isinstance(required_keys, (builtins.list, tuple)):
        for req in required_keys:
            if not has(config_data, req):
                raise ValueError(f"Missing required config key: {req!r}")
    elif isinstance(required_keys, dict):
        for req_k in keys(required_keys, flat=True):
            if not has(config_data, req_k):
                raise ValueError(f"Missing required config key: {req_k!r}")

    return True


def export(
    source: Union[str, Path, dict, Box],
    target: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Xuất cấu hình ra file đích."""
    if isinstance(source, (str, Path)):
        return convert(source, target, overwrite=overwrite, verbose=verbose)
    elif isinstance(source, (dict, Box)):
        data = source.to_dict() if isinstance(source, Box) else source
        return save(target, data, overwrite=overwrite, verbose=verbose)
    else:
        raise TypeError("source must be a file path (str | Path) or dict | Box")
