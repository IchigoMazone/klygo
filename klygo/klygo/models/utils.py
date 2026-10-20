"""
Các hàm tiện ích và tiền xử lý dùng chung cho các mô hình AI (`klygo.models.utils`).
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Any, List, Dict, Union, Tuple, Optional, Mapping, Sequence
from copy import deepcopy
import PIL.Image
from box import Box

from klygo import files


import functools

def suppress_warnings(func=None):
    """
    Decorator hoặc Context Manager tắt warning của Python và các AI framework.
    """
    class SuppressContext:
        def __enter__(self):
            suppress_ai_warnings()
            self._ctx = warnings.catch_warnings()
            self._ctx.__enter__()
            warnings.filterwarnings("ignore")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)

        def __call__(self, fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self:
                    return fn(*args, **kwargs)
            return wrapper

    if func is not None:
        return SuppressContext()(func)
    return SuppressContext()


def suppress_ai_warnings() -> None:
    """
    Configure quiet AI-library logging without importing optional frameworks.

    Importing Torch or Transformers merely to change their logger would defeat
    Klygo's lazy optional-dependency contract. Python logging configuration is
    name-based, so these levels also apply if a framework is imported later.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    for logger_name in [
        "accelerate",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "transformers",
        "urllib3",
        "torch",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


RESERVED_METADATA_KEYS = frozenset(
    {
        "backend",
        "class",
        "config",
        "details",
        "flags",
        "model_id",
        "implementation",
        "num_params",
        "option",
        "priority",
        "parameter_groups",
        "profile",
        "revision",
        "task",
    }
)


def normalize_flags(value: Mapping[str, bool] | Sequence[str]) -> Dict[str, bool]:
    """Validate parameter groups and their predict-time override permissions.

    Mappings preserve an explicit permission for every group.  A sequence is
    accepted for compatibility and grants predict-time overrides to every
    declared group.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError("flags must be a mapping or a sequence of group names")

    items = value.items() if isinstance(value, Mapping) else ((name, True) for name in value)
    normalized: Dict[str, bool] = {}
    for raw_name, permission in items:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("parameter group names must not be empty")
        if name in RESERVED_METADATA_KEYS:
            raise ValueError(f"Reserved metadata key {name!r} cannot be a parameter group")
        if name in normalized:
            raise ValueError(f"Duplicate parameter group {name!r}")
        if not isinstance(permission, bool):
            raise TypeError(f"Flag permission for {name!r} must be a bool")
        normalized[name] = permission

    if not normalized:
        raise ValueError("At least one parameter group is required")
    return normalized


def normalize_implementation(value: Any) -> Optional[str]:
    """Return an import path for a model class declaration."""
    if value is None:
        return None
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise TypeError("implementation must be a class, non-empty import path, or None")


def normalize_metadata(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return canonical model metadata with parameter groups at the top level.

    The former ``config={...}`` wrapper is accepted for compatibility and
    flattened without overwriting explicitly supplied top-level groups.
    """
    result = deepcopy(dict(value or {}))
    legacy_profile = result.pop("profile", None)
    if legacy_profile is not None:
        if not isinstance(legacy_profile, Mapping):
            raise TypeError("metadata['profile'] must be a mapping")
        result.setdefault("details", deepcopy(dict(legacy_profile)))
    legacy_config = result.pop("config", None)
    if legacy_config is not None:
        if not isinstance(legacy_config, Mapping):
            raise TypeError("metadata['config'] must be a mapping")
        for group, parameters in legacy_config.items():
            result.setdefault(group, deepcopy(parameters))
    legacy_groups = result.pop("parameter_groups", None)
    if legacy_groups is not None:
        result.setdefault("flags", legacy_groups)
    if "flags" in result:
        result["flags"] = normalize_flags(result["flags"])
    if "class" in result:
        result["class"] = normalize_implementation(result["class"])
    return result


def parameter_groups(metadata: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return parameter mappings in the order declared by ``flags``."""
    discovered: Dict[str, Dict[str, Any]] = {}
    for name, parameters in metadata.items():
        if name in RESERVED_METADATA_KEYS:
            continue
        if isinstance(parameters, Mapping):
            discovered[str(name)] = deepcopy(dict(parameters))

    declaration = metadata.get("flags")
    if declaration is None:
        return discovered

    names = normalize_flags(declaration)
    undeclared = tuple(name for name in discovered if name not in names)
    if undeclared:
        raise ValueError(f"Metadata contains undeclared parameter groups: {undeclared}")
    return {name: discovered.get(name, {}) for name in names}


def runtime_groups(metadata: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return groups whose parameters may be overridden by ``predict``."""
    declaration = metadata.get("flags")
    if declaration is None:
        return tuple(parameter_groups(metadata))
    permissions = normalize_flags(declaration)
    return tuple(name for name, allowed in permissions.items() if allowed)


def normalize_priority(
    groups: Mapping[str, Any],
    value: Optional[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Validate the original parameter names accepted without group prefixes."""
    available = tuple(str(name) for name in groups)
    result: Dict[str, List[str]] = {}
    owners: Dict[str, str] = {}

    for raw_group, raw_names in dict(value or {}).items():
        group = str(raw_group)
        if group not in groups:
            raise ValueError(
                f"Priority group {group!r} is not present in metadata. "
                f"Available groups: {available}."
            )
        if isinstance(raw_names, str):
            names = (raw_names,)
        else:
            try:
                names = tuple(str(name) for name in raw_names)
            except TypeError as exc:
                raise TypeError(
                    f"Priority parameters for group {group!r} must be an iterable of names."
                ) from exc

        normalized = []
        for name in names:
            if not name or name.startswith("_") or name.endswith("_"):
                raise ValueError(f"Invalid priority parameter name {name!r}.")
            previous = owners.get(name)
            if previous is not None and previous != group:
                raise ValueError(
                    f"Priority parameter {name!r} belongs to multiple groups: "
                    f"{previous!r} and {group!r}."
                )
            owners[name] = group
            if name not in normalized:
                normalized.append(name)
        result[group] = normalized
    return result


def resolve_metadata(
    metadata: Mapping[str, Any],
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    runtime: bool = False,
) -> Box:
    """Merge call parameters into a copied metadata object.

    Priority parameters use their original names. Every other parameter must
    use ``<group>_<name>`` or be supplied as a mapping under the group name.
    Invalid and duplicate destinations raise ``ValueError`` instead of being
    discarded.
    """
    resolved = normalize_metadata(metadata)
    groups = parameter_groups(resolved)
    if not groups:
        if kwargs:
            raise ValueError("Model metadata does not define any parameter groups.")
        return Box(resolved)

    raw_priority = resolved.get("priority")
    priority = normalize_priority(groups, raw_priority)
    resolved["priority"] = priority if raw_priority is not None else None
    priority_owner = {
        parameter: group
        for group, parameters in priority.items()
        for parameter in parameters
    }
    destinations: Dict[Tuple[str, str], str] = {}
    group_names = sorted(groups, key=len, reverse=True)
    mutable_groups = set(runtime_groups(resolved)) if runtime else set(groups)

    def ensure_mutable(group: str) -> None:
        if group not in mutable_groups:
            raise ValueError(
                f"Parameter group {group!r} is locked at predict time. "
                f"Runtime-overridable groups: {tuple(runtime_groups(resolved))}."
            )

    for raw_name, value in dict(kwargs or {}).items():
        name = str(raw_name)
        if name in groups:
            ensure_mutable(name)
            if not isinstance(value, Mapping):
                raise TypeError(f"Parameter group {name!r} must be a mapping.")
            for parameter, parameter_value in value.items():
                destination = (name, str(parameter))
                previous = destinations.get(destination)
                if previous is not None:
                    raise ValueError(
                        f"Parameter {name}.{parameter} was provided more than once "
                        f"using {previous!r} and {name!r}."
                    )
                destinations[destination] = name
                groups[name][str(parameter)] = parameter_value
            continue

        owner = priority_owner.get(name)
        parameter = name
        source = name
        if owner is None:
            for group in group_names:
                prefix = f"{group}_"
                if name.startswith(prefix) and len(name) > len(prefix):
                    owner = group
                    parameter = name[len(prefix):]
                    break

        if owner is None:
            available = tuple(groups)
            raise ValueError(
                f"Parameter {name!r} is neither a priority parameter nor prefixed "
                f"with an available group. Available groups: {available}."
            )

        ensure_mutable(owner)

        destination = (owner, parameter)
        previous = destinations.get(destination)
        if previous is not None:
            raise ValueError(
                f"Parameter {owner}.{parameter} was provided more than once "
                f"using {previous!r} and {source!r}."
            )
        destinations[destination] = source
        groups[owner][parameter] = value

    for group, parameters in groups.items():
        resolved[group] = parameters
    return Box(resolved)

def resolve_images(
    source: Any,
    step: int = 1,
    max_frames: Optional[int] = None,
    stream: bool = False,
) -> Tuple[Union[List[PIL.Image.Image], Any], bool]:
    """
    Tự động phân giải nguồn dữ liệu đầu vào thông qua klygo.media.
    Hỗ trợ stream=True để trả về Generator (chống văng RAM khi đọc video lớn).
    """
    from klygo import media

    step = max(1, int(step))

    # Xử lý luồng Generator/Stream
    if stream:
        raw_stream = None
        total_frames = None
        fps_val = 30.0
        if isinstance(source, (str, Path)):
            raw_stream = media.stream(
                source,
                sample_rate=step,
                max_frames=max_frames,
                verbose=False,
            )
            is_single = raw_stream.source_type == "image"
            return raw_stream, is_single
        elif hasattr(source, "__iter__") and not isinstance(source, (list, tuple)):
            raw_stream = source
            total_frames = getattr(source, "total_frames", None)
            fps_val = getattr(source, "fps", 30.0)
        else:
            raw_stream = source if isinstance(source, (list, tuple)) else [source]
            total_frames = len(raw_stream)

        def _build_generator():
            count = 0
            for idx, item in enumerate(raw_stream):
                if max_frames is not None and count >= max_frames:
                    break
                if idx % step == 0:
                    count += 1
                    if isinstance(item, media.LazyImage):
                        yield item
                    else:
                        pil_img = media.to_pil(item).convert("RGB")
                        if hasattr(item, "path"):
                            pil_img.path = item.path
                        elif isinstance(item, (str, Path)):
                            pil_img.path = files.path(item)
                        yield pil_img

        is_single = False
        if isinstance(source, (str, Path)):
            is_single = files.extension(str(source)).lower() not in media.VIDEO_SUFFIXES
        elif isinstance(source, PIL.Image.Image):
            is_single = True

        return _build_generator(), is_single

    # Xử lý luồng List (RAM tiêu chuẩn) - Giữ nguyên logic cũ
    if isinstance(source, (str, Path)):
        loaded = media.load(source, verbose=False)
        raw_list = loaded if isinstance(loaded, list) else [loaded]
        is_single = len(raw_list) == 1 and files.extension(str(source)).lower() not in media.VIDEO_SUFFIXES
    elif isinstance(source, PIL.Image.Image):
        return [source.convert("RGB")], True
    elif hasattr(source, "shape"):
        if getattr(source, "ndim", 0) in (2, 3):
            return [media.to_pil(source).convert("RGB")], True
        elif getattr(source, "ndim", 0) == 4:
            batch_list = [media.to_pil(img).convert("RGB") for img in source]
            if step > 1:
                batch_list = batch_list[::step]
            if max_frames is not None and max_frames > 0:
                batch_list = batch_list[:max_frames]
            return batch_list, False
        raw_list = list(source)
        is_single = False
    elif isinstance(source, (list, tuple)):
        raw_list = list(source)
        is_single = len(raw_list) == 1
    elif hasattr(source, "__iter__"):
        raw_list = list(source)
        is_single = len(raw_list) == 1
    else:
        raise TypeError(f"Đầu vào '{type(source).__name__}' không hợp lệ.")

    if not raw_list:
        return [], is_single

    if step > 1:
        raw_list = raw_list[::step]
    if max_frames is not None and max_frames > 0:
        raw_list = raw_list[:max_frames]

    cleaned_images = []
    for item in raw_list:
        if isinstance(item, media.LazyImage):
            cleaned_images.append(item)
            continue
        pil_img = media.to_pil(item).convert("RGB")
        if hasattr(item, "path"):
            pil_img.path = item.path
        elif isinstance(item, (str, Path)):
            pil_img.path = files.path(item)
        cleaned_images.append(pil_img)

    return cleaned_images, is_single


def normalize_prompt(text_prompt: Union[str, List[str]]) -> List[str]:
    """Chuẩn hóa prompt nhãn từ khóa thành danh sách List[str]."""
    from klygo.validators import validate_type
    validate_type(text_prompt, (str, list, tuple), "prompt")
    if isinstance(text_prompt, str):
        return [text_prompt.strip()]
    return [str(p).strip() for p in text_prompt if str(p).strip()]
