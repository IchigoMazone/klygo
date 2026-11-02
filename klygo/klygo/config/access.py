"""Nested key access and mutation helpers."""

from __future__ import annotations

import copy
from collections.abc import MutableMapping
from typing import Any

from box import Box

from ._utils import resolve_key_path, to_dict


def get(config_data: dict[str, Any] | Box, key_path: str, default: Any = None) -> Any:
    """Return a nested configuration value by dot-separated path.

    Parameters
    ----------
    config_data : dict or box.Box
        Configuration to inspect.
    key_path : str
        Dot-separated path such as ``"model.optimizer.lr"``.
    default : Any, default=None
        Value returned when the path does not exist.

    Returns
    -------
    Any
        Stored value or ``default`` when any path component is missing.

    Examples
    --------
    >>> from klygo import config
    >>> config.get({"model": {"batch": 16}}, "model.batch")
    16
    """
    current: Any = config_data
    for part in resolve_key_path(key_path):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, Box) and part in current:
            current = current[part]
        else:
            return default
    return current


def set(config_data: dict[str, Any] | Box, key_path: str, value: Any) -> Box:
    """Return a copied configuration with one nested value assigned.

    Missing intermediate dictionaries are created. The input mapping is not
    mutated, making the function safe for reusable defaults.

    Parameters
    ----------
    config_data : dict or box.Box
        Base configuration.
    key_path : str
        Dot-separated destination path.
    value : Any
        Value to store.

    Returns
    -------
    box.Box
        Updated independent configuration.

    Examples
    --------
    >>> from klygo import config
    >>> config.set({}, "model.batch", 32).model.batch
    32
    """
    result = copy.deepcopy(to_dict(config_data))
    current = result
    parts = resolve_key_path(key_path)
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(value)
    return Box(result)


def has(config_data: dict[str, Any] | Box, key_path: str) -> bool:
    """Return whether every component of a nested key path exists.

    Unlike checking the result of :func:`get`, this distinguishes a missing
    key from a key explicitly storing ``None``.

    Examples
    --------
    >>> from klygo import config
    >>> config.has({"model": {"name": None}}, "model.name")
    True
    """
    sentinel = object()
    return get(config_data, key_path, sentinel) is not sentinel


def delete(config_data: MutableMapping[str, Any] | Box, key_path: str) -> bool:
    """Delete one nested key in place.

    Parameters
    ----------
    config_data : mutable mapping or box.Box
        Configuration to modify.
    key_path : str
        Dot-separated path to remove.

    Returns
    -------
    bool
        ``True`` when a key was removed; ``False`` when the path was absent.

    Raises
    ------
    TypeError
        If ``config_data`` is not mutable.

    Examples
    --------
    >>> from klygo import config
    >>> data = {"model": {"batch": 16}}
    >>> config.delete(data, "model.batch")
    True
    """
    if not isinstance(config_data, (MutableMapping, Box)):
        raise TypeError("config_data must be a mutable mapping or Box")
    current: Any = config_data
    parts = resolve_key_path(key_path)
    for part in parts[:-1]:
        if not isinstance(current, (MutableMapping, Box)) or part not in current:
            return False
        current = current[part]
    if isinstance(current, (MutableMapping, Box)) and parts[-1] in current:
        del current[parts[-1]]
        return True
    return False
