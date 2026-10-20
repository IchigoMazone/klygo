"""Configuration collection views and comparisons."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

from box import Box

from ._utils import to_dict
from .io import load
from .structure import flatten


def keys(config_data: dict[str, Any] | Box, flat: bool = False) -> list[str]:
    """Return top-level or flattened configuration keys.

    Examples
    --------
    >>> from klygo import config
    >>> config.keys({"model": {"batch": 16}}, flat=True)
    ['model.batch']
    """
    if not isinstance(flat, bool):
        raise TypeError("flat must be a bool")
    data = to_dict(config_data)
    return builtins.list(flatten(data).keys() if flat else data.keys())


def values(config_data: dict[str, Any] | Box, flat: bool = False) -> list[Any]:
    """Return top-level or flattened configuration values.

    Examples
    --------
    >>> from klygo import config
    >>> config.values({"model": {"batch": 16}}, flat=True)
    [16]
    """
    if not isinstance(flat, bool):
        raise TypeError("flat must be a bool")
    data = to_dict(config_data)
    return builtins.list(flatten(data).values() if flat else data.values())


def items(config_data: dict[str, Any] | Box, flat: bool = False) -> list[tuple[str, Any]]:
    """Return top-level or flattened configuration items.

    Examples
    --------
    >>> from klygo import config
    >>> config.items({"model": {"batch": 16}}, flat=True)
    [('model.batch', 16)]
    """
    if not isinstance(flat, bool):
        raise TypeError("flat must be a bool")
    data = to_dict(config_data)
    return builtins.list(flatten(data).items() if flat else data.items())


def diff(
    config1: dict[str, Any] | Box | str | Path,
    config2: dict[str, Any] | Box | str | Path,
) -> dict[str, dict[str, Any]]:
    """Compare two configuration objects or files by flattened leaf path.

    Returns dictionaries named ``added``, ``removed``, and ``modified``.
    Modified entries contain ``from`` and ``to`` values.

    Examples
    --------
    >>> from klygo import config
    >>> config.diff({"x": 1}, {"x": 2})["modified"]
    {'x': {'from': 1, 'to': 2}}
    """
    first = load(config1, verbose=False) if isinstance(config1, (str, Path)) else config1
    second = load(config2, verbose=False) if isinstance(config2, (str, Path)) else config2
    left = flatten(to_dict(first, name="config1"))
    right = flatten(to_dict(second, name="config2"))
    return {
        "added": {key: value for key, value in right.items() if key not in left},
        "removed": {key: value for key, value in left.items() if key not in right},
        "modified": {
            key: {"from": left[key], "to": right[key]}
            for key in left
            if key in right and left[key] != right[key]
        },
    }
