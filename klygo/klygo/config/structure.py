"""Transform nested configuration structures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from box import Box

from ._utils import flatten_pairs, resolve_key_path, to_dict


def flatten(config_data: Mapping[str, Any] | Box, sep: str = ".") -> dict[str, Any]:
    """Flatten nested mappings into leaf paths.

    Empty dictionaries are retained as leaf values, allowing a subsequent
    :func:`unflatten` call to reconstruct the same structure.

    Parameters
    ----------
    config_data : mapping or box.Box
        Nested configuration.
    sep : str, default='.'
        Non-empty separator placed between path components.

    Returns
    -------
    dict
        Flat mapping of paths to leaf values.

    Examples
    --------
    >>> from klygo import config
    >>> config.flatten({"model": {"batch": 16}})
    {'model.batch': 16}
    """
    if not isinstance(sep, str) or not sep:
        raise ValueError("sep must be a non-empty string")
    return dict(flatten_pairs(to_dict(config_data), sep=sep))


def unflatten(flat_dict: Mapping[str, Any], sep: str = ".") -> dict[str, Any]:
    """Expand flat key paths into nested dictionaries.

    Raises ``ValueError`` when paths conflict, for example when both ``"a"``
    and ``"a.b"`` are present.

    Parameters
    ----------
    flat_dict : mapping
        Flat path/value mapping.
    sep : str, default='.'
        Non-empty path separator.

    Returns
    -------
    dict
        Nested configuration.

    Examples
    --------
    >>> from klygo import config
    >>> config.unflatten({"model.batch": 16})
    {'model': {'batch': 16}}
    """
    data = to_dict(flat_dict, name="flat_dict")
    result: dict[str, Any] = {}
    for key, value in data.items():
        parts = resolve_key_path(key, sep=sep)
        current = result
        for part in parts[:-1]:
            existing = current.get(part)
            if existing is not None and not isinstance(existing, dict):
                raise ValueError(f"conflicting configuration path: {key!r}")
            current = current.setdefault(part, {})
        if parts[-1] in current and isinstance(current[parts[-1]], dict):
            raise ValueError(f"conflicting configuration path: {key!r}")
        current[parts[-1]] = value
    return result
