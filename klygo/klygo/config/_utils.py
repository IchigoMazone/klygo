"""Private normalization helpers shared by config API groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from box import Box


def to_dict(value: Mapping[str, Any] | Box, *, name: str = "config_data") -> dict[str, Any]:
    """Return a plain dictionary or raise a consistent public-facing error."""
    if isinstance(value, Box):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or Box, got {type(value).__name__}")
    return dict(value)


def resolve_key_path(key_path: str, sep: str = ".") -> list[str]:
    """Validate and split one nested configuration key path."""
    if not isinstance(key_path, str) or not key_path.strip():
        raise ValueError("key_path must be a non-empty string")
    if not isinstance(sep, str) or not sep:
        raise ValueError("sep must be a non-empty string")
    parts = [part for part in key_path.split(sep) if part]
    if not parts:
        raise ValueError("key_path must contain at least one key")
    return parts


def flatten_pairs(
    data: Mapping[str, Any],
    *,
    parent: str = "",
    sep: str = ".",
) -> list[tuple[str, Any]]:
    """Return leaf key/value pairs from a nested mapping."""
    pairs: list[tuple[str, Any]] = []
    for key, value in data.items():
        key_text = str(key)
        full_key = f"{parent}{sep}{key_text}" if parent else key_text
        if isinstance(value, Mapping) and value:
            pairs.extend(flatten_pairs(value, parent=full_key, sep=sep))
        else:
            pairs.append((full_key, value))
    return pairs
