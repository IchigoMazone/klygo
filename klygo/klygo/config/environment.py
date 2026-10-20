"""Environment-variable configuration overlays."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from box import Box

from ._utils import to_dict
from .mapping import update
from .structure import unflatten


def _parse_env_value(value: str) -> Any:
    """Decode JSON-like scalar values and retain ordinary strings."""
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def from_env(
    config_data: Mapping[str, Any] | Box | None = None,
    prefix: str = "KLYGO_",
    sep: str = "_",
    parse_values: bool = False,
) -> dict[str, Any]:
    """Overlay matching environment variables onto a configuration.

    Environment names are stripped of ``prefix``, lower-cased, and split into
    nested paths using ``sep``. By default values remain strings, matching
    normal operating-system environment semantics. Set ``parse_values=True``
    to decode JSON scalars such as integers, booleans, lists, and ``null``.

    Parameters
    ----------
    config_data : mapping, box.Box, or None
        Optional base configuration.
    prefix : str, default='KLYGO_'
        Only variables beginning with this prefix are considered.
    sep : str, default='_'
        Environment-name hierarchy separator.
    parse_values : bool, default=False
        Decode JSON-compatible values when true.

    Returns
    -------
    dict
        New configuration with environment overrides.

    Examples
    --------
    >>> import os
    >>> from klygo import config
    >>> os.environ["APP_MODEL_BATCH"] = "32"
    >>> config.from_env(prefix="APP_", parse_values=True)["model"]["batch"]
    32
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(sep, str) or not sep:
        raise ValueError("sep must be a non-empty string")
    if not isinstance(parse_values, bool):
        raise TypeError("parse_values must be a bool")
    base = {} if config_data is None else to_dict(config_data)
    flat_updates: dict[str, Any] = {}
    for name, value in os.environ.items():
        if name.startswith(prefix):
            key = name[len(prefix):].lower().replace(sep, ".")
            if key:
                flat_updates[key] = _parse_env_value(value) if parse_values else value
    if not flat_updates:
        return base
    return update(base, unflatten(flat_updates), deep=True).to_dict()
