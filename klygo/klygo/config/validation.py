"""Configuration key and lightweight schema validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from box import Box

from .access import get, has
from .structure import flatten


def validate(
    config_data: Mapping[str, Any] | Box,
    required_keys: Sequence[str] | Mapping[str, Any] | None = None,
) -> bool:
    """Validate required paths and optional expected value contracts.

    A sequence requires each named path to exist. A mapping uses flattened
    paths as requirements; mapping leaf values may be types, tuples of types,
    predicates, or literal expected values.

    Parameters
    ----------
    config_data : mapping or box.Box
        Configuration to validate.
    required_keys : sequence, mapping, or None
        Required paths or lightweight schema.

    Returns
    -------
    bool
        ``True`` when validation succeeds.

    Raises
    ------
    ValueError
        If a required path is missing or fails its contract.
    TypeError
        If ``required_keys`` has an unsupported type.

    Examples
    --------
    >>> from klygo import config
    >>> config.validate({"model": {"batch": 16}}, {"model.batch": int})
    True
    """
    if required_keys is None:
        return True
    if isinstance(required_keys, Mapping):
        requirements = flatten(required_keys)
        for path, expected in requirements.items():
            if not has(config_data, path):
                raise ValueError(f"missing required config key: {path!r}")
            actual = get(config_data, path)
            if isinstance(expected, type) or (
                isinstance(expected, tuple) and expected and all(isinstance(item, type) for item in expected)
            ):
                if not isinstance(actual, expected):
                    raise ValueError(f"config key {path!r} must be {expected}, got {type(actual).__name__}")
            elif callable(expected):
                if not expected(actual):
                    raise ValueError(f"config key {path!r} failed validation")
            elif expected is not None and actual != expected:
                raise ValueError(f"config key {path!r} must equal {expected!r}, got {actual!r}")
        return True
    if isinstance(required_keys, Sequence) and not isinstance(required_keys, (str, bytes)):
        for path in required_keys:
            if not isinstance(path, str):
                raise TypeError("required key paths must be strings")
            if not has(config_data, path):
                raise ValueError(f"missing required config key: {path!r}")
        return True
    raise TypeError("required_keys must be a sequence, mapping, or None")
