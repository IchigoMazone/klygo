"""Configuration merge and update operations."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from box import Box

from ._utils import to_dict


def update(
    config_data: Mapping[str, Any] | Box,
    updates: Mapping[str, Any] | Box,
    deep: bool = True,
) -> Box:
    """Return a configuration updated from another mapping.

    Parameters
    ----------
    config_data : mapping or box.Box
        Base configuration. It is never modified.
    updates : mapping or box.Box
        Replacement values.
    deep : bool, default=True
        Recursively merge nested mappings. When ``False``, top-level values
        replace existing values in full.

    Returns
    -------
    box.Box
        Independent updated configuration.

    Examples
    --------
    >>> from klygo import config
    >>> config.update({"model": {"batch": 16}}, {"model": {"batch": 32}}).model.batch
    32
    """
    if not isinstance(deep, bool):
        raise TypeError("deep must be a bool")
    result = copy.deepcopy(to_dict(config_data))
    replacement = to_dict(updates, name="updates")
    for key, value in replacement.items():
        if deep and isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = update(result[key], value, deep=True).to_dict()
        else:
            result[key] = copy.deepcopy(value)
    return Box(result)


def merge(*configs: Mapping[str, Any] | Box, deep: bool = True) -> Box:
    """Merge configurations from left to right.

    Later configurations take precedence. Inputs remain unchanged.

    Parameters
    ----------
    configs : mapping or box.Box
        Zero or more configurations.
    deep : bool, default=True
        Recursively combine nested mappings.

    Returns
    -------
    box.Box
        Merged configuration. With no inputs, an empty ``Box`` is returned.

    Examples
    --------
    >>> from klygo import config
    >>> config.merge({"model": {"batch": 16}}, {"model": {"epochs": 10}}).model.epochs
    10
    """
    result = Box()
    for index, value in enumerate(configs):
        result = update(result, to_dict(value, name=f"configs[{index}]"), deep=deep)
    return result
