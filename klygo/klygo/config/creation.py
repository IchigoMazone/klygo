"""Default configuration construction and file creation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from box import Box

from .io import load, save
from .mapping import update


_DEFAULT_CONFIG = {
    "default": {"root": "./data"},
    "model": {
        "name": "yolov8n",
        "epochs": 100,
        "batch": 16,
        "lr": 0.01,
    },
}


def defaults(default_data: Mapping[str, Any] | Box | None = None) -> dict[str, Any]:
    """Return a fresh copy of Klygo's default configuration.

    Parameters
    ----------
    default_data : mapping, box.Box, or None
        Optional deep overrides applied to the built-in defaults.

    Returns
    -------
    dict
        Independent dictionary safe for caller mutation.

    Examples
    --------
    >>> from klygo import config
    >>> config.defaults({"model": {"batch": 32}})["model"]["batch"]
    32
    """
    base = copy.deepcopy(_DEFAULT_CONFIG)
    if default_data is None:
        return base
    return update(base, default_data, deep=True).to_dict()


def create(
    path: str | Path,
    default_data: Mapping[str, Any] | Box | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Box:
    """Create and reload a configuration file from defaults.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination configuration file.
    default_data : mapping, box.Box, or None
        Optional deep overrides for :func:`defaults`.
    overwrite : bool, default=False
        Replace an existing file.
    verbose : bool, default=True
        Enable file progress indicators.

    Returns
    -------
    box.Box
        Configuration reloaded from the created file.

    Examples
    --------
    >>> from klygo import config
    >>> config.create("settings.yaml", overwrite=True, verbose=False)
    """
    save(path, defaults(default_data), overwrite=overwrite, verbose=verbose)
    return load(path, verbose=verbose)
