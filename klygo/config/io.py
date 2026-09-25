"""Configuration file input, output, conversion, and export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from box import Box

from klygo import files

from ._utils import to_dict


def _expand_root_paths(data: dict[str, Any]) -> None:
    """Expand leading-dot strings relative to ``default.root`` in place."""
    default = data.get("default")
    if not isinstance(default, Mapping) or "root" not in default:
        return
    root = Path(str(default["root"]))

    def visit(value: dict[str, Any], *, inside_default: bool = False) -> None:
        for key, child in value.items():
            if key == "default" and value is data:
                continue
            if isinstance(child, dict):
                visit(child, inside_default=inside_default)
            elif isinstance(child, str) and child.startswith("./"):
                value[key] = str(root / child[2:])

    visit(data)


def load(path: str | Path, verbose: bool = True) -> Box:
    """Load a configuration file as a dot-accessible ``Box``.

    The serialization format is inferred by :func:`klygo.files.load`. When a
    mapping contains ``default.root``, string values beginning with ``./`` are
    expanded relative to that root without changing absolute paths.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing structured configuration file.
    verbose : bool, default=True
        Enable the underlying file progress indicator.

    Returns
    -------
    box.Box
        Loaded configuration with attribute and mapping access.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If the decoded document is not a mapping.

    Examples
    --------
    >>> from klygo import config
    >>> cfg = config.load("settings.yaml", verbose=False)
    >>> cfg.model.name
    'yolo'
    """
    decoded = files.load(path, verbose=verbose)
    if not isinstance(decoded, Mapping):
        raise TypeError("configuration root must be a mapping")
    data = dict(decoded)
    _expand_root_paths(data)
    return Box(data)


def save(
    path: str | Path,
    data: Mapping[str, Any] | Box,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Save configuration data using the destination extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Its extension selects the serializer.
    data : mapping or box.Box
        Configuration data to encode.
    overwrite : bool, default=False
        Replace an existing destination when true.
    verbose : bool, default=True
        Enable the underlying file progress indicator.

    Returns
    -------
    pathlib.Path
        Destination path.

    Examples
    --------
    >>> from klygo import config
    >>> config.save("settings.json", {"debug": False}, overwrite=True)
    """
    destination = files.path(path)
    files.save(destination, to_dict(data, name="data"), overwrite=overwrite, verbose=verbose)
    return destination


def convert(
    source: str | Path,
    target: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Convert a configuration file to another supported format.

    Parameters
    ----------
    source : str or pathlib.Path
        Existing input configuration.
    target : str or pathlib.Path
        Destination whose extension selects the output format.
    overwrite : bool, default=False
        Replace an existing destination.
    verbose : bool, default=True
        Enable progress indicators.

    Returns
    -------
    pathlib.Path
        Destination path.

    Examples
    --------
    >>> from klygo import config
    >>> config.convert("settings.yaml", "settings.toml", overwrite=True)
    """
    loaded = load(source, verbose=verbose)
    return save(target, loaded, overwrite=overwrite, verbose=verbose)


def export(
    source: str | Path | Mapping[str, Any] | Box,
    target: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Export a configuration object or file to a destination file.

    File inputs are converted through :func:`convert`; mapping and ``Box``
    inputs are serialized directly through :func:`save`.

    Examples
    --------
    >>> from klygo import config
    >>> config.export({"model": "yolo"}, "settings.yaml", overwrite=True)
    """
    if isinstance(source, (str, Path)):
        return convert(source, target, overwrite=overwrite, verbose=verbose)
    if isinstance(source, (Mapping, Box)):
        return save(target, source, overwrite=overwrite, verbose=verbose)
    raise TypeError("source must be a path, mapping, or Box")
