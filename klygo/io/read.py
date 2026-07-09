from pathlib import Path
from typing import Any
import json

import tomlkit
from ruamel.yaml import YAML
from tqdm import tqdm

from klygo.validators.io import ReadFile

_yaml = YAML()


def _read_with_bar(path: Path, verbose: bool, desc: str, parser_func: Any) -> Any:
    bar = (
        tqdm(
            total=1,
            desc=desc,
            unit="file",
            colour="cyan",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        if verbose
        else None
    )

    try:
        res = parser_func()
        if bar is not None:
            bar.update(1)
        return res
    finally:
        if bar is not None:
            bar.close()


def read_yaml(path: str | Path, verbose: bool = True) -> Any:
    """Read a YAML or YML file and return its contents.

    Parameters
    ----------
    path : str or Path
        Path to the ``.yaml`` or ``.yml`` file.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    Any
        Parsed contents of the file (usually a dict or list).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a file or has an unsupported extension.

    Examples
    --------
    >>> data = read_yaml("config.yaml")
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            return _yaml.load(f)

    return _read_with_bar(params.path, params.verbose, "Reading YAML", _parse)


def read_json(path: str | Path, verbose: bool = True) -> Any:
    """Read a JSON file and return its contents.

    Parameters
    ----------
    path : str or Path
        Path to the ``.json`` file.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    Any
        Parsed contents of the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a file or has an unsupported extension.

    Examples
    --------
    >>> data = read_json("config.json")
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _read_with_bar(params.path, params.verbose, "Reading JSON", _parse)


def read_toml(path: str | Path, verbose: bool = True) -> Any:
    """Read a TOML file and return its contents.

    Parameters
    ----------
    path : str or Path
        Path to the ``.toml`` file.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    Any
        Parsed contents of the file (usually a dict).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a file or has an unsupported extension.

    Examples
    --------
    >>> data = read_toml("config.toml")
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            return tomlkit.load(f).unwrap()

    return _read_with_bar(params.path, params.verbose, "Reading TOML", _parse)


def read_file(path: str | Path, verbose: bool = True) -> Any:
    """Read any supported config file, auto-detecting the format from its extension.

    Supported formats: ``.yaml``, ``.yml``, ``.json``, ``.toml``.

    Parameters
    ----------
    path : str or Path
        Path to the config file.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    Any
        Parsed contents of the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a supported format.

    Examples
    --------
    >>> data = read_file("config.toml")   # format auto-detected
    >>> data = read_file("params.json")
    """
    params = ReadFile(path=path, verbose=verbose)
    suffix = params.path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return read_yaml(params.path, verbose=params.verbose)
    elif suffix == ".json":
        return read_json(params.path, verbose=params.verbose)
    elif suffix == ".toml":
        return read_toml(params.path, verbose=params.verbose)

    raise ValueError(f"Unsupported extension: {suffix}")
