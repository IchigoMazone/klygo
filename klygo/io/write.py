from pathlib import Path
from typing import Any
import json

import tomlkit
from ruamel.yaml import YAML
from tqdm import tqdm

from klygo.validators.io import WriteFile

_yaml = YAML()


def _write_with_bar(
    path: Path,
    data: Any,
    overwrite: bool,
    verbose: bool,
    desc: str,
    writer_func: Any,
) -> None:
    bar = (
        tqdm(
            total=1,
            desc=desc,
            unit="file",
            colour="green",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        if verbose
        else None
    )

    try:
        writer_func()
        if bar is not None:
            bar.update(1)
    finally:
        if bar is not None:
            bar.close()


def write_yaml(
    path: str | Path,
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Write data to a YAML file.

    Parameters
    ----------
    path : str or Path
        Destination path (must end in ``.yaml`` or ``.yml``).
        Parent directories are created automatically.
    data : Any
        Data to serialise (usually a dict or list).
    overwrite : bool, optional
        If *True*, overwrite the file if it exists.
        Default is *False*.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``path`` has an unsupported extension.
    FileExistsError
        If ``path`` exists and ``overwrite`` is *False*.

    Examples
    --------
    >>> write_yaml("out/config.yaml", {"lr": 0.001, "epochs": 10})
    """
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            _yaml.dump(data, f)

    _write_with_bar(
        params.path,
        params.data,
        params.overwrite,
        params.verbose,
        "Writing YAML",
        _write,
    )


def write_json(
    path: str | Path,
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Write data to a JSON file.

    Parameters
    ----------
    path : str or Path
        Destination path (must end in ``.json``).
        Parent directories are created automatically.
    data : Any
        JSON-serialisable data.
    overwrite : bool, optional
        If *True*, overwrite the file if it exists.
        Default is *False*.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``path`` has an unsupported extension.
    FileExistsError
        If ``path`` exists and ``overwrite`` is *False*.

    Examples
    --------
    >>> write_json("out/config.json", {"lr": 0.001})
    """
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    _write_with_bar(
        params.path,
        params.data,
        params.overwrite,
        params.verbose,
        "Writing JSON",
        _write,
    )


def write_toml(
    path: str | Path,
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Write data to a TOML file.

    Parameters
    ----------
    path : str or Path
        Destination path (must end in ``.toml``).
        Parent directories are created automatically.
    data : Any
        Dict-like data to serialise as TOML.
    overwrite : bool, optional
        If *True*, overwrite the file if it exists.
        Default is *False*.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``path`` has an unsupported extension.
    FileExistsError
        If ``path`` exists and ``overwrite`` is *False*.

    Examples
    --------
    >>> write_toml("out/config.toml", {"lr": 0.001})
    """
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            tomlkit.dump(data, f)

    _write_with_bar(
        params.path,
        params.data,
        params.overwrite,
        params.verbose,
        "Writing TOML",
        _write,
    )


def write_file(
    path: str | Path,
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Write data to any supported config file, auto-detecting the format.

    Supported formats: ``.yaml``, ``.yml``, ``.json``, ``.toml``.

    Parameters
    ----------
    path : str or Path
        Destination path. Format is inferred from the file extension.
        Parent directories are created automatically.
    data : Any
        Data to write.
    overwrite : bool, optional
        If *True*, overwrite the file if it exists.
        Default is *False*.
    verbose : bool, optional
        If *True* (default), display a progress bar.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``path`` has an unsupported extension.
    FileExistsError
        If ``path`` exists and ``overwrite`` is *False*.

    Examples
    --------
    >>> write_file("out/config.json", {"lr": 0.001})   # writes JSON
    >>> write_file("out/params.toml", {"batch": 32})   # writes TOML
    """
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    suffix = params.path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        write_yaml(
            params.path,
            data,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )
    elif suffix == ".json":
        write_json(
            params.path,
            data,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )
    elif suffix == ".toml":
        write_toml(
            params.path,
            data,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )
