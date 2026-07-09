from pathlib import Path
from typing import Any

from klygo.validators import validate_type

_SUPPORTED = {".yaml", ".yml", ".json", ".toml"}


class ReadFile:
    """Validate parameters for reading a config/data file."""

    def __init__(self, path: str | Path, verbose: bool = True) -> None:
        validate_type(path, (str, Path), "path")
        validate_type(verbose, bool, "verbose")
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"path must be a file, got directory: {path}")
        if path.suffix.lower() not in _SUPPORTED:
            raise ValueError(
                f"unsupported format: {path.suffix!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        self.path = path
        self.verbose = verbose


class WriteFile:
    """Validate parameters for writing a config/data file."""

    def __init__(
        self,
        path: str | Path,
        data: Any,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        validate_type(path, (str, Path), "path")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")
        path = Path(path)

        if path.suffix.lower() not in _SUPPORTED:
            raise ValueError(
                f"unsupported format: {path.suffix!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"file already exists: {path}. Use overwrite=True to replace it."
            )

        self.path = path
        self.data = data
        self.overwrite = overwrite
        self.verbose = verbose


class ConfigSource:
    """Validate the source path for Config.__init__."""

    def __init__(self, src: str | Path) -> None:
        validate_type(src, (str, Path), "src")
        src = Path(src)

        if not src.exists():
            raise FileNotFoundError(f"config file does not exist: {src}")
        if not src.is_file():
            raise ValueError(f"src must be a file, got directory: {src}")
        if src.suffix.lower() not in _SUPPORTED:
            raise ValueError(
                f"unsupported config format: {src.suffix!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        self.src = src


class ExportFile:
    """Validate parameters for Config.export_file."""

    def __init__(
        self,
        name: str,
        suffix: str,
        output_dir: str | Path = ".",
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        validate_type(name, str, "name")
        validate_type(suffix, str, "suffix")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not suffix.startswith("."):
            suffix = f".{suffix}"

        if suffix.lower() not in _SUPPORTED:
            raise ValueError(
                f"unsupported export format: {suffix!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        self.name = name
        self.suffix = suffix
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.verbose = verbose
