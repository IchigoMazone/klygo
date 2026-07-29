from pathlib import Path
from typing import Any

from klygo.validators import validate_type

_SUPPORTED = {
    ".yaml", ".yml",
    ".json", ".jsonl",
    ".toml",
    ".csv",
    ".txt", ".log",
    ".ini", ".cfg", ".properties",
    ".env",
    ".xml",
    ".pkl", ".pickle",
}


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
        suf = path.suffix.lower()
        if not suf and path.name.lower().startswith(".env"):
            suf = ".env"

        if suf not in _SUPPORTED:
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

        suf = path.suffix.lower()
        if not suf and path.name.lower().startswith(".env"):
            suf = ".env"

        if suf not in _SUPPORTED:
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
