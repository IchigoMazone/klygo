import os
from pathlib import Path

from klygo import files
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


class ConfigSource:
    """Validate the file path for Config.__init__."""

    def __init__(self, config_path: str | os.PathLike[str]) -> None:
        validate_type(config_path, (str, os.PathLike), "config_path")
        resolved_path = files.path(config_path)

        if not files.exists(resolved_path):
            raise FileNotFoundError(f"config file does not exist: {resolved_path}")
        if not files.is_file(resolved_path):
            raise ValueError(
                f"config_path must be a file, got directory: {resolved_path}"
            )
        suf = files.extension(resolved_path).lower()
        if not suf and files.name(resolved_path).lower().startswith(".env"):
            suf = ".env"

        if suf not in _SUPPORTED:
            raise ValueError(
                f"unsupported config format: {files.extension(resolved_path)!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        self.config_path: Path = resolved_path


class ExportFile:
    """Validate parameters for Config.export_file."""

    def __init__(
        self,
        name: str,
        suffix: str,
        output_dir: str | os.PathLike[str] = ".",
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        validate_type(name, str, "name")
        validate_type(suffix, str, "suffix")
        validate_type(output_dir, (str, os.PathLike), "output_dir")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not suffix.startswith("."):
            suffix = f".{suffix}"

        if suffix.lower() not in _SUPPORTED:
            raise ValueError(
                f"unsupported export format: {suffix!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        resolved_dir = files.path(output_dir)
        if files.exists(resolved_dir) and not files.is_dir(resolved_dir):
            raise ValueError(f"output_dir must be a directory, got file: {resolved_dir}")

        self.name = name
        self.suffix = suffix
        self.output_dir: Path = resolved_dir
        self.overwrite = overwrite
        self.verbose = verbose
