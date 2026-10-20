from pathlib import Path
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

    def __init__(self, config_path: str | Path) -> None:
        validate_type(config_path, (str, Path), "config_path")
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"config file does not exist: {config_path}")
        if not config_path.is_file():
            raise ValueError(
                f"config_path must be a file, got directory: {config_path}"
            )
        suf = config_path.suffix.lower()
        if not suf and config_path.name.lower().startswith(".env"):
            suf = ".env"

        if suf not in _SUPPORTED:
            raise ValueError(
                f"unsupported config format: {config_path.suffix!r}, "
                f"supported: {sorted(_SUPPORTED)}"
            )

        self.config_path = config_path


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

        output_dir = Path(output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        self.name = name
        self.suffix = suffix
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.verbose = verbose
