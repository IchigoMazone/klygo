from pathlib import Path
from klygo.validators import validate_type

_SUPPORTED_FORMATS = {
    ".zip", ".tar", ".gz", ".tgz", ".txz", ".7z", ".rar",
    ".tar.gz", ".tar.xz", ".tar.bz2", ".tbz2"
}


def _check_archive_path(archive_path: Path) -> None:
    """
    Check if archive path exists, is a file, and has a supported extension.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"archive_path does not exist: {archive_path}")
    if not archive_path.is_file():
        raise ValueError(
            f"archive_path must be a file, got directory: {archive_path}"
        )
    filename = archive_path.name.lower()
    has_valid_ext = any(filename.endswith(ext) for ext in _SUPPORTED_FORMATS)
    if not has_valid_ext:
        raise ValueError(
            f"unsupported archive format: {archive_path.suffix!r}, "
            f"supported: {sorted(_SUPPORTED_FORMATS)}"
        )


# Alias for backward compatibility
_check_zip_path = _check_archive_path


class Compress:

    COMPRESS_FORMATS = {
        "zip", "tar", "tar.gz", "tgz", "tar.xz", "txz", "tar.bz2", "tbz2", "7z", "gz"
    }

    def __init__(
        self,
        source: str | Path,
        output_path: str | Path,
        format: str,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(output_path, (str, Path), "output_path")
        validate_type(format, str, "format")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        source = Path(source)
        output_path = Path(output_path)

        if not source.exists():
            raise FileNotFoundError(f"source does not exist: {source}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        self.source = source
        self.output_path = output_path
        self.format = format
        self.overwrite = overwrite
        self.verbose = verbose


class Extract:

    def __init__(
        self,
        archive_path: str | Path,
        output_dir: str | Path,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        archive_path = Path(archive_path)
        output_dir = Path(output_dir)

        _check_archive_path(archive_path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        self.archive_path = archive_path
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.verbose = verbose


class ListFiles:

    def __init__(self, archive_path: str | Path) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        archive_path = Path(archive_path)
        _check_archive_path(archive_path)
        self.archive_path = archive_path


class ExtractFile:

    def __init__(
        self,
        archive_path: str | Path,
        filename: str,
        output_dir: str | Path,
        overwrite: bool,
    ) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(filename, str, "filename")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")

        archive_path = Path(archive_path)
        output_dir = Path(output_dir)

        _check_archive_path(archive_path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        self.archive_path = archive_path
        self.filename = filename
        self.output_dir = output_dir
        self.overwrite = overwrite


class GetInfo:

    def __init__(self, archive_path: str | Path) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        archive_path = Path(archive_path)
        _check_archive_path(archive_path)
        self.archive_path = archive_path


class Test:

    def __init__(self, archive_path: str | Path) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        archive_path = Path(archive_path)
        _check_archive_path(archive_path)
        self.archive_path = archive_path


class Add:

    def __init__(
        self,
        archive_path: str | Path,
        files: "str | Path | list",
        verbose: bool,
    ) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(verbose, bool, "verbose")

        archive_path = Path(archive_path)
        _check_archive_path(archive_path)

        if isinstance(files, (str, Path)):
            files = [files]
        elif not isinstance(files, list):
            raise TypeError(
                f"files must be str, Path, or list, got {type(files).__name__}"
            )

        normalized: list[Path] = []
        for f in files:
            validate_type(f, (str, Path), "files item")
            fp = Path(f)
            if not fp.exists():
                raise FileNotFoundError(f"file does not exist: {fp}")
            normalized.append(fp)

        self.archive_path = archive_path
        self.files = normalized
        self.verbose = verbose


class Remove:

    def __init__(
        self,
        archive_path: str | Path,
        files: "str | list[str]",
    ) -> None:
        validate_type(archive_path, (str, Path), "archive_path")

        archive_path = Path(archive_path)
        _check_archive_path(archive_path)

        if isinstance(files, str):
            files = [files]
        elif not isinstance(files, list):
            raise TypeError(
                f"files must be str or list[str], got {type(files).__name__}"
            )
        for f in files:
            validate_type(f, str, "files item")

        self.archive_path = archive_path
        self.files: list[str] = files


class Merge:

    def __init__(
        self,
        archive_paths: list,
        output_path: str | Path,
        overwrite: bool,
    ) -> None:
        validate_type(output_path, (str, Path), "output_path")
        validate_type(overwrite, bool, "overwrite")

        if not isinstance(archive_paths, list) or len(archive_paths) < 2:
            raise TypeError("archive_paths must be a list with at least 2 archives")

        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        normalized: list[Path] = []
        for s in archive_paths:
            validate_type(s, (str, Path), "archive_paths item")
            sp = Path(s)
            _check_archive_path(sp)
            normalized.append(sp)

        self.archive_paths = normalized
        self.output_path = output_path
        self.overwrite = overwrite


class Search:

    def __init__(self, archive_path: str | Path, pattern: str) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(pattern, str, "pattern")

        archive_path = Path(archive_path)
        _check_archive_path(archive_path)

        self.archive_path = archive_path
        self.pattern = pattern


class Split:

    def __init__(
        self,
        archive_path: str | Path,
        size: int | float,
        output_dir: str | Path,
        overwrite: bool,
    ) -> None:
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(size, (int, float), "size")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")

        archive_path = Path(archive_path)
        output_dir = Path(output_dir)

        _check_archive_path(archive_path)

        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        if size <= 0:
            raise ValueError(f"size must be a positive number, got {size}")

        self.archive_path = archive_path
        self.size = float(size)
        self.output_dir = output_dir
        self.overwrite = overwrite
