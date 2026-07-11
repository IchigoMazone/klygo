from pathlib import Path
from klygo.validators import validate_type

_SUPPORTED_FORMATS = {".zip"}


def _check_zip_source(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"source does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"source must be a file, got directory: {source}")
    if source.suffix.lower() not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported archive format: {source.suffix!r}, "
            f"supported: {sorted(_SUPPORTED_FORMATS)}"
        )


class Compress:

    COMPRESS_FORMATS = {"zip"}
    SUPPORTED_FORMATS = {"zip": ".zip"}

    def __init__(
        self,
        source: str | Path,
        output: str | Path,
        format: str,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(output, (str, Path), "output")
        validate_type(format, str, "format")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        source = Path(source)
        output = Path(output)

        if not source.exists():
            raise FileNotFoundError(f"source does not exist: {source}")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if format not in self.COMPRESS_FORMATS:
            raise ValueError(
                f"unsupported format: {format!r}, "
                f"supported: {sorted(self.COMPRESS_FORMATS)}"
            )
        expected_suffix = self.SUPPORTED_FORMATS[format]
        if not str(output).lower().endswith(expected_suffix):
            raise ValueError(f"output suffix must be '{expected_suffix}'")

        self.source = source
        self.output = output
        self.format = format
        self.overwrite = overwrite
        self.verbose = verbose


class Extract:

    def __init__(
        self,
        source: str | Path,
        output: str | Path,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(output, (str, Path), "output")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        source = Path(source)
        output = Path(output)

        _check_zip_source(source)

        self.source = source
        self.output = output
        self.overwrite = overwrite
        self.verbose = verbose


class ListFiles:

    def __init__(self, source: str | Path) -> None:
        validate_type(source, (str, Path), "source")
        source = Path(source)
        _check_zip_source(source)
        self.source = source


class ExtractFile:

    def __init__(
        self,
        source: str | Path,
        filename: str,
        output: str | Path,
        overwrite: bool,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(filename, str, "filename")
        validate_type(output, (str, Path), "output")
        validate_type(overwrite, bool, "overwrite")

        source = Path(source)
        output = Path(output)

        _check_zip_source(source)

        self.source = source
        self.filename = filename
        self.output = output
        self.overwrite = overwrite


class GetInfo:

    def __init__(self, source: str | Path) -> None:
        validate_type(source, (str, Path), "source")
        source = Path(source)
        _check_zip_source(source)
        self.source = source


class Test:

    def __init__(self, source: str | Path) -> None:
        validate_type(source, (str, Path), "source")
        source = Path(source)
        _check_zip_source(source)
        self.source = source


class Add:

    def __init__(
        self,
        source: str | Path,
        files: "str | Path | list",
        verbose: bool,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(verbose, bool, "verbose")

        source = Path(source)
        _check_zip_source(source)

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

        self.source = source
        self.files = normalized
        self.verbose = verbose


class Remove:

    def __init__(
        self,
        source: str | Path,
        files: "str | list[str]",
    ) -> None:
        validate_type(source, (str, Path), "source")

        source = Path(source)
        _check_zip_source(source)

        if isinstance(files, str):
            files = [files]
        elif not isinstance(files, list):
            raise TypeError(
                f"files must be str or list[str], got {type(files).__name__}"
            )
        for f in files:
            validate_type(f, str, "files item")

        self.source = source
        self.files: list[str] = files


class Merge:

    def __init__(
        self,
        sources: list,
        output: str | Path,
        overwrite: bool,
    ) -> None:
        validate_type(output, (str, Path), "output")
        validate_type(overwrite, bool, "overwrite")

        if not isinstance(sources, list) or len(sources) < 2:
            raise TypeError("sources must be a list with at least 2 archives")

        output = Path(output)

        if not str(output).lower().endswith(".zip"):
            raise ValueError("output suffix must be '.zip'")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")

        normalized: list[Path] = []
        for s in sources:
            validate_type(s, (str, Path), "sources item")
            sp = Path(s)
            _check_zip_source(sp)
            normalized.append(sp)

        self.sources = normalized
        self.output = output
        self.overwrite = overwrite


class Search:

    def __init__(self, source: str | Path, pattern: str) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(pattern, str, "pattern")

        source = Path(source)
        _check_zip_source(source)

        self.source = source
        self.pattern = pattern


class Split:

    def __init__(
        self,
        source: str | Path,
        size: int | float,
        output_dir: str | Path,
        overwrite: bool,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(size, (int, float), "size")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")

        source = Path(source)
        output_dir = Path(output_dir)

        _check_zip_source(source)

        if size <= 0:
            raise ValueError(f"size must be a positive number, got {size}")

        self.source = source
        self.size = int(size * 1024 * 1024)
        self.output_dir = output_dir
        self.overwrite = overwrite