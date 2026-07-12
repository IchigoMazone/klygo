from pathlib import Path
from klygo.validators import validate_type

_SUPPORTED_FORMATS = {".zip"}


def _check_zip_path(archive_path: Path) -> None:
    """
    Tác dụng:
    - Thực hiện chức năng _check_zip_path

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"archive_path does not exist: {archive_path}")
    if not archive_path.is_file():
        raise ValueError(
            f"archive_path must be a file, got directory: {archive_path}"
        )
    if archive_path.suffix.lower() not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported archive format: {archive_path.suffix!r}, "
            f"supported: {sorted(_SUPPORTED_FORMATS)}"
        )


class Compress:

    COMPRESS_FORMATS = {"zip"}
    SUPPORTED_FORMATS = {"zip": ".zip"}

    def __init__(
        self,
        source: str | Path,
        output_path: str | Path,
        format: str,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - source: File hoặc thư mục đầu vào
        - output_path: Đường dẫn file đầu ra
        - format: Tham số format của hàm
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
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
        if format not in self.COMPRESS_FORMATS:
            raise ValueError(
                f"unsupported format: {format!r}, "
                f"supported: {sorted(self.COMPRESS_FORMATS)}"
            )
        expected_suffix = self.SUPPORTED_FORMATS[format]
        if not str(output_path).lower().endswith(expected_suffix):
            raise ValueError(f"output suffix must be '{expected_suffix}'")

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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ
        - output_dir: Đường dẫn thư mục đầu ra
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        archive_path = Path(archive_path)
        output_dir = Path(output_dir)

        _check_zip_path(archive_path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        self.archive_path = archive_path
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.verbose = verbose


class ListFiles:

    def __init__(self, archive_path: str | Path) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        archive_path = Path(archive_path)
        _check_zip_path(archive_path)
        self.archive_path = archive_path


class ExtractFile:

    def __init__(
        self,
        archive_path: str | Path,
        filename: str,
        output_dir: str | Path,
        overwrite: bool,
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ
        - filename: Tham số filename của hàm
        - output_dir: Đường dẫn thư mục đầu ra
        - overwrite: Trạng thái cho phép ghi đè

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(filename, str, "filename")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")

        archive_path = Path(archive_path)
        output_dir = Path(output_dir)

        _check_zip_path(archive_path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        self.archive_path = archive_path
        self.filename = filename
        self.output_dir = output_dir
        self.overwrite = overwrite


class GetInfo:

    def __init__(self, archive_path: str | Path) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        archive_path = Path(archive_path)
        _check_zip_path(archive_path)
        self.archive_path = archive_path


class Test:

    def __init__(self, archive_path: str | Path) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        archive_path = Path(archive_path)
        _check_zip_path(archive_path)
        self.archive_path = archive_path


class Add:

    def __init__(
        self,
        archive_path: str | Path,
        files: "str | Path | list",
        verbose: bool,
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ
        - files: Tham số files của hàm
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(verbose, bool, "verbose")

        archive_path = Path(archive_path)
        _check_zip_path(archive_path)

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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ
        - files: Tham số files của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")

        archive_path = Path(archive_path)
        _check_zip_path(archive_path)

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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_paths: Danh sách đường dẫn file lưu trữ
        - output_path: Đường dẫn file đầu ra
        - overwrite: Trạng thái cho phép ghi đè

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(output_path, (str, Path), "output_path")
        validate_type(overwrite, bool, "overwrite")

        if not isinstance(archive_paths, list) or len(archive_paths) < 2:
            raise TypeError("archive_paths must be a list with at least 2 archives")

        output_path = Path(output_path)

        if not str(output_path).lower().endswith(".zip"):
            raise ValueError("output_path suffix must be '.zip'")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        normalized: list[Path] = []
        for s in archive_paths:
            validate_type(s, (str, Path), "archive_paths item")
            sp = Path(s)
            _check_zip_path(sp)
            normalized.append(sp)

        self.archive_paths = normalized
        self.output_path = output_path
        self.overwrite = overwrite


class Search:

    def __init__(self, archive_path: str | Path, pattern: str) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ
        - pattern: Tham số pattern của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(pattern, str, "pattern")

        archive_path = Path(archive_path)
        _check_zip_path(archive_path)

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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - archive_path: Đường dẫn file lưu trữ
        - size: Tham số size của hàm
        - output_dir: Đường dẫn thư mục đầu ra
        - overwrite: Trạng thái cho phép ghi đè

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(archive_path, (str, Path), "archive_path")
        validate_type(size, (int, float), "size")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(overwrite, bool, "overwrite")

        archive_path = Path(archive_path)
        output_dir = Path(output_dir)

        _check_zip_path(archive_path)

        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got file: {output_dir}")

        if size <= 0:
            raise ValueError(f"size must be a positive number, got {size}")

        self.archive_path = archive_path
        self.size = int(size * 1024 * 1024)
        self.output_dir = output_dir
        self.overwrite = overwrite
