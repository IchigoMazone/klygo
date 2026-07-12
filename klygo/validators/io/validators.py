from pathlib import Path
from typing import Any

from klygo.validators import validate_type

_SUPPORTED = {".yaml", ".yml", ".json", ".toml"}


class ReadFile:
    """Validate parameters for reading a config/data file."""

    def __init__(self, path: str | Path, verbose: bool = True) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - path: Đường dẫn file
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - path: Đường dẫn file
        - data: Dữ liệu cần xử lý
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
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
    """Validate the file path for Config.__init__."""

    def __init__(self, config_path: str | Path) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - config_path: Đường dẫn file cấu hình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(config_path, (str, Path), "config_path")
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"config file does not exist: {config_path}")
        if not config_path.is_file():
            raise ValueError(
                f"config_path must be a file, got directory: {config_path}"
            )
        if config_path.suffix.lower() not in _SUPPORTED:
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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - name: Tham số name của hàm
        - suffix: Tham số suffix của hàm
        - output_dir: Đường dẫn thư mục đầu ra
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
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
