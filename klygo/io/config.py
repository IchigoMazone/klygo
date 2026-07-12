from pathlib import Path
from functools import reduce
from operator import getitem
from typing import Any

from box import Box

from klygo.validators.io import ConfigSource, ExportFile
from klygo.io.read import read_file
from klygo.io.write import write_file


class Config:
    def __init__(self, config_path: str | Path) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - config_path: Đường dẫn file cấu hình

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        self._params = ConfigSource(config_path=config_path)
        self._keys: list = []
        self._cfg: dict = {}

    @property
    def config_path(self) -> Path:
        """
        Tác dụng:
        - Thực hiện chức năng config_path

        Đầu vào:
        - self: Đối tượng hiện tại

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        return self._params.config_path

    def _assign(self, keys: list, cfg: dict, value: Any | None = None) -> None:
        """
        Tác dụng:
        - Thực hiện chức năng _assign

        Đầu vào:
        - self: Đối tượng hiện tại
        - keys: Tham số keys của hàm
        - cfg: Tham số cfg của hàm
        - value: Tham số value của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        cur = reduce(getitem, keys[:-1], cfg)
        cur[keys[-1]] = value

    def _get_value(self, keys: list, cfg: dict) -> Any:
        """
        Tác dụng:
        - Thực hiện chức năng _get_value

        Đầu vào:
        - self: Đối tượng hiện tại
        - keys: Tham số keys của hàm
        - cfg: Tham số cfg của hàm

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        return reduce(getitem, keys, cfg)

    def _traverse(self, tree: dict) -> None:
        """
        Tác dụng:
        - Thực hiện chức năng _traverse

        Đầu vào:
        - self: Đối tượng hiện tại
        - tree: Tham số tree của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        for key, value in tree.items():
            if key == "default":
                continue

            self._keys.append(key)

            if len(self._keys) > 1:
                current = self._get_value(self._keys, self._cfg)
                if isinstance(current, str) and current.startswith("."):
                    root = self._cfg["default"]["root"]
                    value = root + current[1:]
                    self._assign(self._keys, self._cfg, value)

            if isinstance(value, dict):
                self._traverse(value)

            self._keys.pop()

    def read(self, verbose: bool = True) -> Box:
        """
        Tác dụng:
        - Đọc và xử lý dữ liệu cấu hình

        Đầu vào:
        - self: Đối tượng hiện tại
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        self._cfg = dict(read_file(self._params.config_path, verbose=verbose))

        if "default" in self._cfg and isinstance(self._cfg["default"], dict) and "root" in self._cfg["default"]:
            self._traverse(self._cfg)

        return Box(self._cfg)

    def imread(self, verbose: bool = True) -> Box:
        """
        Tác dụng:
        - Đọc dữ liệu cấu hình bằng tên hàm tương thích cũ

        Đầu vào:
        - self: Đối tượng hiện tại
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        import warnings
        warnings.warn(
            "`imread` is deprecated and will be removed in a future version. "
            "Use `read()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.read(verbose=verbose)

    def to_dict(self) -> dict[str, Any]:
        """
        Tác dụng:
        - Chuyển dữ liệu cấu hình thành dictionary

        Đầu vào:
        - self: Đối tượng hiện tại

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        return dict(self._cfg)

    def to_json(self, indent: int = 4) -> str:
        """
        Tác dụng:
        - Chuyển dữ liệu cấu hình thành chuỗi JSON

        Đầu vào:
        - self: Đối tượng hiện tại
        - indent: Tham số indent của hàm

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        import json
        return json.dumps(self._cfg, ensure_ascii=False, indent=indent)

    @staticmethod
    def create_default(
        path: str | Path,
        default_data: dict[str, Any] | None = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> "Config":
        """
        Tác dụng:
        - Tạo file cấu hình mặc định

        Đầu vào:
        - path: Đường dẫn file
        - default_data: Tham số default_data của hàm
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        if default_data is None:
            default_data = {
                "default": {
                    "root": "./data",
                },
                "model": {
                    "name": "yolov8n",
                    "epochs": 100,
                    "batch": 16,
                    "lr": 0.01,
                }
            }
        path = Path(path)
        write_file(path, default_data, overwrite=overwrite, verbose=verbose)
        return Config(path)

    def export_file(
        self,
        name: str,
        suffix: str,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Tác dụng:
        - Xuất cấu hình sang định dạng file khác

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
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        params = ExportFile(
            name=name,
            suffix=suffix,
            output_dir=output_dir if output_dir is not None else ".",
            overwrite=overwrite,
            verbose=verbose,
        )

        if params.suffix == self._params.config_path.suffix:
            raise ValueError(
                f"Export suffix {params.suffix!r} must be different "
                f"from source suffix {self._params.config_path.suffix!r}."
            )

        # Resolve output directory
        if output_dir is not None:
            out_dir = Path(output_dir)
        elif "default" in self._cfg and isinstance(self._cfg["default"], dict) and "root" in self._cfg["default"]:
            out_dir = Path(self._cfg["default"]["root"])
        else:
            out_dir = Path(".")

        file_path = out_dir / f"{params.name}{params.suffix}"
        data = read_file(self._params.config_path, verbose=False)
        write_file(
            file_path,
            data,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )
