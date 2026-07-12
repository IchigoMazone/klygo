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
    """
    Tác dụng:
    - Thực hiện chức năng _write_with_bar

    Đầu vào:
    - path: Đường dẫn file
    - data: Dữ liệu cần xử lý
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình
    - desc: Tham số desc của hàm
    - writer_func: Tham số writer_func của hàm

    Đầu ra:
    - Không trả về dữ liệu

    Nguồn: TrinhNhuNhat_12072026.
    """
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
    """
    Tác dụng:
    - Ghi dữ liệu vào file YAML

    Đầu vào:
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
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        """
        Tác dụng:
        - Thực hiện chức năng _write

        Đầu vào:
        - Không có tham số đầu vào

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
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
    """
    Tác dụng:
    - Ghi dữ liệu vào file JSON

    Đầu vào:
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
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        """
        Tác dụng:
        - Thực hiện chức năng _write

        Đầu vào:
        - Không có tham số đầu vào

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
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
    """
    Tác dụng:
    - Ghi dữ liệu vào file TOML

    Đầu vào:
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
    params = WriteFile(
        path=path,
        data=data,
        overwrite=overwrite,
        verbose=verbose,
    )
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        """
        Tác dụng:
        - Thực hiện chức năng _write

        Đầu vào:
        - Không có tham số đầu vào

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
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
    """
    Tác dụng:
    - Ghi dữ liệu vào file cấu hình được hỗ trợ

    Đầu vào:
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
