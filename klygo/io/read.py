from pathlib import Path
from typing import Any
import json

import tomlkit
from ruamel.yaml import YAML
from tqdm import tqdm

from klygo.validators.io import ReadFile

_yaml = YAML()


def _read_with_bar(path: Path, verbose: bool, desc: str, parser_func: Any) -> Any:
    """
    Tác dụng:
    - Thực hiện chức năng _read_with_bar

    Đầu vào:
    - path: Đường dẫn file
    - verbose: Trạng thái hiển thị tiến trình
    - desc: Tham số desc của hàm
    - parser_func: Tham số parser_func của hàm

    Đầu ra:
    - Kết quả xử lý của hàm

    Nguồn: TrinhNhuNhat_12072026.
    """
    bar = (
        tqdm(
            total=1,
            desc=desc,
            unit="file",
            colour="cyan",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        if verbose
        else None
    )

    try:
        res = parser_func()
        if bar is not None:
            bar.update(1)
        return res
    finally:
        if bar is not None:
            bar.close()


def read_yaml(path: str | Path, verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file YAML

    Đầu vào:
    - path: Đường dẫn file
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        """
        Tác dụng:
        - Thực hiện chức năng _parse

        Đầu vào:
        - Không có tham số đầu vào

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        with open(params.path, "r", encoding="utf-8") as f:
            return _yaml.load(f)

    return _read_with_bar(params.path, params.verbose, "Reading YAML", _parse)


def read_json(path: str | Path, verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file JSON

    Đầu vào:
    - path: Đường dẫn file
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        """
        Tác dụng:
        - Thực hiện chức năng _parse

        Đầu vào:
        - Không có tham số đầu vào

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        with open(params.path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _read_with_bar(params.path, params.verbose, "Reading JSON", _parse)


def read_toml(path: str | Path, verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file TOML

    Đầu vào:
    - path: Đường dẫn file
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        """
        Tác dụng:
        - Thực hiện chức năng _parse

        Đầu vào:
        - Không có tham số đầu vào

        Đầu ra:
        - Kết quả xử lý của hàm

        Nguồn: TrinhNhuNhat_12072026.
        """
        with open(params.path, "r", encoding="utf-8") as f:
            return tomlkit.load(f).unwrap()

    return _read_with_bar(params.path, params.verbose, "Reading TOML", _parse)


def read_file(path: str | Path, verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file cấu hình được hỗ trợ

    Đầu vào:
    - path: Đường dẫn file
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = ReadFile(path=path, verbose=verbose)
    suffix = params.path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return read_yaml(params.path, verbose=params.verbose)
    elif suffix == ".json":
        return read_json(params.path, verbose=params.verbose)
    elif suffix == ".toml":
        return read_toml(params.path, verbose=params.verbose)

    raise ValueError(f"Unsupported extension: {suffix}")
