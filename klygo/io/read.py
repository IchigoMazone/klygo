import csv
import json
from pathlib import Path
from typing import Any, List, Dict, Union

import tomlkit
from ruamel.yaml import YAML

from klygo.utils.progress import ProgressBar
from klygo.validators.io import ReadFile

_yaml = YAML()


def _read_with_bar(path: Path, verbose: bool, desc: str, parser_func: Any) -> Any:
    """
    Thực hiện đọc file kèm thanh tiến trình cyan mượt.
    """
    with ProgressBar(total=1, desc=desc, unit="file", verbose=verbose, colour="cyan") as pbar:
        res = parser_func()
        pbar.update(1)
        return res


def read_yaml(path: Union[str, Path], verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file YAML (.yaml, .yml) trả về Dictionary hoặc List.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .yaml, .yml

    Đầu vào:
    - path [str | Path]: Đường dẫn file YAML cần đọc.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [Any] Dữ liệu cấu hình đã parse (Dict / List).

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi path không tồn tại.
    - ValueError: Phát sinh khi định dạng file không phải YAML hợp lệ.

    Ví dụ:
    >>> import klygo.io as io
    >>> data = io.read_yaml("config.yaml")
    >>> print(type(data))
    <class 'dict'>

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            return _yaml.load(f)

    return _read_with_bar(params.path, params.verbose, "Reading YAML", _parse)


def read_json(path: Union[str, Path], verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file JSON (.json) trả về Dictionary hoặc List.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .json

    Đầu vào:
    - path [str | Path]: Đường dẫn file JSON cần đọc.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [Any] Dữ liệu cấu hình/đối tượng đã parse (Dict / List).

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi path không tồn tại.
    - ValueError: Phát sinh khi nội dung JSON không hợp lệ.

    Ví dụ:
    >>> import klygo.io as io
    >>> data = io.read_json("data.json")

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _read_with_bar(params.path, params.verbose, "Reading JSON", _parse)


def read_jsonl(path: Union[str, Path], verbose: bool = True) -> List[Any]:
    """
    Tác dụng:
    - Đọc dữ liệu từ file JSON Lines (.jsonl) — mỗi dòng là một đối tượng JSON độc lập.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .jsonl

    Đầu vào:
    - path [str | Path]: Đường dẫn file JSONL cần đọc.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [List[Any]] Danh sách các đối tượng JSON parsed từ từng dòng.

    Ví dụ:
    >>> import klygo.io as io
    >>> lines = io.read_jsonl("dataset.jsonl")

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        records = []
        with open(params.path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(json.loads(line_str))
        return records

    return _read_with_bar(params.path, params.verbose, "Reading JSONL", _parse)


def read_toml(path: Union[str, Path], verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Đọc dữ liệu từ file cấu hình TOML (.toml) trả về Dictionary.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .toml

    Đầu vào:
    - path [str | Path]: Đường dẫn file TOML cần đọc.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [Any] Dữ liệu cấu hình đã parse.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi path không tồn tại.

    Ví dụ:
    >>> import klygo.io as io
    >>> cfg = io.read_toml("pyproject.toml")

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            return tomlkit.load(f).unwrap()

    return _read_with_bar(params.path, params.verbose, "Reading TOML", _parse)


def read_csv(path: Union[str, Path], verbose: bool = True) -> List[Dict[str, str]]:
    """
    Tác dụng:
    - Đọc file văn bản bảng biểu CSV (.csv) trả về danh sách các hàng dưới dạng Dictionary.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .csv

    Đầu vào:
    - path [str | Path]: Đường dẫn file CSV cần đọc.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [List[Dict[str, str]]] Danh sách các bản ghi (row) có key là tên cột (header).

    Ví dụ:
    >>> import klygo.io as io
    >>> rows = io.read_csv("labels.csv")

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    return _read_with_bar(params.path, params.verbose, "Reading CSV", _parse)


def read_txt(path: Union[str, Path], as_lines: bool = False, verbose: bool = True) -> Union[str, List[str]]:
    """
    Tác dụng:
    - Đọc file văn bản thuần (.txt, .log) dưới dạng chuỗi hoặc danh sách các dòng.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .txt, .log

    Đầu vào:
    - path [str | Path]: Đường dẫn file văn bản cần đọc.
    - as_lines [bool]: Trả về danh sách các dòng (List[str]) thay vì chuỗi đơn (str). Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [str | List[str]] Nội dung file văn bản.

    Ví dụ:
    >>> import klygo.io as io
    >>> text = io.read_txt("readme.txt")
    >>> lines = io.read_txt("log.txt", as_lines=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)

    def _parse():
        with open(params.path, "r", encoding="utf-8") as f:
            if as_lines:
                return [line.rstrip("\r\n") for line in f]
            return f.read()

    return _read_with_bar(params.path, params.verbose, "Reading TXT", _parse)


def read_file(path: Union[str, Path], verbose: bool = True) -> Any:
    """
    Tác dụng:
    - Tự động nhận diện định dạng từ đuôi file mở rộng và đọc dữ liệu tương ứng.

    Định dạng tương thích:
    - Đầu vào hỗ trợ: .yaml, .yml, .json, .jsonl, .toml, .csv, .txt, .log

    Đầu vào:
    - path [str | Path]: Đường dẫn file cần đọc.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [Any] Dữ liệu đã đọc tương ứng với loại file.

    Ngoại lệ:
    - ValueError: Phát sinh khi định dạng đuôi file không thuộc danh sách hỗ trợ.

    Ví dụ:
    >>> import klygo.io as io
    >>> data1 = io.read_file("config.yaml")
    >>> data2 = io.read_file("users.json")
    >>> data3 = io.read_file("dataset.jsonl")
    >>> data4 = io.read_file("settings.toml")

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = ReadFile(path=path, verbose=verbose)
    suffix = params.path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return read_yaml(params.path, verbose=params.verbose)
    elif suffix == ".json":
        return read_json(params.path, verbose=params.verbose)
    elif suffix == ".jsonl":
        return read_jsonl(params.path, verbose=params.verbose)
    elif suffix == ".toml":
        return read_toml(params.path, verbose=params.verbose)
    elif suffix == ".csv":
        return read_csv(params.path, verbose=params.verbose)
    elif suffix in (".txt", ".log"):
        return read_txt(params.path, verbose=params.verbose)

    raise ValueError(f"Unsupported extension: {suffix}")

