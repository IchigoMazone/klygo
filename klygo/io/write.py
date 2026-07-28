import csv
import json
from pathlib import Path
from typing import Any, List, Dict, Union, Optional

import tomlkit
from ruamel.yaml import YAML

from klygo.utils.progress import ProgressBar
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
    Thực hiện ghi file kèm thanh tiến trình cyan mượt.
    """
    with ProgressBar(total=1, desc=desc, unit="file", verbose=verbose, colour="cyan") as pbar:
        writer_func()
        pbar.update(1)


def write_yaml(
    path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Ghi dữ liệu vào file YAML (.yaml, .yml).

    Định dạng tương thích:
    - Đích hỗ trợ: .yaml, .yml

    Đầu vào:
    - path [str | Path]: Đường dẫn file YAML cần ghi.
    - data [Any]: Dữ liệu (Dict / List) cần xuất ra file.
    - overwrite [bool]: Cho phép ghi đè nếu file đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.io as io
    >>> io.write_yaml("config.yaml", {"name": "app", "version": 1}, overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            _yaml.dump(data, f)

    _write_with_bar(params.path, params.data, params.overwrite, params.verbose, "Writing YAML", _write)


def write_json(
    path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
    indent: int = 4,
) -> None:
    """
    Tác dụng:
    - Ghi dữ liệu vào file JSON (.json) với định dạng đẹp mắt (indentation).

    Định dạng tương thích:
    - Đích hỗ trợ: .json

    Đầu vào:
    - path [str | Path]: Đường dẫn file JSON cần ghi.
    - data [Any]: Dữ liệu cần xuất ra.
    - overwrite [bool]: Cho phép ghi đè. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.
    - indent [int]: Số khoảng trắng thò lùi đầu dòng. Mặc định: 4.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.io as io
    >>> io.write_json("data.json", {"items": [1, 2, 3]}, overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    _write_with_bar(params.path, params.data, params.overwrite, params.verbose, "Writing JSON", _write)


def write_jsonl(
    path: Union[str, Path],
    data: List[Any],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Ghi danh sách các đối tượng/dict vào file JSON Lines (.jsonl).

    Định dạng tương thích:
    - Đích hỗ trợ: .jsonl

    Đầu vào:
    - path [str | Path]: Đường dẫn file JSONL cần ghi.
    - data [List[Any]]: Danh sách các đối tượng/dict cần ghi từng dòng.
    - overwrite [bool]: Cho phép ghi đè. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.io as io
    >>> io.write_jsonl("dataset.jsonl", [{"id": 1}, {"id": 2}], overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    _write_with_bar(params.path, params.data, params.overwrite, params.verbose, "Writing JSONL", _write)


def write_toml(
    path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Ghi dữ liệu dictionary vào file cấu hình TOML (.toml).

    Định dạng tương thích:
    - Đích hỗ trợ: .toml

    Đầu vào:
    - path [str | Path]: Đường dẫn file TOML cần ghi.
    - data [Any]: Dữ liệu dictionary cấu hình.
    - overwrite [bool]: Cho phép ghi đè. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.io as io
    >>> io.write_toml("config.toml", {"title": "App"}, overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            tomlkit.dump(data, f)

    _write_with_bar(params.path, params.data, params.overwrite, params.verbose, "Writing TOML", _write)


def write_csv(
    path: Union[str, Path],
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Ghi danh sách các dictionary thành file bảng biểu CSV (.csv).

    Định dạng tương thích:
    - Đích hỗ trợ: .csv

    Đầu vào:
    - path [str | Path]: Đường dẫn file CSV cần ghi.
    - data [List[Dict[str, Any]]]: Danh sách các bản ghi dictionary.
    - fieldnames [List[str] | None]: Danh sách tên cột (header). Nếu None tự trích xuất từ phần tử đầu tiên.
    - overwrite [bool]: Cho phép ghi đè. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.io as io
    >>> rows = [{"name": "Alice", "score": 90}, {"name": "Bob", "score": 85}]
    >>> io.write_csv("results.csv", rows, overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    params.path.parent.mkdir(parents=True, exist_ok=True)

    if not fieldnames and data and isinstance(data[0], dict):
        fieldnames = list(data[0].keys())

    def _write():
        with open(params.path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or [])
            writer.writeheader()
            writer.writerows(data)

    _write_with_bar(params.path, params.data, params.overwrite, params.verbose, "Writing CSV", _write)


def write_txt(
    path: Union[str, Path],
    data: Union[str, List[str]],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Ghi chuỗi văn bản hoặc danh sách các dòng văn bản vào file (.txt, .log).

    Định dạng tương thích:
    - Đích hỗ trợ: .txt, .log

    Đầu vào:
    - path [str | Path]: Đường dẫn file văn bản cần ghi.
    - data [str | List[str]]: Chuỗi nội dung hoặc danh sách từng dòng.
    - overwrite [bool]: Cho phép ghi đè. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.io as io
    >>> io.write_txt("notes.txt", "Hello World!", overwrite=True)
    >>> io.write_txt("lines.txt", ["Line 1", "Line 2"], overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    params.path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        with open(params.path, "w", encoding="utf-8") as f:
            if isinstance(data, list):
                f.write("\n".join(str(item) for item in data) + "\n")
            else:
                f.write(str(data))

    _write_with_bar(params.path, params.data, params.overwrite, params.verbose, "Writing TXT", _write)


def write_file(
    path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Tự động nhận diện định dạng từ đuôi file mở rộng và ghi dữ liệu ra file tương ứng.

    Định dạng tương thích:
    - Đích hỗ trợ: .yaml, .yml, .json, .jsonl, .toml, .csv, .txt, .log

    Đầu vào:
    - path [str | Path]: Đường dẫn file cần xuất.
    - data [Any]: Dữ liệu cần xuất ra.
    - overwrite [bool]: Cho phép ghi đè nếu file đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - ValueError: Phát sinh khi định dạng đuôi file không thuộc danh sách hỗ trợ.

    Ví dụ:
    >>> import klygo.io as io
    >>> io.write_file("config.yaml", {"a": 1}, overwrite=True)
    >>> io.write_file("output.json", [1, 2, 3], overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    params = WriteFile(path=path, data=data, overwrite=overwrite, verbose=verbose)
    suffix = params.path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        write_yaml(params.path, data, overwrite=params.overwrite, verbose=params.verbose)
    elif suffix == ".json":
        write_json(params.path, data, overwrite=params.overwrite, verbose=params.verbose)
    elif suffix == ".jsonl":
        write_jsonl(params.path, data, overwrite=params.overwrite, verbose=params.verbose)
    elif suffix == ".toml":
        write_toml(params.path, data, overwrite=params.overwrite, verbose=params.verbose)
    elif suffix == ".csv":
        write_csv(params.path, data, overwrite=params.overwrite, verbose=params.verbose)
    elif suffix in (".txt", ".log"):
        write_txt(params.path, data, overwrite=params.overwrite, verbose=params.verbose)
    else:
        raise ValueError(f"Unsupported extension: {suffix}")

