import builtins
import configparser
import csv
import hashlib
import json
import os
import pickle
import shutil
import xml.etree.ElementTree as ET
import urllib.request
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Tuple, Union

from ruamel.yaml import YAML

from klygo.archive.human_size import human_size as _human_size
from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type

_yaml = YAML()

_DATA_SUFFIXES = {
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


# =========================================================================
# Helper Functions
# =========================================================================

def _read_with_bar(path: Path, verbose: bool, desc: str, parser_func: Any) -> Any:
    with ProgressBar(total=1, desc=desc, unit="file", verbose=verbose, colour="cyan") as pbar:
        res = parser_func()
        pbar.update(1)
        return res


def _write_with_bar(path: Path, data: Any, overwrite: bool, verbose: bool, desc: str, writer_func: Any) -> None:
    with ProgressBar(total=1, desc=desc, unit="file", verbose=verbose, colour="cyan") as pbar:
        writer_func()
        pbar.update(1)


# =========================================================================
# Data I/O & Conversion
# =========================================================================

def load(
    path: Union[str, Path],
    as_lines: bool = False,
    verbose: bool = True,
) -> Any:
    """
    Tác dụng:
    - Tự động đọc dữ liệu từ file dựa theo phần mở rộng đuôi file mở rộng.
    - Hỗ trợ giải mã cấu trúc dữ liệu thô (dict, list, str) của 14 định dạng file phổ biến.

    Định dạng tương thích:
    - Hỗ trợ: YAML (.yaml, .yml), JSON (.json), JSON Lines (.jsonl), TOML (.toml), CSV (.csv),
      Text (.txt), Log (.log), INI (.ini), Config (.cfg), Properties (.properties),
      Environment (.env), XML (.xml), Pickle (.pkl, .pickle).

    Đầu vào:
    - path [str | Path]: Đường dẫn file nguồn cần đọc.
    - as_lines [bool]: Chọn True nếu muốn đọc file .txt/.log thành danh sách các dòng (list[str]). Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar màu Cyan trong console. Mặc định: True.

    Đầu ra:
    - [Any]: Trả về đối tượng dữ liệu đã đọc (dict, list, str hoặc object đã unpickle).

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi file path không tồn tại trên hệ thống.
    - ValueError: Phát sinh khi path là thư mục hoặc định dạng file không được hỗ trợ.

    Ví dụ:
    >>> import klygo.files as files

    # Ví dụ 1: Đọc file YAML cấu hình
    >>> config = files.load("config.yaml")

    # Ví dụ 2: Đọc file JSON danh sách người dùng
    >>> users = files.load("data.json")

    # Ví dụ 3: Đọc file log thành danh sách các dòng mà không hiển thị thanh tiến trình
    >>> lines = files.load("app.log", as_lines=True, verbose=False)
    """
    validate_type(path, (str, Path), "path")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Path must be a file: {p}")

    suffix = p.suffix.lower()
    if not suffix and p.name.lower().startswith(".env"):
        suffix = ".env"

    if suffix not in _DATA_SUFFIXES:
        raise ValueError(f"Unsupported format: {suffix!r}, supported: {sorted(_DATA_SUFFIXES)}")

    if suffix in (".yaml", ".yml"):
        def _parse_yaml():
            with open(p, "r", encoding="utf-8") as f:
                return _yaml.load(f)
        return _read_with_bar(p, verbose, "Reading YAML", _parse_yaml)

    elif suffix == ".json":
        def _parse_json():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return _read_with_bar(p, verbose, "Reading JSON", _parse_json)

    elif suffix == ".jsonl":
        def _parse_jsonl():
            records = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        records.append(json.loads(line_str))
            return records
        return _read_with_bar(p, verbose, "Reading JSONL", _parse_jsonl)

    elif suffix == ".toml":
        import tomlkit
        def _parse_toml():
            with open(p, "r", encoding="utf-8") as f:
                return tomlkit.load(f).unwrap()
        return _read_with_bar(p, verbose, "Reading TOML", _parse_toml)

    elif suffix == ".csv":
        def _parse_csv():
            with open(p, "r", encoding="utf-8", newline="") as f:
                return builtins.list(csv.DictReader(f))
        return _read_with_bar(p, verbose, "Reading CSV", _parse_csv)

    elif suffix in (".txt", ".log"):
        def _parse_txt():
            with open(p, "r", encoding="utf-8") as f:
                if as_lines:
                    return [line.rstrip("\r\n") for line in f]
                return f.read()
        return _read_with_bar(p, verbose, "Reading TXT", _parse_txt)

    elif suffix in (".ini", ".cfg", ".properties"):
        def _parse_ini():
            cp = configparser.ConfigParser()
            cp.read(p, encoding="utf-8")
            return {s: dict(cp[s]) for s in cp.sections()}
        return _read_with_bar(p, verbose, "Reading INI", _parse_ini)

    elif suffix == ".env":
        def _parse_env():
            env_dict = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str and not line_str.startswith("#") and "=" in line_str:
                        k, v = line_str.split("=", 1)
                        env_dict[k.strip()] = v.strip().strip("'\"")
            return env_dict
        return _read_with_bar(p, verbose, "Reading ENV", _parse_env)

    elif suffix == ".xml":
        def _parse_xml():
            tree = ET.parse(p)
            root = tree.getroot()

            def _elem_to_dict(elem):
                children = builtins.list(elem)
                if not children and not elem.attrib:
                    return {elem.tag: elem.text.strip() if elem.text else ""}

                res_dict = {}
                if elem.attrib:
                    for ak, av in elem.attrib.items():
                        res_dict[f"@{ak}"] = av
                if elem.text and elem.text.strip():
                    res_dict["#text"] = elem.text.strip()

                for child in children:
                    c_dict = _elem_to_dict(child)
                    for ck, cv in c_dict.items():
                        if ck in res_dict:
                            if not isinstance(res_dict[ck], builtins.list):
                                res_dict[ck] = [res_dict[ck]]
                            res_dict[ck].append(cv)
                        else:
                            res_dict[ck] = cv
                return {elem.tag: res_dict}

            return _elem_to_dict(root)
        return _read_with_bar(p, verbose, "Reading XML", _parse_xml)

    elif suffix in (".pkl", ".pickle"):
        def _parse_pkl():
            with open(p, "rb") as f:
                return pickle.load(f)
        return _read_with_bar(p, verbose, "Reading Pickle", _parse_pkl)


def save(
    path: Union[str, Path],
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
    indent: int = 4,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """
    Tác dụng:
    - Tự động ghi và mã hóa dữ liệu ra file dựa theo phần mở rộng đuôi file mở rộng.
    - Tự động tạo thư mục cha nếu chưa tồn tại.

    Định dạng tương thích:
    - Đầu ra hỗ trợ: YAML, JSON, JSONL, TOML, CSV, TXT, LOG, INI, CFG, PROPERTIES, ENV, XML, PKL, PICKLE.

    Đầu vào:
    - path [str | Path]: Đường dẫn file đầu ra cần lưu.
    - data [Any]: Dữ liệu cần ghi ra file (dict, list, str...).
    - overwrite [bool]: Cho phép ghi đè nếu file đầu ra đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar trong console. Mặc định: True.
    - indent [int]: Số khoảng trắng thụt lề khi xuất định dạng JSON. Mặc định: 4.
    - fieldnames [list[str] | None]: Danh sách tên cột tương ứng khi xuất file CSV. Mặc định: None.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - FileExistsError: Phát sinh khi path đã tồn tại và overwrite=False.
    - ValueError: Phát sinh khi định dạng đuôi file không được hỗ trợ.

    Ví dụ:
    >>> import klygo.files as files

    # Ví dụ 1: Ghi dictionary ra file YAML
    >>> files.save("config.yaml", {"model": "yolov8", "batch": 16}, overwrite=True)

    # Ví dụ 2: Ghi file biến môi trường .env
    >>> files.save(".env", {"PORT": "8080", "HOST": "localhost"}, overwrite=True)
    """
    validate_type(path, (str, Path), "path")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    p = Path(path)
    suffix = p.suffix.lower()
    if not suffix and p.name.lower().startswith(".env"):
        suffix = ".env"

    if suffix not in _DATA_SUFFIXES:
        raise ValueError(f"Unsupported export format: {suffix!r}, supported: {sorted(_DATA_SUFFIXES)}")

    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p}. Use overwrite=True to replace it.")

    p.parent.mkdir(parents=True, exist_ok=True)

    if suffix in (".yaml", ".yml"):
        def _write_yaml():
            with open(p, "w", encoding="utf-8") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    _yaml.dump(data, f)
        _write_with_bar(p, data, overwrite, verbose, "Writing YAML", _write_yaml)

    elif suffix == ".json":
        def _write_json():
            with open(p, "w", encoding="utf-8") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    json.dump(data, f, ensure_ascii=False, indent=indent)
        _write_with_bar(p, data, overwrite, verbose, "Writing JSON", _write_json)

    elif suffix == ".jsonl":
        def _write_jsonl():
            with open(p, "w", encoding="utf-8") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        _write_with_bar(p, data, overwrite, verbose, "Writing JSONL", _write_jsonl)

    elif suffix == ".toml":
        import tomlkit
        def _write_toml():
            with open(p, "w", encoding="utf-8") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    tomlkit.dump(data, f)
        _write_with_bar(p, data, overwrite, verbose, "Writing TOML", _write_toml)

    elif suffix == ".csv":
        fn = fieldnames
        if not fn and data and isinstance(data[0], dict):
            fn = builtins.list(data[0].keys())
        def _write_csv():
            with open(p, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fn or [])
                writer.writeheader()
                writer.writerows(data)
        _write_with_bar(p, data, overwrite, verbose, "Writing CSV", _write_csv)

    elif suffix in (".txt", ".log"):
        def _write_txt():
            with open(p, "w", encoding="utf-8") as f:
                if isinstance(data, (builtins.list, tuple)):
                    f.write("\n".join(str(item) for item in data) + "\n")
                else:
                    f.write(str(data))
        _write_with_bar(p, data, overwrite, verbose, "Writing TXT", _write_txt)

    elif suffix in (".ini", ".cfg", ".properties"):
        def _write_ini():
            cp = configparser.ConfigParser()
            if isinstance(data, dict):
                for section, options in data.items():
                    if isinstance(options, dict):
                        cp[section] = {str(k): str(v) for k, v in options.items()}
            with open(p, "w", encoding="utf-8") as f:
                cp.write(f)
        _write_with_bar(p, data, overwrite, verbose, "Writing INI", _write_ini)

    elif suffix == ".env":
        def _write_env():
            with open(p, "w", encoding="utf-8") as f:
                if isinstance(data, dict):
                    for k, v in data.items():
                        f.write(f"{k}={v}\n")
        _write_with_bar(p, data, overwrite, verbose, "Writing ENV", _write_env)

    elif suffix == ".xml":
        def _write_xml():
            def _dict_to_elem(tag, val):
                elem = ET.Element(tag)
                if isinstance(val, dict):
                    for k, v in val.items():
                        if k.startswith("@"):
                            elem.set(k[1:], str(v))
                        elif k == "#text":
                            elem.text = str(v)
                        elif isinstance(v, builtins.list):
                            for item in v:
                                elem.append(_dict_to_elem(k, item))
                        else:
                            elem.append(_dict_to_elem(k, v))
                else:
                    elem.text = str(val)
                return elem

            if isinstance(data, dict) and len(data) == 1:
                rt_tag = builtins.list(data.keys())[0]
                rt_elem = _dict_to_elem(rt_tag, data[rt_tag])
            else:
                rt_elem = _dict_to_elem("root", data)

            tree = ET.ElementTree(rt_elem)
            tree.write(p, encoding="utf-8", xml_declaration=True)
        _write_with_bar(p, data, overwrite, verbose, "Writing XML", _write_xml)

    elif suffix in (".pkl", ".pickle"):
        def _write_pkl():
            with open(p, "wb") as f:
                pickle.dump(data, f)
        _write_with_bar(p, data, overwrite, verbose, "Writing Pickle", _write_pkl)


def convert(
    source: Union[str, Path],
    target: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Tác dụng:
    - Chuyển đổi dữ liệu từ file nguồn sang định dạng file đích.
    - Tự động xác định kiểu đọc/ghi theo đuôi mở rộng của hai file.

    Định dạng tương thích:
    - Chuyển đổi qua lại giữa: YAML, JSON, JSONL, TOML, CSV, TXT, LOG, INI, CFG, PROPERTIES, ENV, XML, PKL, PICKLE.

    Đầu vào:
    - source [str | Path]: Đường dẫn file nguồn.
    - target [str | Path]: Đường dẫn file đích.
    - overwrite [bool]: Cho phép ghi đè nếu file đích đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn Path đến file đích đã chuyển đổi.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.convert("config.yaml", "config.json", overwrite=True)
    """
    data = load(source, verbose=verbose)
    save(target, data, overwrite=overwrite, verbose=verbose)
    return Path(target)


def download(
    source: Union[str, Path],
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Tác dụng:
    - Tải tập tin từ URL Internet về Colab/Server HOẶC tải tập tin từ Colab/Server về máy tính cá nhân (PC).
    - Giữ nguyên tên file gốc (không đổi tên). Để đổi tên file sau khi tải, sử dụng hàm files.rename().
    - Chỉ hỗ trợ tải tập tin (file), không hỗ trợ thư mục (không nén).

    Đầu vào:
    - source [str | Path]: Đường dẫn URL tập tin hoặc đường dẫn file cục bộ trên Colab/Server.
    - output_dir [str | Path]: Thư mục đầu ra chứa file tải về. Mặc định: '.' (thư mục hiện tại).
    - overwrite [bool]: Cho phép ghi đè nếu file đầu ra đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar trong console. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn Path đến tập tin đã tải về.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi file nguồn cục bộ không tồn tại.
    - ValueError: Phát sinh khi nguồn là thư mục thay vì tập tin.
    - FileExistsError: Phát sinh khi file đầu ra đã tồn tại và overwrite=False.

    Ví dụ:
    >>> import klygo.files as files

    # Ví dụ 1: Tải file từ URL Internet về thư mục "downloads/" giữ nguyên tên file gốc
    >>> files.download("https://example.com/model.pt", output_dir="downloads", overwrite=True)

    # Ví dụ 2: Tải file từ Google Colab về trực tiếp máy tính cá nhân (PC)
    >>> files.download("results.json")
    """
    validate_type(source, (str, Path), "source")
    validate_type(output_dir, (str, Path), "output_dir")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    src_str = str(source)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TH1: Tải từ URL Internet (http, https, ftp)
    if src_str.startswith(("http://", "https://", "ftp://")):
        parsed = urlparse(src_str)
        filename = os.path.basename(parsed.path) or "downloaded_file"
        dst_p = out_dir / filename

        if dst_p.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {dst_p}. Use overwrite=True to replace it.")

        req = urllib.request.Request(src_str, headers={"User-Agent": "klygo/2.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with open(dst_p, "wb") as f, ProgressBar(
                total=total_size if total_size > 0 else None,
                desc=f"Downloading {dst_p.name}",
                unit="B",
                unit_scale=True,
                verbose=verbose,
                colour="cyan",
            ) as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))

        return dst_p

    # TH2: Tải từ file cục bộ (Colab / Server / Local)
    src_p = Path(source)
    if not src_p.exists():
        raise FileNotFoundError(f"Source file does not exist: {src_p}")

    if src_p.is_dir():
        raise ValueError(f"download() only supports single files, not directories: {src_p}")

    # Nếu đang chạy trong Google Colab và lưu ở thư mục mặc định '.' -> Kích hoạt tải về máy tính PC
    try:
        from google.colab import files as colab_files
        colab_files.download(str(src_p))
        return src_p
    except ImportError:
        pass

    # Copy file cục bộ sang output_dir giữ nguyên tên file
    dst_p = out_dir / src_p.name
    if dst_p.resolve() != src_p.resolve():
        copy(src_p, dst_p, overwrite=overwrite)

    return dst_p


# =========================================================================
# File Status & Type Checks
# =========================================================================

def exists(path: Union[str, Path]) -> bool:
    """
    Tác dụng:
    - Kiểm tra xem file hoặc thư mục tại đường dẫn chỉ định có tồn tại hay không.

    Đầu vào:
    - path [str | Path]: Đường dẫn cần kiểm tra.

    Đầu ra:
    - [bool]: True nếu tồn tại, ngược lại False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.exists("config.yaml")
    True
    """
    return Path(path).exists()


def is_file(path: Union[str, Path]) -> bool:
    """
    Tác dụng:
    - Kiểm tra xem đường dẫn có phải là một tập tin (file) hợp lệ hay không.

    Đầu vào:
    - path [str | Path]: Đường dẫn cần kiểm tra.

    Đầu ra:
    - [bool]: True nếu là file, ngược lại False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.is_file("config.yaml")
    True
    """
    return Path(path).is_file()


def is_dir(path: Union[str, Path]) -> bool:
    """
    Tác dụng:
    - Kiểm tra xem đường dẫn có phải là một thư mục (directory) hay không.

    Đầu vào:
    - path [str | Path]: Đường dẫn cần kiểm tra.

    Đầu ra:
    - [bool]: True nếu là thư mục, ngược lại False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.is_dir("dataset/")
    True
    """
    return Path(path).is_dir()


# =========================================================================
# Traversal & Directory Operations
# =========================================================================

def list(
    path: Union[str, Path] = ".",
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    Tác dụng:
    - Liệt kê danh sách các tập tin và thư mục con bên trong đường dẫn chỉ định.

    Đầu vào:
    - path [str | Path]: Đường dẫn thư mục cần duyệt. Mặc định: '.'.
    - pattern [str]: Mẫu lọc tên file khớp wildcard. Mặc định: '*'.
    - recursive [bool]: Duyệt đệ quy sâu vào tất cả các thư mục con. Mặc định: False.

    Đầu ra:
    - [List[Path]]: Danh sách các đường dẫn Path sắp xếp theo thứ tự bảng chữ cái.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi path không tồn tại.
    - ValueError: Phát sinh khi path không phải thư mục.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.list("dataset/", pattern="*.json")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Path must be a directory: {p}")

    iterator = p.rglob(pattern) if recursive else p.glob(pattern)
    return sorted(iterator, key=lambda x: str(x).lower())


def find(
    path: Union[str, Path] = ".",
    pattern: str = "*",
    recursive: bool = True,
) -> List[Path]:
    """
    Tác dụng:
    - Tìm kiếm tất cả các file khớp với mẫu wildcard chỉ định bên trong thư mục.

    Đầu vào:
    - path [str | Path]: Thư mục gốc bắt đầu tìm kiếm. Mặc định: '.'.
    - pattern [str]: Mẫu wildcard tìm kiếm (ví dụ: '*.jpg', '*.json'). Mặc định: '*'.
    - recursive [bool]: Tìm kiếm sâu trong các thư mục con. Mặc định: True.

    Đầu ra:
    - [List[Path]]: Danh sách các file tìm thấy dưới dạng Path.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.find("dataset/", pattern="*.jpg")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    iterator = p.rglob(pattern) if recursive else p.glob(pattern)
    return sorted((f for f in iterator if f.is_file()), key=lambda x: str(x).lower())


def walk(
    path: Union[str, Path] = ".",
) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    """
    Tác dụng:
    - Duyệt cây thư mục theo cơ chế generator tiết kiệm bộ nhớ (tương tự os.walk).

    Đầu vào:
    - path [str | Path]: Thư mục gốc bắt đầu duyệt. Mặc định: '.'.

    Đầu ra:
    - [Generator]: Trả về tuple (root_dir, dir_names, file_names) cho từng thư mục.

    Ví dụ:
    >>> import klygo.files as files
    >>> for root, dirs, filenames in files.walk("dataset/"):
    ...     print(root, filenames)
    """
    return os.walk(str(path))


def mkdir(
    path: Union[str, Path],
    parents: bool = True,
    exist_ok: bool = True,
) -> Path:
    """
    Tác dụng:
    - Tạo thư mục mới trên ổ đĩa hệ thống.

    Đầu vào:
    - path [str | Path]: Đường dẫn thư mục cần tạo.
    - parents [bool]: Tự động tạo các thư mục cha nếu chưa tồn tại. Mặc định: True.
    - exist_ok [bool]: Bỏ qua thông báo lỗi nếu thư mục đã tồn tại từ trước. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn thư mục Path đã tạo.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.mkdir("output/reports/2026", parents=True, exist_ok=True)
    """
    p = Path(path)
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return p


# =========================================================================
# File System Actions
# =========================================================================

def copy(
    source: Union[str, Path],
    target: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """
    Tác dụng:
    - Sao chép tập tin hoặc toàn bộ thư mục từ vị trí nguồn sang vị trí đích.

    Đầu vào:
    - source [str | Path]: Đường dẫn nguồn.
    - target [str | Path]: Đường dẫn đích.
    - overwrite [bool]: Cho phép ghi đè/thay thế nếu đích đã tồn tại. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn Path đến đối tượng đã sao chép.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi source không tồn tại.
    - FileExistsError: Phát sinh khi target đã tồn tại và overwrite=False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.copy("config.yaml", "backup/config.yaml", overwrite=True)
    """
    src_p = Path(source)
    dst_p = Path(target)

    if not src_p.exists():
        raise FileNotFoundError(f"Source does not exist: {src_p}")

    if dst_p.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {dst_p}")

    if src_p.is_file():
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
    else:
        if dst_p.exists() and overwrite:
            shutil.rmtree(dst_p)
        shutil.copytree(src_p, dst_p)

    return dst_p


def move(
    source: Union[str, Path],
    target: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """
    Tác dụng:
    - Di chuyển tập tin hoặc thư mục sang vị trí đích mới.

    Đầu vào:
    - source [str | Path]: Đường dẫn nguồn.
    - target [str | Path]: Đường dẫn đích.
    - overwrite [bool]: Cho phép ghi đè nếu đích đã tồn tại. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn Path mới sau khi di chuyển.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi source không tồn tại.
    - FileExistsError: Phát sinh khi target đã tồn tại và overwrite=False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.move("old_dir/data.json", "new_dir/data.json", overwrite=True)
    """
    src_p = Path(source)
    dst_p = Path(target)

    if not src_p.exists():
        raise FileNotFoundError(f"Source does not exist: {src_p}")

    if dst_p.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {dst_p}")
        if dst_p.is_file():
            os.remove(dst_p)
        else:
            shutil.rmtree(dst_p)

    dst_p.parent.mkdir(parents=True, exist_ok=True)
    res = shutil.move(str(src_p), str(dst_p))
    return Path(res)


def rename(
    path: Union[str, Path],
    new_name_or_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Tác dụng:
    - Đổi tên tập tin/thư mục hoặc chuyển sang tên mới.

    Đầu vào:
    - path [str | Path]: Đường dẫn tập tin/thư mục hiện tại.
    - new_name_or_path [str | Path]: Tên mới hoặc đường dẫn mới.
    - overwrite [bool]: Cho phép ghi đè nếu tên mới đã tồn tại. Mặc định: False.

    Đầu ra:
    - [Path]: Đường dẫn Path mới sau khi đổi tên.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi path không tồn tại.
    - FileExistsError: Phát sinh khi tên mới đã tồn tại và overwrite=False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.rename("draft.txt", "final.txt", overwrite=True)
    """
    src_p = Path(path)
    if not src_p.exists():
        raise FileNotFoundError(f"Path does not exist: {src_p}")

    target = Path(new_name_or_path)
    if len(target.parts) == 1:
        dst_p = src_p.parent / target
    else:
        dst_p = target

    if dst_p.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {dst_p}")

    if dst_p.exists() and overwrite:
        if dst_p.is_file():
            os.remove(dst_p)
        else:
            shutil.rmtree(dst_p)

    src_p.rename(dst_p)
    return dst_p


def remove(
    path: Union[str, Path],
    recursive: bool = True,
    missing_ok: bool = True,
) -> None:
    """
    Tác dụng:
    - Xóa tập tin hoặc toàn bộ thư mục khỏi ổ đĩa hệ thống.

    Đầu vào:
    - path [str | Path]: Đường dẫn tập tin hoặc thư mục cần xóa.
    - recursive [bool]: Xóa đệ quy nếu đường dẫn là thư mục. Mặc định: True.
    - missing_ok [bool]: Bỏ qua thông báo lỗi nếu đường dẫn không tồn tại. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.remove("temp.txt", missing_ok=True)
    """
    p = Path(path)
    if not p.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        os.remove(p)
    elif p.is_dir():
        if recursive:
            shutil.rmtree(p)
        else:
            os.rmdir(p)


# =========================================================================
# Metadata, Size, Hash & Comparison
# =========================================================================

def size(path: Union[str, Path], human: bool = False) -> Union[int, str]:
    """
    Tác dụng:
    - Tính tổng dung lượng của tập tin hoặc toàn bộ thư mục.

    Đầu vào:
    - path [str | Path]: Đường dẫn tập tin hoặc thư mục.
    - human [bool]: Nếu True, trả về chuỗi đọc được (ví dụ: '1.5 MB'). Nếu False, trả về số byte. Mặc định: False.

    Đầu ra:
    - [int | str]: Dung lượng dưới dạng số byte (int) hoặc chuỗi thân thiện (str).

    Ví dụ:
    >>> import klygo.files as files
    >>> files.size("config.yaml", human=True)
    '1.20 KB'
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        total_bytes = p.stat().st_size
    else:
        total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    if human:
        return _human_size(total_bytes)
    return total_bytes


def hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Tác dụng:
    - Tính mã checksum hash của một tập tin.

    Đầu vào:
    - path [str | Path]: Đường dẫn tập tin cần tính hash.
    - algorithm [str]: Thuật toán hash ('md5', 'sha256', 'sha1'). Mặc định: 'md5'.

    Đầu ra:
    - [str]: Chuỗi mã checksum Hex.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi path không tồn tại.
    - ValueError: Phát sinh khi path không phải là file.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.hash("data.json", algorithm="md5")
    'd41d8cd98f00b204e9800998ecf8427e'
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Path must be a file to hash: {p}")

    hasher = hashlib.new(algorithm)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Tác dụng:
    - Lấy thông tin metadata chi tiết của tập tin hoặc thư mục.

    Đầu vào:
    - path [str | Path]: Đường dẫn tập tin hoặc thư mục.

    Đầu ra:
    - [Dict[str, Any]]: Dictionary chứa các thuộc tính metadata (name, stem, extension, size, human_size, is_file, is_dir, created, modified, hash...).

    Ví dụ:
    >>> import klygo.files as files
    >>> info_dict = files.info("config.yaml")
    >>> print(info_dict["size"])
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    stat = p.stat()
    sz = size(p)
    return {
        "name": p.name,
        "stem": p.stem,
        "extension": p.suffix,
        "parent": p.parent,
        "size": sz,
        "human_size": _human_size(sz),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "hash": hash(p) if p.is_file() else None,
    }


def compare(
    path1: Union[str, Path],
    path2: Union[str, Path],
    by: str = "hash",
) -> bool:
    """
    Tác dụng:
    - So sánh nội dung giữa 2 tập tin xem có giống nhau hay không.

    Đầu vào:
    - path1 [str | Path]: Đường dẫn tập tin thứ nhất.
    - path2 [str | Path]: Đường dẫn tập tin thứ hai.
    - by [str]: Phương thức so sánh: 'hash' (so sánh mã MD5) hoặc 'content' (so sánh byte-by-byte). Mặc định: 'hash'.

    Đầu ra:
    - [bool]: True nếu 2 file có nội dung hoàn toàn giống nhau, ngược lại False.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.compare("file1.txt", "file2.txt", by="hash")
    True
    """
    p1 = Path(path1)
    p2 = Path(path2)

    if not p1.exists():
        raise FileNotFoundError(f"Path does not exist: {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"Path does not exist: {p2}")

    if p1.stat().st_size != p2.stat().st_size:
        return False

    if by == "hash":
        return hash(p1) == hash(p2)
    elif by == "content":
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            for chunk1, chunk2 in zip(iter(lambda: f1.read(65536), b""), iter(lambda: f2.read(65536), b"")):
                if chunk1 != chunk2:
                    return False
        return True
    else:
        raise ValueError(f"Invalid compare mode: {by!r}. Expected 'hash' or 'content'.")


# =========================================================================
# Path Property Helpers
# =========================================================================

def name(path: Union[str, Path]) -> str:
    """
    Tác dụng:
    - Trích xuất tên tập tin/thư mục kèm phần mở rộng từ đường dẫn.

    Đầu vào:
    - path [str | Path]: Đường dẫn cần xử lý.

    Đầu ra:
    - [str]: Tên tập tin/thư mục (ví dụ: 'sample.jpg').

    Ví dụ:
    >>> import klygo.files as files
    >>> files.name("dataset/images/sample.jpg")
    'sample.jpg'
    """
    return Path(path).name


def stem(path: Union[str, Path]) -> str:
    """
    Tác dụng:
    - Trích xuất tên tập tin không bao gồm phần mở rộng (đuôi file).

    Đầu vào:
    - path [str | Path]: Đường dẫn cần xử lý.

    Đầu ra:
    - [str]: Tên tập tin không có đuôi (ví dụ: 'sample').

    Ví dụ:
    >>> import klygo.files as files
    >>> files.stem("dataset/images/sample.jpg")
    'sample'
    """
    return Path(path).stem


def extension(path: Union[str, Path]) -> str:
    """
    Tác dụng:
    - Trích xuất phần mở rộng đuôi file (suffix) từ đường dẫn.

    Đầu vào:
    - path [str | Path]: Đường dẫn cần xử lý.

    Đầu ra:
    - [str]: Đuôi phần mở rộng (ví dụ: '.jpg').

    Ví dụ:
    >>> import klygo.files as files
    >>> files.extension("dataset/images/sample.jpg")
    '.jpg'
    """
    return Path(path).suffix


def parent(path: Union[str, Path]) -> Path:
    """
    Tác dụng:
    - Lấy đường dẫn thư mục cha chứa tập tin hoặc thư mục hiện tại.

    Đầu vào:
    - path [str | Path]: Đường dẫn cần xử lý.

    Đầu ra:
    - [Path]: Đường dẫn Path đến thư mục cha.

    Ví dụ:
    >>> import klygo.files as files
    >>> files.parent("dataset/images/sample.jpg")
    WindowsPath('dataset/images')
    """
    return Path(path).parent
