import builtins
import configparser
import csv
import hashlib
import json
import os
import pickle
import shutil
import xml.etree.ElementTree as ET
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
# Data I/O & Conversion
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


def load(
    path: Union[str, Path],
    as_lines: bool = False,
    verbose: bool = True,
) -> Any:
    """
    Tự động đọc file dữ liệu/cấu hình dựa theo định dạng đuôi file mở rộng.
    Hỗ trợ 14 định dạng: YAML, JSON, JSONL, TOML, CSV, TXT, LOG, INI, CFG, PROPERTIES, ENV, XML, PKL, PICKLE.
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
    Tự động ghi dữ liệu ra file theo đuôi mở rộng.
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
                _yaml.dump(data, f)
        _write_with_bar(p, data, overwrite, verbose, "Writing YAML", _write_yaml)

    elif suffix == ".json":
        def _write_json():
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
        _write_with_bar(p, data, overwrite, verbose, "Writing JSON", _write_json)

    elif suffix == ".jsonl":
        def _write_jsonl():
            with open(p, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        _write_with_bar(p, data, overwrite, verbose, "Writing JSONL", _write_jsonl)

    elif suffix == ".toml":
        import tomlkit
        def _write_toml():
            with open(p, "w", encoding="utf-8") as f:
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
    Chuyển đổi file dữ liệu từ định dạng này sang định dạng khác.
    """
    data = load(source, verbose=verbose)
    save(target, data, overwrite=overwrite, verbose=verbose)
    return Path(target)


# =========================================================================
# File Status & Type Checks
# =========================================================================

def exists(path: Union[str, Path]) -> bool:
    return Path(path).exists()


def is_file(path: Union[str, Path]) -> bool:
    return Path(path).is_file()


def is_dir(path: Union[str, Path]) -> bool:
    return Path(path).is_dir()


# =========================================================================
# Traversal & Directory Operations
# =========================================================================

def list(
    path: Union[str, Path] = ".",
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
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
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    iterator = p.rglob(pattern) if recursive else p.glob(pattern)
    return sorted((f for f in iterator if f.is_file()), key=lambda x: str(x).lower())


def walk(
    path: Union[str, Path] = ".",
) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    return os.walk(str(path))


def mkdir(
    path: Union[str, Path],
    parents: bool = True,
    exist_ok: bool = True,
) -> Path:
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
    return Path(path).name


def stem(path: Union[str, Path]) -> str:
    return Path(path).stem


def extension(path: Union[str, Path]) -> str:
    return Path(path).suffix


def parent(path: Union[str, Path]) -> Path:
    return Path(path).parent
