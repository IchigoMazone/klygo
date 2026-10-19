"""Structured-data loading, saving, and format conversion."""

from __future__ import annotations

import configparser
import csv
import json
import os
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Generator, List, Optional

import yaml

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type

from ._typing import PathInput
from .filesystem import exists, is_file, is_symlink, mkdir, remove
from .paths import extension, name, parent, path as _to_path

_DATA_SUFFIXES = {
    ".yaml", ".yml", ".json", ".jsonl", ".toml", ".csv", ".txt",
    ".log", ".ini", ".cfg", ".properties", ".env", ".xml", ".pkl",
    ".pickle",
}


def _suffix(candidate: Path) -> str:
    suffix = extension(candidate).lower()
    return ".env" if not suffix and name(candidate).lower().startswith(".env") else suffix


def _run_with_progress(verbose: bool, description: str, function):
    with ProgressBar(total=1, desc=description, unit="file", verbose=verbose, colour="cyan") as progress:
        result = function()
        progress.update(1)
        return result


def _xml_to_data(element: ET.Element):
    children = list(element)
    if not children and not element.attrib:
        return {element.tag: element.text.strip() if element.text else ""}
    result = {f"@{key}": value for key, value in element.attrib.items()}
    if element.text and element.text.strip():
        result["#text"] = element.text.strip()
    for child in children:
        child_data = _xml_to_data(child)
        for key, value in child_data.items():
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)
            else:
                result[key] = value
    return {element.tag: result}


def _data_to_xml(tag: str, value: Any) -> ET.Element:
    element = ET.Element(tag)
    if isinstance(value, dict):
        for key, item in value.items():
            if key.startswith("@"):
                element.set(key[1:], str(item))
            elif key == "#text":
                element.text = str(item)
            elif isinstance(item, list):
                for entry in item:
                    element.append(_data_to_xml(key, entry))
            else:
                element.append(_data_to_xml(key, item))
    else:
        element.text = str(value)
    return element


def load(path: PathInput, as_lines: bool = False, verbose: bool = True) -> Any:
    """Load structured data from a file using its filename extension.

    The format is inferred from the final extension. Text formats use UTF-8. YAML, JSON, JSON Lines, TOML, CSV, text, INI-like, ENV, XML, and Pickle files are supported.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    as_lines : bool, default=False
        Return TXT and LOG files as lines without newline characters.
    verbose : bool, default=True
        Display a progress indicator.

    Returns
    -------
    Any
        Decoded content; the concrete type depends on the format.

    Raises
    ------
    TypeError
        If ``path`` is not path-like.
    FileNotFoundError
        If the input does not exist.
    ValueError
        If the input is not a file or its format is unsupported.

    Notes
    -----
    Loading Pickle data can execute arbitrary code. Only load Pickle files
    obtained from trusted sources.

    See Also
    --------
    save
    convert

    Examples
    --------
    >>> from klygo import files
    >>> config = files.load("config.json", verbose=False)
    """
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not is_file(candidate):
        raise ValueError(f"Path must be a file: {candidate}")
    suffix = _suffix(candidate)
    if suffix not in _DATA_SUFFIXES:
        raise ValueError(f"Unsupported format: {suffix!r}, supported: {sorted(_DATA_SUFFIXES)}")

    def read():
        if suffix in (".yaml", ".yml"):
            with candidate.open("r", encoding="utf-8") as stream:
                return yaml.safe_load(stream)
        if suffix == ".json":
            with candidate.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        if suffix == ".jsonl":
            with candidate.open("r", encoding="utf-8") as stream:
                return [json.loads(line) for line in stream if line.strip()]
        if suffix == ".toml":
            import tomlkit

            with candidate.open("r", encoding="utf-8") as stream:
                return tomlkit.load(stream).unwrap()
        if suffix == ".csv":
            with candidate.open("r", encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        if suffix in (".txt", ".log"):
            with candidate.open("r", encoding="utf-8") as stream:
                return [line.rstrip("\r\n") for line in stream] if as_lines else stream.read()
        if suffix in (".ini", ".cfg", ".properties"):
            parser = configparser.ConfigParser()
            parser.read(candidate, encoding="utf-8")
            return {section: dict(parser[section]) for section in parser.sections()}
        if suffix == ".env":
            result = {}
            with candidate.open("r", encoding="utf-8") as stream:
                for line in stream:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        key, value = stripped.split("=", 1)
                        result[key.strip()] = value.strip().strip("'\"")
            return result
        if suffix == ".xml":
            return _xml_to_data(ET.parse(candidate).getroot())
        with candidate.open("rb") as stream:
            return pickle.load(stream)

    return _run_with_progress(verbose, f"Reading {suffix.lstrip('.').upper()}", read)


def iter_load(path: PathInput) -> Generator[Any, None, None]:
    """Yield individual JSONL values or CSV rows using UTF-8 decoding.

    JSONL blank lines are ignored. CSV records are dictionaries of strings,
    with the first record supplying field names, as in ``load``.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Existing JSONL or CSV file. Other formats are intentionally unsupported.

    Returns
    -------
    generator[Any, None, None]
        JSON values or CSV row dictionaries, read one record at a time.

    Raises
    ------
    TypeError
        If ``path`` is not a string or a text-returning path-like object.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the input is not a file or its format is unsupported.
    json.JSONDecodeError
        If a JSONL record is invalid; raised when that record is consumed.
    OSError
        If the file cannot be opened or read during iteration.

    Notes
    -----
    The path and format are validated immediately. The stream opens when
    iteration starts and closes on exhaustion, an error, or ``close()``.
    Use ``contextlib.closing`` if iteration may stop early. Memory use depends
    on the current record, not the complete file.

    See Also
    --------
    load
    save

    Examples
    --------
    >>> from contextlib import closing
    >>> from klygo import files
    >>> with closing(files.iter_load("records.jsonl")) as records:
    ...     for record in records:
    ...         print(record)
    """
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not is_file(candidate):
        raise ValueError(f"Path must be a file: {candidate}")
    suffix = _suffix(candidate)
    if suffix not in (".jsonl", ".csv"):
        raise ValueError("iter_load() supports only .jsonl and .csv files")

    def generate():
        with candidate.open("r", encoding="utf-8", newline="") as stream:
            if suffix == ".csv":
                yield from csv.DictReader(stream)
            else:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

    return generate()


def save(
    path: PathInput,
    data: Any,
    overwrite: bool = False,
    verbose: bool = True,
    indent: int = 4,
    fieldnames: Optional[List[str]] = None,
    *,
    atomic: bool = True,
) -> None:
    """Save structured data using the destination filename extension.

    The format is inferred from the final extension. Missing parent directories are created before writing.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    data : Any
        Python object to serialize.
    overwrite : bool, default=False
        Allow an existing destination to be replaced.
    verbose : bool, default=True
        Display a progress indicator.
    indent : int, default=4
        Number of spaces used to indent JSON output.
    fieldnames : list[str] or None, default=None
        Ordered CSV column names; inferred from the first row when omitted.
    atomic : bool, default=True
        Serialize to a sibling temporary file, then publish the completed file.

    Returns
    -------
    None
        The destination is written as a side effect.

    Raises
    ------
    TypeError
        If validated arguments have invalid types.
    FileExistsError
        If the destination exists and overwrite is disabled.
    ValueError
        If the destination format is unsupported.
    OSError
        If writing or publication fails, including unsupported hard links.

    Notes
    -----
    Atomic mode leaves an existing destination unchanged if serialization fails.
    With ``overwrite=True`` the destination entry (including a symlink) is
    replaced, rather than writing through a symlink. With ``overwrite=False``
    publication uses a hard link and refuses even a concurrently created target.
    Filesystems without hard-link support raise ``OSError`` in that mode.
    Temporary files are removed after success or failure. Atomic mode uses the
    temporary file's permissions, does not preserve destination metadata, and
    does not guarantee durability across power loss.

    See Also
    --------
    load
    convert

    Examples
    --------
    >>> from klygo import files
    >>> files.save("config.json", {"batch": 16}, verbose=False)
    """
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")
    validate_type(atomic, bool, "atomic")
    candidate = _to_path(path)
    suffix = _suffix(candidate)
    if suffix not in _DATA_SUFFIXES:
        raise ValueError(
            f"Unsupported export format: {suffix!r}, supported: {sorted(_DATA_SUFFIXES)}"
        )
    if (exists(candidate) or is_symlink(candidate)) and not overwrite:
        raise FileExistsError(
            f"File already exists: {candidate}. Use overwrite=True to replace it."
        )
    mkdir(parent(candidate), parents=True, exist_ok=True)

    def write(destination):
        if suffix in (".yaml", ".yml"):
            with destination.open("w", encoding="utf-8") as stream:
                stream.write(data) if isinstance(data, str) else yaml.safe_dump(
                    data, stream, sort_keys=False, allow_unicode=True
                )
        elif suffix == ".json":
            with destination.open("w", encoding="utf-8") as stream:
                stream.write(data) if isinstance(data, str) else json.dump(
                    data, stream, ensure_ascii=False, indent=indent
                )
        elif suffix == ".jsonl":
            with destination.open("w", encoding="utf-8") as stream:
                if isinstance(data, str):
                    stream.write(data)
                else:
                    for item in data:
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif suffix == ".toml":
            import tomlkit

            with destination.open("w", encoding="utf-8") as stream:
                stream.write(data) if isinstance(data, str) else tomlkit.dump(data, stream)
        elif suffix == ".csv":
            columns = fieldnames or (list(data[0].keys()) if data and isinstance(data[0], dict) else [])
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(data)
        elif suffix in (".txt", ".log"):
            with destination.open("w", encoding="utf-8") as stream:
                stream.write("\n".join(str(item) for item in data) + "\n" if isinstance(data, (list, tuple)) else str(data))
        elif suffix in (".ini", ".cfg", ".properties"):
            parser = configparser.ConfigParser()
            if isinstance(data, dict):
                for section, options in data.items():
                    if isinstance(options, dict):
                        parser[section] = {str(key): str(value) for key, value in options.items()}
            with destination.open("w", encoding="utf-8") as stream:
                parser.write(stream)
        elif suffix == ".env":
            with destination.open("w", encoding="utf-8") as stream:
                if isinstance(data, dict):
                    for key, value in data.items():
                        stream.write(f"{key}={value}\n")
        elif suffix == ".xml":
            if isinstance(data, dict) and len(data) == 1:
                root_tag = next(iter(data))
                root = _data_to_xml(root_tag, data[root_tag])
            else:
                root = _data_to_xml("root", data)
            ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
        else:
            with destination.open("wb") as stream:
                pickle.dump(data, stream)

    def publish():
        if not atomic:
            write(candidate)
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name(candidate)}.", suffix=".tmp", dir=parent(candidate)
        )
        temporary = _to_path(temporary_name, expand_user=False)
        try:
            os.close(descriptor)
            write(temporary)
            if overwrite:
                os.replace(temporary, candidate)
            else:
                # Link publication fails if any entry already occupies the name.
                # An exists-then-replace sequence could overwrite another writer.
                os.link(temporary, candidate)
        finally:
            remove(temporary, missing_ok=True)

    _run_with_progress(verbose, f"Writing {suffix.lstrip('.').upper()}", publish)


def convert(
    source: PathInput,
    target: PathInput,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Convert data between two supported structured-data formats.

    The source is decoded with ``load`` and the resulting object is encoded with ``save``.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Source URL or filesystem path.
    target : str or os.PathLike[str]
        Destination filesystem path.
    overwrite : bool, default=False
        Allow an existing destination to be replaced.
    verbose : bool, default=True
        Display a progress indicator.

    Returns
    -------
    pathlib.Path
        Path to the converted file.

    Raises
    ------
    FileNotFoundError
        If the source does not exist.
    FileExistsError
        If the target exists and overwrite is disabled.
    ValueError
        If either format is unsupported.

    See Also
    --------
    load
    save

    Examples
    --------
    >>> from klygo import files
    >>> files.convert("config.yaml", "config.json", verbose=False)
    """
    data = load(source, verbose=verbose)
    save(target, data, overwrite=overwrite, verbose=verbose)
    return _to_path(target)


__all__ = ["load", "iter_load", "save", "convert"]
