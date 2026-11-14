# `files.info`

```python
files.info(path)
```

Collect common metadata for a file or directory.

## Contract

Returns name, stem, extension, parent, size, timestamps, and type flags without reading file contents. Computing directory size still traverses its descendants. Use `files.hash()` separately when a checksum is needed, and `utils.tabulate()` when you want to display the returned dictionary as a formatted Rich table.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.

## Returns

A metadata dictionary.

## Errors and edge cases

Raises `FileNotFoundError` when the path does not exist.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.info(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`info.py`](../../../examples/files/info.py) for an executable example.

## Tests

See [`test_info.py`](../../../test/files/test_info.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.info."""

from klygo import files, utils

metadata = files.info("README.md")
utils.tabulate(metadata, title="File information")
print(metadata["name"], metadata["human_size"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert "hash" not in metadata
```
