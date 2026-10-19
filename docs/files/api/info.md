# `files.info`

```python
files.info(path, *, verbose=False)
```

Collect common metadata for a file or directory.

## Contract

Returns name, stem, extension, parent, size, timestamps, and type flags without reading file contents. With `verbose=True`, it also prints the dictionary as a Rich table. Computing directory size still traverses its descendants. Use `files.hash()` separately when a checksum is needed.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
- `verbose`: `bool`, default `False` — print a Rich table; keyword-only.

## Returns

A metadata dictionary.

## Errors and edge cases

Raises `FileNotFoundError` when the path does not exist and `TypeError` for a non-boolean `verbose`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.info(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`info.py`](../../../examples/files/info.py) for an executable example.

## Tests

See [`test_info.py`](../../../test/files/test_info.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.info."""

from klygo import files

metadata = files.info("README.md", verbose=True)
print(metadata["name"], metadata["human_size"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert "hash" not in metadata

# Disable display when only the dictionary is needed.
quiet = files.info("README.md")
assert quiet == metadata
```
