# `files.info`

```python
files.info(path)
```

Collect common metadata for a file or directory.

## Contract

Returns name, stem, extension, parent, size, timestamps, type flags, and a file hash.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

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

from klygo import files

metadata = files.info("README.md")
print(metadata["name"], metadata["human_size"], metadata["hash"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert isinstance(metadata["hash"], str)
```
