# `files.hash`

```python
files.hash(path, algorithm='sha256')
```

Calculate a streaming checksum for a file.

## Contract

Uses SHA-256 by default. Other algorithms supported by hashlib, including MD5 for compatibility, can be requested explicitly.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
- `algorithm`: `str` — algorithm accepted by `hashlib`.

## Returns

A string result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing path and `ValueError` for directories or unknown algorithms.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.hash(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`hash.py`](../../../examples/files/hash.py) for an executable example.

## Tests

See [`test_hash.py`](../../../test/files/test_hash.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.hash."""

from klygo import files

print(files.hash("README.md", algorithm="sha256"))


# Additional cases
# Select any algorithm supported by hashlib.
md5 = files.hash("README.md", algorithm="md5")
sha256 = files.hash("README.md")
assert len(md5) == 32 and len(sha256) == 64
```
