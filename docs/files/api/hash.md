# `files.hash`

```python
files.hash(path, algorithm='md5')
```

Calculate a streaming checksum for a file.

## Contract

Accepts algorithms supported by hashlib, such as md5, sha1, and sha256.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
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
md5 = files.hash("README.md")
sha256 = files.hash("README.md", algorithm="sha256")
assert len(md5) == 32 and len(sha256) == 64
```
