# `files.with_extension`

```python
files.with_extension(value, new_extension)
```

Return a Path with a replaced extension.

## Contract

Returns a new path without changing the filesystem. To rename an existing entry, use [`files.rename()`](rename.md) or `Path.rename()`.

Recognizes .tar.gz, .tar.xz, and .tar.bz2 as compound archive extensions.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `value`: `str | os.PathLike[str]` — path-like value to transform.
- `new_extension`: `str` — extension with or without a leading dot.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `ValueError` when the extension contains a path separator.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.with_extension(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`with_extension.py`](../../../examples/files/with_extension.py) for an executable example.

## Tests

See [`test_with_extension.py`](../../../test/files/test_with_extension.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.with_extension."""

from klygo import files

print(files.with_extension("archives/dataset.tar.gz", ".zip"))


# Additional cases
assert files.with_extension("image.jpg", "png") == files.path("image.png")
assert files.with_extension("archive.tar.gz", ".zip") == files.path("archive.zip")
assert files.with_extension("README", ".md") == files.path("README.md")
```
