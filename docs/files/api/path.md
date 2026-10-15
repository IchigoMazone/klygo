# `files.path`

```python
files.path(value, expand_user=True)
```

Convert a string or text-returning path-like object to a native `pathlib.Path`.

## Contract

Expands a leading user home marker when `expand_user=True`. Returns a standard
`Path` (a `WindowsPath` or `PosixPath` on the current platform), so native
properties and methods are immediately available. Does not create a filesystem
entry, expand environment variables, make a path absolute, or resolve `..`.

See [Working with pathlib.Path](../pathlib.md) for native methods, symlink
examples, Python compatibility, and differences from the Klygo helpers.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `value`: `str | os.PathLike[str]` — path-like value to transform.
- `expand_user`: `bool` — expand a leading `~`.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `TypeError` when the value is not a string or a text-returning `os.PathLike` object.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.path(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`path.py`](../../../examples/files/path.py) for an executable example.

## Tests

See [`test_path.py`](../../../test/files/test_path.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.path."""

from klygo import files

value = files.path("dataset/images")
print(value, type(value))


# Additional cases
home = files.path("~")
literal = files.path("~", expand_user=False)
assert home != literal and literal.name == "~"

# Custom filesystem inputs use __fspath__, not their string representation.
class DatasetPath:
    def __fspath__(self):
        return "dataset/images/cat.jpg"


native = files.path(DatasetPath())
assert native.name == "cat.jpg"
assert native.parent.as_posix() == "dataset/images"
```
