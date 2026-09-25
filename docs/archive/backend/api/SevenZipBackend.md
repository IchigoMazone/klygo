# `backend.SevenZipBackend`

```python
from klygo.archive.backend import SevenZipBackend

backend = SevenZipBackend()
```

Read and write 7-Zip (`.7z`) archives via the optional `py7zr` library.

## Contract

`SevenZipBackend` provides read and write operations for `.7z` archives when `py7zr` is installed. The class can be imported and inspected without `py7zr`. When an operation requires `py7zr` and it is not installed, an actionable `ImportError` is raised.

## Parameters

- `SevenZipBackend()` takes no constructor parameters.
- Key class attributes:
  - `format_name`: `"7z"`.
  - `capabilities`: Supports `compress=True` with `include_root`, and password-based extraction.

## Returns

`SevenZipBackend`
    A newly constructed backend adapter for 7-Zip archives.

## Errors and edge cases

Raises `ImportError` if `py7zr` is missing when performing actual archive operations. Raises `FileExistsError` if the output already exists and `overwrite=False`.

## AI usage guidance

Install `klygo[py7zr]` when working with `.7z` archives. Use high-level functions like `archive.extract("data.7z")` in user scripts.

## Example

See [`SevenZipBackend.py`](../../../../examples/archive/backend/SevenZipBackend.py).

## Tests

See [`test_SevenZipBackend.py`](../../../../test/archive/backend/test_SevenZipBackend.py).

## Complete executable example

```python
from klygo.archive.backend import SevenZipBackend

backend = SevenZipBackend()
assert backend.format_name == "7z"
assert backend.capabilities.compress is True
assert backend.capabilities.add is False
```
