# `backend.ArchiveBackend`

```python
from klygo.archive.backend import ArchiveBackend
```

Abstract base contract implemented by all archive format adapters.

## Contract

`ArchiveBackend` defines the mandatory read interface (extraction, listing, searching, metadata inspection, and integrity testing) and optional mutation interface (`compress`, `add`, `remove`, `merge`, `split_by_size`). Concrete backends must set a canonical `format_name` and declare their supported operations and option flags via `capabilities` (`BackendCapabilities`). Methods accept pre-normalized `pathlib.Path` objects.

## Parameters

- `format_name`: `str` — Canonical format identifier (e.g. `"zip"`, `"tar.gz"`).
- `capabilities`: `BackendCapabilities` — Immutable declaration of supported operations and non-default option sets.
- Methods:
  - `require_operation(operation)`: Validates that an optional operation is supported.
  - `validate_option(operation, option, value, default)`: Validates that a non-default option value is accepted.
  - Mandatory read methods: `extract`, `extract_file`, `list_files`, `iter_files`, `search`, `get_info`, `test`.
  - Optional mutating methods: `compress`, `add`, `remove`, `merge`, `split_by_size`.

## Returns

`ArchiveBackend` is an abstract base class (`ABC`). Instances of concrete subclasses provide format-specific execution for archive operations.

## Errors and edge cases

Calling an unimplemented optional mutation raises `UnsupportedOperationError`. Supplying a non-default value for an unsupported option raises `UnsupportedOptionError`. Extraction rejects unsafe paths (path traversal) with `ValueError`.

## AI usage guidance

Do not use `ArchiveBackend` directly in user application code. Use high-level functions from `klygo.archive` or `archive.open()`. Subclass `ArchiveBackend` only when implementing a new custom archive format adapter.

## Example

See [`ArchiveBackend.py`](../../../../examples/archive/backend/ArchiveBackend.py).

## Tests

See [`test_ArchiveBackend.py`](../../../../test/archive/backend/test_ArchiveBackend.py).

## Complete executable example

```python
from klygo.archive.backend import ArchiveBackend, BackendCapabilities

class DummyBackend(ArchiveBackend):
    format_name = "dummy"
    capabilities = BackendCapabilities(compress=False)

    def extract(self, archive_path, output_dir, **kwargs): pass
    def extract_file(self, archive_path, filename, output_dir, **kwargs): pass
    def list_files(self, archive_path): return []
    def iter_files(self, archive_path): yield from ()
    def search(self, archive_path, pattern, **kwargs): return []
    def get_info(self, archive_path): return {"format": "dummy"}
    def test(self, archive_path, **kwargs): return True

backend = DummyBackend()
assert backend.format_name == "dummy"
assert backend.capabilities.compress is False
assert backend.test("mock.dummy") is True
```
