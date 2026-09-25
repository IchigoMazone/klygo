# `ArchiveBackend`

```python
from klygo.archive.backend import ArchiveBackend
```

Abstract contract for archive adapters. A concrete backend must implement all
read operations. Creation and mutation are optional and default to
`UnsupportedOperationError`.

## Required class attributes

```python
format_name: str
capabilities: BackendCapabilities
```

`format_name` must be canonical and stable. `capabilities` must accurately
describe every optional operation and changed option value the implementation
honors.

## Required read methods

```python
extract(archive_path, output_dir, password=None, include=None, exclude=None,
        overwrite=False, verbose=True)
extract_file(archive_path, filename, output_dir, password=None, overwrite=False)
list_files(archive_path)
iter_files(archive_path)
search(archive_path, pattern, regex=False, case_sensitive=True)
get_info(archive_path)
test(archive_path, raise_exception=False)
```

Backend methods receive normalized `pathlib.Path` values. They should not repeat
facade-level string/path conversion. Extraction implementations must reject
unsafe member paths and enforce `overwrite=False` before writing.

## Optional mutation methods

```python
compress(...)
add(...)
remove(...)
merge(...)
split_by_size(...)
```

Override a method only when the corresponding capability is `True`. Leaving the
base implementation in place gives callers a consistent structured exception.

## Validation helpers

`require_operation(operation)` rejects disabled operations before filesystem
work begins. `validate_option(operation, option, value, default)` rejects a
changed option that is absent from the operation's option set.

See `examples/archive/backend/ArchiveBackend.py` for a minimal custom backend.

