# `backend.UnsupportedOptionError`

```python
from klygo.archive.backend import UnsupportedOptionError

raise UnsupportedOptionError(format_name, operation, option)
```

Report a non-default option unsupported by an archive backend.

## Contract

`UnsupportedOptionError` subclasses standard Python `ValueError`. A backend accepts an unsupported option when it retains its public default value, but rejects any non-default value by raising this exception. It exposes `format_name`, `operation`, and `option` as instance attributes.

## Parameters

- `format_name`: `str` — Canonical format identifier.
- `operation`: `str` — Operation name for which validation was invoked.
- `option`: `str` — Name of the unsupported option.

## Returns

`UnsupportedOptionError`
    Exception instance representing an unsupported option configuration.

## Errors and edge cases

Raised when `ArchiveBackend.validate_option()` detects that `value != default` and `option` is not in the operation's capability option set.

## AI usage guidance

Check whether backend options are supported before passing specialized flags like `password`, `method`, or custom compression parameters.

## Example

See [`UnsupportedOptionError.py`](../../../../examples/archive/backend/UnsupportedOptionError.py).

## Tests

See [`test_UnsupportedOptionError.py`](../../../../test/archive/backend/test_UnsupportedOptionError.py).

## Complete executable example

```python
from klygo.archive.backend import UnsupportedOptionError, GZipBackend

backend = GZipBackend()
try:
    backend.validate_option("compress", "include_root", False, True)
except UnsupportedOptionError as exc:
    assert exc.format_name == "gz"
    assert exc.operation == "compress"
    assert exc.option == "include_root"
    assert isinstance(exc, ValueError)
```
