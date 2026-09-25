# `UnsupportedOperationError`

Raised when an archive format cannot perform a requested operation.

```python
from klygo.archive.backend import GZipBackend, UnsupportedOperationError

backend = GZipBackend()
try:
    backend.require_operation("add")
except UnsupportedOperationError as error:
    print(error.format_name)  # gz
    print(error.operation)    # add
```

The exception subclasses `NotImplementedError` and exposes `format_name` and
`operation` for programmatic handling. Prefer catching this concrete type when
showing format-specific fallback behavior.

