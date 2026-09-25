# `UnsupportedOptionError`

Raised when a caller changes an option that a backend does not implement.

```python
from klygo.archive.backend import GZipBackend, UnsupportedOptionError

backend = GZipBackend()
try:
    backend.validate_option("compress", "include_root", False, True)
except UnsupportedOptionError as error:
    print(error.format_name, error.operation, error.option)
```

Passing the public default is allowed even when the option is absent from the
capability set. This is what permits a common facade signature without silently
ignoring behavior explicitly requested by the caller.

