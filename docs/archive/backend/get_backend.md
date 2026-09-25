# `get_backend`

```python
get_backend(path, format_hint=None) -> ArchiveBackend
```

Creates a new registered backend. `format_hint`, when supplied, takes priority
over the path and accepts `tgz`, `txz`, and `tbz2` aliases.

```python
from klygo.archive.backend import TarBackend, get_backend

backend = get_backend("unused", format_hint="tgz")
assert isinstance(backend, TarBackend)
assert backend.format_name == "tar.gz"
```

The function returns a new object on every call. `ValueError` is raised for an
unknown format or when path-based detection fails.

