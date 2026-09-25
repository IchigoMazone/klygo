# `detect_format`

```python
detect_format(path) -> str
```

Returns a canonical format identifier using compound extensions first, then
magic bytes for existing files, then a simple suffix fallback.

Supported results are `zip`, `tar`, `tar.gz`, `tar.xz`, `tar.bz2`, `gz`, `7z`,
and `rar`. Aliases such as `.tgz` are normalized to their canonical value.

```python
from klygo.archive.backend import detect_format

assert detect_format("dataset.tgz") == "tar.gz"
assert detect_format("backup.zip") == "zip"
```

`ValueError` is raised when neither the file content nor its name identifies a
supported format.

