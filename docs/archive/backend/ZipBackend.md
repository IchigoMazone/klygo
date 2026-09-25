# `ZipBackend`

Feature-complete standard-library ZIP adapter.

```python
from pathlib import Path
from klygo.archive.backend import ZipBackend

backend = ZipBackend()
backend.compress(Path("dataset"), Path("dataset.zip"), verbose=False)
print(backend.list_files(Path("dataset.zip")))
```

Supported compression methods are `deflated` (default), `stored`, `bzip2`, and
`lzma`. Compression levels range from 0 through 9. It supports password-based
reading, include/exclude extraction filters, conflict-aware add, remove, merge,
and split.

`on_conflict="rename"` creates deterministic `_dup1`, `_dup2`, … member names;
`overwrite` rebuilds the archive without the old entry; `skip` leaves it intact.

Direct methods require `Path` arguments. Prefer `klygo.archive` unless explicit
backend control is necessary.

