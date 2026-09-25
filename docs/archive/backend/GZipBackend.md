# `GZipBackend`

Single-file GZip stream adapter.

```python
from pathlib import Path
from klygo.archive.backend import GZipBackend

backend = GZipBackend()
backend.compress(Path("weights.bin"), Path("weights.gz"), verbose=False)
assert backend.list_files(Path("weights.gz")) == ["weights"]
```

GZip supports compression levels 0–9 but does not represent a directory tree.
Its logical member name comes from the archive filename. Directories, member
mutation, merging, and splitting are unsupported; use `TarBackend("tar.gz")`
for multi-file archives.

