# `TarBackend`

Standard-library adapter for `tar`, `tar.gz`, `tar.xz`, and `tar.bz2`.

```python
from pathlib import Path
from klygo.archive.backend import TarBackend

backend = TarBackend("tar.gz")
backend.compress(Path("dataset"), Path("dataset.tar.gz"), verbose=False)
```

The selected variant is instance state because compressed TAR formats differ in
their accepted compression options. Compressed variants support levels 1–9;
raw TAR rejects a changed `compresslevel` because no compression occurs.

Adding always rebuilds the archive. This provides identical conflict behavior
for raw and compressed TAR and avoids unsafe append attempts on compressed
streams. Extraction uses path containment checks and Python's `data` filter.

