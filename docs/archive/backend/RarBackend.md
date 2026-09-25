# `RarBackend`

Optional read-only RAR adapter backed by `rarfile`.

```bash
pip install "klygo[rarfile]"
uv add "klygo[rarfile]"
uv pip install "klygo[rarfile]"
```

Use `uv add` inside a uv-managed project. Use `uv pip install` for a direct
installation into the active environment without updating project metadata.

```python
from pathlib import Path
from klygo.archive.backend import RarBackend

backend = RarBackend()
print(backend.list_files(Path("dataset.rar")))
```

Extraction, password-based reading, listing, searching, metadata, and integrity
tests are supported. Compression and all mutation operations intentionally
raise `UnsupportedOperationError`.

Depending on the RAR version and host, `rarfile` may also require an external
extraction program. That requirement belongs to the optional dependency rather
than the Klygo backend contract.
