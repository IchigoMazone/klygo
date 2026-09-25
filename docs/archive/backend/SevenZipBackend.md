# `SevenZipBackend`

Optional 7-Zip adapter backed by `py7zr`.

```bash
pip install "klygo[py7zr]"
uv add "klygo[py7zr]"
uv pip install "klygo[py7zr]"
```

Use `uv add` inside a uv-managed project. Use `uv pip install` for a direct
installation into the active environment without updating project metadata.

```python
from pathlib import Path
from klygo.archive.backend import SevenZipBackend

backend = SevenZipBackend()
backend.compress(Path("dataset"), Path("dataset.7z"), verbose=False)
```

The class can be imported and its capabilities inspected without the optional
dependency. Operations that need it raise an actionable `ImportError`.

Creation, extraction, password-based reading, listing, searching, metadata, and
integrity tests are implemented. Mutation operations remain disabled so they do
not silently provide weaker conflict or atomicity semantics than ZIP and TAR.
