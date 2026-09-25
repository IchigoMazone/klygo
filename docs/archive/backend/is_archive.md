# `is_archive`

```python
is_archive(path) -> bool
```

Non-raising probe built on `detect_format`. It returns `False` for unsupported,
missing, or unreadable paths whose names do not identify a supported archive.

```python
from klygo.archive.backend import is_archive

assert is_archive("dataset.zip")
assert not is_archive("notes.txt")
```

Use `detect_format` instead when the failure reason should be visible.

