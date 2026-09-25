# `BackendCapabilities`

```python
BackendCapabilities(
    compress=False,
    add=False,
    remove=False,
    merge=False,
    split=False,
    compress_options=frozenset(),
    extract_options=frozenset(),
    add_options=frozenset(),
)
```

Immutable capability declaration attached to every backend.

Operation flags answer whether a backend can perform a mutation. Option sets
answer whether a non-default facade option has real behavior for that backend.
Defaults do not need to be listed because they request no backend-specific
change.

```python
from klygo.archive.backend import get_backend

backend = get_backend("dataset.zip")
if backend.capabilities.add:
    print("members can be added")

print(sorted(backend.capabilities.compress_options))
```

The dataclass is frozen and slotted. Treat it as metadata; do not modify it at
runtime.

