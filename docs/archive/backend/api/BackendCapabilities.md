# `backend.BackendCapabilities`

```python
from klygo.archive.backend import BackendCapabilities

capabilities = BackendCapabilities(
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

Immutable capability declaration attached to every archive backend.

## Contract

`BackendCapabilities` is a frozen dataclass with slots that explicitly enumerates supported mutating operations and non-universal option flags for an archive backend. Read operations (extract, list, search, metadata, test) are universal and mandatory in `ArchiveBackend`, so only optional mutation operations and variable options are declared here.

## Parameters

- `compress`: `bool`, default=False — Whether archive creation is supported.
- `add`: `bool`, default=False — Whether adding members to an existing archive is supported.
- `remove`: `bool`, default=False — Whether deleting members from an existing archive is supported.
- `merge`: `bool`, default=False — Whether same-format fast merging is supported.
- `split`: `bool`, default=False — Whether size-limited splitting is supported.
- `compress_options`: `frozenset[str]`, default=frozenset() — Supported non-default `compress` options (e.g. `{"compresslevel", "method"}`).
- `extract_options`: `frozenset[str]`, default=frozenset() — Supported non-default `extract` options (e.g. `{"password", "include", "exclude"}`).
- `add_options`: `frozenset[str]`, default=frozenset() — Supported non-default `add` options (e.g. `{"on_conflict"}`).

## Returns

`BackendCapabilities`
    An immutable instance describing supported operations and option parameters.

## Errors and edge cases

`BackendCapabilities` is frozen (`frozen=True`); attempting to modify any attribute at runtime raises `FrozenInstanceError`.

## AI usage guidance

Inspect `backend.capabilities` when writing tools, dynamic dispatchers, or tests to verify format support before invoking operations like `add()`, `remove()`, or `merge()`.

## Example

See [`BackendCapabilities.py`](../../../../examples/archive/backend/BackendCapabilities.py).

## Tests

See [`test_BackendCapabilities.py`](../../../../test/archive/backend/test_BackendCapabilities.py).

## Complete executable example

```python
from klygo.archive.backend import BackendCapabilities

caps = BackendCapabilities(
    compress=True,
    add=True,
    compress_options=frozenset({"compresslevel", "method"}),
)

assert caps.compress is True
assert caps.remove is False
assert "compresslevel" in caps.compress_options
```
