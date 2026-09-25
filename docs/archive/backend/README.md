# Archive backend API

`klygo.archive.backend` is the extension layer below the public
`klygo.archive` facade. Application code should normally use the facade;
backend APIs are intended for capability inspection, controlled format
selection, backend development, and low-level testing.

## Selection flow

1. `detect_format(path)` returns a canonical format name.
2. `get_backend(path, format_hint=None)` creates the registered adapter.
3. The facade calls `require_operation()` and `validate_option()`.
4. The selected backend performs the format-specific operation.

## Capability matrix

| Backend | Read/extract | Compress | Add | Remove | Merge | Split | Optional dependency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ZipBackend` | yes | yes | yes | yes | yes | yes | none |
| `TarBackend` | yes | yes | yes | yes | yes | yes | none |
| `GZipBackend` | yes | yes | no | no | no | no | none |
| `SevenZipBackend` | yes | yes | no | no | no | no | `py7zr` |
| `RarBackend` | yes | no | no | no | no | no | `rarfile` plus a compatible extraction tool when required |

Install optional format support independently:

```bash
# pip
pip install "klygo[py7zr]"
pip install "klygo[rarfile]"

# add Klygo to a uv-managed project
uv add "klygo[py7zr]"
uv add "klygo[rarfile]"

# install into the current uv environment without changing project metadata
uv pip install "klygo[py7zr]"
uv pip install "klygo[rarfile]"
```

Both extras can be selected together as `klygo[py7zr,rarfile]`.

Read operations (`extract`, `extract_file`, `list_files`, `iter_files`,
`search`, `get_info`, and `test`) are mandatory in the abstract contract and
therefore are not capability flags.

## Public symbols

- [`ArchiveBackend`](ArchiveBackend.md)
- [`BackendCapabilities`](BackendCapabilities.md)
- [`UnsupportedOperationError`](UnsupportedOperationError.md)
- [`UnsupportedOptionError`](UnsupportedOptionError.md)
- [`detect_format`](detect_format.md)
- [`is_archive`](is_archive.md)
- [`get_backend`](get_backend.md)
- [`ZipBackend`](ZipBackend.md)
- [`TarBackend`](TarBackend.md)
- [`GZipBackend`](GZipBackend.md)
- [`SevenZipBackend`](SevenZipBackend.md)
- [`RarBackend`](RarBackend.md)

## Stability boundary

The symbols exported by `klygo.archive.backend.__all__` form the supported
backend API. Registry constants, private helpers, and implementation modules are
internal and may change without preserving compatibility.
