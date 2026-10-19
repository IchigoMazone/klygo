# Working with `pathlib.Path`

`files.path()` returns Python's standard `pathlib.Path`, using the current
operating system's concrete path type. Native properties and methods are
available immediately. Klygo does not add methods such as `.load()` or `.save()`
to the returned object: pass it to `files.load(p)` or `files.save(p, data)`.

This guide targets Python 3.10 and newer. Method availability and some details
of filesystem errors depend on the Python version and platform.

## Path construction and input types

`files.path(value, expand_user=True)` accepts `str` and `os.PathLike[str]`,
including `Path` and custom objects whose `__fspath__()` returns a string.
Byte paths are not supported. All public file APIs accept these text path inputs.

`files.path()` expands `~` by default. It does not create entries, expand
environment variables, make paths absolute, or resolve `..`. `files.normalize()`
normalizes components lexically; `files.resolve()` makes paths absolute and
resolves symlinks. Relative paths are interpreted from the working directory.

Home expansion remains specific to `files.path()` and the path helpers that
call it. I/O, metadata, filesystem operations, `join()`, and `download()` do not
implicitly expand `~`; pass `files.path("~/data")` when expansion is needed.

```python
from pathlib import Path
from klygo import files

p = files.path("dataset/images/cat.jpg")
assert isinstance(p, Path)
assert p.name == "cat.jpg"
assert p.parent.as_posix() == "dataset/images"
assert files.path("~", expand_user=False) == Path("~")
```

`str(p)` uses native separators (including backslashes on Windows).
Use `p.as_posix()` for a display string with `/`. Parsing follows the host's
path rules; use `PureWindowsPath` or `PurePosixPath` for lexical manipulation
of another operating system's paths.

## Properties and methods

Properties have no parentheses; methods do. Path construction and lexical
inspection do not require the represented file to exist.

| Native API | Meaning | Related Klygo API |
|---|---|---|
| `p.name`, `p.stem`, `p.suffix` | Filename, stem, final suffix | `name()`, `stem()`, `extension()` |
| `p.suffixes` | List of suffixes | `extensions()` returns a tuple |
| `p.parent`, `p.parents` | Parent and ancestors | `parent()`, `parents()`; the latter returns a tuple |
| `p.parts`, `p.drive`, `p.anchor` | Components and path root information | Use native properties |
| `p.exists()`, `p.is_file()`, `p.is_dir()` | Inspect the target | `exists()`, `is_file()`, `is_dir()` |
| `p.is_symlink()` | Inspect the link itself, including a broken link | Use the native method |
| `p / "child"`, `p.joinpath("child")` | Combine path components | `join()` |
| `p.stat()`, `p.lstat()` | Target metadata or link metadata | `info()` additionally computes file or directory size; use `hash()` separately for a checksum |

For Python 3.14+, status predicates such as `exists()` suppress all `OSError`
exceptions and return `False`. Earlier versions propagate some errors. Use
`stat()` when the distinction between missing and inaccessible matters.

## Path changes and filesystem changes

`with_name()`, `with_stem()`, and `with_suffix()` return new path objects.
They do not rename files. Use `p.rename(destination)` or `files.rename()` for
an actual rename. Both interfaces keep their own overwrite and destination
interpretation rules; they are not interchangeable in every case.

```python
from pathlib import Path
from klygo import files

p = files.path("dataset/cat.jpg")
q = p.with_name("dog.jpg")
assert p.name == "cat.jpg"
assert q.name == "dog.jpg"

archive = files.path("backup.tar.gz")
assert archive.with_suffix(".zip").name == "backup.tar.zip"
assert files.with_extension(archive, "zip").name == "backup.zip"
```

`files.with_extension()` accepts a suffix with or without the leading dot,
and replaces `.tar.gz`, `.tar.xz`, or `.tar.bz2` as one unit. In contrast,
`files.compound_extension()` joins *all* suffixes: `report.v1.json` produces
`.v1.json`. The latter is not a registry of recognized archive formats.

| Operation | Difference |
|---|---|
| `p.relative_to(base)` | Lexical operation; with default options, raises `ValueError` outside the base. Python 3.12+ adds `walk_up=True`. |
| `files.relative(p, base)` | Uses `os.path.relpath`; can return `..`, and can fail across Windows drives. |
| `p.is_relative_to(base)` | Lexical check; does not resolve symlinks or `..`. |
| `files.is_within(p, base)` | Resolves both paths by default. With `resolve_paths=False`, normalizes lexically before checking. |

A containment check describes the paths at the time of the check; it does not
reserve them or prevent a later filesystem change. Likewise, `unique_path()`
returns an unused candidate, not an exclusively reserved filename.

## Native I/O and Klygo workflows

| Need | Native API | Klygo behavior to consider |
|---|---|---|
| Text or binary I/O | `open()`, `read_text()`, `write_text()`, `read_bytes()`, `write_bytes()` | `load()` and `save()` select structured formats by extension |
| Empty file / modification time | `touch()` | Parent directory must already exist |
| Create a directory | `mkdir(parents=False, exist_ok=False)` returns `None` | `files.mkdir()` defaults both options to `True` and returns `Path` |
| Traverse children or glob matches | `iterdir()`, `glob()`, `rglob()` | `find()` returns a sorted file list; `iter_find()` yields files and prunes excluded directories |
| Compare file identity | `samefile(other)` | `files.compare()` compares contents, not identity |
| Incremental structured records | Use streams and format parsers | `iter_load()` yields JSONL values or CSV records |

`write_text()` and `write_bytes()` overwrite content. Native whole-file reads
load the complete content into memory. Use a context manager with `open()`
for stream access. Glob matches can include directories; traversal order is
not guaranteed unless explicitly sorted.

`files.save(..., atomic=True)` writes a complete temporary file before
publication; see its [contract](api/save.md) for overwrite, hard-link,
symlink, and metadata behavior. `files.info()` returns metadata without
reading file contents; use `files.hash()` when a checksum is needed.

Keep `files.copy()` and `files.move()` when targeting Python 3.10: native
`Path.copy()` and `Path.move()` were added in Python 3.14 and have their own
metadata and overwrite behavior.

## Symbolic links

A symlink stores a target path. `is_file()`, `is_dir()`, and `exists()` follow
that target by default; `is_symlink()` inspects the link. A broken link has
`exists() == False` while `is_symlink() == True`.

Use `link.symlink_to(target)` to create a link, `link.readlink()` to inspect its
stored target, and `link.resolve()` to follow it. Relative symlink targets are
interpreted relative to the link's parent. An absolute target is often clearer
in a first example. `link.unlink()` removes the link without deleting its target.
`files.remove(link)` also handles broken links; `files.unique_path(link)` treats
a broken link's name as occupied.

Windows may require Developer Mode or symlink privileges. For a directory link,
pass `target_is_directory=True`, particularly when the target does not yet exist.
The runnable example below skips only expected capability/permission failures.

## Complete executable example

The example uses temporary directories and can be run repeatedly. Its source is
[`native_pathlib.py`](../../examples/files/native_pathlib.py). It covers native
operations, format-aware I/O, and both live and broken symlinks when supported.

```python
"""Executable examples of native pathlib integration with klygo.files."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

p = files.path("dataset/images/cat.jpg")
assert isinstance(p, Path)
assert p.name == "cat.jpg"
assert p.parent.as_posix() == "dataset/images"
assert files.path("~", expand_user=False) == Path("~")
assert files.path("~") == Path.home()

archive = files.path("backup.tar.gz")
assert archive.with_suffix(".zip").name == "backup.tar.zip"
assert files.with_extension(archive, "zip").name == "backup.zip"
assert files.compound_extension("report.v1.json") == ".v1.json"

with TemporaryDirectory() as directory:
    root = files.path(directory)
    note = root / "note.txt"
    note.write_text("Hello\n", encoding="utf-8")
    assert note.is_file()
    assert note.read_text(encoding="utf-8") == "Hello\n"

    renamed = note.with_name("renamed.txt")
    assert note.exists() and not renamed.exists()
    note.rename(renamed)
    assert renamed.exists() and not note.exists()

    config = root / "config.json"
    files.save(config, {"epochs": 10}, atomic=True, verbose=False)
    assert files.load(config, verbose=False) == {"epochs": 10}
    assert files.info(config)["size"] > 0
    assert files.is_within(config, root)
    assert not files.is_within(root / ".." / "outside.txt", root)

    link = root / "linked.txt"
    try:
        link.symlink_to(renamed.resolve())
    except OSError as error:
        if getattr(error, "winerror", None) != 1314 and error.errno not in (1, 13, 95):
            raise
        print("Symlink example skipped: this environment cannot create symlinks.")
    else:
        assert link.is_symlink() and link.is_file()
        assert link.readlink() == renamed.resolve()
        assert link.read_text(encoding="utf-8") == "Hello\n"
        renamed.unlink()
        assert not link.exists() and link.is_symlink()
        assert files.unique_path(link) != link
        files.remove(link, missing_ok=False)
        assert not link.is_symlink()
```

## Further reading

- [Path construction](api/path.md)
- [Streaming file discovery](api/iter_find.md)
- [Streaming structured data](api/iter_load.md)
- [Python 3.10 pathlib reference](https://docs.python.org/3.10/library/pathlib.html)
- [Current Python pathlib reference](https://docs.python.org/3/library/pathlib.html)
