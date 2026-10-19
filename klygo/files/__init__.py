"""Work with paths, filesystems, metadata, and structured data.

Colab tutorial:
    https://colab.research.google.com/drive/1-Oo8ERqSuxns1OfZAdHY5jVMTrLVpJG-?usp=sharing

Public API (43 functions):
    Structured data I/O (io.py):
          1. load(...)                 Read a file using its extension.
          2. iter_load(...)            Stream JSONL or CSV records.
          3. save(...)                 Write structured data using its extension.
          4. convert(...)              Convert between structured-data formats.

    Downloads (download.py):
          5. download(...)             Download a URL or copy a local file.

    Filesystem (filesystem.py):
          6. exists(...)               Check whether a path exists.
          7. is_file(...)              Check for a regular file.
          8. is_dir(...)               Check for a directory.
          9. is_symlink(...)           Check for a symbolic link, including a broken one.
         10. list_entries(...)         List files and directories by pattern.
         11. find(...)                 Find matching regular files.
         12. iter_find(...)            Stream matching files with directory exclusions.
         13. walk(...)                 Walk a directory tree.
         14. mkdir(...)                Create a directory and optional parents.
         15. copy(...)                 Copy a file or directory.
         16. move(...)                 Move a file or directory.
         17. rename(...)               Rename or relocate a path.
         18. remove(...)               Remove a file or directory.

    Metadata (metadata.py):
         19. info(...)                 Return detailed metadata.
         20. size(...)                 Calculate file or directory size.
         21. hash(...)                 Calculate a file checksum.
         22. compare(...)              Compare two files by hash or content.

    Paths (paths.py):
         23. path(...)                 Convert an input to pathlib.Path.
         24. join(...)                 Join path components.
         25. normalize(...)            Normalize a path lexically.
         26. resolve(...)              Resolve to an absolute path.
         27. relative(...)             Compute a relative path.
         28. as_posix(...)             Represent a path with forward slashes.
         29. name(...)                 Return the final path component.
         30. stem(...)                 Return the filename without its last suffix.
         31. extension(...)            Return the last filename suffix.
         32. extensions(...)           Return all filename suffixes.
         33. compound_extension(...)   Return all suffixes joined.
         34. parent(...)               Return the immediate parent.
         35. parents(...)              Return all ancestor paths.
         36. is_absolute(...)          Check whether a path is absolute.
         37. is_within(...)            Check resolved path containment.
         38. common_path(...)          Find the longest common parent.
         39. replace_root(...)         Map a path to another root.
         40. with_name(...)            Replace the final path component.
         41. with_stem(...)            Replace the filename stem.
         42. with_extension(...)       Replace the filename extension.
         43. unique_path(...)          Find an unused numbered path.

Inputs accept strings and text-returning os.PathLike[str] objects.
Structured I/O covers JSON, JSONL, YAML, TOML, CSV, text, and more.
"""

from .download import download
from .filesystem import (
    exists,
    is_file,
    is_dir,
    is_symlink,
    list_entries,
    find,
    iter_find,
    walk,
    mkdir,
    copy,
    move,
    rename,
    remove,
)
from .io import load, iter_load, save, convert
from .metadata import (
    info,
    size,
    hash,
    compare,
)
from .paths import (
    as_posix,
    name,
    stem,
    extension,
    parent,
    path,
    join,
    normalize,
    resolve,
    relative,
    is_within,
    common_path,
    replace_root,
    with_name,
    with_stem,
    with_extension,
    unique_path,
    extensions,
    compound_extension,
    parents,
    is_absolute,
)

__all__ = [
    "load",
    "iter_load",
    "save",
    "convert",
    "download",
    "exists",
    "is_file",
    "is_dir",
    "is_symlink",
    "list_entries",
    "find",
    "iter_find",
    "walk",
    "mkdir",
    "copy",
    "move",
    "rename",
    "remove",
    "info",
    "size",
    "hash",
    "compare",
    "as_posix",
    "name",
    "stem",
    "extension",
    "parent",
    "path",
    "join",
    "normalize",
    "resolve",
    "relative",
    "is_within",
    "common_path",
    "replace_root",
    "with_name",
    "with_stem",
    "with_extension",
    "unique_path",
    "extensions",
    "compound_extension",
    "parents",
    "is_absolute",
]
