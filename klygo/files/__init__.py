"""Filesystem Management & Multi-Format Data I/O Utilities (`klygo.files`).

Interactive Google Colab Tutorial:
    https://colab.research.google.com/drive/1-Oo8ERqSuxns1OfZAdHY5jVMTrLVpJG-?usp=sharing

Supported Structured Data I/O Formats:
    - Structured & Config: YAML (.yaml, .yml), JSON (.json), JSON Lines (.jsonl), TOML (.toml)
    - Tabular & Text: CSV (.csv), TXT (.txt), LOG (.log), INI (.ini), CFG (.cfg), PROPERTIES (.properties)
    - Environment & Markup: ENV (.env), XML (.xml), Pickle (.pkl, .pickle)

Supported File Download Formats (`files.download`):
    - All binary and arbitrary formats (AI Models: .pt, .onnx, .safetensors; Archives: .zip, .tar.gz, .7z;
      Media: .mp4, .png, .jpg; Datasets & Binaries: .parquet, .db, .whl, etc.)

Public APIs (39 Functions):
    Structured Data I/O:
        1.  load(path, ...)                  - Automatically load structured data based on file extension.
        2.  save(path, data, ...)            - Save structured data using the destination file extension.
        3.  convert(source, target, ...)     - Direct format conversion between structured data files.
        4.  download(source, output_dir, ...) - Download file from URL/Colab or copy local file with progress bar.

    Filesystem Operations:
        5.  exists(path)                     - Check whether a file or directory exists.
        6.  is_file(path)                    - Check if path points to a regular file.
        7.  is_dir(path)                     - Check if path points to a directory.
        8.  list_entries(path, ...)          - List matching files and directories in a directory.
        9.  find(path, ...)                  - Find files matching a wildcard pattern recursively.
        10. walk(path)                       - Walk directory tree as a generator (os.walk wrapper).
        11. mkdir(path, ...)                 - Create a new directory and missing parent directories.
        12. copy(source, target, ...)        - Copy a file or directory recursively.
        13. move(source, target, ...)        - Move a file or directory to a new target.
        14. rename(path, new_name, ...)      - Rename or relocate a file or directory.
        15. remove(path, ...)                - Remove a file or directory recursively.

    File Metadata & Inspection:
        16. info(path)                       - Detailed metadata dictionary (size, hashes, timestamps, etc.).
        17. size(path, ...)                  - Total size of a file or directory (bytes or human-readable format).
        18. hash(path, ...)                  - Compute file checksum digest (MD5, SHA256, etc.).
        19. compare(path1, path2, ...)       - Compare contents of two files by checksum or binary content.

    Path Manipulation & Inspection:
        20. name(path)                       - Final path component including file extension.
        21. stem(path)                       - Final path component without its last extension.
        22. extension(path)                  - Final filename extension (including leading dot).
        23. parent(path)                     - Immediate parent directory of a path.
        24. path(value, ...)                 - Convert input value to a normalized pathlib.Path object.
        25. join(*parts)                     - Join multiple path components together.
        26. normalize(path)                  - Lexically normalize path separators and dot components.
        27. resolve(path, ...)               - Resolve path to an absolute normalized path.
        28. relative(path, start)            - Compute relative path from a starting directory.
        29. is_within(path, root)            - Check whether a path is contained within a root directory.
        30. common_path(paths)               - Find the longest common parent directory of multiple paths.
        31. replace_root(path, old_root, new_root) - Map a path from one root directory to another.
        32. with_name(path, name)            - Return path with replaced final component.
        33. with_stem(path, stem)            - Return path with replaced stem.
        34. with_extension(path, extension)  - Return path with replaced extension (supports .tar.gz).
        35. unique_path(path, ...)           - Generate an unused candidate path with numeric suffix.
        36. extensions(path)                 - Return all filename extensions as a tuple.
        37. compound_extension(path)         - Return all filename extensions joined together.
        38. parents(path)                    - Return tuple of ancestor paths from nearest to farthest.
        39. is_absolute(path)                - Check whether a path is absolute.
"""

from .download import download
from .filesystem import (
    exists,
    is_file,
    is_dir,
    list_entries,
    find,
    walk,
    mkdir,
    copy,
    move,
    rename,
    remove,
)
from .io import load, save, convert
from .metadata import (
    info,
    size,
    hash,
    compare,
)
from .paths import (
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
    "save",
    "convert",
    "download",
    "exists",
    "is_file",
    "is_dir",
    "list_entries",
    "find",
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
