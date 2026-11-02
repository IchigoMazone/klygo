"""Backward-compatible aggregate for the split :mod:`klygo.files` modules.

New code should import the public namespace with ``from klygo import files``.
Direct imports from ``klygo.files.operations`` remain supported.
"""

from .download import download
from .filesystem import (
    copy,
    exists,
    find,
    is_dir,
    is_file,
    list_entries,
    mkdir,
    move,
    remove,
    rename,
    walk,
)
from .io import convert, load, save
from .metadata import compare, hash, info, size
from .paths import (
    common_path,
    compound_extension,
    extension,
    extensions,
    is_absolute,
    is_within,
    join,
    name,
    normalize,
    parent,
    parents,
    path,
    relative,
    replace_root,
    resolve,
    stem,
    unique_path,
    with_extension,
    with_name,
    with_stem,
)

__all__ = [
    "load", "save", "convert", "download", "exists", "is_file", "is_dir",
    "list_entries", "find", "walk", "mkdir", "copy", "move", "rename",
    "remove", "info", "size", "hash", "compare", "name", "stem",
    "extension", "parent", "path", "join", "normalize", "resolve",
    "relative", "is_within", "common_path", "replace_root", "with_name",
    "with_stem", "with_extension", "unique_path", "extensions",
    "compound_extension", "parents", "is_absolute",
]
