"""Create, extract, inspect, and transform archives.

Colab tutorial:
    https://colab.research.google.com/drive/1z5cuH35jFOkTtSkS4IcufF8ZygnylEJS?usp=sharing

Public API (21 functions, 1 class):
    I/O (io.py):
          1. compress(...)             Create an archive from a file or directory.
          2. extract(...)              Extract all or selected archive members.
          3. extract_file(...)         Extract one member by exact name.
          4. extract_matching(...)     Extract members matching a pattern.

    Metadata (metadata.py):
          5. list_files(...)           Return all stored member names.
          6. iter_files(...)           Yield stored member names lazily.
          7. search(...)               Find members by glob or regular expression.
          8. get_info(...)             Report format, size, and member metadata.
          9. test(...)                 Check archive integrity.
         10. verify(...)               Build a detailed validation report.
         11. compare(...)              Compare member names in two archives.
         12. detect_format(...)        Identify an archive by name or content.
         13. is_archive(...)           Check whether a file is a supported archive.

    Operations (operations.py):
         14. add(...)                  Add files or directories to an archive.
         15. remove(...)               Remove named members from an archive.
         16. merge(...)                Combine multiple archives.
         17. split_by_size(...)        Create size-limited archive parts.
         18. convert(...)              Convert to another archive format.
         19. recompress(...)           Rebuild with a new format or compression level.
         20. copy(...)                 Copy an archive without extracting it.

    Context (context.py):
         21. open(...)                 Open an ArchiveFile context wrapper.
         22. ArchiveFile               Class for repeated archive operations.

Formats: ZIP, TAR, compressed TAR, and single-file GZip.
Optional: 7-Zip via py7zr; read-only RAR via rarfile.
Filesystem work uses klygo.files; backends handle archive contents.
"""

from .io import compress, extract, extract_file, extract_matching
from .metadata import (
    compare,
    detect_format,
    get_info,
    is_archive,
    iter_files,
    list_files,
    search,
    test,
    verify,
)
from .operations import (
    add,
    convert,
    copy,
    merge,
    recompress,
    remove,
    split_by_size,
)
from .context import open_archive as open, ArchiveFile

__all__ = [
    "compress",
    "extract",
    "extract_file",
    "extract_matching",
    "list_files",
    "iter_files",
    "search",
    "get_info",
    "test",
    "verify",
    "compare",
    "detect_format",
    "is_archive",
    "add",
    "remove",
    "merge",
    "split_by_size",
    "convert",
    "recompress",
    "copy",
    "open",
    "ArchiveFile",
]
