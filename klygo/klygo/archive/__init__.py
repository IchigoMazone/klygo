"""Archive Creation, Extraction & Transformation Utilities (`klygo.archive`).

Interactive Google Colab Tutorial:
    https://colab.research.google.com/drive/1CYtOv1nz-lujPiQA_f50HRwEdN5FVVnE?usp=sharing

Supported Archive Formats:
    - Read & Write: ZIP (.zip), TAR (.tar), TAR.GZ (.tar.gz, .tgz), TAR.XZ (.tar.xz, .txz),
      TAR.BZ2 (.tar.bz2, .tbz2), and single-file GZip (.gz).
    - Optional Read & Write: 7-Zip (.7z), provided by the optional `py7zr` package.
    - Optional Read-Only: RAR (.rar), provided by the optional `rarfile` package.

Public APIs (21 Functions, 1 Class):
    Archive Creation & Extraction:
        1.  compress(source, output_path, ...)       - Create an archive from a file or directory.
        2.  extract(archive_path, output_dir, ...)   - Extract all or selected archive members.
        3.  extract_file(archive_path, filename, ...) - Extract one exact archive member.
        4.  extract_matching(archive_path, pattern, ...) - Extract members matching a wildcard.

    Member Discovery, Metadata & Validation:
        5.  list_files(archive_path)                 - Return all stored member names.
        6.  iter_files(archive_path)                 - Lazily iterate over stored member names.
        7.  search(archive_path, pattern, ...)       - Search member names by glob or regex.
        8.  get_info(archive_path)                   - Return format, size, ratio, and member metadata.
        9.  test(archive_path, ...)                  - Test archive integrity.
        10. verify(archive_path)                     - Build a high-level verification report.
        11. compare(archive1, archive2)              - Compare archive member-name sets.
        12. detect_format(path)                      - Detect format from extension or magic bytes.
        13. is_archive(path)                         - Check whether a path is a supported archive.

    Archive Modification & Transformation:
        14. add(archive_path, files, ...)            - Add files or directories to an archive.
        15. remove(archive_path, files)              - Remove named members from an archive.
        16. merge(archive_paths, output_path, ...)   - Merge multiple archives.
        17. split_by_size(archive_path, size, ...)   - Split an archive into size-limited parts.
        18. convert(source_path, target_path, ...)   - Convert an archive to another format.
        19. recompress(source_path, target_path, ...) - Rebuild with another level or format.
        20. copy(source_path, target_path, ...)      - Copy an archive without extracting it.

    Object Interface:
        21. open(archive_path)                       - Create an `ArchiveFile` context wrapper.
        22. ArchiveFile(archive_path)                - Object-oriented archive interface (class).
"""

from .io import compress, extract, extract_file, extract_matching
from .metadata import (
    compare,
    get_info,
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
from .backend import detect_format, is_archive

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
