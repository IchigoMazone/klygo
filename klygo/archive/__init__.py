"""
Bộ công cụ quản lý Archive đa định dạng (ZIP, TAR, TAR.GZ, TAR.XZ, GZ, 7Z, RAR).

Demo Google Colab:
https://colab.research.google.com/drive/1CYtOv1nz-lujPiQA_f50HRwEdN5FVVnE?usp=sharing

Nguồn: TrinhNhuNhat_28072026.
"""

from klygo.archive.compress import compress

from klygo.archive.extract import extract, extract_file, extract_matching
from klygo.archive.list import list_files, iter_files, search
from klygo.archive.info import get_info, test, verify
from klygo.archive.modify import add, remove
from klygo.archive.transform import merge, split_by_size, convert, recompress, copy
from klygo.archive.compare import compare
from klygo.archive.context import open_archive as open, ArchiveFile
from klygo.archive.backend import detect_format, is_archive
from klygo.archive.human_size import human_size

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
    "add",
    "remove",
    "merge",
    "split_by_size",
    "convert",
    "recompress",
    "copy",
    "compare",
    "open",
    "ArchiveFile",
    "detect_format",
    "is_archive",
    "human_size",
]
