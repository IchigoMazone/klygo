"""
Bộ công cụ quản lý Archive đa định dạng (ZIP, TAR, TAR.GZ, TAR.XZ, GZ, 7Z, RAR).

Hướng dẫn sử dụng (Google Colab):
https://colab.research.google.com/drive/1CYtOv1nz-lujPiQA_f50HRwEdN5FVVnE?usp=sharing

Định dạng hỗ trợ:
  - Nén (Compress): ZIP, TAR, TAR.GZ, TAR.XZ, GZ (chỉ file lẻ), 7Z
  - Giải nén (Extract): ZIP, TAR, TAR.GZ, TAR.XZ, GZ, 7Z, RAR (chỉ đọc/giải nén)

Danh sách hàm public (23 APIs):
  1.  compress(source, output_path, ...) - Nén file/thư mục
  2.  extract(archive_path, output_dir, ...) - Giải nén toàn bộ archive
  3.  extract_file(archive_path, filename, ...) - Giải nén 1 file cụ thể bằng Streaming I/O
  4.  extract_matching(archive_path, pattern, ...) - Giải nén file theo wildcard
  5.  list_files(archive_path) - Lấy danh sách file trong archive
  6.  iter_files(archive_path) - Duyệt file bằng Generator (tiết kiệm RAM)
  7.  search(archive_path, pattern, regex=False) - Tìm kiếm file theo wildcard hoặc Regex
  8.  get_info(archive_path) - Lấy metadata chi tiết (tỷ lệ nén, kích thước...)
  9.  test(archive_path) - Kiểm tra tính toàn vẹn CRC (True/False)
  10. verify(archive_path) - Báo cáo xác minh toàn diện từng file
  11. add(archive_path, files, on_conflict='rename') - Thêm file vào archive
  12. remove(archive_path, files) - Xóa file khỏi archive
  13. merge(archive_paths, output_path) - Gộp nhiều archive thành 1 (hỗ trợ cross-format)
  14. split_by_size(archive_path, size) - Chia archive thành nhiều phần theo MB
  15. convert(source_path, target_path) - Chuyển đổi định dạng archive
  16. recompress(source_path, target_path, level=6) - Nén lại với level khác
  17. copy(source_path, target_path) - Sao chép file archive an toàn
  18. compare(archive1, archive2) - So sánh nội dung 2 archive (added/removed/common)
  19. open(archive_path) - Context Manager mở archive làm việc dạng OOP
  20. ArchiveFile(archive_path) - Lớp OOP thao tác archive
  21. detect_format(archive_path) - Tự động nhận dạng định dạng qua Magic Bytes
  22. is_archive(archive_path) - Kiểm tra file có phải archive không
  23. human_size(size_in_bytes) - Đổi dung lượng byte sang chuỗi KB/MB/GB

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
info = get_info
list = list_files

__all__ = [
    "compress",
    "extract",
    "extract_file",
    "extract_matching",
    "list_files",
    "list",
    "iter_files",
    "search",
    "get_info",
    "info",
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
