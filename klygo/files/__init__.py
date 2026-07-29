"""
Bộ công cụ Thao tác File & Thư mục đa năng (`klygo.files`).

Danh sách 22 APIs:
  1. load(path, ...) - Đọc dữ liệu (YAML, JSON, JSONL, TOML, CSV, TXT, LOG) hoặc ảnh
  2. save(path, data, ...) - Ghi dữ liệu ra file dựa theo đuôi mở rộng
  3. convert(source, target, ...) - Chuyển đổi định dạng file dữ liệu
  4. exists(path) - Kiểm tra sự tồn tại
  5. is_file(path) - Kiểm tra có phải file
  6. is_dir(path) - Kiểm tra có phải thư mục
  7. list(path, ...) - Liệt kê file/thư mục
  8. find(path, ...) - Tìm kiếm file theo pattern
  9. walk(path) - Duyệt cây thư mục
  10. mkdir(path, ...) - Tạo thư mục
  11. copy(source, target, ...) - Sao chép file/thư mục
  12. move(source, target, ...) - Di chuyển file/thư mục
  13. rename(path, new_name, ...) - Đổi tên file/thư mục
  14. remove(path, ...) - Xóa file/thư mục
  15. info(path) - Chi tiết thông tin file/thư mục
  16. size(path, ...) - Dung lượng file/thư mục
  17. hash(path, ...) - Tính mã checksum hash
  18. compare(path1, path2, ...) - So sánh 2 file
  19. name(path) - Tên file kèm đuôi
  20. stem(path) - Tên file không kèm đuôi
  21. extension(path) - Đuôi file mở rộng
  22. parent(path) - Thư mục cha
"""

from .operations import (
    load,
    save,
    convert,
    exists,
    is_file,
    is_dir,
    list,
    find,
    walk,
    mkdir,
    copy,
    move,
    rename,
    remove,
    info,
    size,
    hash,
    compare,
    name,
    stem,
    extension,
    parent,
)

__all__ = [
    "load",
    "save",
    "convert",
    "exists",
    "is_file",
    "is_dir",
    "list",
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
]
