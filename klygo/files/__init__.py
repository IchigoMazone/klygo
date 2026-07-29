"""
Bộ công cụ Quản lý File System & I/O Dữ liệu đa định dạng (`klygo.files`).

Hướng dẫn sử dụng (Google Colab):
https://colab.research.google.com/drive/1-Oo8ERqSuxns1OfZAdHY5jVMTrLVpJG-?usp=sharing

Định dạng hỗ trợ (14 loại):
  - YAML (.yaml, .yml), JSON (.json), JSON Lines (.jsonl), TOML (.toml)
  - CSV (.csv), TXT (.txt), LOG (.log), INI (.ini), CFG (.cfg), PROPERTIES (.properties)
  - ENV (.env), XML (.xml), Pickle (.pkl, .pickle)

Danh sách 22 APIs:
  1.  load(path, ...) - Đọc file dữ liệu tự động theo phần mở rộng đuôi file
  2.  save(path, data, ...) - Ghi dữ liệu ra file dựa theo đuôi mở rộng
  3.  convert(source, target, ...) - Chuyển đổi định dạng file dữ liệu
  4.  exists(path) - Kiểm tra sự tồn tại của file hoặc thư mục
  5.  is_file(path) - Kiểm tra đường dẫn có phải là file không
  6.  is_dir(path) - Kiểm tra đường dẫn có phải là thư mục không
  7.  list(path, ...) - Liệt kê các tập tin/thư mục con
  8.  find(path, ...) - Tìm kiếm file theo mẫu wildcard
  9.  walk(path) - Duyệt cây thư mục dạng generator
  10. mkdir(path, ...) - Tạo thư mục mới trên ổ đĩa
  11. copy(source, target, ...) - Sao chép file hoặc thư mục
  12. move(source, target, ...) - Di chuyển file hoặc thư mục
  13. rename(path, new_name, ...) - Đổi tên file hoặc thư mục
  14. remove(path, ...) - Xóa file hoặc thư mục
  15. info(path) - Chi tiết thông tin metadata (dung lượng, hash, time...)
  16. size(path, ...) - Dung lượng file hoặc thư mục (bytes/human)
  17. hash(path, ...) - Tính mã checksum hash MD5/SHA256
  18. compare(path1, path2, ...) - So sánh nội dung 2 file
  19. name(path) - Tên file/thư mục kèm phần mở rộng
  20. stem(path) - Tên file không kèm phần mở rộng
  21. extension(path) - Phần mở rộng đuôi file
  22. parent(path) - Thư mục cha chứa file/thư mục
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
