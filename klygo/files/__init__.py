"""
Bộ công cụ Quản lý File System & I/O Dữ liệu đa định dạng (`klygo.files`).

Hướng dẫn sử dụng (Google Colab):
https://colab.research.google.com/drive/1-Oo8ERqSuxns1OfZAdHY5jVMTrLVpJG-?usp=sharing

Định dạng I/O giải mã dữ liệu hỗ trợ (14 loại):
  - YAML (.yaml, .yml), JSON (.json), JSON Lines (.jsonl), TOML (.toml)
  - CSV (.csv), TXT (.txt), LOG (.log), INI (.ini), CFG (.cfg), PROPERTIES (.properties)
  - ENV (.env), XML (.xml), Pickle (.pkl, .pickle)

Định dạng Tải xuống (`files.download`):
  - Hỗ trợ TẤT CẢ các định dạng tập tin nhị phân (AI Models: .pt, .onnx, .safetensors;
    File nén: .zip, .tar.gz, .7z; Media: .mp4, .png, .jpg; Data: .parquet, .db, .whl, v.v.)

Danh sách 23 APIs:
  1.  load(path, ...) - Đọc file dữ liệu tự động theo phần mở rộng đuôi file
  2.  save(path, data, ...) - Ghi dữ liệu ra file dựa theo đuôi mở rộng
  3.  convert(source, target, ...) - Chuyển đổi định dạng file dữ liệu
  4.  download(source, output_dir, ...) - Tải tập tin bất kỳ từ URL/Colab về máy/thư mục có ProgressBar (giữ nguyên tên gốc)
  5.  exists(path) - Kiểm tra sự tồn tại của file hoặc thư mục
  6.  is_file(path) - Kiểm tra đường dẫn có phải là file không
  7.  is_dir(path) - Kiểm tra đường dẫn có phải là thư mục không
  8.  list(path, ...) - Liệt kê các tập tin/thư mục con
  9.  find(path, ...) - Tìm kiếm file theo mẫu wildcard
  10. walk(path) - Duyệt cây thư mục dạng generator
  11. mkdir(path, ...) - Tạo thư mục mới trên ổ đĩa
  12. copy(source, target, ...) - Sao chép file hoặc thư mục
  13. move(source, target, ...) - Di chuyển file hoặc thư mục
  14. rename(path, new_name, ...) - Đổi tên file hoặc thư mục
  15. remove(path, ...) - Xóa file hoặc thư mục
  16. info(path) - Chi tiết thông tin metadata (dung lượng, hash, time...)
  17. size(path, ...) - Dung lượng file hoặc thư mục (bytes/human)
  18. hash(path, ...) - Tính mã checksum hash MD5/SHA256
  19. compare(path1, path2, ...) - So sánh nội dung 2 file
  20. name(path) - Tên file/thư mục kèm phần mở rộng
  21. stem(path) - Tên file không kèm phần mở rộng
  22. extension(path) - Phần mở rộng đuôi file
  23. parent(path) - Thư mục cha chứa file/thư mục
"""

from .operations import (
    load,
    save,
    convert,
    download,
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
    "download",
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
