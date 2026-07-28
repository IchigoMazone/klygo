"""
Bộ công cụ Đọc/Ghi dữ liệu (I/O) và quản lý File Cấu hình đa định dạng.

Định dạng hỗ trợ:
  - Đọc/Ghi dữ liệu: YAML (.yaml, .yml), JSON (.json), JSONL (.jsonl), TOML (.toml), CSV (.csv), TXT/LOG (.txt, .log)
  - Đọc tập tin ảnh: PNG, JPG, JPEG, WEBP, BMP, TIF, TIFF (PIL / OpenCV)

Danh sách hàm public (16 APIs):
  1. Config(config_path) - Quản lý cấu hình thông minh (Box dot-notation, default inheritance)
  2. read_file(path, ...) - Tự động nhận diện đuôi file và đọc dữ liệu
  3. read_yaml(path, ...) - Đọc file YAML
  4. read_json(path, ...) - Đọc file JSON
  5. read_jsonl(path, ...) - Đọc file JSON Lines (.jsonl)
  6. read_toml(path, ...) - Đọc file TOML
  7. read_csv(path, ...) - Đọc file CSV
  8. read_txt(path, ...) - Đọc file văn bản (.txt, .log)
  9. read_images(source, ...) - Đọc 1 hoặc nhiều ảnh (PIL / OpenCV)
  10. write_file(path, data, ...) - Tự động nhận diện đuôi file và ghi dữ liệu
  11. write_yaml(path, data, ...) - Ghi dữ liệu ra file YAML
  12. write_json(path, data, ...) - Ghi dữ liệu ra file JSON
  13. write_jsonl(path, data, ...) - Ghi danh sách ra file JSON Lines (.jsonl)
  14. write_toml(path, data, ...) - Ghi dữ liệu ra file TOML
  15. write_csv(path, data, ...) - Ghi danh sách dict ra file CSV
  16. write_txt(path, data, ...) - Ghi chuỗi/danh sách ra file văn bản (.txt, .log)

Nguồn: TrinhNhuNhat_28072026.
"""

from .config import Config
from .read import (
    read_yaml,
    read_json,
    read_jsonl,
    read_toml,
    read_csv,
    read_txt,
    read_file,
)
from .read_images import read_images
from .write import (
    write_yaml,
    write_json,
    write_jsonl,
    write_toml,
    write_csv,
    write_txt,
    write_file,
)

__all__ = [
    # Config class
    "Config",
    # Read
    "read_yaml",
    "read_json",
    "read_jsonl",
    "read_toml",
    "read_csv",
    "read_txt",
    "read_file",
    "read_images",
    # Write
    "write_yaml",
    "write_json",
    "write_jsonl",
    "write_toml",
    "write_csv",
    "write_txt",
    "write_file",
]
