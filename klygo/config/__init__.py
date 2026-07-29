"""
Bộ công cụ Quản lý File Cấu hình đa định dạng (`klygo.config`).

Danh sách 16 APIs:
  1. load(path, ...) - Đọc file cấu hình trả về Box dot-notation
  2. save(path, data, ...) - Ghi dữ liệu cấu hình ra file
  3. convert(source, target, ...) - Chuyển đổi định dạng file cấu hình
  4. create(path, ...) - Tạo file cấu hình mặc định
  5. defaults(...) - Lấy dictionary cấu hình mặc định
  6. merge(*configs, ...) - Gộp nhiều dictionary cấu hình
  7. update(config_data, updates, ...) - Cập nhật thông số cấu hình
  8. get(config_data, key_path, ...) - Truy xuất giá trị theo dot-notation
  9. set(config_data, key_path, value) - Gán giá trị theo dot-notation
  10. has(config_data, key_path) - Kiểm tra sự tồn tại của key path
  11. delete(config_data, key_path) - Xóa key path khỏi cấu hình
  12. keys(config_data, flat=False) - Danh sách các key (hỗ trợ flat path)
  13. values(config_data, flat=False) - Danh sách các giá trị
  14. items(config_data, flat=False) - Danh sách các cặp (key, value)
  15. validate(config_data, ...) - Kiểm tra tính hợp lệ của cấu hình
  16. export(source, target, ...) - Xuất cấu hình sang định dạng khác
"""

from .config import Config
from .operations import (
    load,
    save,
    convert,
    create,
    defaults,
    merge,
    update,
    get,
    set,
    has,
    delete,
    keys,
    values,
    items,
    validate,
    export,
)

__all__ = [
    "Config",
    "load",
    "save",
    "convert",
    "create",
    "defaults",
    "merge",
    "update",
    "get",
    "set",
    "has",
    "delete",
    "keys",
    "values",
    "items",
    "validate",
    "export",
]
