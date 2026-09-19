"""
Bộ công cụ Quản lý File Cấu hình đa định dạng (`klygo.config`).

Hướng dẫn sử dụng (Google Colab):
https://colab.research.google.com/drive/1-aOofq_ZwLi00gRXnBLbJ4raupc6OZKm?usp=sharing

Định dạng hỗ trợ (5 loại):
  - YAML (.yaml, .yml), JSON (.json), TOML (.toml), INI (.ini, .cfg, .properties), ENV (.env)

Danh sách 20 APIs:
  1.  load(path, ...) - Đọc file cấu hình trả về đối tượng Box dot-notation
  2.  save(path, data, ...) - Ghi dữ liệu cấu hình ra file
  3.  convert(source, target, ...) - Chuyển đổi trực tiếp giữa các định dạng file cấu hình
  4.  create(path, ...) - Tạo file cấu hình mặc định mới trên đĩa
  5.  defaults() - Lấy dictionary chứa cấu hình mặc định của hệ thống
  6.  merge(*configs, ...) - Gộp đệ quy nhiều dictionary cấu hình lại với nhau
  7.  update(config_data, updates, ...) - Cập nhật thông số cấu hình bằng dictionary mới
  8.  get(config_data, key_path, ...) - Truy xuất giá trị theo chuỗi đường dẫn dot-notation ('a.b.c')
  9.  set(config_data, key_path, value) - Gán/thiết lập giá trị tại đường dẫn dot-notation
  10. has(config_data, key_path) - Kiểm tra sự tồn tại của key path ('a.b.c')
  11. delete(config_data, key_path) - Xóa một key path khỏi đối tượng cấu hình
  12. keys(config_data, flat=False) - Danh sách các key (hỗ trợ dạng phẳng flat=True)
  13. values(config_data, flat=False) - Danh sách các giá trị trong cấu hình
  14. items(config_data, flat=False) - Danh sách các cặp (key, value)
  15. validate(config_data, ...) - Kiểm tra tính hợp lệ và đầy đủ của cấu hình
  16. export(source, target, ...) - Xuất file cấu hình sang định dạng khác
  17. diff(config1, config2) - So sánh sự khác biệt chi tiết giữa 2 file/object cấu hình
  18. flatten(config_data, sep='.') - Phẳng hóa cấu hình lồng nhau thành dictionary 1 cấp
  19. unflatten(flat_dict, sep='.') - Khôi phục dictionary dạng phẳng trở lại dạng lồng nhau
  20. from_env(config_data, prefix, ...) - Đọc biến môi trường hệ thống OS ghi đè vào cấu hình
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
    diff,
    flatten,
    unflatten,
    from_env,
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
    "diff",
    "flatten",
    "unflatten",
    "from_env",
]
