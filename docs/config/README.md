# klygo.config - Bộ công cụ Quản lý File Cấu hình đa định dạng

`klygo.config` cung cấp lớp OOP `Config` và 20 APIs chuyên biệt giúp khởi tạo, đọc/ghi, hợp nhất, cập nhật, so sánh, phẳng hóa và thao tác dữ liệu cấu hình theo dạng dot-notation.

---

# 1. Config(path=None)

Lớp đối tượng Wrapper OOP quản lý file cấu hình dạng dot-notation (`cfg.model.batch`), tự động giải quyết đường dẫn gốc (`default.root`), hỗ trợ đọc, ghi, cập nhật và xuất định dạng file.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| path | str \| Path \| None | None | Đường dẫn tới file cấu hình. Nếu None, tự động tạo cấu hình mặc định. |

## Các phương thức chính

- `cfg.read(verbose=True)`: Đọc và trả về đối tượng cấu hình Box dot-notation.
- `cfg.update(updates, deep=True)`: Cập nhật thông số cấu hình.
- `cfg.to_dict()`: Chuyển đổi cấu hình thành dictionary thuần Python.
- `cfg.to_json(indent=4)`: Chuyển đổi cấu hình thành chuỗi JSON.
- `cfg.export_file(output_name, ext='.json', output_dir=None, overwrite=False)`: Xuất cấu hình thành file mới.
- `Config.create_default(path, overwrite=False)`: Hàm tĩnh tạo đối tượng Config với dữ liệu mặc định.

## Ví dụ

```python
from klygo.config import Config

cfg = Config("config.yaml")
data = cfg.read()
print(data.model.name)

cfg.update({"model": {"batch": 32}})
cfg.export_file("exported_config", ext=".json", overwrite=True)
```

---

# 2. config.load(path, expand_root=True, verbose=True)

Đọc file cấu hình (`.yaml`, `.json`, `.toml`, `.ini`, `.env`) và trả về đối tượng `Box` hỗ trợ truy cập dạng dot-notation (`cfg.model.batch`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| path | str \| Path | Bắt buộc | Đường dẫn tới file cấu hình cần đọc. |
| expand_root | bool | True | Tự động giải quyết đường dẫn root mặc định (`default.root`). |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi đọc. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| Box | Đối tượng cấu hình hỗ trợ truy cập dạng dot-notation. |

## Ví dụ

```python
import klygo.config as config

cfg = config.load("config.yaml")
print(cfg.model.name)
print(cfg.training.batch_size)
```

---

# 3. config.save(path, data, overwrite=False, verbose=True, indent=4)

Ghi dữ liệu cấu hình ra file đĩa dựa theo đuôi mở rộng file.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| path | str \| Path | Bắt buộc | Đường dẫn file cấu hình đầu ra. |
| data | Any | Bắt buộc | Dữ liệu cấu hình cần lưu (dict, Box...). |
| overwrite | bool | False | Cho phép ghi đè nếu file đầu ra đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi ghi. |
| indent | int | 4 | Thụt lề khi ghi file JSON. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| None |

## Ví dụ

```python
import klygo.config as config

data = {"model": {"name": "yolov8n", "batch": 16}}
config.save("config.yaml", data, overwrite=True)
```

---

# 4. config.convert(source, target, overwrite=False, verbose=True)

Chuyển đổi trực tiếp định dạng giữa 2 file cấu hình (vd: từ `.yaml` sang `.json` hoặc `.toml`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| source | str \| Path | Bắt buộc | Đường dẫn file cấu hình nguồn. |
| target | str \| Path | Bắt buộc | Đường dẫn file cấu hình đích. |
| overwrite | bool | False | Cho phép ghi đè nếu file đích đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi chuyển đổi. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.config as config

config.convert("config.yaml", "config.json", overwrite=True)
```

---

# 5. config.create(path, default_data=None, overwrite=False, verbose=True)

Tạo một file cấu hình mặc định chuẩn mới trên đĩa.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| path | str \| Path | Bắt buộc | Đường dẫn file cấu hình cần tạo. |
| default_data | dict \| None | None | Dữ liệu mặc định ban đầu. Nếu None, dùng cấu hình mặc định hệ thống. |
| overwrite | bool | False | Cho phép ghi đè nếu file đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi tạo. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.config as config

config.create("default_config.yaml", overwrite=True)
```

---

# 6. config.defaults()

Lấy dictionary chứa bộ thông số cấu hình mặc định chuẩn của hệ thống.

## Tham số

Hàm không nhận tham số đầu vào.

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| dict | Dictionary chứa bộ cấu hình mặc định. |

## Ví dụ

```python
import klygo.config as config

default_cfg = config.defaults()
print(default_cfg)
```

---

# 7. config.merge(*configs, deep=True)

Gộp đệ quy nhiều dictionary hoặc đối tượng cấu hình lại với nhau.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| *configs | dict \| Box | Bắt buộc | Các dictionary/Box cấu hình cần gộp (truyền 2 hoặc nhiều hơn). |
| deep | bool | True | Gộp đệ quy vào các sub-dictionary lồng nhau. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| dict | Dictionary mới sau khi đã gộp tất cả cấu hình. |

## Ví dụ

```python
import klygo.config as config

cfg1 = {"model": {"name": "yolo", "batch": 16}}
cfg2 = {"model": {"batch": 32, "lr": 0.001}}

merged = config.merge(cfg1, cfg2, deep=True)
print(merged)
```

---

# 8. config.update(config_data, updates, deep=True)

Cập nhật các tham số trong cấu hình hiện tại bằng dictionary thông số mới.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình gốc cần cập nhật. |
| updates | dict \| Box | Bắt buộc | Dữ liệu thông số mới dùng để cập nhật. |
| deep | bool | True | Cập nhật đệ quy vào các dictionary lồng nhau. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| dict \| Box | Dữ liệu cấu hình đã được cập nhật. |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"batch": 16}}
config.update(cfg, {"model": {"batch": 64}})
```

---

# 9. config.get(config_data, key_path, default=None)

Truy xuất giá trị theo chuỗi đường dẫn dot-notation (vd: `"model.backbone.layers"`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| key_path | str | Bắt buộc | Chuỗi đường dẫn dot-notation (vd: 'model.batch'). |
| default | Any | None | Giá trị trả về mặc định nếu key_path không tồn tại. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Any |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"batch": 16}}
batch_size = config.get(cfg, "model.batch", default=32)
```

---

# 10. config.set(config_data, key_path, value)

Gán hoặc thiết lập giá trị tại đường dẫn dot-notation (tự động tạo các dict lồng nhau nếu chưa có).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| key_path | str | Bắt buộc | Chuỗi đường dẫn dot-notation (vd: 'model.arch.backbone'). |
| value | Any | Bắt buộc | Giá trị cần gán. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| None |

## Ví dụ

```python
import klygo.config as config

cfg = {}
config.set(cfg, "model.batch", 64)
```

---

# 11. config.has(config_data, key_path)

Kiểm tra xem một key path chỉ định có tồn tại trong cấu hình hay không.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| key_path | str | Bắt buộc | Chuỗi đường dẫn dot-notation cần kiểm tra. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| bool | True nếu key path tồn tại, ngược lại False. |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"batch": 16}}
print(config.has(cfg, "model.batch")) # -> True
```

---

# 12. config.delete(config_data, key_path)

Xóa một key path khỏi đối tượng cấu hình.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| key_path | str | Bắt buộc | Chuỗi đường dẫn dot-notation cần xóa. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| bool | True nếu xóa thành công, ngược lại False. |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"batch": 16, "temp": 1}}
config.delete(cfg, "model.temp")
```

---

# 13. config.keys(config_data, flat=False)

Lấy danh sách các key trong cấu hình.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| flat | bool | False | Nếu True, trả về danh sách key dạng đường dẫn phẳng (vd: 'model.batch'). |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| list[str] |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"name": "yolo", "batch": 16}}
print(config.keys(cfg, flat=True)) # -> ['model.name', 'model.batch']
```

---

# 14. config.values(config_data, flat=False)

Lấy danh sách các giá trị tương ứng trong cấu hình.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| flat | bool | False | Nếu True, duyệt đệ quy lấy danh sách giá trị phẳng. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| list[Any] |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"name": "yolo", "batch": 16}}
print(config.values(cfg, flat=True)) # -> ['yolo', 16]
```

---

# 15. config.items(config_data, flat=False)

Lấy danh sách các cặp (key, value) trong cấu hình.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình. |
| flat | bool | False | Nếu True, trả về danh sách cặp (flat_key_path, value). |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| list[tuple[str, Any]] |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"name": "yolo", "batch": 16}}
print(config.items(cfg, flat=True)) # -> [('model.name', 'yolo'), ('model.batch', 16)]
```

---

# 16. config.validate(config_data, schema=None)

Kiểm tra xem dữ liệu cấu hình có hợp lệ và chứa đầy đủ các trường yêu cầu hay không.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình cần kiểm tra. |
| schema | dict \| None | None | Mẫu schema kiểm tra. Nếu None, kiểm tra các trường bắt buộc mặc định. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| bool | True nếu cấu hình hợp lệ, ngược lại False. |

## Ví dụ

```python
import klygo.config as config

cfg = {"model": {"name": "yolo"}}
print(config.validate(cfg)) # -> True
```

---

# 17. config.export(source, target, overwrite=False, verbose=True)

Xuất file cấu hình ra định dạng mới trên đĩa.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| source | str \| Path | Bắt buộc | Đường dẫn file cấu hình nguồn. |
| target | str \| Path | Bắt buộc | Đường dẫn file cấu hình mới cần xuất. |
| overwrite | bool | False | Cho phép ghi đè nếu file đích đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi xuất. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.config as config

config.export("config.yaml", "config_backup.json", overwrite=True)
```

---

# 18. config.diff(config1, config2)

So sánh sự khác biệt chi tiết giữa 2 file hoặc 2 đối tượng cấu hình.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config1 | str \| Path \| dict \| Box | Bắt buộc | File hoặc đối tượng cấu hình thứ nhất. |
| config2 | str \| Path \| dict \| Box | Bắt buộc | File hoặc đối tượng cấu hình thứ hai. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| dict[str, dict] | Trả về dictionary gồm: 'added', 'removed', 'modified'. |

## Ví dụ

```python
import klygo.config as config

cfg1 = {"model": {"batch": 16}}
cfg2 = {"model": {"batch": 32, "lr": 0.001}}

diff_dict = config.diff(cfg1, cfg2)
print("File khác nhau:", diff_dict)
```

---

# 19. config.flatten(config_data, sep='.')

Chuyển đổi cấu trúc dictionary cấu hình lồng nhau thành dictionary dạng phẳng với key theo dạng dot-notation.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box | Bắt buộc | Dữ liệu cấu hình lồng nhau. |
| sep | str | '.' | Ký tự phân cách các cấp key. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| dict[str, Any] |

## Ví dụ

```python
import klygo.config as config

nested = {"model": {"arch": {"name": "yolo"}, "batch": 16}}
flat = config.flatten(nested)
# -> {'model.arch.name': 'yolo', 'model.batch': 16}
```

---

# 20. config.unflatten(flat_dict, sep='.')

Khôi phục dictionary dạng phẳng (dot-notation) trở lại cấu trúc dictionary lồng nhau ban đầu.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| flat_dict | dict | Bắt buộc | Dictionary dạng phẳng (ví dụ: {'model.batch': 16}). |
| sep | str | '.' | Ký tự phân cách các cấp key. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| dict[str, Any] |

## Ví dụ

```python
import klygo.config as config

flat = {"model.batch": 16}
nested = config.unflatten(flat)
# -> {'model': {'batch': 16}}
```

---

# 21. config.from_env(config_data=None, prefix='KLYGO_', sep='_')

Đọc các biến môi trường hệ thống OS (bắt đầu bằng prefix) và ghi đè/cập nhật vào cấu hình.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| config_data | dict \| Box \| None | None | Cấu hình gốc cần cập nhật (nếu None sẽ tạo mới). |
| prefix | str | 'KLYGO_' | Tiền tố tên biến môi trường cần đọc. |
| sep | str | '_' | Ký tự phân cách các cấp key trong tên biến môi trường. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| dict[str, Any] |

## Ví dụ

```python
import os, klygo.config as config

os.environ["KLYGO_MODEL_BATCH"] = "64"
cfg = config.from_env(prefix="KLYGO_")
# -> {'model': {'batch': '64'}}
```
