# klygo.io

Module đọc/ghi các định dạng file cấu hình (YAML, JSON, TOML) và quản lý phân giải đường dẫn tự động.

```python
import klygo.io as io
```

---

## Hướng dẫn chi tiết từng API

### 1. read_yaml
Đọc file YAML/YML và trả về dữ liệu Python.
* **Cú pháp:** `io.read_yaml(path, verbose=True) -> Any`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = io.read_yaml("config.yaml", verbose=True)
print(data)
```

### 2. read_json
Đọc file JSON và trả về dữ liệu Python.
* **Cú pháp:** `io.read_json(path, verbose=True) -> Any`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = io.read_json("config.json", verbose=False)
print(data)
```

### 3. read_toml
Đọc file TOML và trả về dữ liệu Python.
* **Cú pháp:** `io.read_toml(path, verbose=True) -> Any`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = io.read_toml("config.toml", verbose=True)
print(data)
```

### 4. read_file
Đọc file cấu hình và tự động nhận diện định dạng dựa vào phần mở rộng (`.yaml`, `.yml`, `.json`, `.toml`).
* **Cú pháp:** `io.read_file(path, verbose=True) -> Any`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

# Tự động đọc và parse file dựa vào phần mở rộng của file
data = io.read_file("params.json", verbose=True)
```

### 5. write_yaml
Ghi dữ liệu thành file YAML. Tự động tạo thư mục cha của tệp đích nếu chưa tồn tại.
* **Cú pháp:** `io.write_yaml(path, data, overwrite=False, verbose=True)`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = {"learning_rate": 0.001, "epochs": 100}
io.write_yaml("out/config.yaml", data, overwrite=True, verbose=True)
```

### 6. write_json
Ghi dữ liệu thành file JSON với định dạng thụt lề thụt dòng.
* **Cú pháp:** `io.write_json(path, data, overwrite=False, verbose=True)`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = {"learning_rate": 0.001, "epochs": 100}
io.write_json("out/config.json", data, overwrite=True, verbose=True)
```

### 7. write_toml
Ghi dữ liệu thành file TOML.
* **Cú pháp:** `io.write_toml(path, data, overwrite=False, verbose=True)`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = {"learning_rate": 0.001, "epochs": 100}
io.write_toml("out/config.toml", data, overwrite=True, verbose=True)
```

### 8. write_file
Ghi dữ liệu vào file cấu hình, tự động nhận diện định dạng ghi dựa vào phần mở rộng của file.
* **Cú pháp:** `io.write_file(path, data, overwrite=False, verbose=True)`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

data = {"learning_rate": 0.001, "epochs": 100}
# Tự động ghi thành YAML
io.write_file("out/config.yaml", data, overwrite=True, verbose=True)
```

### 9. Config
Lớp khởi tạo đối tượng cấu hình hỗ trợ tự động mở rộng đường dẫn tương đối và chuyển đổi định dạng.
* **Khởi tạo:** `config = io.Config(src)` (với `src` là đường dẫn tệp cấu hình nguồn).

#### .read
Đọc cấu hình nguồn, trả về đối tượng `Box` hỗ trợ truy cập bằng thuộc tính (dot-notation).
* **Cú pháp:** `config.read(verbose=True) -> Box`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

config = io.Config("config.yaml")
cfg = config.read(verbose=True)
print(cfg.dataset.train_path)
```

#### .imread (Deprecated)
Đọc cấu hình nguồn (alias tương thích ngược của `.read()`). Ném cảnh báo `DeprecationWarning`.

#### .to_dict
Chuyển đổi dữ liệu cấu hình đã parse thành một `dict` Python tiêu chuẩn.
* **Cú pháp:** `config.to_dict() -> dict`

#### .to_json
Xuất cấu hình đã parse ra chuỗi định dạng JSON.
* **Cú pháp:** `config.to_json(indent=4) -> str`

#### .create_default (Static Method)
Khởi tạo tự động một file cấu hình YAML mẫu (default template) và trả về đối tượng `Config` tương ứng.
* **Cú pháp:** `Config.create_default(path, default_data=None, overwrite=False, verbose=True) -> Config`

#### .export_file
Xuất cấu hình hiện tại sang định dạng cấu hình khác.
* **Cú pháp:** `config.export_file(name, suffix, output_dir=None, overwrite=False, verbose=True)`
* **Ví dụ sử dụng:**
```python
import klygo.io as io

config = io.Config("config.yaml")
config.read()
# Xuất sang file config.json trong thư mục 'out'
config.export_file("config", ".json", output_dir="out/", overwrite=True)
```
