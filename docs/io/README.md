# klygo.io

`klygo.io` đọc/ghi cấu hình YAML, JSON, TOML; đọc ảnh bằng PIL hoặc OpenCV; và cung cấp lớp `Config` để truy cập cấu hình dạng thuộc tính.

```python
import klygo.io as io
```

## 1. Đọc file cấu hình

### API theo định dạng

```python
yaml_data = io.read_yaml("config.yaml", verbose=False)
json_data = io.read_json("config.json", verbose=False)
toml_data = io.read_toml("config.toml", verbose=False)
```

### Tự nhận định dạng

`read_file` chọn parser dựa trên phần mở rộng `.yaml`, `.yml`, `.json` hoặc `.toml`:

```python
data = io.read_file("config.yaml", verbose=True)
```

Các hàm đọc phát sinh `FileNotFoundError` nếu file không tồn tại và `ValueError` nếu định dạng không được hỗ trợ.

## 2. Ghi file cấu hình

```python
data = {
    "model": "Grounding-Dino/232M",
    "threshold": 0.25,
}

io.write_yaml("configs/model.yaml", data, overwrite=True)
io.write_json("configs/model.json", data, overwrite=True)
io.write_toml("configs/model.toml", data, overwrite=True)
```

`write_file` tự chọn writer theo đuôi file:

```python
io.write_file(
    path="configs/model.yaml",
    data=data,
    overwrite=True,
    verbose=False,
)
```

Thư mục cha được tạo tự động. Nếu file đã tồn tại và `overwrite=False`, hàm phát sinh `FileExistsError`.

## 3. `read_images`

```python
io.read_images(
    source,
    recursive=False,
    backend="pil",
)
```

`source` nhận một file ảnh hoặc thư mục ảnh. Các định dạng hỗ trợ: BMP, JPEG, PNG, TIFF và WebP.

### Backend PIL mặc định

```python
images = io.read_images("dataset/images")

print(type(images[0]))  # PIL.Image.Image
print(images[0].mode)   # RGB
```

Đây là backend phù hợp để truyền vào Model:

```python
from klygo.models import Model

model = Model()
images = io.read_images("dataset/images", backend="pil")
predictions = model.predict(images, prompt="apple.")
```

### Backend OpenCV

```python
images = io.read_images(
    source="dataset/images",
    backend="opencv",
)

print(type(images[0]))   # numpy.ndarray
print(images[0].shape)   # height, width, channels
```

Ảnh OpenCV có thứ tự màu BGR:

```python
import cv2 as cv

gray = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
```

Model cũng nhận trực tiếp danh sách ảnh OpenCV và tự chuyển BGR sang RGB.

### Đọc thư mục con

```python
images = io.read_images(
    source="dataset/images",
    recursive=True,
    backend="pil",
)
```

`recursive=False` chỉ đọc file ngay trong thư mục `source`; `True` đọc toàn bộ cây thư mục. File được sắp xếp theo tên trước khi đọc.

### Đọc một file ảnh

```python
images = io.read_images("image.jpg")
image = images[0]
```

Hàm luôn trả danh sách, kể cả khi source chỉ có một file.

## 4. `Config`

### Đọc và truy cập dạng thuộc tính

```yaml
# config.yaml
default:
  root: ./data
dataset:
  images: ./images
model:
  threshold: 0.25
```

```python
config = io.Config("config.yaml")
cfg = config.read(verbose=False)

print(cfg.model.threshold)
print(cfg.dataset.images)
```

Khi cấu hình có `default.root`, chuỗi bắt đầu bằng `.` được mở rộng theo root đó.

### Lấy đường dẫn file cấu hình

```python
print(config.config_path)
```

### Chuyển sang dictionary hoặc JSON string

```python
data = config.to_dict()
json_text = config.to_json(indent=2)
```

Phải gọi `read()` trước khi chuyển đổi để dữ liệu nội bộ được nạp.

### Tạo cấu hình mặc định

```python
config = io.Config.create_default(
    path="configs/default.yaml",
    overwrite=True,
)

cfg = config.read()
```

Có thể cung cấp template riêng:

```python
config = io.Config.create_default(
    path="configs/app.json",
    default_data={"app": {"debug": True}},
    overwrite=True,
)
```

### Xuất sang định dạng khác

```python
config.read()
config.export_file(
    name="model_config",
    suffix=".toml",
    output_dir="exports",
    overwrite=True,
)
```

Các suffix hỗ trợ là `.yaml`, `.yml`, `.json` và `.toml`.

## 5. Danh sách API public

| API | Công dụng |
|---|---|
| `read_yaml`, `read_json`, `read_toml` | Đọc cấu hình theo định dạng cụ thể |
| `read_file` | Tự chọn parser theo đuôi file |
| `read_images` | Đọc file/thư mục ảnh bằng PIL hoặc OpenCV |
| `write_yaml`, `write_json`, `write_toml` | Ghi cấu hình theo định dạng cụ thể |
| `write_file` | Tự chọn writer theo đuôi file |
| `Config` | Quản lý, chuyển đổi và xuất cấu hình |
