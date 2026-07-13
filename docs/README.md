# Tài liệu klygo

## Bắt đầu nhanh

```bash
pip install klygo
```

Hoặc cài source bằng uv:

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
uv sync
```

## Tài liệu theo package

- [klygo.archive](archive/README.md): xử lý file ZIP.
- [klygo.datasets](datasets/README.md): quản lý dataset YOLO.
- [klygo.io](io/README.md): cấu hình và đọc ảnh PIL/OpenCV.
- [klygo.models](models/README.md): predict, detect và crop trên ảnh/video.
- [klygo.visualize](visualize/README.md): hiển thị, bbox, crop và biểu đồ.
- [klygo.validators](validators/README.md): lớp kiểm tra tham số và `validate_type`.
- [klygo.utils](utils/README.md): helper nội bộ dùng chung.

## Quy ước API

- `source`: file hoặc thư mục đầu vào.
- `*_path`: file cụ thể.
- `*_dir`: thư mục cụ thể.
- `target`: đầu ra có thể là file hoặc thư mục.
- Hàm có `overwrite=False` không thay thế đầu ra đã tồn tại.
- Hàm có `verbose=True` hiển thị tiến trình.

## Workflow tham khảo

```python
from klygo.io import read_images
from klygo.models import Model
from klygo.visualize import read_crops, read_detections, show_image

model = Model()
images = read_images("images", backend="pil")
predictions = model.predict(images, prompt="apple.")
detected = model.detect(
    images,
    annotated_dir="detected",
    prompt="apple.",
    metadata=True,
)
saved_detections = read_detections("detected")
crop_results = model.crop(images, target="crops", prompt="apple.")
grid = read_crops("crops")
show_image(grid)
```
