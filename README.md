# klygo

Thư viện Python hỗ trợ xử lý ZIP, quản lý dataset YOLO, đọc/ghi cấu hình và ảnh, nhận diện vật thể zero-shot, crop và trực quan hóa kết quả.

## Cài đặt

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
uv sync
```

## Các package

- `klygo.archive`: nén, giải nén, tìm kiếm, kiểm tra, chỉnh sửa, gộp và chia ZIP.
- `klygo.datasets`: partition, repartition, unpartition, merge, split và remap dataset YOLO.
- `klygo.io`: đọc/ghi YAML, JSON, TOML; đọc ảnh bằng PIL hoặc OpenCV.
- `klygo.models`: nhận diện zero-shot trên ảnh, thư mục ảnh và video.
- `klygo.visualize`: hiển thị, vẽ bbox, crop, đọc crop và thống kê dataset.

## Sử dụng nhanh

### Archive

```python
import klygo.archive as ar

ar.compress("dataset", "dataset.zip", overwrite=True)
ar.extract("dataset.zip", output_dir="extracted", overwrite=True)
files = ar.search("dataset.zip", "images/*.jpg")
```

### Dataset YOLO

```python
import klygo.datasets as ds

ds.partition(
    source="raw_dataset.zip",
    target="dataset",
    ratios=(0.8, 0.1, 0.1),
    overwrite=True,
)

info = ds.get_dataset_info("dataset")
print(info)
```

### Đọc cấu hình và ảnh

```python
import klygo.io as io

config = io.Config("config.yaml").read()
pil_images = io.read_images("dataset/images", backend="pil")
opencv_images = io.read_images("dataset/images", backend="opencv")
```

### Model

```python
from klygo.models import Model

model = Model()

predictions = model.predict(
    source="dataset/images",
    prompt="car. person.",
)

detected_frames = model.detect(
    source="traffic.mp4",
    output_path="detected.mp4",
    annotated_dir="annotated",
    dataset_dir="generated_dataset",
    prompt="car. person.",
)

detection_results = model.detect(
    source="dataset/images",
    prompt="car. person.",
    metadata=True,
)

crop_results = model.crop(
    source="traffic.mp4",
    target="crops",
    prompt="car. person.",
)
```

### Visualize

```python
import klygo.visualize as vis

grid = vis.read_crops("crops")  # lưới 5 x 5 mặc định
vis.show_image(grid, title="Detected objects")

saved_detections = vis.read_detections("annotated")
```

## Quy ước đường dẫn

- `source`: file hoặc thư mục đầu vào.
- `*_path`: đường dẫn file.
- `*_dir`: đường dẫn thư mục.
- `target`: file hoặc thư mục đầu ra linh hoạt.

## Tài liệu chi tiết

- [Archive](docs/archive/README.md)
- [Datasets](docs/datasets/README.md)
- [IO](docs/io/README.md)
- [Models](docs/models/README.md)
- [Visualize](docs/visualize/README.md)
- [Mục lục tài liệu](docs/README.md)
