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
- `klygo.config`: quản lý cấu hình đa định dạng với dot-notation.
- `klygo.datasets`: partition, repartition, unpartition, merge, split và remap dataset YOLO.
- `klygo.files`: bộ công cụ 39 hàm thao tác file, thư mục và path, hỗ trợ 14 định dạng dữ liệu (YAML, JSON, TOML, CSV, INI, ENV, XML, Pickle...).
- `klygo.media`: xử lý và tải/lưu tập tin hình ảnh và truyền thông.
- `klygo.processing`: pipeline xử lý ảnh lazy đa backend, quality filtering và phục hồi tọa độ.
- `klygo.postprocessing`: lọc, chỉnh nhãn, sửa box, NMS và pipeline xử lý kết quả nhận diện.
- `klygo.models`: nạp và chạy mô hình nhận diện trên ảnh, thư mục ảnh và video.
- `klygo.outputs`: kiểu kết quả chuẩn hóa `Box`, `Detection`, `Detections` và `Crops`.
- `klygo.visual`: hiển thị ảnh, vẽ bounding box và thống kê dataset.

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

### File, Cấu hình & Truyền thông

```python
import klygo.files as files
import klygo.media as media
import klygo.config as config
from klygo.config import Config

cfg = config.load("config.yaml")
data = files.load("data.json")
files.save("output.env", {"PORT": "8080"}, overwrite=True)
pil_images = media.load("dataset/images", backend="pil")
```

### Lazy processing

```python
from klygo import media
from klygo import processing as P

image = media.load("input.jpg")[0]

pipeline = P.compose([
    P.auto_orient(),
    P.letterbox((640, 640)),
    P.clahe(),
    P.normalize(scale=255),
])

processed = pipeline(image)  # Chưa decode pixel.
array = P.materialize(processed, output="array", cache=False)
```

Áp dụng cùng pipeline cho cả collection mà vẫn giữ lazy loading:

```python
images = media.load("dataset/images").transform(pipeline)
selected = images.where(lambda image: image.path.suffix.lower() == ".jpg")
print(selected.total_frames, selected.loaded_count)
```

Với video dài, dùng `media.stream()` để không tạo danh sách frame toàn bộ video:

```python
frames = (
    media.stream("input.mp4", sample_rate=3)
    .where_quality(min_sharpness=80)
    .transform(pipeline)
)

for frame in frames:
    result = model.predict(frame)
```

### Model và kết quả nhận diện

```python
from klygo import models

model = models.load("grounding-dino-tiny")
results = model.predict("traffic.mp4", prompt="car. person.", stream=True)

# Xử lý tuần tự để không giữ toàn bộ video trong RAM.
for frame in results:
    print(frame.labels, frame.scores)

# Dự đoán lại và lưu video đã vẽ bounding box.
model.predict(
    "traffic.mp4",
    prompt="car. person.",
    stream=True,
).save("detected.mp4")
```

### Post-processing kết quả

```python
from klygo import postprocessing as post

result = results.get_frame(120)
result = (
    result
    .relabel("car", ids=[1, 4, 7])
    .filter_score(0.4)
    .clip()
    .remove_invalid(min_area=16)
    .nms(iou=0.5)
    .top(100)
)

cleanup = post.compose(
    post.threshold(0.4),
    post.clip(),
    post.remove_invalid(min_area=16),
    post.nms(iou=0.5),
    post.top(100),
)

clean_results = cleanup(results)  # Vẫn lazy nếu results là stream.
```

### Visualize

```python
from PIL import Image
from klygo import visual

image = Image.open("traffic.jpg")
annotated = visual.draw_bboxes(
    image,
    boxes=[[20, 20, 180, 160]],
    labels=["car"],
    scores=[0.95],
)
visual.show_image(annotated, title="Detected objects")
```

## Quy ước đường dẫn

- `source`: file hoặc thư mục đầu vào.
- `*_path`: đường dẫn file.
- `*_dir`: đường dẫn thư mục.
- `target`: file hoặc thư mục đầu ra linh hoạt.
