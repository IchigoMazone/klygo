# klygo

Thư viện Python hỗ trợ nén/giải nén file, xử lý bộ dữ liệu YOLO, đọc/ghi file cấu hình và nhận diện vật thể zero-shot.

---

## Tính năng chính

- klygo.archive: Các thao tác với file nén ZIP (nén, giải nén, tìm kiếm, gộp, chia, thêm, xóa file) hỗ trợ hiển thị thanh tiến trình.
- klygo.datasets: Các công cụ quản lý và xử lý bộ dữ liệu YOLO (phân chia train/val/test, chia lại tỷ lệ tại chỗ, unpartition phẳng hóa, gộp nhiều dataset có ánh xạ lại class ID, và tách dataset theo class/tỷ lệ con).
- klygo.io: Đọc và ghi các file cấu hình YAML, JSON, TOML, tự động tạo thư mục, tự động phân giải đường dẫn tương đối và truy cập dạng thuộc tính qua Box.
- klygo.models: Lớp hỗ trợ nhận diện vật thể zero-shot (dựa trên Grounding DINO) trên ảnh và video, tự động xuất dữ liệu theo định dạng YOLO.
- klygo.visualize: Trực quan hóa hình ảnh, vẽ bounding box, hiển thị kết quả predict, crop vật thể theo nhãn YOLO ra file nén ZIP và biểu đồ hóa phân bổ dữ liệu.

---

## Cấu trúc thư mục

```
klygo/
├── archive/        # Các tiện ích nén và giải nén
├── datasets/       # Công cụ quản lý và phân chia bộ dữ liệu YOLO
├── io/             # Bộ đọc/ghi cấu hình YAML, JSON, TOML
├── models/         # Lớp nhận diện vật thể (Kernel)
├── visualize/      # Trực quan hóa hình ảnh & crop bboxes
└── validators/     # Bộ xác thực dữ liệu đầu vào
```

---

## Hướng dẫn sử dụng nhanh

### 1. Nén và giải nén (klygo.archive as ar)

```python
import klygo.archive as ar

# Nén thư mục và giải nén
ar.compress("src_directory/", "archive.zip")
ar.extract("archive.zip", output="destination_dir/", overwrite=True)

# Tìm kiếm file trong archive
files = ar.search("archive.zip", "images/*.jpg")
```

Xem tài liệu đầy đủ tại docs/archive/README.md.

---

### 2. Quản lý bộ dữ liệu YOLO (klygo.datasets)

```python
from klygo.datasets import partition, repartition, unpartition, merge, split, remap_classes, get_dataset_info

# 1. Phân chia tập dữ liệu thô thành train/val/test
partition(source="raw_data.zip", output="dataset/", ratios=(0.8,), overwrite=True)

# 2. Phẳng hóa tập dữ liệu đã chia về dạng raw ban đầu
unpartition(source="dataset/", output="raw_flat.zip", overwrite=True)

# 3. Hợp nhất hai bộ dữ liệu khác class tự động đổi class ID
merge(sources=["data1/", "data2.zip"], output="merged.zip", overwrite=True)

# 4. Tách nhỏ bộ dữ liệu theo từng class đối tượng riêng biệt
split(source="merged.zip", output_dir="split_classes/", by_class=True, overwrite=True)

# 5. Ánh xạ và đổi tên lớp dữ liệu
remap_classes(source="merged.zip", output="remapped.zip", class_map={0: 1, "apple": "red_apple"}, overwrite=True)

# 6. Lấy thông tin chi tiết của bộ dữ liệu YOLO
info = get_dataset_info("merged.zip")
print(info)
```

Xem tài liệu đầy đủ tại docs/datasets/README.md.

---

### 3. File cấu hình (klygo.io as io)

```python
import klygo.io as io

# Ghi file cấu hình
config_data = {"learning_rate": 0.01, "epochs": 50}
io.write_file("configs/train_params.toml", config_data, overwrite=True)

# Đọc file cấu hình và truy cập bằng thuộc tính
cfg = io.Config("configs/train_params.toml").read()
print(cfg.learning_rate)  # 0.01
```

Xem tài liệu đầy đủ tại docs/io/README.md.

---

### 4. Mô hình nhận diện (klygo.models.Kernel)

```python
from klygo.models import Kernel

# Khởi tạo mô hình Grounding DINO
kernel = Kernel()

# Chạy nhận diện trên video và xuất ra bộ dữ liệu YOLO
kernel.detect(
    input_path="input_video.mp4",
    prompt="car. traffic light.",
    save_yolo_dir="dataset/"
)
```

Xem tài liệu đầy đủ tại docs/models/README.md.

---

### 5. Trực quan hóa & Cắt vật thể (klygo.visualize)

```python
import klygo.visualize as vis

# Hiển thị ảnh kèm bounding box YOLO chỉ định
vis.visualize_dataset_image(
    image_path="dataset/images/frame_0.jpg",
    label_path="dataset/labels/frame_0.txt",
    classes=["car", "bus"]
)

# Cắt các vật thể trong ảnh và lưu vào file ZIP
vis.crop_objects(
    image_path="dataset/images/frame_0.jpg",
    label_path="dataset/labels/frame_0.txt",
    classes=["car", "bus"],
    output_zip="crops.zip",
    overwrite=True
)

# Đọc ngược lại danh sách ảnh crop từ file ZIP
cropped_dict = vis.read_cropped_objects("crops.zip")
```

Xem tài liệu đầy đủ tại docs/visualize/README.md.

---

## Cài đặt

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
uv sync
```

---

## Tài liệu chi tiết

Tất cả hướng dẫn đầy đủ đều được lưu tại thư mục docs:
* Hướng dẫn nén & giải nén: docs/archive/README.md
* Hướng dẫn xử lý bộ dữ liệu YOLO: docs/datasets/README.md
* Hướng dẫn cấu hình & IO: docs/io/README.md
* Hướng dẫn nhận diện vật thể: docs/models/README.md
* Hướng dẫn trực quan hóa & biểu diễn ảnh: docs/visualize/README.md
