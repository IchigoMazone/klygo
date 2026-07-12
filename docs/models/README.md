# klygo.models

Module cung cấp mô hình học sâu hỗ trợ thị giác máy tính và nhận diện vật thể zero-shot.

```python
import cv2 as cv

from klygo.models import Model
```

---

## Lớp Model

Lớp Model hỗ trợ nhận diện vật thể zero-shot sử dụng các mô hình Transformers (như Grounding DINO).
* **Khởi tạo:**
```python
from klygo.models import Model

# Tải mô hình mặc định Grounding DINO
model = Model(model="Grounding-Dino/232M")
```

---

## Hướng dẫn chi tiết các phương thức

### 1. predict
Nhận diện vật thể trên một hoặc nhiều ảnh tĩnh bằng câu lệnh gợi ý (Text Prompt).
* **Cú pháp:** `kernel.predict(images, prompt="apple.", return_tensors="pt", box_threshold=0.25, text_threshold=0.25) -> list[dict]`
* **Tham số:**
  * `images`: Một ảnh PIL hoặc danh sách ảnh PIL.
  * `prompt`: Câu lệnh prompt dạng chuỗi (ví dụ: `"dog. cat."`).
  * `box_threshold`: Ngưỡng độ tin cậy của box phát hiện.
* **Ví dụ sử dụng:**
```python
from PIL import Image
from klygo.models import Model

model = Model()
img = Image.open("traffic.jpg")

# Dự đoán xe hơi và xe buýt trên ảnh
results = model.predict(img, prompt="car. bus.", box_threshold=0.25)
print(results[0]["labels"])
```

### 2. detect
Chạy nhận diện vật thể trên từng frame của video, vẽ khung nhận diện, và/hoặc tự động lưu cấu trúc nhãn YOLO cho việc huấn luyện.
* **Cú pháp:** `kernel.detect(input_path, output_path=None, prompt="apple.", return_tensors="pt", box_threshold=0.25, text_threshold=0.25, desc="Processing", annotated_dir=None, dataset_dir=None)`
* **Tham số:**
  * `input_path`: Đường dẫn video nguồn.
  * `output_path`: Đường dẫn lưu video kết quả đã vẽ khung.
  * `annotated_dir`: Thư mục lưu các frame đã vẽ bounding box và label.
  * `dataset_dir`: Thư mục xuất bộ dataset YOLO (chứa `images`, `labels` và `data.yaml`).
  * `box_color`: Màu bounding box theo định dạng BGR của OpenCV.
  * `text_color`: Màu label theo định dạng BGR của OpenCV.
  * `thickness`: Độ dày nét của bounding box và label.
  * `font`: Font chữ bằng constant font của OpenCV.
  * `font_scale`: Kích thước chữ theo font scale của OpenCV.
* **Ví dụ sử dụng:**
```python
from klygo.models import Model

model = Model()

# Chạy trên video và xuất bộ dữ liệu YOLO sang thư mục 'dataset/'
frames = model.detect(
    input_path="traffic.mp4",
    output_path="traffic_output.mp4",
    prompt="car. pedestrian.",
    dataset_dir="dataset/",
    box_color=(0, 255, 0),
    text_color=(0, 0, 255),
    thickness=2,
    font=cv.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
)
```

### 3. crop
Cắt toàn bộ đối tượng được model phát hiện trên từng frame của video và lưu theo thư mục label.

```python
crop_images = model.crop(
    input_path="traffic.mp4",
    target="crops/",
    prompt="car. pedestrian.",
    box_threshold=0.25,
    text_threshold=0.25,
    max_area=0.5,
    padding=4,
)
```

Hàm luôn trả `list[PIL.Image.Image]`. Khi có `target`, ảnh cũng được lưu theo cấu trúc `crops/<label>/frame_<index>_crop_<index>.jpg`; khi không truyền `target`, hàm không ghi file.

### 4. set_input_kwargs
Cấu hình thêm các tham số đầu vào truyền trực tiếp cho processor của HuggingFace.
* **Cú pháp:** `kernel.set_input_kwargs(**kwargs)`
* **Ví dụ sử dụng:**
```python
from klygo.models import Model

model = Model()
# Thêm cấu hình padding cho processor trước khi chạy predict/detect
model.set_input_kwargs(padding="max_length")
```

### 5. set_output_kwargs
Cấu hình thêm các tham số đầu ra truyền trực tiếp cho post-processor của HuggingFace.
* **Cú pháp:** `kernel.set_output_kwargs(**kwargs)`
* **Ví dụ sử dụng:**
```python
from klygo.models import Model

model = Model()
# Cấu hình kích thước đích mong muốn cho post-processor
model.set_output_kwargs(target_sizes=[(640, 480)])
```
