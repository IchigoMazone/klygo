# klygo.models

Module cung cấp mô hình học sâu hỗ trợ thị giác máy tính và nhận diện vật thể zero-shot.

```python
from klygo.models import Kernel
```

---

## Lớp Kernel

Lớp Kernel hỗ trợ nhận diện vật thể zero-shot sử dụng các mô hình Transformers (như Grounding DINO).
* **Khởi tạo:**
```python
from klygo.models import Kernel

# Tải mô hình mặc định Grounding DINO
kernel = Kernel(model="Grounding-Dino/232M")
```

---

## Hướng dẫn chi tiết các phương thức

### 1. predict
Nhận diện vật thể trên một hoặc nhiều ảnh tĩnh bằng câu lệnh gợi ý (Text Prompt).
* **Cú pháp:** `kernel.predict(images, prompt="apple.", return_tensors="pt", threshold=0.25, text_threshold=0.25) -> list[dict]`
* **Tham số:**
  * `images`: Một ảnh PIL hoặc danh sách ảnh PIL.
  * `prompt`: Câu lệnh prompt dạng chuỗi (ví dụ: `"dog. cat."`).
  * `threshold`: Ngưỡng độ tin cậy của box phát hiện.
* **Ví dụ sử dụng:**
```python
from PIL import Image
from klygo.models import Kernel

kernel = Kernel()
img = Image.open("traffic.jpg")

# Dự đoán xe hơi và xe buýt trên ảnh
results = kernel.predict(img, prompt="car. bus.", threshold=0.25)
print(results[0]["labels"])
```

### 2. detect
Chạy nhận diện vật thể trên từng frame của video, vẽ khung nhận diện, và/hoặc tự động lưu cấu trúc nhãn YOLO cho việc huấn luyện.
* **Cú pháp:** `kernel.detect(input_path, output_path=None, prompt="apple.", return_tensors="pt", threshold=0.25, text_threshold=0.25, desc="Processing", save_frame_dir=None, save_yolo_dir=None)`
* **Tham số:**
  * `input_path`: Đường dẫn video nguồn.
  * `output_path`: Đường dẫn lưu video kết quả đã vẽ khung.
  * `save_frame_dir`: Thư mục lưu các ảnh frame tĩnh của video.
  * `save_yolo_dir`: Thư mục tự động xuất bộ dataset YOLO (chứa thư mục `images`, `labels` và file cấu hình `data.yaml` tự động).
* **Ví dụ sử dụng:**
```python
from klygo.models import Kernel

kernel = Kernel()

# Chạy trên video và xuất bộ dữ liệu YOLO sang thư mục 'dataset/'
kernel.detect(
    input_path="traffic.mp4",
    output_path="traffic_output.mp4",
    prompt="car. pedestrian.",
    save_yolo_dir="dataset/"
)
```

### 3. set_input_kwargs
Cấu hình thêm các tham số đầu vào truyền trực tiếp cho processor của HuggingFace.
* **Cú pháp:** `kernel.set_input_kwargs(**kwargs)`
* **Ví dụ sử dụng:**
```python
from klygo.models import Kernel

kernel = Kernel()
# Thêm cấu hình padding cho processor trước khi chạy predict/detect
kernel.set_input_kwargs(padding="max_length")
```

### 4. set_output_kwargs
Cấu hình thêm các tham số đầu ra truyền trực tiếp cho post-processor của HuggingFace.
* **Cú pháp:** `kernel.set_output_kwargs(**kwargs)`
* **Ví dụ sử dụng:**
```python
from klygo.models import Kernel

kernel = Kernel()
# Cấu hình kích thước đích mong muốn cho post-processor
kernel.set_output_kwargs(target_sizes=[(640, 480)])
```
