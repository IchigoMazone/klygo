# klygo.visualize

Module cung cấp các API trực quan hóa hình ảnh, vẽ bounding box, thống kê bộ dữ liệu, trích xuất vật thể (crop) ra file ZIP và nạp ngược lại.

```python
import klygo.visualize as vis
```

---

## Hướng dẫn chi tiết từng API

### 1. show_image
Hiển thị hình ảnh bằng OpenCV hoặc Matplotlib (hoàn toàn tương thích và khuyên dùng trên Colab).
* **Cú pháp:** `vis.show_image(image, title="Image", backend="matplotlib", figsize=(10, 10))`
* **Tham số:**
  * `image`: Ảnh dạng PIL Image hoặc Numpy Array (OpenCV format).
  * `backend`: `"matplotlib"` (mặc định - hỗ trợ Jupyter/Colab) hoặc `"cv2"` (sử dụng cửa sổ OpenCV).
* **Ví dụ:**
```python
from PIL import Image
import klygo.visualize as vis

img = Image.open("traffic.jpg")
vis.show_image(img, title="Hải Phòng Traffic", backend="matplotlib")
```

### 2. visualize_prediction
Nhận diện và hiển thị kết quả predict từ `Model.predict()`.
* **Cú pháp:** `vis.visualize_prediction(image, prediction, backend="matplotlib", box_color=(255, 0, 0), text_color=(255, 0, 0), thickness=2, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.6)`
* **Ví dụ:**
```python
from klygo.models import Model
import klygo.visualize as vis

model = Model()
img = Image.open("traffic.jpg")

pred = model.predict(img, prompt="car. bus.")[0]
vis.visualize_prediction(img, pred, backend="matplotlib")
```

### 3. visualize_dataset_image
Vẽ và hiển thị nhãn của một ảnh cụ thể được chỉ định từ tập dữ liệu YOLO.
* **Cú pháp:** `vis.visualize_dataset_image(image_path, label_path, classes, backend="matplotlib", box_color=(255, 0, 0), text_color=(255, 0, 0), thickness=2, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.6)`
* **Tham số:**
  * `image_path`: Đường dẫn tới file ảnh cụ thể.
  * `label_path`: Đường dẫn tới file nhãn `.txt` YOLO cụ thể tương ứng.
  * `classes`: Danh sách tên lớp đối tượng (lấy từ `data.yaml`).
* **Ví dụ:**
```python
import klygo.visualize as vis

classes = ["car", "bus", "pedestrian"]
vis.visualize_dataset_image(
    image_path="dataset/images/train/frame_0.jpg",
    label_path="dataset/labels/train/frame_0.txt",
    classes=classes,
    backend="matplotlib"
)
```

### 4. crop_objects
Cắt tất cả các vật thể được gắn nhãn trong ảnh và lưu trực tiếp thành các file ảnh con nén trong một file ZIP.
* **Cú pháp:** `vis.crop_objects(image_path, label_path, classes, output_path, overwrite=False)`
* **Ví dụ:**
```python
import klygo.visualize as vis

vis.crop_objects(
    image_path="dataset/images/train/frame_0.jpg",
    label_path="dataset/labels/train/frame_0.txt",
    classes=["car", "bus"],
    output_path="extracted_cars.zip",
    overwrite=True
)
# File ZIP đầu ra sẽ chứa các file như: car_crop_0.jpg, car_crop_1.jpg, bus_crop_0.jpg...
```

### 5. read_cropped_objects
Đọc nhanh file ZIP chứa các vật thể đã cắt bằng `crop_objects` và trả về một từ điển đối tượng để sử dụng tiếp.
* **Cú pháp:** `vis.read_cropped_objects(zip_path) -> dict[str, list[PIL.Image.Image]]`
* **Ví dụ:**
```python
import klygo.visualize as vis

crops = vis.read_cropped_objects("extracted_cars.zip")
# crops['car'] -> chứa list các ảnh PIL Image của class car
for car_img in crops.get("car", []):
    vis.show_image(car_img, backend="matplotlib")
```

### crop_dataset
Cắt toàn bộ object từ dataset YOLO dạng phẳng hoặc có `train/val/test`.

```python
crops = vis.crop_dataset(
    source="dataset/",  # hoặc dataset.zip
    target="crops/",    # tùy chọn
    padding=4,
)

vis.show_image(crops[0])
```

Hàm luôn trả `list[PIL.Image.Image]`. Khi có `target`, ảnh cũng được lưu theo thư mục class.

### 6. plot_dataset_stats
Quét và thống kê số lượng đối tượng của từng lớp trong toàn bộ dataset và vẽ biểu đồ cột trực quan.
* **Cú pháp:** `vis.plot_dataset_stats(source)`
* **Tham số:**
  * `source`: Đường dẫn file ZIP hoặc thư mục chứa dataset YOLO (hỗ trợ cả dạng flat thô hoặc đã chia split `train/val/test`).
* **Ví dụ:**
```python
import klygo.visualize as vis

vis.plot_dataset_stats("dataset.zip")
```

---

## 7. unpartition (Thuộc module `klygo.datasets`)
Chuyển đổi một bộ dữ liệu đã được chia phân hoạch (`train/val/test`) quay trở lại dạng phẳng thô ban đầu (toàn bộ ảnh gom vào `images/` và nhãn vào `labels/`).
* **Cú pháp:** `unpartition(source, target, overwrite=False, verbose=True)`
* **Ví dụ:**
```python
from klygo.datasets import unpartition

# Chuyển dataset dạng split về dạng flat thô
unpartition("split_dataset/", "raw_dataset.zip", overwrite=True)
```
