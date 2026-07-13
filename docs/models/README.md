# klygo.models

`klygo.models` cung cấp lớp `Model` để nhận diện vật thể zero-shot trên ảnh, thư mục ảnh và video. Model hiện dùng Grounding DINO thông qua Hugging Face Transformers.

```python
from klygo.models import Model

model = Model()
```

## 1. Khởi tạo

```python
model = Model(model="Grounding-Dino/232M")
```

Model tự chọn CUDA nếu PyTorch nhận được GPU, ngược lại dùng CPU. Tên `Locate-Anything/3B` đã được khai báo nhưng chưa triển khai và sẽ phát sinh `NotImplementedError`.

## 2. Những loại `source` được hỗ trợ

| Source | `predict` | `detect` | `crop` |
|---|---:|---:|---:|
| Một ảnh PIL RGB | Có | Có | Có |
| Một mảng OpenCV BGR | Có | Có | Có |
| Danh sách ảnh PIL/OpenCV | Có | Có | Có |
| File ảnh | Có | Có | Có |
| Thư mục ảnh | Có | Có | Có |
| File video | Không | Có | Có |

Các định dạng ảnh hỗ trợ gồm BMP, JPEG, PNG, TIFF và WebP. Video hỗ trợ AVI, M4V, MKV, MOV, MP4 và WebM.

## 3. Đọc ảnh trước khi chạy Model

Mặc định `read_images` trả ảnh PIL RGB phù hợp để truyền vào Model:

```python
from klygo.io import read_images

images = read_images(
    source="dataset/images",
    recursive=True,
    backend="pil",
)

predictions = model.predict(images, prompt="car. person.")
detected_images = model.detect(images, prompt="car. person.")
crop_results = model.crop(images, prompt="car. person.")
```

Backend OpenCV trả các mảng NumPy BGR. Model vẫn nhận được và tự chuyển sang PIL RGB:

```python
opencv_images = read_images("dataset/images", backend="opencv")
predictions = model.predict(opencv_images, prompt="car.")
```

Không bắt buộc gọi `read_images`: cả ba method đều có thể nhận trực tiếp file hoặc thư mục ảnh.

## 4. `predict` — dự đoán trên ảnh

```text
model.predict(
    source,
    prompt="apple.",
    return_tensors="pt",
    box_threshold=0.25,
    text_threshold=0.25,
    max_area=None,
) -> list[dict]
```

Mỗi ảnh đầu vào tương ứng với một dictionary:

```python
{
    "boxes": Tensor,     # các bbox x1, y1, x2, y2
    "scores": Tensor,    # độ tin cậy
    "labels": list[str], # nhãn từ prompt
}
```

### Một ảnh PIL

```python
from PIL import Image

image = Image.open("traffic.jpg").convert("RGB")
prediction = model.predict(image, prompt="car. bus. person.")[0]

print(prediction["boxes"])
print(prediction["scores"])
print(prediction["labels"])
```

### Một file ảnh

```python
prediction = model.predict(
    source="traffic.jpg",
    prompt="car. traffic light.",
)[0]
```

### Một thư mục ảnh

```python
predictions = model.predict(
    source="dataset/images",
    prompt="apple. orange.",
)
```

Thứ tự kết quả tương ứng với thứ tự file được sắp xếp theo tên.

### Nhiều ảnh có sẵn trong bộ nhớ

```python
images = [Image.open("a.jpg"), Image.open("b.jpg")]
predictions = model.predict(images, prompt="cat. dog.")
```

### Giới hạn diện tích bbox

`max_area` nằm trong khoảng `0..1` và là tỷ lệ diện tích bbox tối đa so với toàn ảnh:

```python
prediction = model.predict(
    "traffic.jpg",
    prompt="car.",
    max_area=0.4,  # chỉ giữ bbox có diện tích <= 40% ảnh
)[0]
```

## 5. `detect` — vẽ kết quả dự đoán

```text
model.detect(
    source,
    output_path=None,
    prompt="apple.",
    return_tensors="pt",
    box_threshold=0.25,
    text_threshold=0.25,
    desc="Processing",
    annotated_dir=None,
    dataset_dir=None,
    max_area=None,
    box_color=(255, 0, 0),
    text_color=(255, 0, 0),
    thickness=2,
    font=cv.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    metadata=False,
) -> list[PIL.Image.Image] | list[dict]
```

Mặc định hàm trả danh sách ảnh PIL đã vẽ bbox, label và score. Khi
`metadata=True`, mỗi phần tử gồm `image`, `boxes`, `scores` và `labels`.

### Detect một ảnh hoặc thư mục ảnh

```python
detected = model.detect(
    source="dataset/images",
    prompt="car. person.",
)

detected[0].show()
```

### Lưu từng ảnh đã annotate

```python
detected = model.detect(
    source="dataset/images",
    annotated_dir="annotated",
    prompt="car. person.",
)
```

Kết quả lưu tại `annotated/frame_0.jpg`, `annotated/frame_1.jpg`, ... và
`annotated/metadata.json`. Metadata được lưu kể cả khi tham số `metadata=False`.

### Lấy và đọc lại metadata detect

```python
from klygo.visualize import read_detections

direct_results = model.detect(
    source="dataset/images",
    prompt="car. person.",
    metadata=True,
)

saved_results = read_detections("annotated")
zipped_results = read_detections("annotated.zip")

for item in saved_results:
    item["image"].show()
    print(item["boxes"], item["scores"], item["labels"])
```

`read_detections()` cũng nhận trực tiếp `direct_results` và luôn chuẩn hóa đầu
ra thành `list[dict]` có cùng bốn trường trên.

### Detect video và lưu video kết quả

```python
detected_frames = model.detect(
    source="traffic.mp4",
    output_path="traffic_detected.mp4",
    prompt="car. bus. person.",
)
```

`output_path` chỉ dùng được khi `source` là video. Với ảnh hoặc thư mục ảnh, hãy dùng `annotated_dir`.

### Tạo dataset YOLO tự động

```python
detected_frames = model.detect(
    source="traffic.mp4",
    dataset_dir="dataset",
    prompt="car. person.",
)
```

Cấu trúc đầu ra:

```text
dataset/
├── images/
│   ├── frame_0.jpg
│   └── ...
├── labels/
│   ├── frame_0.txt
│   └── ...
└── data.yaml
```

`data.yaml` sử dụng `train: images`. Ảnh trong `images` là ảnh gốc chưa annotate; file TXT chứa nhãn YOLO chuẩn hóa.

### Vừa lưu video, frame annotate và dataset

```python
detected_frames = model.detect(
    source="traffic.mp4",
    output_path="outputs/detected.mp4",
    annotated_dir="outputs/annotated",
    dataset_dir="outputs/dataset",
    prompt="car. person.",
)
```

### Tùy chỉnh màu và font

Màu sử dụng thứ tự BGR của OpenCV:

```python
import cv2 as cv

detected = model.detect(
    source="traffic.jpg",
    prompt="car.",
    box_color=(0, 255, 0),
    text_color=(0, 0, 255),
    thickness=3,
    font=cv.FONT_HERSHEY_DUPLEX,
    font_scale=0.8,
)
```

## 6. `crop` — dự đoán và cắt đối tượng

```text
model.crop(
    source,
    target=None,
    prompt="apple.",
    return_tensors="pt",
    box_threshold=0.25,
    text_threshold=0.25,
    desc="Cropping",
    max_area=None,
    padding=0,
    overwrite=False,
) -> list[dict]
```

Mỗi đối tượng được trả dưới dạng:

```python
{
    "image": PIL.Image.Image,
    "score": 0.93,
    "label": "apple",
}
```

### Crop một ảnh

```python
crop_results = model.crop(
    source="orchard.jpg",
    prompt="apple. orange.",
    padding=4,
)

for result in crop_results:
    result["image"].show()
    print(result["label"], result["score"])
```

### Crop thư mục ảnh và lưu theo label

```python
crop_results = model.crop(
    source="dataset/images",
    target="crops",
    prompt="apple. orange.",
)
```

```text
crops/
├── metadata.json
├── apple/
│   ├── frame_0_crop_0.jpg
│   └── ...
└── orange/
    └── ...
```

### Crop video

```python
crop_results = model.crop(
    source="orchard.mp4",
    target="crops",
    prompt="apple.",
    box_threshold=0.3,
    text_threshold=0.25,
    max_area=0.5,
    padding=8,
)
```

Có `target` hay không thì hàm vẫn trả đầy đủ `image`, `score`, `label`.
Khi có `target`, hàm lưu thêm `metadata.json` để giữ `score`, `label` và đường
dẫn của từng ảnh crop.

Nếu `target` đã có dữ liệu, hàm phát sinh `FileExistsError` để tránh ghi đè
ngoài ý muốn. Chỉ xóa và tạo lại thư mục khi chủ động bật `overwrite=True`:

```python
crop_results = model.crop(
    source="orchard.mp4",
    target="crops",
    overwrite=True,
)
```

`metadata.json` được ghi atomic: file tạm chỉ thay thế file chính sau khi hoàn
tất việc ghi dữ liệu.

### Đọc lại và tạo lưới crop

```python
from klygo.visualize import read_crops, show_image

grid_image = read_crops("crops")  # mặc định 5 cột x 5 hàng
show_image(grid_image)

apple_grid = read_crops("crops", label="apple")
show_image(apple_grid)

all_crops_by_label = read_crops("crops", grid=None)
apple_images = all_crops_by_label.get("apple", [])

saved_results = read_crops("crops", metadata=True)
for item in saved_results:
    print(item["label"], item["score"])
```

`metadata=True` trả `list[dict]` có cùng cấu trúc `image`, `score`, `label` như
kết quả trực tiếp của `model.crop()`. Cách này dùng được với cả thư mục và ZIP.

## 7. Threshold

- `box_threshold`: độ tin cậy tối thiểu của bbox, từ `0` đến `1`.
- `text_threshold`: độ phù hợp tối thiểu giữa bbox và prompt, từ `0` đến `1`.
- `max_area`: tỷ lệ diện tích bbox tối đa so với ảnh, từ `0` đến `1`; `None` để không lọc diện tích.

Prompt nhiều lớp nên ngăn cách bằng dấu chấm:

```python
prompt="car. bus. person. traffic light."
```

## 8. Cấu hình tham số nâng cao

```python
model.set_input_kwargs(padding="max_length")
model.set_output_kwargs(top_k=100)
```

Các giá trị được truyền trực tiếp vào processor và post-processor của Hugging Face. Chỉ dùng các keyword mà model/processor hiện tại hỗ trợ.

## 9. Lưu ý hiệu năng

- `read_images` trả danh sách nên đọc thư mục rất lớn sẽ dùng nhiều RAM.
- `predict` xử lý toàn bộ danh sách ảnh trong một batch; GPU có thể hết VRAM nếu danh sách quá lớn.
- `detect` và `crop` đọc video tuần tự, nhưng vẫn giữ toàn bộ kết quả trả về trong RAM.
- Khi chỉ cần file kết quả của video dài, danh sách trả về vẫn tồn tại cho tới khi biến được giải phóng.

## 10. Ngoại lệ thường gặp

```python
# Không hỗ trợ video trong predict
model.predict("video.mp4")  # ValueError

# output_path chỉ dành cho video
model.detect("images/", output_path="result.mp4")  # ValueError

# Threshold phải nằm trong 0..1
model.predict("image.jpg", box_threshold=1.5)  # ValueError

# padding không được âm
model.crop("image.jpg", padding=-1)  # ValueError
```
