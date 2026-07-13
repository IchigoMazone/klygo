# klygo.visualize

`klygo.visualize` hiển thị ảnh, vẽ bbox, trực quan hóa dự đoán và nhãn YOLO, cắt object, đọc lại crop và vẽ thống kê dataset.

```python
import klygo.visualize as vis
```

## 1. `show_image`

```python
vis.show_image(
    image,
    title="Image",
    backend="matplotlib",
    figsize=(10, 10),
)
```

Nhận ảnh PIL RGB hoặc mảng OpenCV BGR.

### Matplotlib — phù hợp notebook/Colab

```python
from PIL import Image

image = Image.open("traffic.jpg")
vis.show_image(image, title="Traffic")
```

### OpenCV — cửa sổ desktop

```python
import cv2 as cv

image = cv.imread("traffic.jpg")
vis.show_image(image, title="Traffic", backend="opencv")
```

Mọi giá trị backend khác `matplotlib` hiện được xử lý bằng OpenCV.

## 2. `draw_bboxes`

Vẽ bbox `x1, y1, x2, y2` và trả ảnh cùng kiểu với đầu vào:

```python
annotated = vis.draw_bboxes(
    image=image,
    bboxes=[[20, 30, 180, 220]],
    labels=["car"],
    scores=[0.92],
    box_color=(0, 255, 0),
    text_color=(0, 0, 255),
    thickness=2,
    font_scale=0.7,
)
```

- Đầu vào PIL → trả PIL.
- Đầu vào NumPy/OpenCV → trả NumPy.
- `labels` và `scores` có thể bỏ qua.
- Màu dùng thứ tự BGR của OpenCV.

## 3. `visualize_prediction`

Vẽ và hiển thị trực tiếp một kết quả của `Model.predict`:

```python
from PIL import Image
from klygo.models import Model

model = Model()
image = Image.open("traffic.jpg").convert("RGB")
prediction = model.predict(image, prompt="car. bus.")[0]

vis.visualize_prediction(
    image=image,
    prediction=prediction,
    backend="matplotlib",
)
```

Có thể truyền thêm `box_color`, `text_color`, `thickness`, `font`, `font_scale`, `figsize`.

## 4. `visualize_dataset_image`

Đọc một cặp ảnh/nhãn YOLO rồi hiển thị bbox:

```python
vis.visualize_dataset_image(
    image_path="dataset/images/train/frame_0.jpg",
    label_path="dataset/labels/train/frame_0.txt",
    classes=["car", "bus", "person"],
    backend="matplotlib",
)
```

Nếu file nhãn không tồn tại, ảnh vẫn được hiển thị mà không có bbox.

## 5. `crop_image`

Cắt object của một ảnh từ nhãn YOLO:

```python
crops = vis.crop_image(
    image_path="dataset/images/frame_0.jpg",
    label_path="dataset/labels/frame_0.txt",
    classes=["car", "person"],
    target="crops",
    padding=4,
)
```

Hàm luôn trả `list[PIL.Image.Image]`. Khi có `target`, ảnh được lưu thêm theo class:

```text
crops/
├── car/frame_0_crop_0.jpg
└── person/frame_0_crop_0.jpg
```

Không muốn lưu file:

```python
crops = vis.crop_image(
    "image.jpg",
    "image.txt",
    ["apple"],
)
```

## 6. `crop_dataset`

Cắt toàn bộ object trong dataset YOLO dạng phẳng hoặc `train/val/test`:

```python
crops = vis.crop_dataset(
    source="dataset",       # hoặc dataset.zip
    target="dataset_crops", # có thể bỏ qua
    padding=4,
)
```

Hàm luôn trả danh sách ảnh PIL và có thể nhận dataset thư mục hoặc ZIP.

## 7. `read_crops`

Đọc crop từ thư mục, ZIP hoặc kết quả trả về của `Model.crop`:

```python
vis.read_crops(
    source,
    grid=(5, 5),
    metadata=False,
    label=None,
)
```

### Lưới mặc định 5 × 5

```python
grid_image = vis.read_crops("crops")
vis.show_image(grid_image)
```

Hàm lấy tối đa 25 ảnh đầu, căn giữa ảnh trong từng ô và dùng nền trắng cho ô trống.

### Lưới cố định

```python
grid_image = vis.read_crops("crops.zip", grid=(3, 3))
```

Lấy tối đa 9 ảnh đầu.

### Số cột cố định, số hàng tự động

```python
grid_image = vis.read_crops("crops", grid=(3,))
```

Ba cột và đủ số hàng để chứa toàn bộ ảnh.

### Chỉ đọc một class

```python
apple_grid = vis.read_crops("crops", label="apple")
vis.show_image(apple_grid)

apple_images = vis.read_crops("crops", grid=None, label="apple")
```

`label` hoạt động giống nhau với thư mục, ZIP và kết quả của `Model.crop`.

### Dictionary theo label

```python
crops_by_label = vis.read_crops("crops", grid=None)

for image in crops_by_label.get("apple", []):
    vis.show_image(image)
```

### Đọc trực tiếp đầu ra của `Model.crop`

```python
from klygo.models import Model

model = Model()
results = model.crop(
    source="orchard.mp4",
    target="crops",
    prompt="apple. orange.",
)

grid_image = vis.read_crops(results)
crops_by_label = vis.read_crops(results, grid=None)
```

Dù `source` là thư mục, ZIP hay kết quả của `Model.crop`, hàm luôn trả ảnh PIL
dạng lưới khi có `grid`, hoặc `dict[label, list[PIL.Image]]` khi `grid=None`.
Ở chế độ mặc định, trường `score` vẫn nằm trong biến `results` ban đầu nhưng
không được đưa vào kết quả của `read_crops`.

### Đọc đầy đủ metadata

Khi `Model.crop` có `target`, model lưu thêm `metadata.json`. Có thể khôi phục
cùng một cấu trúc `image`, `score`, `label` từ kết quả trực tiếp, thư mục hoặc
ZIP bằng `metadata=True`:

```python
results = model.crop("orchard.mp4", target="crops")

direct_records = vis.read_crops(results, metadata=True)
saved_records = vis.read_crops("crops", metadata=True)
apple_records = vis.read_crops("crops", metadata=True, label="apple")

for item in saved_records:
    print(item["label"], item["score"])
    vis.show_image(item["image"])
```

Với dataset crop cũ không có `metadata.json`, `score` nhận giá trị `None`.
Khi `metadata=True`, hàm luôn trả danh sách record và không tạo lưới; tham số
`grid` chỉ được sử dụng khi `metadata=False`.

## 8. `read_detections`

Đọc kết quả đã annotate cùng bbox, score và label từ kết quả trực tiếp của
`Model.detect(metadata=True)`, thư mục hoặc ZIP:

```python
results = model.detect(
    source="traffic.mp4",
    annotated_dir="detected",
    metadata=True,
)

direct_records = vis.read_detections(results)
saved_records = vis.read_detections("detected")
zipped_records = vis.read_detections("detected.zip")

for item in saved_records:
    vis.show_image(item["image"])
    print(item["boxes"])
    print(item["scores"])
    print(item["labels"])
```

Nếu đọc thư mục hoặc ZIP detect cũ không có `metadata.json`, hàm vẫn trả ảnh;
`boxes`, `scores` và `labels` là các danh sách rỗng.

## 9. `plot_dataset_stats`

Đếm số object theo class rồi hiển thị biểu đồ cột:

```python
vis.plot_dataset_stats(
    source="dataset.zip",  # hoặc thư mục
    figsize=(12, 6),
)
```

Hỗ trợ dataset phẳng và dataset có `train/val/test`.

## 10. Workflow hoàn chỉnh

### Predict → visualize

```python
from klygo.io import read_images
from klygo.models import Model

model = Model()
image = read_images("traffic.jpg")[0]
prediction = model.predict(image, prompt="car. person.")[0]
vis.visualize_prediction(image, prediction)
```

### Model crop → grid

```python
model.crop(
    source="traffic.mp4",
    target="crops",
    prompt="car. person.",
)

grid = vis.read_crops("crops", grid=(4,))
vis.show_image(grid, title="Detected objects")
```

## 11. Danh sách API public

```text
show_image, draw_bboxes, visualize_prediction,
visualize_dataset_image, crop_image, crop_dataset,
read_crops, read_detections, plot_dataset_stats
```
