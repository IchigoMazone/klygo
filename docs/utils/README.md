# klygo.utils

`klygo.utils` chứa helper dùng chung giữa các package:

- Chuẩn hóa ảnh PIL/OpenCV và đọc frame video.
- Đọc bbox YOLO, crop và lưu crop.
- Tạo ảnh lưới.
- Quét, sao chép, di chuyển và ánh xạ file dataset.

Các hàm đều mang prefix `_` hoặc nằm trong module nội bộ, không được export qua `klygo.utils.__init__` và không cam kết ổn định như API public.

Người dùng nên gọi API cấp cao tương ứng:

```python
from klygo.io import read_images
from klygo.models import Model
from klygo.visualize import crop_dataset, crop_image, read_crops
```

Không nên import trực tiếp `_normalize_images`, `_crop_boxes`, `_save_crops` hoặc các helper dataset trong code ứng dụng.
