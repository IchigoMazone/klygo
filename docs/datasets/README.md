# klygo.datasets — Phân chia, Repartition, Gộp và Tách bộ dữ liệu YOLO

Module `klygo.datasets` cung cấp các công cụ mạnh mẽ và tối ưu để xử lý, quản lý và tổ chức các tập dữ liệu ảnh nhãn theo chuẩn YOLO. Module này hỗ trợ từ việc phân chia tập train/val/test ban đầu, chia lại tỷ lệ dữ liệu tại chỗ, hợp nhất nhiều bộ dữ liệu (tự động giải quyết xung đột tên và ánh xạ lại class ID) cho đến tách nhỏ dữ liệu theo các nhóm class hoặc tỷ lệ con.

---

## Mục lục

- [1. partition — Phân chia tập Train/Val/Test](#1-partition--phan-chia-tap-trainvaltest)
- [2. repartition — Chia lại tỷ lệ tại chỗ](#2-repartition--chia-lai-ty-le-tai-cho)
- [3. merge — Hợp nhất nhiều bộ dữ liệu](#3-merge--hop-nhat-nhieu-bo-du-lieu)
- [4. split — Chia nhỏ dữ liệu theo Class hoặc Tỷ lệ](#4-split--chia-nho-du-lieu-theo-class-hoac-ty-le)
- [5. remap_classes — Ánh xạ và đổi tên lớp dữ liệu](#5-remap_classes--anh-xa-va-doi-ten-lop-du-lieu)
- [6. get_dataset_info — Lấy thông tin chi tiết bộ dữ liệu](#6-get_dataset_info--lay-thong-tin-chi-tiet-bo-du-lieu)

---

## 1. partition — Phân chia tập Train/Val/Test

Hàm `partition` dùng để phân chia một thư mục hoặc tệp nén chứa ảnh và nhãn (unpartitioned) thành các tập con `train`, `val`, và `test` theo tỷ lệ ngẫu nhiên có tính lặp lại (reproducible seed).

### Cách sử dụng
```python
from klygo.datasets import partition

partition(
    source="path/to/data.zip",     # Tệp ZIP hoặc thư mục chứa ảnh & nhãn gốc
    output="path/to/output_dir",   # Thư mục đích chứa kết quả chia tập
    ratios=(0.8,),                 # Tỷ lệ tập train. Val/test sẽ chia đều phần còn lại
    overwrite=True,                # Ghi đè nếu output đã tồn tại
    verbose=True
)
```

### Ràng buộc tỷ lệ `ratios`
- **1 tỷ lệ `(train,)`**: Giá trị nằm trong khoảng `[0.5, 0.95]`. `val = test = (1 - train) / 2`.
- **2 tỷ lệ `(train, val)`**: Tổng giá trị nằm trong khoảng `[0.525, 0.975]`. `test = 1 - (train + val)`.
- **3 tỷ lệ `(train, val, test)`**: Tổng của 3 tỷ lệ bắt buộc phải bằng `1.0`.

---

## 2. repartition — Chia lại tỷ lệ tại chỗ

Hàm `repartition` cho phép điều chỉnh và phân chia lại tỷ lệ của một bộ dữ liệu đã được chia tập sẵn trước đó (đã có cấu trúc `images/train`, `images/val`, v.v.). Hàm này hỗ trợ ghi đè trực tiếp (in-place) một cách an toàn.

### Cách sử dụng
```python
from klygo.datasets import repartition

repartition(
    source="path/to/partitioned_dataset",  # Thư mục dataset đã chia tập
    output="path/to/partitioned_dataset",  # Lưu ghi đè trực tiếp (in-place)
    ratios=(0.7, 0.2, 0.1),                # Tỷ lệ mới
    overwrite=True,
    verbose=True
)
```

---

## 3. merge — Hợp nhất nhiều bộ dữ liệu

Hàm `merge` cho phép hợp nhất nhiều bộ dữ liệu thô (chưa chia tập) khác nhau thành một bộ dữ liệu nén duy nhất dưới dạng tệp `.zip`.

### Các tính năng đặc biệt:
1. **Tránh xung đột tên file**: Tự động thêm tiền tố `src{index}_` (ví dụ: `src0_frame_1.jpg`, `src1_frame_1.jpg`) cho cả file ảnh và file nhãn tương ứng để tránh bị ghi đè dữ liệu.
2. **Hợp nhất Class & Tự động ánh xạ lại Class ID**: Đọc danh sách lớp từ file `data.yaml` của tất cả các nguồn, tổng hợp thành một danh sách lớp toàn cục duy nhất. Nếu class ID trong file nhãn `.txt` của nguồn nào cần thay đổi để khớp với danh sách lớp toàn cục mới, hàm sẽ tự động cập nhật lại nội dung file `.txt` tương ứng.

### Cách sử dụng
```python
from klygo.datasets import merge

merge(
    sources=["path/to/dataset1", "path/to/dataset2.zip"], # Danh sách các nguồn
    output="path/to/merged_dataset.zip",                  # Tệp ZIP đích
    overwrite=True,
    verbose=True
)
```

---

## 4. split — Chia nhỏ dữ liệu theo Class hoặc Tỷ lệ

Hàm `split` cho phép phân chia nhỏ bộ dữ liệu hiện có (chấp nhận cả dữ liệu thô hoặc đã chia tập) thành nhiều tệp ZIP dữ liệu thô độc lập. Hàm hỗ trợ hai chế độ hoạt động:

### Chế độ A: Tách theo Class / Nhóm Class
Tách dữ liệu dựa trên danh sách các lớp đối tượng. Các file nhãn `.txt` con sẽ được lọc bỏ nhãn thừa và ánh xạ lại chỉ số class ID cục bộ khớp với file `data.yaml` con mới.
```python
from klygo.datasets import split

# 1. Tách từng class riêng biệt (Mỗi class thành 1 file ZIP: split_apple.zip, split_orange.zip, ...)
split(
    source="path/to/dataset.zip",
    output_dir="path/to/output_dir",
    by_class=True,
    overwrite=True
)

# 2. Tách theo nhóm class tùy chỉnh
split(
    source="path/to/dataset.zip",
    output_dir="path/to/output_dir",
    class_groups=[["apple", "orange"], ["banana"]],
    overwrite=True
)
```

### Chế độ B: Tách theo Tỷ lệ (Subset Splitting)
Phân tách ngẫu nhiên toàn bộ dữ liệu thành nhiều phần độc lập không trùng lặp (ví dụ: tách thành 3 phần bằng nhau, mỗi phần chiếm 33% dữ liệu).
```python
from klygo.datasets import split

split(
    source="path/to/dataset.zip",
    output_dir="path/to/output_dir",
    ratios=(0.33, 0.33, 0.34), # Chia dataset làm 3 phần
    overwrite=True
)
```

---


## 5. remap_classes — Ánh xạ và đổi tên lớp dữ liệu

Hàm `remap_classes` giúp thay đổi mã chỉ mục class ID (ví dụ: đổi từ `0` sang `1`) hoặc đổi tên lớp trực tiếp trên file cấu hình `data.yaml` và các file nhãn `.txt` tương ứng.

### Các tính năng chính:
1. **Đổi ID lớp**: Ánh xạ lại chỉ số ID trong các file nhãn `.txt` (ví dụ: `{0: 1}` sẽ đổi tất cả đối tượng lớp 0 thành lớp 1).
2. **Đổi tên lớp**: Cập nhật lại chuỗi tên ở dạng từ điển (`names: {id: name}`) ở file `data.yaml` (ví dụ: `{"apple": "red_apple"}`), giúp tránh phải điền các class trống giả như `class_0` nếu có chỉ số class ID bị khuyết.
3. **Hỗ trợ Đầu ra Linh hoạt**: Tự động nhận diện đuôi `.zip` để nén đầu ra hoặc lưu vào thư mục đích.

### Cách sử dụng
```python
from klygo.datasets import remap_classes

remap_classes(
    source="path/to/dataset.zip",
    output="path/to/remapped_dataset.zip",
    class_map={0: 1, "apple": "red_apple"}, # Đổi ID 0 thành 1 và đổi tên lớp
    overwrite=True
)
```

---

## 6. get_dataset_info — Lấy thông tin chi tiết bộ dữ liệu

Hàm `get_dataset_info` phân tích tệp cấu hình `data.yaml` và cấu trúc tệp của bộ dữ liệu YOLO (chấp nhận cả tệp nén `.zip` hoặc thư mục thông thường) để trả về thông tin chi tiết.

### Dữ liệu trả về:
- `classes`: Danh sách tên các lớp đối tượng.
- `nc`: Số lượng các lớp đối tượng.
- `total_images`: Tổng số lượng file ảnh quét được.
- `total_labels`: Số lượng ảnh đã có file nhãn `.txt` đi kèm.
- `unlabeled_images`: Số lượng ảnh trống chưa có nhãn.
- `splits`: Từ điển thống kê số lượng ảnh thuộc các tập con (ví dụ: `{'train': 288, 'val': 36, 'test': 37}` hoặc `{'raw': 361}`).

### Cách sử dụng
```python
from klygo.datasets import get_dataset_info

info = get_dataset_info("path/to/dataset.zip")
print(info)
```


