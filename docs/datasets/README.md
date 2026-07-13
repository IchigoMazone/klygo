# klygo.datasets

`klygo.datasets` quản lý dataset YOLO dạng phẳng hoặc đã chia `train/val/test`. Hầu hết API nhận `source` là thư mục hoặc ZIP; đầu ra linh hoạt dùng `target`, `output_path` hoặc `output_dir` tùy loại kết quả.

```python
import klygo.datasets as ds
```

## 1. Cấu trúc dataset hỗ trợ

Dạng phẳng:

```text
dataset/
├── images/
├── labels/
└── data.yaml
```

Dạng đã partition:

```text
dataset/
├── images/train|val|test/
├── labels/train|val|test/
└── data.yaml
```

`data.yaml` nên có `names` dạng list hoặc dictionary.

## 2. `partition`

Chia dataset thành `train`, `val`, `test`:

```python
ds.partition(
    source="raw_dataset.zip",
    target="partitioned_dataset",
    ratios=(0.8, 0.1, 0.1),
    overwrite=True,
    verbose=True,
)
```

`target` có thể là thư mục hoặc file `.zip`.

Quy tắc `ratios`:

- `(train,)`: train từ `0.5` đến `0.95`; phần còn lại chia đều cho val/test.
- `(train, val)`: tổng train + val từ `0.525` đến `0.975`; phần còn lại là test.
- `(train, val, test)`: tổng bắt buộc bằng `1.0`.
- Mọi giá trị phải viết dạng float, ví dụ `0.8`, không phải `8`.

Ví dụ chỉ truyền train:

```python
ds.partition("raw_dataset", "partitioned.zip", ratios=(0.8,))
```

## 3. `repartition`

Chia lại dataset đã có split theo tỷ lệ mới:

```python
ds.repartition(
    source="partitioned_dataset",
    target="repartitioned_dataset",
    ratios=(0.7, 0.2, 0.1),
    overwrite=True,
)
```

Có thể ghi lại cùng đường dẫn:

```python
ds.repartition(
    source="partitioned_dataset",
    target="partitioned_dataset",
    ratios=(0.5, 0.5, 0.0),
    overwrite=True,
)
```

## 4. `unpartition`

Đưa dataset `train/val/test` về dạng phẳng:

```python
ds.unpartition(
    source="partitioned_dataset",
    target="flat_dataset",
    overwrite=True,
)
```

Xuất trực tiếp ZIP:

```python
ds.unpartition(
    source="partitioned_dataset.zip",
    target="flat_dataset.zip",
    overwrite=True,
)
```

## 5. `merge`

Gộp nhiều dataset YOLO và tự ánh xạ class ID:

```python
ds.merge(
    sources=["dataset_a", "dataset_b.zip"],
    output_path="merged_dataset.zip",
    overwrite=True,
    verbose=True,
)
```

Hàm thực hiện:

- Tổng hợp danh sách class không trùng lặp.
- Ánh xạ lại class ID trong từng file nhãn.
- Thêm prefix nguồn để tránh trùng tên ảnh.
- Tạo `data.yaml` mới.

Khác với `klygo.archive.merge`, hàm này hiểu cấu trúc YOLO.

## 6. `split`

Tạo nhiều dataset ZIP độc lập. Đây không phải `partition`: `partition` tạo train/val/test trong một dataset, còn `split` tạo nhiều dataset riêng.

### Tách mỗi class thành một ZIP

```python
ds.split(
    source="dataset.zip",
    output_dir="class_datasets",
    by_class=True,
    overwrite=True,
)
```

Kết quả ví dụ: `split_apple.zip`, `split_orange.zip`.

### Tách theo nhóm class

```python
ds.split(
    source="dataset.zip",
    output_dir="groups",
    class_groups=[
        ["apple", "orange"],
        ["banana"],
    ],
    overwrite=True,
)
```

### Tách thành các phần theo tỷ lệ

```python
ds.split(
    source="dataset.zip",
    output_dir="parts",
    ratios=(0.5, 0.3, 0.2),
    overwrite=True,
)
```

Các tỷ lệ của `split` phải có ít nhất hai giá trị và tổng bằng `1.0`.

## 7. `remap_classes`

Đổi class ID, đổi tên class hoặc thực hiện cả hai:

```python
ds.remap_classes(
    source="dataset.zip",
    target="remapped.zip",
    class_map={
        0: 1,
        "apple": "red_apple",
    },
    overwrite=True,
    verbose=True,
)
```

Đổi tên nhưng giữ ID:

```python
ds.remap_classes(
    source="dataset",
    target="renamed_dataset",
    class_map={"apple": "fresh_apple"},
    overwrite=True,
)
```

`target` nhận thư mục hoặc ZIP.

## 8. `get_dataset_info`

```python
info = ds.get_dataset_info("dataset.zip")

print(info["classes"])
print(info["nc"])
print(info["total_images"])
print(info["total_labels"])
print(info["unlabeled_images"])
print(info["splits"])
```

Ví dụ kết quả:

```python
{
    "classes": ["apple"],
    "nc": 1,
    "total_images": 361,
    "total_labels": 361,
    "unlabeled_images": 0,
    "splits": {"train": 288, "val": 36, "test": 37},
}
```

## 9. Workflow phổ biến

### Dataset thô → partition → kiểm tra

```python
ds.partition(
    source="raw.zip",
    target="dataset",
    ratios=(0.8, 0.1, 0.1),
    overwrite=True,
)

print(ds.get_dataset_info("dataset"))
```

### Gộp → đổi tên class → partition

```python
ds.merge(
    sources=["source_a.zip", "source_b"],
    output_path="merged.zip",
    overwrite=True,
)

ds.remap_classes(
    source="merged.zip",
    target="renamed.zip",
    class_map={"car": "vehicle"},
    overwrite=True,
)

ds.partition(
    source="renamed.zip",
    target="final_dataset",
    ratios=(0.8, 0.1, 0.1),
    overwrite=True,
)
```

## 10. Danh sách API public

```text
partition, repartition, unpartition, merge, split,
remap_classes, get_dataset_info
```
