# klygo.archive

`klygo.archive` cung cấp các thao tác ZIP: nén, giải nén, liệt kê, tìm kiếm, kiểm tra, chỉnh sửa, gộp và chia theo dung lượng.

```python
import klygo.archive as ar
```

## 1. `compress`

```python
ar.compress(
    source,
    output_path,
    format="zip",
    overwrite=False,
    verbose=True,
)
```

Nén một file hoặc thư mục:

```python
ar.compress("dataset", "dataset.zip", overwrite=True)
ar.compress("config.yaml", "config.zip", overwrite=True)
```

Hiện định dạng được hỗ trợ là ZIP. `output_path` phải là đường dẫn file.

## 2. `extract`

Giải nén toàn bộ ZIP vào thư mục:

```python
ar.extract(
    archive_path="dataset.zip",
    output_dir="dataset",
    overwrite=True,
    verbose=True,
)
```

## 3. `extract_file`

Giải nén một file cụ thể và làm phẳng tên file tại thư mục đích:

```python
files = ar.list_files("dataset.zip")
ar.extract_file(
    archive_path="dataset.zip",
    filename=files[0],
    output_dir="single",
    overwrite=True,
)
```

`filename` phải khớp chính xác tên nằm trong ZIP.

## 4. `list_files` và `search`

```python
all_files = ar.list_files("dataset.zip")
jpg_files = ar.search("dataset.zip", "images/*.jpg")
```

`search` sử dụng glob pattern trên đường dẫn bên trong ZIP.

## 5. `get_info`

```python
info = ar.get_info("dataset.zip")

print(info["file_count"])
print(info["compressed_size"])
print(info["uncompressed_size"])
```

Hàm chỉ đọc metadata, không giải nén dữ liệu.

## 6. `test`

Kiểm tra CRC của ZIP:

```python
is_valid = ar.test("dataset.zip")
print(is_valid)
```

Kết quả `True` nghĩa là không phát hiện file lỗi.

## 7. `add`

Thêm một file, thư mục hoặc danh sách nguồn vào ZIP đã tồn tại:

```python
ar.add("dataset.zip", "new_image.jpg")

ar.add(
    "dataset.zip",
    ["new_image.jpg", "new_label.txt", "metadata/"],
    verbose=True,
)
```

## 8. `remove`

Xóa một hoặc nhiều entry theo tên bên trong ZIP:

```python
ar.remove("dataset.zip", "temp.txt")
ar.remove("dataset.zip", ["temp.txt", "logs/debug.log"])
```

Nên dùng `list_files()` để lấy đúng tên entry trước khi xóa.

## 9. `merge`

Gộp nhiều ZIP thành một ZIP:

```python
ar.merge(
    archive_paths=["part_1.zip", "part_2.zip"],
    output_path="merged.zip",
    overwrite=True,
    verbose=True,
)
```

Đây là gộp archive thuần túy. Để gộp dataset YOLO và ánh xạ class ID, dùng `klygo.datasets.merge`.

## 10. `split_by_size`

Chia một ZIP thành các ZIP nhỏ hơn theo dung lượng nén tối đa, tính bằng MB:

```python
parts = ar.split_by_size(
    archive_path="large_dataset.zip",
    size=100,
    output_dir="parts",
    overwrite=True,
    verbose=True,
)

print(parts)
```

`size` nhận số nguyên hoặc số thực. Kết quả là danh sách đường dẫn các ZIP đã tạo.

## 11. `human_size`

```python
print(ar.human_size(0))        # 0.00 B
print(ar.human_size(1024))     # 1.00 KB
print(ar.human_size(1048576))  # 1.00 MB
```

## 12. Ghi đè và tiến trình

- `overwrite=False`: bảo vệ file/thư mục đích đã tồn tại.
- `overwrite=True`: cho phép thay thế đầu ra.
- `verbose=True`: hiển thị thanh tiến trình cho thao tác dài.
- Các API chỉ đọc như `list_files`, `search`, `get_info`, `test` không có `overwrite`.

## 13. Danh sách API public

```text
compress, extract, extract_file, list_files, search,
get_info, test, add, remove, merge, split_by_size, human_size
```
