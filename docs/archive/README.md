# klygo.archive

Module xử lý file nén ZIP. Cung cấp 11 API để thao tác với archive.

```python
import klygo.archive as ar
```

---

## Hướng dẫn chi tiết từng API

### 1. compress
Nén một file hoặc toàn bộ thư mục thành file ZIP.
* **Cú pháp:** `ar.compress(source, output, format="zip", overwrite=False, verbose=True)`
* **Tham số:**
  * `source`: File hoặc thư mục cần nén.
  * `output`: Đường dẫn lưu file ZIP đầu ra.
  * `overwrite`: Ghi đè nếu file ZIP đích đã tồn tại.
  * `verbose`: Hiển thị thanh tiến trình màu xanh lá.
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Nén cả thư mục, hiển thị tiến trình, cho phép ghi đè
ar.compress("data_dir/", "data.zip", overwrite=True, verbose=True)
```

### 2. extract
Giải nén toàn bộ nội dung archive vào một thư mục.
* **Cú pháp:** `ar.extract(source, output=".", overwrite=False, verbose=True)`
* **Tham số:**
  * `source`: Đường dẫn file ZIP đầu vào.
  * `output`: Thư mục đích giải nén.
  * `overwrite`: Ghi đè lên các tệp đã có tại thư mục đích.
  * `verbose`: Hiển thị thanh tiến trình màu xanh lam.
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Giải nén toàn bộ file ZIP vào thư mục 'extracted'
ar.extract("data.zip", output="extracted/", overwrite=True, verbose=True)
```

### 3. extract_file
Giải nén một file cụ thể từ file ZIP.
* **Cú pháp:** `ar.extract_file(source, filename, output=".", overwrite=False)`
* **Tham số:**
  * `source`: Đường dẫn file ZIP.
  * `filename`: Tên/đường dẫn chính xác của file cần giải nén bên trong ZIP.
  * `output`: Thư mục đích giải nén.
  * `overwrite`: Ghi đè lên tệp nếu đã tồn tại tại đích.
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Chỉ giải nén tệp 'images/001.jpg' trong file nén ra thư mục 'out'
ar.extract_file("data.zip", "images/001.jpg", output="out/", overwrite=True)
```

### 4. list_files
Liệt kê toàn bộ đường dẫn file chứa trong archive.
* **Cú pháp:** `ar.list_files(source) -> list[str]`
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Lấy danh sách tất cả các file bên trong ZIP
files = ar.list_files("data.zip")
print(files)
```

### 5. search
Tìm kiếm các file bên trong archive khớp với biểu thức glob pattern.
* **Cú pháp:** `ar.search(source, pattern) -> list[str]`
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Tìm tất cả file ảnh JPEG trong thư mục images bên trong file nén
jpg_files = ar.search("data.zip", "images/*.jpg")
print(jpg_files)
```

### 6. get_info
Trả về metadata tổng quan của file nén (không giải nén).
* **Cú pháp:** `ar.get_info(source) -> dict`
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Lấy thông tin dung lượng gốc, dung lượng nén, tỷ lệ nén...
info = ar.get_info("data.zip")
print(f"Số lượng file: {info['file_count']}")
print(f"Kích thước nén: {info['compressed_size']} bytes")
```

### 7. test
Kiểm tra tính toàn vẹn của tệp ZIP bằng CRC32 checksum.
* **Cú pháp:** `ar.test(source) -> bool`
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Kiểm tra xem file ZIP có bị lỗi bit-rot hay hỏng hóc hay không
try:
    ar.test("data.zip")
    print("File ZIP hoàn toàn bình thường")
except ValueError as e:
    print(f"File ZIP bị lỗi: {e}")
```

### 8. add
Thêm một hoặc nhiều file/thư mục vào tệp ZIP đã tồn tại.
* **Cú pháp:** `ar.add(source, files, verbose=True)`
* **Tham số:**
  * `source`: File ZIP hiện có.
  * `files`: Tên file hoặc list các đường dẫn cần thêm.
  * `verbose`: Hiển thị thanh tiến trình màu vàng.
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Thêm file ảnh mới và file cấu hình mới vào ZIP
ar.add("data.zip", ["new_image.jpg", "config.json"], verbose=True)
```

### 9. remove
Xóa một hoặc nhiều file ra khỏi tệp ZIP.
* **Cú pháp:** `ar.remove(source, files)`
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Xóa tệp log tạm khỏi file nén
ar.remove("data.zip", "temp_log.txt")
```

### 10. merge
Gộp hai hoặc nhiều file ZIP nguồn thành một file ZIP mới duy nhất.
* **Cú pháp:** `ar.merge(sources, output, overwrite=False, verbose=True)`
* **Tham số:**
  * `sources`: Danh sách các đường dẫn file ZIP nguồn.
  * `output`: Đường dẫn ZIP gộp đầu ra.
  * `verbose`: Hiển thị thanh tiến trình màu magenta.
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Gộp part1.zip và part2.zip thành merged.zip
ar.merge(["part1.zip", "part2.zip"], "merged.zip", overwrite=True, verbose=True)
```

### 11. split
Chia nhỏ một file ZIP lớn thành các phần nhỏ hơn có dung lượng nén giới hạn.
* **Cú pháp:** `ar.split(source, size, output_dir=".", overwrite=False, verbose=True) -> list[str]`
* **Tham số:**
  * `size`: Kích thước nén tối đa của mỗi phần (tính bằng MB, hỗ trợ cả số thực).
  * `output_dir`: Thư mục lưu các phần được tạo ra.
* **Ví dụ sử dụng:**
```python
import klygo.archive as ar

# Chia nhỏ file ZIP thành các phần tối đa 10 MB, lưu vào thư mục parts
parts = ar.split("large_data.zip", size=10, output_dir="parts/", overwrite=True)
print(parts)  # ['parts/large_data_part_001.zip', ...]
```
