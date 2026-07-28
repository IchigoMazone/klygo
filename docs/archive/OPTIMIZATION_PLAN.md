# Kế Hoạch Cải Tiến & Nâng Cấp Module `klygo.archive`

## I. Mục Tiêu Tổng Quát

Nâng cấp `klygo.archive` từ một module xử lý ZIP cơ bản thành một thư viện **Archive Engine đa nền tảng** với API thống nhất, hiệu năng cao, tiết kiệm bộ nhớ RAM và dễ dàng mở rộng.

**Các nguyên tắc thiết kế:**
1. **Một API cho nhiều định dạng**: Thống nhất giao diện cho tất cả loại file lưu trữ.
2. **Tự động nhận dạng định dạng**: Không cần truyền thủ công `format="zip"`.
3. **Streaming I/O mặc định**: Không nạp toàn bộ file vào RAM (`read()` full), sử dụng buffer chunking.
4. **Hỗ trợ Archive dung lượng lớn**: Dùng Generator/Iterator để xử lý file ZIP/TAR hàng triệu file hoặc dung lượng hàng trăm GB.
5. **An toàn bảo mật**: Tự động chống lỗ hổng Path Traversal / Zip-Slip (`../`).
6. **Thanh tiến trình (Progress Bar) chuyên nghiệp**: Màu xanh da trời (Cyan/Light Blue), hiển thị ETA, tốc độ MB/s, tự động đổi đơn vị.

---

## II. Hỗ Trợ Đa Định Dạng Archive (Backend Architecture)

Kiến trúc backend đa nền tảng (`ArchiveBackend`):

```text
ArchiveBackend (Base Class)
│
├── ZipBackend         (.zip)
├── TarBackend         (.tar, .tar.gz, .tgz, .tar.xz, .txz)
├── SevenZipBackend    (.7z)
├── GZipBackend        (.gz)
└── RarBackend         (.rar - Read/Extract Only)
```

### Bảng định dạng hỗ trợ:

| Định dạng | Compress | Extract | Ghi chú |
|---|:---:|:---:|---|
| `.zip` | ✅ | ✅ | Chuẩn mặc định |
| `.tar` | ✅ | ✅ | Chuẩn Linux |
| `.tar.gz` / `.tgz` | ✅ | ✅ | Source code, dataset |
| `.tar.xz` / `.txz` | ✅ | ✅ | Tỷ lệ nén cao |
| `.7z` | ✅ | ✅ | Tỷ lệ nén rất cao (dùng py7zr hoặc lib) |
| `.gz` | ✅ | ✅ | Dành cho 1 file đơn lẻ |
| `.rar` | ❌ | ✅ | Chỉ đọc / giải nén (dùng rarfile / unrar) |

---

## III. Chi Tiết Nâng Cấp Các API Hiện Có

### 1. `compress`
- **Tự động nhận dạng backend** qua đuôi file đầu ra (ví dụ `.7z`, `.tar.gz`, `.zip`).
- **Các tham số mới**:
  - `compresslevel: int`: Mức độ nén (1-9).
  - `method: str`: Thuật toán nén tùy chọn.
  - `preserve_timestamp: bool = True`: Giữ nguyên mtime/ctime của file.
  - `preserve_permissions: bool = True`: Giữ nguyên quyền POSIX/permission.
  - `follow_symlinks: bool = False`: Xử lý symlink.
- **Tối ưu RAM**: Dùng Generator quét thư mục thay vì `rglob` gom `list` vào RAM.

### 2. `extract`
- **Bảo mật Zip-Slip**: Tự động chặn và vô hiệu hóa các đường dẫn chứa `../` hoặc đường dẫn tuyệt đối ghi đè ra ngoài `output_dir`.
- **Các tham số mới**:
  - `password: str | None = None`: Mở khóa archive có mật khẩu.
  - `include: str | list[str] | None = None`: Chỉ giải nén các file khớp mẫu.
  - `exclude: str | list[str] | None = None`: Bỏ qua các file khớp mẫu.
  - `preserve_timestamp: bool = True` & `preserve_permissions: bool = True`.

### 3. `extract_file`
- **Streaming I/O**: Dùng `open()` + `shutil.copyfileobj` để giải nén theo chunk (không nạp toàn bộ file vào RAM).
- **Tra cứu $O(1)$**: Sử dụng `getinfo()` trực tiếp thay vì kiểm tra `in list`.

### 4. `list_files` & `search`
- `list_files`: Hỗ trợ Generator với `iter_files` bên dưới.
- `search`: Bổ sung tham số `regex: bool = False` và `case_sensitive: bool = True`.

### 5. `get_info`
- Bổ sung các thông tin thống kê chi tiết:
  - `format`: Định dạng file nén (`zip`, `tar.gz`, `7z`...)
  - `compression_algorithm`: Thuật toán nén
  - `encrypted`: Trạng thái mã hóa (có pass hay không)
  - `comment`: Ghi chú đính kèm archive
  - `archive_size` & `human_archive_size`
  - `uncompressed_size` & `human_uncompressed_size`
  - `compressed_size` & `human_compressed_size`
  - `file_count` & `directory_count`
  - `largest_file` & `smallest_file`

### 6. `test`
- Bổ sung `raise_exception: bool = False`. Nếu `False`, trả về `True` (hợp lệ) hoặc `False` (lỗi).

### 7. `add`
- **Streaming I/O** khi chép file.
- Bổ sung `on_conflict: Literal["rename", "overwrite", "skip"] = "rename"`.

### 8. `remove`
- **Streaming I/O** khi tạo archive tạm (không nạp toàn bộ file vào RAM bằng `read()`).

### 9. `merge`
- **Streaming I/O** nén/chép giữa các archive.

### 10. `split_by_size`
- **Streaming I/O** khi tạo các part.
- Tùy chỉnh xử lý khi 1 file lớn hơn `size` quy định của part (tạo part riêng hoặc báo lỗi).

### 11. `human_size`
- Bổ sung `decimal_places: int = 2`.

---

## IV. Bổ Sung Các API Mới

1. `detect_format(path: str | Path) -> str`
   - Nhận diện định dạng archive qua magic bytes / extension (ví dụ: `'tar.gz'`, `'zip'`, `'7z'`).
2. `is_archive(path: str | Path) -> bool`
   - Kiểm tra xem file có phải là file archive hợp lệ được hỗ trợ hay không.
3. `open(path: str | Path) -> ArchiveFile`
   - Context manager wrapper (OOP interface) giúp giữ handle mở archive và gọi liên tiếp các hàm `.list_files()`, `.extract()`, `.search()`, `.get_info()` mà không phải mở/đóng file liên tục.
4. `iter_files(archive_path: str | Path)`
   - Trả về generator duyệt qua các item trong archive mà không nạp toàn bộ danh sách vào RAM.
5. `extract_matching(archive_path: str | Path, pattern: str, output_dir: str | Path = ".")`
   - Giải nén trực tiếp các file khớp mẫu (wildcard/pattern) mà không cần qua 2 bước `search` + `extract_file`.
6. `convert(source_path: str | Path, target_path: str | Path)`
   - Chuyển đổi định dạng archive (ví dụ từ `.zip` sang `.7z` hoặc `.tar.gz`).
7. `recompress(source_path: str | Path, target_path: str | Path)`
   - Nén lại archive với mức độ nén hoặc thuật toán nén khác.
8. `compare(archive1: str | Path, archive2: str | Path) -> dict`
   - So sánh 2 file archive và trả về danh sách: `added_files`, `removed_files`, `modified_files`.
9. `verify(archive_path: str | Path) -> dict`
   - Kiểm tra toàn diện header, CRC, metadata, tính toàn vẹn của archive.

---

## V. Chuẩn Thanh Progress Bar (Console UX)

Tất cả các thao tác nén, giải nén, gộp, tách, chuyển đổi đều sử dụng chung một giao diện Progress Bar màu **Cyan / Light Blue**:

```text
dataset.zip: compressing: ████████████████░░░ 2.35GB / 5.00GB 132MB/s ETA 00:18
dataset.tar.gz: extracting: ███████████████████░ 8500 / 9000 files
```

- **Màu sắc**: Màu xanh da trời (`colour="cyan"`).
- **Trải nghiệm**: Một dòng duy nhất, cập nhật thời gian thực, tốc độ xử lý (MB/s hoặc files/s), ETA đếm ngược, tự động chuyển đổi đơn vị KB/MB/GB.
