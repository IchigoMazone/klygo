# Klygo Documentation Hub

Chào mừng bạn đến với trung tâm tài liệu chính thức của **Klygo**. Thư viện cung cấp bộ công cụ toàn diện từ quản lý tệp tin, nén/giải nén đa định dạng, xử lý cấu hình, tiền xử lý ảnh lazy đến nhận diện vật thể Zero-shot.

---

## 📚 Điều hướng nhanh theo Module

| Module | Mô tả | Hướng dẫn & Tra cứu | Link Colab |
| :--- | :--- | :---: | :---: |
| **[`klygo.archive`](archive/api/README.md)** | Nén, giải nén, in-place mutation, split, merge 8 định dạng (ZIP, TAR, GZ, 7z, RAR) | [Xem Docs Archive](archive/api/README.md) | [Colab](https://colab.research.google.com/drive/1CYtOv1nz-lujPiQA_f50HRwEdN5FVVnE?usp=sharing) |
| **[`klygo.files`](files/api/README.md)** | 39 hàm filesystem, streaming download, hash checksum, I/O 14 định dạng dữ liệu | [Xem Docs Files](files/api/README.md) | [Colab](https://colab.research.google.com/drive/1-Oo8ERqSuxns1OfZAdHY5jVMTrLVpJG-?usp=sharing) |
| **[`klygo.config`](config/api/README.md)** | Quản lý cấu hình đa tầng, dot-notation, env overlay (`KLYGO_*`), validation, class `Config` | [Xem Docs Config](config/api/README.md) | [Colab](https://colab.research.google.com/drive/1-aOofq_ZwLi00gRXnBLbJ4raupc6OZKm?usp=sharing) |
| **[`klygo.processing`](processing.md)** | Pipeline tiền xử lý ảnh lazy đa backend, phục hồi tọa độ bounding box | [Xem Docs Processing](processing.md) | - |
| **[`klygo.utils`](utils/human_size.md)** | Tiện ích định dạng byte `human_size`, thanh tiến trình `ProgressBar` | [Xem Docs Utils](utils/human_size.md) | - |

---

## 🔍 Bảng tra cứu tác vụ tổng hợp (Master Cheatsheet)

<details>
<summary><b>1. Nén & Giải nén (Archive)</b> <i>(Bấm để mở rộng)</i></summary>

| Tác vụ | API khuyến nghị | Ví dụ |
| :--- | :--- | :--- |
| Nén thư mục/file | [`archive.compress()`](archive/api/compress.md) | `ar.compress("folder", "data.zip")` |
| Nén chỉ nội dung (bỏ folder cha) | [`archive.compress()`](archive/api/compress.md) | `ar.compress("folder", "data.zip", include_root=False)` |
| Giải nén an toàn toàn bộ | [`archive.extract()`](archive/api/extract.md) | `ar.extract("data.zip", "output", overwrite=True)` |
| Trích xuất 1 file duy nhất | [`archive.extract_file()`](archive/api/extract_file.md) | `ar.extract_file("data.zip", "sub/file.txt", "out")` |
| Giải nén theo pattern | [`archive.extract_matching()`](archive/api/extract_matching.md) | `ar.extract_matching("data.zip", "*.png", "imgs")` |
| Duyệt file tiết kiệm RAM | [`archive.iter_files()`](archive/api/iter_files.md) | `for name in ar.iter_files("data.tar.gz"): ...` |
| Thêm file vào file nén | [`archive.add()`](archive/api/add.md) | `ar.add("data.zip", "new.txt", on_conflict="rename")` |
| Xóa file trong file nén | [`archive.remove()`](archive/api/remove.md) | `ar.remove("data.zip", ["old.txt"])` |
| Cắt file nén theo MB | [`archive.split_by_size()`](archive/api/split_by_size.md) | `ar.split_by_size("data.zip", 100, "parts")` |
| Gộp nhiều file nén | [`archive.merge()`](archive/api/merge.md) | `ar.merge(["p1.zip", "p2.zip"], "merged.zip")` |
| Chuyển đổi định dạng archive | [`archive.convert()`](archive/api/convert.md) | `ar.convert("data.zip", "data.tar.gz")` |

</details>

<details>
<summary><b>2. Thao tác Filesystem & Đọc/Ghi dữ liệu (Files)</b> <i>(Bấm để mở rộng)</i></summary>

| Tác vụ | API khuyến nghị | Ví dụ |
| :--- | :--- | :--- |
| Nạp dữ liệu tự động (14 format) | [`files.load()`](files/api/load.md) | `data = files.load("data.yaml")` |
| Đọc file text dạng dòng (bỏ `\n`) | [`files.load()`](files/api/load.md) | `lines = files.load("labels.txt", as_lines=True)` |
| Lưu dữ liệu tự động theo đuôi file | [`files.save()`](files/api/save.md) | `files.save("out.json", data, overwrite=True)` |
| Chuyển đổi định dạng trực tiếp | [`files.convert()`](files/api/convert.md) | `files.convert("cfg.yaml", "cfg.json")` |
| Tải file từ URL có progress bar | [`files.download()`](files/api/download.md) | `files.download(url, output_dir="weights")` |
| Chỉ tìm regular files đệ quy | [`files.find()`](files/api/find.md) | `imgs = files.find("data", pattern="*.jpg")` |
| Liệt kê cả file & folder có sort | [`files.list_entries()`](files/api/list_entries.md) | `entries = files.list_entries("data")` |
| Xóa file/folder an toàn đệ quy | [`files.remove()`](files/api/remove.md) | `files.remove("temp", missing_ok=True)` |
| Copy / Di chuyển file hoặc folder | [`files.copy()`](files/api/copy.md) / [`files.move()`](files/api/move.md) | `files.copy(src, dst)`, `files.move(src, dst)` |
| Tính dung lượng (KB, MB, GB) | [`files.size()`](files/api/size.md) | `files.size("weights.pt", human=True)` |
| Tính checksum streaming không tốn RAM | [`files.hash()`](files/api/hash.md) | `files.hash("model.onnx", "sha256")` |
| Kiểm tra chống Path Traversal | [`files.is_within()`](files/api/is_within.md) | `files.is_within(target, root)` |
| Đổi đuôi file chuẩn (kể cả `.tar.gz`) | [`files.with_extension()`](files/api/with_extension.md) | `files.with_extension("a.tar.gz", ".zip")` |

</details>

<details>
<summary><b>3. Quản lý Cấu hình (Config)</b> <i>(Bấm để mở rộng)</i></summary>

| Tác vụ | API khuyến nghị | Ví dụ |
| :--- | :--- | :--- |
| Nạp config có tự động expand `./` | [`config.load()`](config/api/load.md) | `cfg = config.load("settings.yaml")` |
| Đọc giá trị lồng nhau bằng dot path | [`config.get()`](config/api/get.md) | `batch = config.get(cfg, "model.batch", default=16)` |
| Gán giá trị lồng nhau (pure/immutable) | [`config.set()`](config/api/set.md) | `cfg2 = config.set(cfg, "model.batch", 32)` |
| Kiểm tra key tồn tại (kể cả giá trị `None`)| [`config.has()`](config/api/has.md) | `if config.has(cfg, "model.lr"): ...` |
| Ghi đè bằng biến môi trường `KLYGO_*` | [`config.from_env()`](config/api/from_env.md) | `cfg = config.from_env(cfg, parse_values=True)` |
| Hợp nhất nhiều cấu hình đệ quy | [`config.merge()`](config/api/merge.md) | `cfg = config.merge(base, env, cli)` |
| Làm phẳng dict thành dot path | [`config.flatten()`](config/api/flatten.md) | `flat = config.flatten(cfg)` |
| Bung dot path thành dict lồng nhau | [`config.unflatten()`](config/api/unflatten.md) | `nested = config.unflatten(flat)` |
| So sánh sai khác giữa 2 config/file | [`config.diff()`](config/api/diff.md) | `diff = config.diff(c1, c2)` |
| Xác thực schema & kiểu dữ liệu | [`config.validate()`](config/api/validate.md) | `config.validate(cfg, {"model.batch": int})` |
| Giao diện OOP stateful bound-path | [`Config`](config/api/Config.md) | `mgr = Config("settings.yaml"); mgr.read()` |

</details>

---

## 💡 Quy tắc thiết kế cho AI Assistants

1. **Top-Level Imports**: Luôn dùng `from klygo import files, archive, config`.
2. **Immutability**: Functional APIs không gây side-effect lên dict/mapping đầu vào.
3. **Path-Flexible**: Mọi API đều chấp nhận cả `str` và `pathlib.Path`.
