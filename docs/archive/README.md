# `klygo.archive` Documentation

`klygo.archive` provides high-level and low-level utilities for creating, extracting, inspecting, validating, mutating, and transforming multi-format archive files.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/drive/1CYtOv1nz-lujPiQA_f50HRwEdN5FVVnE?usp=sharing)

---

## 1. Supported Formats & Capabilities

<details open>
<summary><b>Bảng so sánh 8 định dạng archive và khả năng hỗ trợ</b> <i>(Bấm để đóng/mở)</i></summary>

| Format | Extensions | Read / Extract | Create / Compress | Mutate (`add`/`remove`) | Fast Merge | Split | Dependencies |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ZIP** | `.zip` | Yes (password) | Yes (levels 0-9) | Yes | Yes | Yes | Standard Library |
| **TAR** | `.tar` | Yes | Yes | Yes (rebuild) | Yes | No | Standard Library |
| **TAR.GZ** | `.tar.gz`, `.tgz` | Yes | Yes (levels 1-9) | Yes (rebuild) | Yes | No | Standard Library |
| **TAR.XZ** | `.tar.xz`, `.txz` | Yes | Yes (levels 1-9) | Yes (rebuild) | Yes | No | Standard Library |
| **TAR.BZ2** | `.tar.bz2`, `.tbz2`| Yes | Yes (levels 1-9) | Yes (rebuild) | Yes | No | Standard Library |
| **GZip** | `.gz` (single file) | Yes | Yes (levels 1-9) | No (stream) | No | No | Standard Library |
| **7-Zip** | `.7z` | Yes (password) | Yes (include_root)| No | No | No | `klygo[py7zr]` |
| **RAR** | `.rar` | Yes (password) | No (read-only) | No | No | No | `klygo[rarfile]` |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Bảng tra cứu nhanh theo tác vụ thường gặp (22 tác vụ)</b> <i>(Bấm để đóng/mở)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Compress a folder/file** | [`compress()`](api/compress.md) | `archive.compress("dataset", "dataset.zip")` |
| **Compress folder contents ONLY (no parent folder)** | [`compress()`](api/compress.md) | `archive.compress("dataset", "data.zip", include_root=False)` |
| **Extract all files safely** | [`extract()`](api/extract.md) | `archive.extract("data.zip", output_dir="extracted", overwrite=True)` |
| **Extract one specific member without unzipping all** | [`extract_file()`](api/extract_file.md) | `archive.extract_file("data.zip", "labels/train.txt", "output")` |
| **Extract members matching wildcard / pattern** | [`extract_matching()`](api/extract_matching.md) | `archive.extract_matching("data.zip", "*.jpg", "images")` |
| **List stored member names** | [`list_files()`](api/list_files.md) | `files = archive.list_files("data.zip")` |
| **Iterate members lazily (Zero-RAM for big archives)** | [`iter_files()`](api/iter_files.md) | `for name in archive.iter_files("huge.tar.gz"): ...` |
| **Search member names by glob or regex** | [`search()`](api/search.md) | `archive.search("data.zip", r"images/.*\.png", regex=True)` |
| **Inspect archive metadata & compression ratio** | [`get_info()`](api/get_info.md) | `info = archive.get_info("data.zip")` |
| **Test archive integrity** | [`test()`](api/test.md) | `is_ok = archive.test("data.zip")` |
| **Generate full verification report** | [`verify()`](api/verify.md) | `report = archive.verify("data.zip")` |
| **Compare member names between two archives** | [`compare()`](api/compare.md) | `diff = archive.compare("v1.zip", "v2.zip")` |
| **Detect archive format from path or magic bytes** | [`detect_format()`](api/detect_format.md) | `fmt = archive.detect_format("dataset.tar.gz")  # 'tar.gz'` |
| **Check if a file is a valid supported archive** | [`is_archive()`](api/is_archive.md) | `if archive.is_archive("data.bin"): ...` |
| **Add files to an existing archive** | [`add()`](api/add.md) | `archive.add("data.zip", "new.txt", on_conflict="rename")` |
| **Remove files from an existing archive** | [`remove()`](api/remove.md) | `archive.remove("data.zip", ["old.txt"])` |
| **Merge multiple archives into one** | [`merge()`](api/merge.md) | `archive.merge(["part1.zip", "part2.zip"], "combined.zip")` |
| **Split a large archive into parts by MB** | [`split_by_size()`](api/split_by_size.md) | `parts = archive.split_by_size("large.zip", size=100, output_dir="parts")` |
| **Convert archive to another format** | [`convert()`](api/convert.md) | `archive.convert("data.zip", "data.tar.gz")` |
| **Recompress with higher compression level** | [`recompress()`](api/recompress.md) | `archive.recompress("data.zip", "compact.zip", compresslevel=9)` |
| **Copy an archive without decompressing** | [`copy()`](api/copy.md) | `archive.copy("data.zip", "backup/data.zip")` |
| **Stateful / Context Manager Object API** | [`ArchiveFile`](api/ArchiveFile.md) / [`open()`](api/open.md) | `with archive.open("data.zip") as opened: ...` |

</details>

---

## 3. Public APIs Reference (21 Functions, 1 Class)

<details>
<summary><b>Tạo & Giải nén Archive (4 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`compress(source, output_path, ...)`](api/compress.md): Create an archive from a file or directory.
- [`extract(archive_path, output_dir, ...)`](api/extract.md): Extract all or filtered members safely.
- [`extract_file(archive_path, filename, ...)`](api/extract_file.md): Streaming extraction of a single archive member.
- [`extract_matching(archive_path, pattern, ...)`](api/extract_matching.md): Extract members matching a glob pattern.

</details>

<details>
<summary><b>Khám phá, Metadata & Kiểm tra Hợp lệ (9 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`list_files(archive_path)`](api/list_files.md): Return all stored member names as a list.
- [`iter_files(archive_path)`](api/iter_files.md): Lazily yield stored member names without loading the whole list.
- [`search(archive_path, pattern, ...)`](api/search.md): Search stored member names via glob or regular expression.
- [`get_info(archive_path)`](api/get_info.md): Comprehensive metadata dictionary (file count, sizes, ratio, encrypted).
- [`test(archive_path, ...)`](api/test.md): Verify archive integrity check.
- [`verify(archive_path)`](api/verify.md): High-level verification summary report.
- [`compare(archive1, archive2)`](api/compare.md): Compare member sets between two archives.
- [`detect_format(path)`](api/detect_format.md): Detect canonical format from extension and file magic bytes.
- [`is_archive(path)`](api/is_archive.md): Non-raising check if a path is a supported archive.

</details>

<details>
<summary><b>Biến đổi & Thay đổi Archive In-Place (7 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`add(archive_path, files, ...)`](api/add.md): Add files or directories to an existing archive.
- [`remove(archive_path, files)`](api/remove.md): Remove named members from an existing archive.
- [`merge(archive_paths, output_path, ...)`](api/merge.md): Merge multiple archives into one.
- [`split_by_size(archive_path, size, ...)`](api/split_by_size.md): Split an archive into fixed-size parts in megabytes.
- [`convert(source_path, target_path, ...)`](api/convert.md): Convert an archive from one format to another.
- [`recompress(source_path, target_path, ...)`](api/recompress.md): Rebuild archive with another compression level or format.
- [`copy(source_path, target_path, ...)`](api/copy.md): Copy archive without extracting or altering data.

</details>

<details>
<summary><b>Giao diện Hướng đối tượng & Backend Subsystem (1 Class, 1 Subpackage)</b> <i>(Bấm để mở rộng)</i></summary>

- [`ArchiveFile(archive_path)`](api/ArchiveFile.md): Stateful wrapper binding an archive to its backend.
- [`open(archive_path)`](api/open.md): Convenience context manager returning an `ArchiveFile` instance.
- **Backend Subsystem (`klygo.archive.backend`)**:
  - [`ArchiveBackend`](backend/ArchiveBackend.md), [`BackendCapabilities`](backend/BackendCapabilities.md)
  - [`ZipBackend`](backend/ZipBackend.md), [`TarBackend`](backend/TarBackend.md), [`GZipBackend`](backend/GZipBackend.md)
  - [`SevenZipBackend`](backend/SevenZipBackend.md), [`RarBackend`](backend/RarBackend.md)
  - [`UnsupportedOperationError`](backend/UnsupportedOperationError.md), [`UnsupportedOptionError`](backend/UnsupportedOptionError.md)

</details>
