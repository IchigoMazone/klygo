# `klygo.files` Documentation

`klygo.files` provides 39 pure, deterministic, and safe filesystem operations, path inspection helpers, streaming download tools, and multi-format structured data I/O.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/drive/1-Oo8ERqSuxns1OfZAdHY5jVMTrLVpJG-?usp=sharing)

---

## 1. Supported Structured Data Formats (14 Formats)

<details open>
<summary><b>Danh sách 14 định dạng dữ liệu hỗ trợ</b> <i>(Bấm để đóng/mở)</i></summary>

| Category | File Extensions | Python Type Returned by `load()` | Notes |
| :--- | :--- | :--- | :--- |
| **JSON** | `.json` | `dict` or `list` | UTF-8 encoded |
| **JSON Lines** | `.jsonl` | `list[dict]` | Evaluated line-by-line |
| **YAML** | `.yaml`, `.yml` | `dict` or `list` | Safe YAML loader (`pyyaml`) |
| **TOML** | `.toml` | `dict` | Preserves type mapping via `tomlkit` |
| **Tabular** | `.csv` | `list[dict]` | Parsed with `csv.DictReader` |
| **Text & Log** | `.txt`, `.log` | `str` or `list[str]` | Use `as_lines=True` to strip newlines |
| **Ini-style Config** | `.ini`, `.cfg`, `.properties` | `dict[str, dict[str, str]]` | Multi-section dictionary |
| **Environment** | `.env` | `dict[str, str]` | Strips quotes and ignores comments |
| **Markup** | `.xml` | `dict` | Two-way bidirectional XML/dict mapping |
| **Binary Serialization** | `.pkl`, `.pickle` | `Any` | Standard Python pickle |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Bảng tra cứu nhanh theo tác vụ thường gặp (25 tác vụ)</b> <i>(Bấm để đóng/mở)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Load structured data by file extension** | [`load()`](api/load.md) | `data = files.load("config.json")` |
| **Read text file as lines (no `\n`)** | [`load()`](api/load.md) | `lines = files.load("labels.txt", as_lines=True)` |
| **Save data automatically by file extension** | [`save()`](api/save.md) | `files.save("data.yaml", payload, overwrite=True)` |
| **Convert between data formats directly** | [`convert()`](api/convert.md) | `files.convert("config.yaml", "config.json")` |
| **Download from URL / Colab / local with progress** | [`download()`](api/download.md) | `files.download(url, output_dir="weights", overwrite=True)` |
| **Check if path exists** | [`exists()`](api/exists.md) | `if files.exists("runs/detect"): ...` |
| **Check if regular file or directory** | [`is_file()`](api/is_file.md) / [`is_dir()`](api/is_dir.md) | `files.is_file(p)`, `files.is_dir(p)` |
| **List both files & subfolders (sorted)** | [`list_entries()`](api/list_entries.md) | `files.list_entries("dataset", pattern="*.json")` |
| **Find regular files ONLY (recursive)** | [`find()`](api/find.md) | `images = files.find("dataset", pattern="*.jpg")` |
| **Walk directory tree lazily** | [`walk()`](api/walk.md) | `for root, dirs, names in files.walk("."): ...` |
| **Create directory safely** | [`mkdir()`](api/mkdir.md) | `files.mkdir("runs/experiment_1")` |
| **Remove file or folder recursively** | [`remove()`](api/remove.md) | `files.remove("runs/temp", missing_ok=True)` |
| **Copy file or directory tree** | [`copy()`](api/copy.md) | `files.copy("source.txt", "backup/source.txt")` |
| **Move file or directory tree** | [`move()`](api/move.md) | `files.move("draft.txt", "output/draft.txt")` |
| **Rename or relocate file** | [`rename()`](api/rename.md) | `files.rename("old.txt", "new.txt")` |
| **Get file or folder size (bytes or KB/MB/GB)** | [`size()`](api/size.md) | `files.size("weights.pt", human=True)  # '14.2 MB'` |
| **Calculate streaming checksum (Zero-RAM)** | [`hash()`](api/hash.md) | `files.hash("model.onnx", algorithm="sha256")` |
| **Get comprehensive file metadata** | [`info()`](api/info.md) | `meta = files.info("model.onnx")` |
| **Compare contents of two files** | [`compare()`](api/compare.md) | `if files.compare("f1.bin", "f2.bin", by="hash"): ...` |
| **Prevent Path Traversal (Zip Slip check)** | [`is_within()`](api/is_within.md) | `files.is_within(target_path, root_dir)` |
| **Find longest common parent directory** | [`common_path()`](api/common_path.md) | `common = files.common_path(["a/b/c", "a/b/d"])` |
| **Map path to a new root directory** | [`replace_root()`](api/replace_root.md) | `files.replace_root("data/train/1.jpg", "data", "backup")` |
| **Handle compound extensions (`.tar.gz`)** | [`compound_extension()`](api/compound_extension.md) | `files.compound_extension("a.tar.gz")  # '.tar.gz'` |
| **Change filename extension cleanly** | [`with_extension()`](api/with_extension.md) | `files.with_extension("a.tar.gz", ".zip")  # 'a.zip'` |
| **Generate unique non-conflicting path** | [`unique_path()`](api/unique_path.md) | `path = files.unique_path("report.txt")  # 'report_1.txt'` |

</details>

---

## 3. Public APIs Reference (39 Functions)

<details>
<summary><b>I/O Dữ liệu cấu trúc & Tải tệp (4 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`load(path, as_lines=False, ...)`](api/load.md): Automatically load structured data based on file extension.
- [`save(path, data, overwrite=False, ...)`](api/save.md): Serialize structured data using destination file extension.
- [`convert(source, target, ...)`](api/convert.md): Format conversion between structured data files.
- [`download(source, output_dir, ...)`](api/download.md): Download URL, Colab, or local file with progress bar.

</details>

<details>
<summary><b>Thao tác Filesystem (11 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`exists(path)`](api/exists.md): Check whether a file or directory exists.
- [`is_file(path)`](api/is_file.md): Check if path identifies a regular file.
- [`is_dir(path)`](api/is_dir.md): Check if path identifies a directory.
- [`list_entries(path, pattern, recursive)`](api/list_entries.md): Deterministically list matching files and directories.
- [`find(path, pattern, recursive)`](api/find.md): Find regular files matching wildcard pattern recursively.
- [`walk(path)`](api/walk.md): Walk directory tree as a generator (`os.walk` wrapper).
- [`mkdir(path, parents, exist_ok)`](api/mkdir.md): Create a new directory and missing parent directories.
- [`copy(source, target, overwrite)`](api/copy.md): Copy a file or directory tree recursively.
- [`move(source, target, overwrite)`](api/move.md): Move a file or directory to a new target.
- [`rename(path, new_name_or_path, overwrite)`](api/rename.md): Rename or relocate a file or directory.
- [`remove(path, recursive, missing_ok)`](api/remove.md): Remove a file or directory recursively.

</details>

<details>
<summary><b>Metadata & Kiểm tra Tệp (4 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`info(path)`](api/info.md): Comprehensive metadata dictionary (sizes, timestamps, checksum).
- [`size(path, human=False)`](api/size.md): Size of file or directory tree (bytes or formatted string).
- [`hash(path, algorithm='md5')`](api/hash.md): Compute streaming checksum digest.
- [`compare(path1, path2, by='hash')`](api/compare.md): Compare two files by checksum or binary content.

</details>

<details>
<summary><b>Xử lý Đường dẫn Thuần túy (20 hàm)</b> <i>(Bấm để mở rộng)</i></summary>

- [`name(path)`](api/name.md): Final path component including extension.
- [`stem(path)`](api/stem.md): Final path component without its last extension.
- [`extension(path)`](api/extension.md): Final filename extension (including leading dot).
- [`parent(path)`](api/parent.md): Immediate parent directory of a path.
- [`path(value, expand_user=True)`](api/path.md): Convert string or path to normalized `pathlib.Path`.
- [`join(*parts)`](api/join.md): Join multiple path components together.
- [`normalize(path)`](api/normalize.md): Lexically normalize path separators and dot components.
- [`resolve(path, strict=False)`](api/resolve.md): Return an absolute normalized path.
- [`relative(path, start='.')`](api/relative.md): Compute relative path from a starting directory.
- [`is_within(path, root)`](api/is_within.md): Check whether a path is safely contained within root.
- [`common_path(paths)`](api/common_path.md): Find the longest common parent directory of multiple paths.
- [`replace_root(path, old_root, new_root)`](api/replace_root.md): Map a path from one root directory to another.
- [`with_name(path, name)`](api/with_name.md): Return path with replaced final component.
- [`with_stem(path, stem)`](api/with_stem.md): Return path with replaced stem.
- [`with_extension(path, extension)`](api/with_extension.md): Return path with replaced extension (handles `.tar.gz`).
- [`unique_path(path)`](api/unique_path.md): Generate an unused candidate path with numeric suffix.
- [`extensions(path)`](api/extensions.md): Return all filename extensions as a tuple.
- [`compound_extension(path)`](api/compound_extension.md): Return all filename extensions joined together.
- [`parents(path)`](api/parents.md): Return tuple of ancestor paths from nearest to farthest.
- [`is_absolute(path)`](api/is_absolute.md): Check whether a path is absolute.

</details>
