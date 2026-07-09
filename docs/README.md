# klygo — Documentation

Chào mừng đến với tài liệu hướng dẫn sử dụng **klygo**.

---

## Mục lục

- [klygo.archive](../docs/archive/README.md) — Nén, giải nén, và xử lý file nén
- [klygo.io](../docs/io/README.md) — Đọc/ghi file cấu hình (YAML, JSON, TOML)
- [klygo.models](../docs/models/README.md) — Nhận diện vật thể & Mô hình học sâu (Kernel)

---

## Cài đặt

```bash
pip install klygo
# hoặc nếu dùng uv
uv add klygo
```

## Cấu trúc thư mục

```
klygo/
├── archive/        # Xử lý file nén (.zip)
│   ├── compress.py
│   ├── extract.py
│   ├── list.py
│   ├── info.py
│   ├── modify.py
│   └── transform.py
├── io/             # Đọc/ghi file cấu hình
│   ├── config.py
│   ├── read.py
│   └── write.py
└── models/         # Nhận diện vật thể
    └── kernel.py
```
