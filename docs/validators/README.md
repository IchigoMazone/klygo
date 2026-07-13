# klygo.validators

Package này chứa các lớp kiểm tra tham số được các API `archive`, `datasets`, `io` và `models` sử dụng nội bộ.

API public trực tiếp là `validate_type`:

```python
from klygo.validators import validate_type

validate_type("dataset.zip", str, "source")
validate_type(0.25, (int, float), "threshold")
```

Nếu giá trị không đúng kiểu, hàm phát sinh `TypeError` kèm tên tham số.

Các validator theo package nằm trong:

```text
klygo.validators.archive
klygo.validators.datasets
klygo.validators.io
klygo.validators.models
```

Chúng phục vụ triển khai nội bộ và không phải lớp cấu hình dành cho người dùng cuối.
