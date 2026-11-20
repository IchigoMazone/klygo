# `reports.partition_summary`

```python
reports.partition_summary(source)
```

Summarize image and annotation counts across dataset partitions or splits.

```python
from tempfile import TemporaryDirectory
from PIL import Image
from klygo import files, reports

with TemporaryDirectory() as tmp:
    root = files.path(tmp)
    files.mkdir(root / "images" / "train")
    files.mkdir(root / "labels" / "train")
    Image.new("RGB", (16, 16)).save(root / "images" / "train" / "a.png")
    files.save(root / "labels" / "train" / "a.txt", "0 0.5 0.5 0.2 0.2\n", overwrite=True, verbose=False)
    print(reports.partition_summary(root))
```

See [`examples/reports/partition_summary.py`](../../../examples/reports/partition_summary.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
