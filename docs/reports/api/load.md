# `reports.load`

```python
reports.load(source, *, format='yolo', verbose=False)
```

Load training logs (`results.csv`, `args.yaml`) from a run directory.

```python
from tempfile import TemporaryDirectory
from klygo import files, reports

with TemporaryDirectory() as tmp:
    run = files.path(tmp) / "run"
    files.mkdir(run)
    files.save(run / "args.yaml", {"epochs": 2, "seed": 42}, overwrite=True, verbose=False)
    files.save(run / "results.csv", [{"epoch": 1, "metrics/mAP50-95(B)": 0.5, "time": 10.0}], overwrite=True, verbose=False)
    print(reports.load(run)["best_epoch"])
```

See [`examples/reports/load.py`](../../../examples/reports/load.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
