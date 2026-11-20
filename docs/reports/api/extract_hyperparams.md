# `reports.extract_hyperparams`

```python
reports.extract_hyperparams(run_dirs, *, ignore=('seed', 'save_dir', 'name'))
```

Extract and cross-check `args.yaml` hyperparameters across runs.

```python
from tempfile import TemporaryDirectory
from klygo import files, reports

with TemporaryDirectory() as tmp:
    r1, r2 = files.path(tmp) / "r1", files.path(tmp) / "r2"
    files.mkdir(r1); files.mkdir(r2)
    files.save(r1 / "args.yaml", {"batch": 16, "seed": 42}, overwrite=True, verbose=False)
    files.save(r2 / "args.yaml", {"batch": 16, "seed": 100}, overwrite=True, verbose=False)
    print(reports.extract_hyperparams([r1, r2])["diffs"])
```

See [`examples/reports/extract_hyperparams.py`](../../../examples/reports/extract_hyperparams.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
