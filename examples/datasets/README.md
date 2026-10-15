# `klygo.datasets` examples

The examples are self-contained and create temporary datasets, so they do not
modify the repository. Run any file from the repository root:

```bash
python examples/datasets/partition.py
python examples/datasets/export.py
```

Each public function has one matching example. Shared sample-data creation is
kept in `_support.py` and is not part of the Klygo public API.
