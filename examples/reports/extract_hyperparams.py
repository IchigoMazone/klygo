"""Runnable example for reports.extract_hyperparams."""

from tempfile import TemporaryDirectory
from klygo import files, reports

with TemporaryDirectory() as tmp:
    r1, r2 = files.path(tmp) / "r1", files.path(tmp) / "r2"
    files.mkdir(r1); files.mkdir(r2)
    files.save(r1 / "args.yaml", {"batch": 16, "seed": 42}, overwrite=True, verbose=False)
    files.save(r2 / "args.yaml", {"batch": 16, "seed": 100}, overwrite=True, verbose=False)
    print(reports.extract_hyperparams([r1, r2])["diffs"])
