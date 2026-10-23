"""Executable example for klygo.files.unique_path."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    existing = files.path(directory) / "result.json"
    files.save(existing, {"ok": True}, verbose=False)
    print(files.unique_path(existing))


# Additional cases
# Compound extensions remain intact when numbering is needed.
with TemporaryDirectory() as directory:
    existing = files.path(directory) / "dataset.tar.gz"
    existing.touch()
    assert files.name(files.unique_path(existing)) == "dataset_1.tar.gz"
    assert files.name(files.unique_path(existing, separator="-", start=4)) == "dataset-4.tar.gz"
