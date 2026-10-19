"""Executable example for klygo.files.save."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    output = files.path(directory) / "config.yaml"
    files.save(output, {"batch": 16}, overwrite=True, verbose=False)
    print(files.load(output, verbose=False))


# Additional cases
# The extension selects the serializer; parent folders are created.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    files.save(root / "nested" / "records.jsonl", [{"id": 1}, {"id": 2}], verbose=False)
    assert files.load(root / "nested" / "records.jsonl", verbose=False) == [{"id": 1}, {"id": 2}]

# Publish a complete replacement only after serialization succeeds.
with TemporaryDirectory() as directory:
    output = files.path(directory) / "config.json"
    files.save(output, {"version": 1}, atomic=True, verbose=False)
    try:
        files.save(output, {"invalid": object()}, overwrite=True, atomic=True, verbose=False)
    except TypeError:
        pass
    assert files.load(output, verbose=False) == {"version": 1}
