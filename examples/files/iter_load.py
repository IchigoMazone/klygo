"""Executable example for klygo.files.iter_load."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "records.jsonl"
    files.save(source, [{"id": 1}, {"id": 2}], verbose=False)
    assert list(files.iter_load(source)) == [{"id": 1}, {"id": 2}]

    # Closing the iterator releases the stream when processing stops early.
    with closing(files.iter_load(source)) as records:
        first = next(records)
        assert first == {"id": 1}
        print(first)

    table = Path(directory) / "records.csv"
    files.save(table, [{"id": "1", "name": "cat"}], verbose=False)
    assert list(files.iter_load(table)) == [{"id": "1", "name": "cat"}]
