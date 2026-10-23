"""Behavioral contract for klygo.files.iter_load."""

import json
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from klygo import files


class TestIterLoad(unittest.TestCase):
    def test_jsonl_values_blank_lines_and_incremental_errors(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "records.jsonl"
            source.write_text('\n{"label": "caf\u00e9"}\n\n42\ninvalid\n', encoding="utf-8")
            with closing(files.iter_load(source)) as rows:
                self.assertEqual(next(rows), {"label": "caf\u00e9"})
                self.assertEqual(next(rows), 42)
                with self.assertRaises(json.JSONDecodeError):
                    next(rows)
                self.assertIsNone(rows.gi_frame)

    def test_csv_quotes_multiline_and_empty_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.csv"
            source.write_bytes(b'name,note\r\ncat,"line 1\nline 2"\r\n')
            self.assertEqual(list(files.iter_load(source)), [{"name": "cat", "note": "line 1\nline 2"}])
            for suffix in (".csv", ".jsonl"):
                empty = root / ("empty" + suffix)
                empty.touch()
                self.assertEqual(list(files.iter_load(empty)), [])

    def test_file_opens_lazily_and_closes_on_early_exit(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "records.jsonl"
            source.write_text('1\n2\n', encoding="utf-8")
            opened = []
            original_open = Path.open

            def track(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                opened.append(stream)
                return stream

            with patch.object(Path, "open", track):
                rows = files.iter_load(source)
                self.assertEqual(opened, [])
                with closing(rows):
                    self.assertEqual(next(rows), 1)
                    self.assertFalse(opened[0].closed)
                self.assertTrue(opened[0].closed)

    def test_errors_are_validated_before_iteration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                files.iter_load(root / "missing.jsonl")
            with self.assertRaises(ValueError):
                files.iter_load(root)
            source = root / "data.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                files.iter_load(source)
            with self.assertRaises(TypeError):
                files.iter_load(123)


if __name__ == "__main__":
    unittest.main()
