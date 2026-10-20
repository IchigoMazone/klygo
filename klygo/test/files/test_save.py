"""Behavioral contract for klygo.files.save."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestSave(unittest.TestCase):
    def test_save_and_overwrite(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "data.json"
            files.save(target, {"a": 1}, verbose=False)
            self.assertEqual(files.load(target, verbose=False), {"a": 1})
            with self.assertRaises(FileExistsError):
                files.save(target, {"a": 2}, verbose=False)
            files.save(target, {"a": 2}, overwrite=True, verbose=False)
            self.assertEqual(files.load(target, verbose=False), {"a": 2})

    def test_every_supported_extension_round_trips(self):
        structured = {"section": {"value": "1"}}
        cases = {
            "data.yaml": ({"a": 1}, {"a": 1}),
            "data.yml": ({"a": 1}, {"a": 1}),
            "data.json": ({"a": 1}, {"a": 1}),
            "data.jsonl": ([{"a": 1}, {"a": 2}], [{"a": 1}, {"a": 2}]),
            "data.toml": (structured, structured),
            "data.csv": ([{"name": "cat", "count": "2"}], [{"name": "cat", "count": "2"}]),
            "data.txt": (["a", "b"], "a\nb\n"),
            "data.log": ("message", "message"),
            "data.ini": (structured, structured),
            "data.cfg": (structured, structured),
            "data.properties": (structured, structured),
            ".env": ({"PORT": "8080"}, {"PORT": "8080"}),
            "data.xml": ({"root": {"value": "1"}}, {"root": {"value": "1"}}),
            "data.pkl": ({"a": 1}, {"a": 1}),
            "data.pickle": ({"a": 1}, {"a": 1}),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, (value, expected) in cases.items():
                with self.subTest(filename=filename):
                    target = root / filename
                    files.save(target, value, verbose=False)
                    self.assertEqual(files.load(target, verbose=False), expected)


if __name__ == "__main__":
    unittest.main()
