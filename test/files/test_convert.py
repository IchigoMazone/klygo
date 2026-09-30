"""Behavioral contract for klygo.files.convert."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestConvert(unittest.TestCase):
    def test_convert_json_to_yaml(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "data.json", root / "data.yaml"
            files.save(source, {"items": [1, 2]}, verbose=False)
            result = files.convert(source, target, verbose=False)
            self.assertEqual(result, target)
            self.assertEqual(files.load(target, verbose=False), {"items": [1, 2]})

    def test_convert_yaml_to_json_with_string_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "settings.yaml", root / "settings.json"
            data = {"enabled": True, "retries": 3}
            files.save(source, data, verbose=False)
            self.assertEqual(files.convert(str(source), target, verbose=False), target)
            self.assertEqual(files.load(target, verbose=False), data)

    def test_existing_target_requires_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.json", root / "target.yaml"
            files.save(source, {"version": 2}, verbose=False)
            files.save(target, {"version": 1}, verbose=False)
            with self.assertRaises(FileExistsError):
                files.convert(source, target, verbose=False)
            files.convert(source, target, overwrite=True, verbose=False)
            self.assertEqual(files.load(target, verbose=False), {"version": 2})

    def test_missing_source_and_unsupported_target_raise_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                files.convert(root / "missing.json", root / "target.yaml", verbose=False)
            source = root / "source.json"
            files.save(source, {"value": 1}, verbose=False)
            with self.assertRaises(ValueError):
                files.convert(source, root / "target.bin", verbose=False)


if __name__ == "__main__":
    unittest.main()

