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


if __name__ == "__main__":
    unittest.main()

