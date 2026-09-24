"""Behavioral contract for klygo.files.download."""

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from klygo import files


class TestDownload(unittest.TestCase):
    def test_local_download(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"klygo")
            result = files.download(source, root / "downloads", verbose=False)
            self.assertEqual(result.name, source.name)
            self.assertEqual(result.read_bytes(), b"klygo")

    def test_rejects_directory(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                files.download(directory, Path(directory) / "out", verbose=False)

    def test_remote_download_and_overwrite_policy(self):
        class Response(BytesIO):
            headers = {"content-length": "6"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", return_value=Response(b"remote")):
                result = files.download(
                    "https://example.test/model.bin",
                    directory,
                    verbose=False,
                )
            self.assertEqual(result.read_bytes(), b"remote")
            with self.assertRaises(FileExistsError):
                files.download(
                    "https://example.test/model.bin",
                    directory,
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()
