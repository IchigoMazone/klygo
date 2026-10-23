"""Behavioral contract for klygo.files.download."""

import unittest
import hashlib
import importlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from klygo import files


download_module = importlib.import_module("klygo.files.download")


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

    def test_incomplete_download_preserves_existing_file(self):
        class Response(BytesIO):
            headers = {"content-length": "10"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.bin"
            target.write_bytes(b"original")
            with patch("urllib.request.urlopen", return_value=Response(b"short")):
                with self.assertRaises(OSError):
                    files.download("https://example.test/model.bin", root, overwrite=True, verbose=False)
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(root.iterdir()), [target])

    def test_sha256_and_timeout(self):
        class Response(BytesIO):
            headers = {"content-length": "6"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        expected = hashlib.sha256(b"remote").hexdigest()
        with TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", return_value=Response(b"remote")) as opener:
                result = files.download(
                    "https://example.test/model.bin", directory, verbose=False,
                    timeout=3, expected_sha256=expected,
                )
            self.assertEqual(result.read_bytes(), b"remote")
            self.assertEqual(opener.call_args.kwargs["timeout"], 3)
            with self.assertRaises(ValueError):
                files.download("https://example.test/model.bin", directory, timeout=0)

    def test_checksum_mismatch_and_retry_leave_no_partial_destination(self):
        class Response(BytesIO):
            headers = {"content-length": "6"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.bin"
            target.write_bytes(b"previous")
            with patch("urllib.request.urlopen", return_value=Response(b"remote")):
                with self.assertRaisesRegex(ValueError, "expected_sha256"):
                    files.download(
                        "https://example.test/model.bin", root,
                        overwrite=True, verbose=False, expected_sha256="0" * 64,
                    )
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(list(root.iterdir()), [target])

            with patch("urllib.request.urlopen", side_effect=[OSError("offline"), Response(b"remote")]) as opener:
                with patch.object(download_module.time, "sleep"):
                    files.download(
                        "https://example.test/model.bin", root,
                        overwrite=True, verbose=False, retries=1,
                    )
            self.assertEqual(opener.call_count, 2)
            self.assertEqual(target.read_bytes(), b"remote")


if __name__ == "__main__":
    unittest.main()
