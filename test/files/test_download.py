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
            self.assertIsNone(result)
            downloaded = root / "downloads" / source.name
            self.assertTrue(downloaded.is_file())
            self.assertEqual(downloaded.read_bytes(), b"klygo")

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
            self.assertIsNone(result)
            self.assertEqual((Path(directory) / "model.bin").read_bytes(), b"remote")
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
            self.assertIsNone(result)
            self.assertEqual((Path(directory) / "model.bin").read_bytes(), b"remote")
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

    def test_google_drive_view_url_and_content_disposition(self):
        class Response(BytesIO):
            def __init__(self, data, headers):
                super().__init__(data)
                self.headers = headers

            def geturl(self):
                return "https://drive.usercontent.google.com/download?id=abc123XYZ"

        resp = Response(
            b"weights-bytes",
            {
                "content-length": "13",
                "content-type": "application/octet-stream",
                "content-disposition": 'attachment; filename="yolov8n.pt"',
            },
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("urllib.request.urlopen", return_value=resp) as opener:
                files.download(
                    "https://drive.google.com/file/d/abc123XYZ/view?usp=sharing",
                    root,
                    verbose=False,
                )
            called_url = opener.call_args.args[0].full_url
            self.assertEqual(called_url, "https://drive.google.com/uc?export=download&id=abc123XYZ&confirm=t")
            saved = root / "yolov8n.pt"
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.read_bytes(), b"weights-bytes")
            self.assertFalse((root / "view").exists())

    def test_google_drive_virus_scan_confirm_form_and_html_rejection(self):
        class Response(BytesIO):
            def __init__(self, data, headers):
                super().__init__(data)
                self.headers = headers

        html_confirm = (
            b'<form id="download-form" action="https://drive.usercontent.google.com/download">'
            b'<input type="hidden" name="id" value="abc123">'
            b'<input type="hidden" name="export" value="download">'
            b'<input type="hidden" name="confirm" value="t">'
            b'<input type="hidden" name="uuid" value="u999">'
            b"</form>"
        )
        first = Response(html_confirm, {"content-type": "text/html; charset=utf-8"})
        second = Response(
            b"large-bin",
            {
                "content-length": "9",
                "content-type": "application/octet-stream",
                "content-disposition": "attachment; filename*=UTF-8''model%20v2.pt",
            },
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("urllib.request.urlopen", side_effect=[first, second]) as opener:
                files.download("https://drive.google.com/open?id=abc123", root, verbose=False)
            self.assertEqual(opener.call_count, 2)
            confirm_url = opener.call_args_list[1].args[0].full_url
            self.assertIn("drive.usercontent.google.com/download", confirm_url)
            self.assertIn("uuid=u999", confirm_url)
            self.assertEqual((root / "model v2.pt").read_bytes(), b"large-bin")

            html_denied = Response(b"<html>Sign in</html>", {"content-type": "text/html"})
            with patch("urllib.request.urlopen", return_value=html_denied):
                with self.assertRaisesRegex(OSError, "HTML page"):
                    files.download("https://drive.google.com/file/d/private123/view", root, verbose=False)


if __name__ == "__main__":
    unittest.main()
