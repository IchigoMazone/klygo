"""Behavioral contract for klygo.files.save."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import patch

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
            "data.md": ("# Report\n\n- fast\n- stable\n", "# Report\n\n- fast\n- stable\n"),
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
                for atomic in (False, True):
                    with self.subTest(filename=filename, atomic=atomic):
                        target = root / str(atomic) / filename
                        files.save(target, value, verbose=False, atomic=atomic)
                        self.assertEqual(files.load(target, verbose=False), expected)

    def test_atomic_failure_preserves_destination_and_cleans_temporary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "data.json"
            target.write_text('{"old": true}', encoding="utf-8")
            with self.assertRaises(TypeError):
                files.save(target, {"first": 1, "bad": object()}, overwrite=True, verbose=False, atomic=True)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}')
            self.assertEqual(list(root.iterdir()), [target])
            target.unlink()
            with self.assertRaises(TypeError):
                files.save(target, {"bad": object()}, verbose=False, atomic=True)
            self.assertEqual(list(root.iterdir()), [])

    def test_default_save_preserves_existing_file_on_serialization_failure(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "data.json"
            target.write_text('{"old": true}', encoding="utf-8")
            with self.assertRaises(TypeError):
                files.save(target, {"bad": object()}, overwrite=True, verbose=False)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}')
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_atomic_overwrite_and_commit_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "data.json"
            files.save(target, {"v": 1}, atomic=True, verbose=False)
            with self.assertRaises(FileExistsError):
                files.save(target, {"v": 2}, atomic=True, verbose=False)
            files.save(target, {"v": 2}, overwrite=True, atomic=True, verbose=False)
            with patch("klygo.files.io.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    files.save(target, {"v": 3}, overwrite=True, atomic=True, verbose=False)
            self.assertEqual(files.load(target, verbose=False), {"v": 2})
            self.assertEqual(list(root.iterdir()), [target])

    def test_atomic_no_overwrite_preserves_concurrently_created_destination(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "data.json"
            link = os.link

            def create_competing_file(source, destination):
                Path(destination).write_text("concurrent writer", encoding="utf-8")
                link(source, destination)

            with patch("klygo.files.io.os.link", side_effect=create_competing_file):
                with self.assertRaises(FileExistsError):
                    files.save(target, {"v": 1}, atomic=True, verbose=False)
            self.assertEqual(target.read_text(encoding="utf-8"), "concurrent writer")
            self.assertEqual(list(root.iterdir()), [target])

    def test_atomic_hard_link_failure_does_not_publish_partial_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "data.json"
            with patch("klygo.files.io.os.link", side_effect=OSError("unsupported")):
                with self.assertRaises(OSError):
                    files.save(target, {"v": 1}, atomic=True, verbose=False)
            self.assertEqual(list(root.iterdir()), [])

    def test_symlink_policy_and_atomic_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text('{"old": true}', encoding="utf-8")
            link = root / "link.json"
            with self.assertRaises(TypeError):
                files.save(target, {}, atomic=1, verbose=False)
            try:
                link.symlink_to(root / "missing")
            except OSError as error:
                if getattr(error, "winerror", None) == 1314 or error.errno in (1, 13, 95):
                    self.skipTest(f"Symlinks unavailable: {error}")
                raise
            for atomic in (False, True):
                with self.assertRaises(FileExistsError):
                    files.save(link, {}, verbose=False, atomic=atomic)
            link.unlink()
            link.symlink_to(target)
            files.save(link, {"new": True}, overwrite=True, atomic=True, verbose=False)
            self.assertFalse(link.is_symlink())
            self.assertEqual(files.load(target, verbose=False), {"old": True})
            self.assertEqual(files.load(link, verbose=False), {"new": True})


if __name__ == "__main__":
    unittest.main()
