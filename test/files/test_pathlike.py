"""Text-returning PathLike inputs work across the public files APIs."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files
from klygo.files import operations


class TextPath:
    def __init__(self, value):
        self.value = str(value)

    def __fspath__(self):
        return self.value

    def __str__(self):
        return "not-the-filesystem-path"


class BytesPath:
    def __fspath__(self):
        return b"data.json"


class TestPathLike(unittest.TestCase):
    def test_path_manipulation_and_native_return_types(self):
        value = TextPath("dataset/archive.tar.gz")
        expected = Path("dataset/archive.tar.gz")
        self.assertEqual(files.path(value), expected)
        self.assertIsInstance(files.path(value), Path)
        for name in ("as_posix", "name", "stem", "extension", "extensions", "compound_extension", "parent", "parents", "normalize", "resolve", "is_absolute"):
            with self.subTest(name=name):
                self.assertEqual(getattr(files, name)(value), getattr(files, name)(expected))
        self.assertEqual(files.join(TextPath("dataset"), TextPath("a.txt")), Path("dataset/a.txt"))
        self.assertEqual(files.relative(value, TextPath("dataset")), Path("archive.tar.gz"))
        self.assertTrue(files.is_within(value, TextPath("dataset")))
        self.assertEqual(files.common_path([value, TextPath("dataset/a.txt")]), Path("dataset"))
        with self.assertRaises(TypeError):
            files.common_path(value)
        self.assertEqual(files.replace_root(value, TextPath("dataset"), TextPath("out")), Path("out/archive.tar.gz"))
        self.assertEqual(files.with_extension(value, "zip"), Path("dataset/archive.zip"))
        self.assertEqual(files.with_name(value, "new.txt"), Path("dataset/new.txt"))
        self.assertEqual(files.with_stem(value, "new"), Path("dataset/new.gz"))
        self.assertEqual(files.path(TextPath("~")), Path.home())
        self.assertEqual(files.path(TextPath("~"), expand_user=False), Path("~"))

    def test_io_metadata_traversal_and_mutations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.jsonl"
            wrapped = TextPath(source)
            files.save(wrapped, [{"v": 1}], atomic=True, verbose=False)
            self.assertEqual(files.load(wrapped, verbose=False), [{"v": 1}])
            self.assertEqual(list(files.iter_load(wrapped)), [{"v": 1}])
            for name in ("exists", "is_file", "size", "hash", "info", "unique_path"):
                with self.subTest(name=name):
                    self.assertEqual(getattr(files, name)(wrapped), getattr(files, name)(source))
            self.assertTrue(files.is_dir(TextPath(root)))
            self.assertFalse(files.is_symlink(TextPath(root)))
            self.assertEqual(files.list_entries(TextPath(root)), [source])
            self.assertEqual(files.find(TextPath(root)), [source])
            self.assertEqual(list(files.iter_find(TextPath(root))), [source])
            self.assertEqual(list(files.walk(TextPath(root)))[0][2], [source.name])
            folder = files.mkdir(TextPath(root / "sub"))
            self.assertIsInstance(folder, Path)
            copied = files.copy(wrapped, TextPath(folder / "copy.jsonl"))
            self.assertTrue(files.compare(wrapped, TextPath(copied)))
            moved = files.move(TextPath(copied), TextPath(root / "moved.jsonl"))
            renamed = files.rename(TextPath(moved), TextPath("renamed.jsonl"))
            downloaded = files.download(wrapped, TextPath(root / "downloads"), verbose=False)
            self.assertEqual(downloaded.read_bytes(), source.read_bytes())
            converted = files.convert(wrapped, TextPath(root / "data.json"), verbose=False)
            self.assertIsInstance(converted, Path)
            files.remove(TextPath(renamed))
            self.assertFalse(renamed.exists())

    def test_bytes_inputs_are_rejected(self):
        for value in (b"data.json", BytesPath()):
            for call in (files.path, files.exists, files.walk, files.load, files.iter_load, files.iter_find):
                with self.subTest(call=call.__name__, value=type(value).__name__), self.assertRaises(TypeError):
                    call(value)
            with self.assertRaises(TypeError):
                files.save(value, {}, verbose=False)

    def test_legacy_namespace_has_matching_exports(self):
        self.assertEqual(set(files.__all__), set(operations.__all__))
        for name in files.__all__:
            self.assertIs(getattr(files, name), getattr(operations, name))


if __name__ == "__main__":
    unittest.main()
