"""Directory and ZIP destination contracts for split and merge."""
import unittest
from tempfile import TemporaryDirectory

from klygo import config, datasets, files
from test.datasets.test_api import make_dataset


class TestOutputDestinations(unittest.TestCase):
    def test_ratio_patterns_preserve_samples_and_labels(self):
        cases = (
            ("default", ("split_part_0", "split_part_1")),
            ("indexed/*", ("0", "1")),
            ("custom/batch_*", ("batch_0", "batch_1")),
            ("suffix/*_data", ("0_data", "1_data")),
            ("zipped/*.zip", ("0.zip", "1.zip")),
            ("named_zip/split_part_*.ZIP", ("split_part_0.ZIP", "split_part_1.ZIP")),
        )
        with TemporaryDirectory() as directory:
            root = files.path(directory)
            source = make_dataset(root / "source")
            expected = None
            for pattern, names in cases:
                with self.subTest(pattern=pattern):
                    target = root / pattern
                    self.assertIsNone(datasets.split(source, target, ratios=(0.5, 0.5), seed=17, verbose=False))
                    parent = files.parent(target) if "*" in pattern else target
                    self.assertEqual([files.name(p) for p in files.list_entries(parent)], list(names))
                    signatures = []
                    for name in names:
                        part = parent / name
                        data = datasets.load(part, verbose=False)
                        signatures.append([(f.source_id, f.labels) for f in data])
                        self.assertEqual(len(data), 2)
                    if expected is None:
                        expected = signatures
                    else:
                        self.assertEqual(signatures, expected)

    def test_class_and_group_patterns(self):
        with TemporaryDirectory() as directory:
            root = files.path(directory)
            source = make_dataset(root / "source")
            datasets.split(source, root / "classes", by_class=True, verbose=False)
            self.assertEqual([files.name(p) for p in files.list_entries(root / "classes")], ["split_apple", "split_banana"])
            datasets.split(source, root / "zips" / "class_*.zip", by_class=True, verbose=False)
            for index, label in enumerate(("apple", "banana")):
                part = root / "zips" / f"class_{index}.zip"
                self.assertEqual(datasets.info(part)["classes"], [label])
                self.assertEqual(datasets.info(part)["total_images"], 2)
            datasets.split(source, root / "groups" / "*_data", class_groups=[["apple", "banana"], ["banana"]], verbose=False)
            self.assertEqual(datasets.info(root / "groups" / "0_data")["total_images"], 4)
            self.assertEqual(datasets.info(root / "groups" / "1_data")["classes"], ["banana"])

    def test_pattern_validation_and_overwrite(self):
        with TemporaryDirectory() as directory:
            root = files.path(directory)
            source = make_dataset(root / "source")
            for pattern in ("bad/**", "bad/part_*_*", "bad/*/part", "bad/*/part_*.zip"):
                with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                    datasets.split(source, root / pattern, ratios=(0.5, 0.5), verbose=False)
            for pattern in ("folders/part_*", "archives/part_*.zip"):
                target = root / pattern
                datasets.split(source, target, ratios=(0.5, 0.5), verbose=False)
                with self.assertRaises(FileExistsError):
                    datasets.split(source, target, ratios=(0.5, 0.5), verbose=False)
                self.assertIsNone(datasets.split(source, target, ratios=(0.5, 0.5), overwrite=True, verbose=False))

    def test_merge_directory_and_zip_reconcile_class_ids(self):
        with TemporaryDirectory() as directory:
            root = files.path(directory)
            first = make_dataset(root / "first", ("apple", "banana"), "first")
            second = make_dataset(root / "second", ("banana", "orange"), "second")
            for name in ("merged", "merged.zip"):
                with self.subTest(name=name):
                    output = root / name
                    self.assertIsNone(datasets.merge([first, second], output, verbose=False))
                    self.assertEqual(datasets.info(output)["classes"], ["apple", "banana", "orange"])
                    data = datasets.load(output, verbose=False)
                    self.assertEqual(len(data), 8)
                    for frame in data:
                        if frame.source_id.startswith("src1_"):
                            for box in frame:
                                self.assertEqual(box.class_id, {"banana": 1, "orange": 2}[box.label])
                    with self.assertRaises(FileExistsError):
                        datasets.merge([first, second], output, verbose=False)
                    self.assertIsNone(datasets.merge([first, second], output, overwrite=True, verbose=False))
            self.assertEqual(config.load(root / "merged" / "data.yaml", verbose=False).path, ".")

    def test_merge_rejects_overlapping_outputs_without_touching_sources(self):
        with TemporaryDirectory() as directory:
            root = files.path(directory)
            first = make_dataset(root / "first")
            second = make_dataset(root / "second")
            for output in (first, first / "nested", root):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlap"):
                    datasets.merge([first, second], output, overwrite=True, verbose=False)
            self.assertEqual(datasets.info(first)["total_images"], 4)
            self.assertEqual(datasets.info(second)["total_images"], 4)
