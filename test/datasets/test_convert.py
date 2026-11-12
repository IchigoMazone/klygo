"""Tests for datasets.convert."""

import tempfile
import unittest
import warnings

from PIL import Image

from klygo import config, datasets, files, media


class ConvertDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = files.path(self.temporary.name)
        self.source = files.join(self.root, "source")
        files.mkdir(self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def sample(self, batch: str, stem: str, class_id: int = 0, *, labeled: bool = True,
               extension: str = ".jpg") -> None:
        image_dir = files.join(self.source, batch, "images")
        label_dir = files.join(self.source, batch, "labels")
        files.mkdir(image_dir)
        files.mkdir(label_dir)
        media.save(files.join(image_dir, stem + extension), Image.new("RGB", (20, 10), "white"),
                   overwrite=True, verbose=False)
        if labeled:
            files.save(files.join(label_dir, stem + ".txt"),
                       [f"{class_id} 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)

    def test_glob_conversion_and_partition_integration(self) -> None:
        self.sample("batch_a", "one")
        self.sample("batch_b", "two", labeled=False)
        target = files.join(self.root, "flat")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = datasets.convert(
                self.source, target,
                tree={"pictures": "batch_*/images", "annotations": "batch_*/labels"},
                classes={0: "item"},
            )
        self.assertEqual(output, files.resolve(target))
        self.assertTrue(files.is_file(files.join(target, "images", "one.jpg")))
        self.assertTrue(files.is_file(files.join(target, "images", "two.jpg")))
        self.assertFalse(files.exists(files.join(target, "labels", "two.txt")))
        self.assertTrue(any("negative samples" in str(item.message) for item in caught))
        metadata = config.load(files.join(target, "data.yaml"), verbose=False).to_dict()
        self.assertEqual((metadata["path"], metadata["train"], metadata["nc"]), (".", "images", 1))

        partitioned = datasets.partition(target, files.join(self.root, "partitioned"),
                                         (0.5, 0.5, 0.0), verbose=False)
        split_metadata = config.load(files.join(partitioned, "data.yaml"), verbose=False).to_dict()
        self.assertEqual((split_metadata["train"], split_metadata["val"]),
                         ("images/train", "images/val"))
        self.assertEqual(split_metadata["path"], ".")
        split_images = files.find(files.join(partitioned, "images"), "*", recursive=True)
        self.assertEqual(len(split_images), 2)

    def test_yaml_metadata_and_class_validation(self) -> None:
        self.sample("batch", "sample", class_id=1)
        config.save(files.join(self.source, "classes.yaml"),
                    {"names": {0: "apple", 1: "banana"}, "nc": 2}, verbose=False)
        output = datasets.convert(
            self.source, files.join(self.root, "flat"),
            tree={"a": "batch/images", "b": "batch/labels", "c": "classes.yaml"},
        )
        self.assertEqual(config.load(files.join(output, "data.yaml"), verbose=False).names,
                         {0: "apple", 1: "banana"})
        with self.assertRaisesRegex(ValueError, "together"):
            datasets.convert(
                self.source, files.join(self.root, "invalid"),
                tree={"a": "batch/images", "b": "batch/labels", "c": "classes.yaml"},
                classes={0: "apple"},
            )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            datasets.convert(
                self.source, files.join(self.root, "invalid_ids"),
                tree={"a": "batch/images", "b": "batch/labels"}, classes={1: "banana"},
            )

    def test_all_conflict_policies_are_sample_atomic(self) -> None:
        self.sample("first", "same", extension=".jpg")
        self.sample("second", "same", extension=".png")
        tree = {"a": ["first/images", "second/images"],
                "b": ["first/labels", "second/labels"]}
        with self.assertRaisesRegex(ValueError, "conflict"):
            datasets.convert(self.source, files.join(self.root, "error"), tree=tree, classes={0: "item"})

        renamed = datasets.convert(self.source, files.join(self.root, "rename"), tree=tree,
                                   classes={0: "item"}, on_conflict="rename")
        self.assertEqual(sorted(files.name(path) for path in files.find(files.join(renamed, "images"))),
                         ["same.jpg", "same_1.png"])
        self.assertEqual(sorted(files.name(path) for path in files.find(files.join(renamed, "labels"))),
                         ["same.txt", "same_1.txt"])

        skipped = datasets.convert(self.source, files.join(self.root, "skip"), tree=tree,
                                   classes={0: "item"}, on_conflict="skip")
        self.assertTrue(files.is_file(files.join(skipped, "images", "same.jpg")))
        overwritten = datasets.convert(self.source, files.join(self.root, "overwrite"), tree=tree,
                                       classes={0: "item"}, on_conflict="overwrite")
        self.assertTrue(files.is_file(files.join(overwritten, "images", "same.png")))

    def test_overwrite_replaces_the_whole_target(self) -> None:
        self.sample("batch", "old")
        target = datasets.convert(self.source, files.join(self.root, "flat"),
                                  tree={"a": "batch/images", "b": "batch/labels"}, classes={0: "item"})
        files.save(files.join(target, "stale.txt"), "stale", verbose=False)
        with self.assertRaises(FileExistsError):
            datasets.convert(self.source, target,
                             tree={"a": "batch/images", "b": "batch/labels"}, classes={0: "item"})
        datasets.convert(self.source, target,
                         tree={"a": "batch/images", "b": "batch/labels"},
                         classes={0: "item"}, overwrite=True)
        self.assertFalse(files.exists(files.join(target, "stale.txt")))

        files.save(files.join(self.source, "batch", "labels", "old.txt"),
                   ["9 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
        with self.assertRaisesRegex(ValueError, "unknown class ID"):
            datasets.convert(self.source, target,
                             tree={"a": "batch/images", "b": "batch/labels"},
                             classes={0: "item"}, overwrite=True)
        self.assertTrue(files.is_file(files.join(target, "images", "old.jpg")))

    def test_missing_metadata_or_inputs_create_a_valid_empty_layout(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = datasets.convert(
                self.source, files.join(self.root, "empty"),
                tree={"a": "missing/images", "b": "missing/labels"},
            )
        self.assertTrue(files.is_dir(files.join(output, "images")))
        self.assertTrue(files.is_dir(files.join(output, "labels")))
        self.assertEqual(config.load(files.join(output, "data.yaml"), verbose=False).names, {})
        self.assertGreaterEqual(len(caught), 4)

        files.mkdir(files.join(self.source, "labels_only"))
        files.save(files.join(self.source, "labels_only", "orphan.txt"),
                   ["0 0.5 0.5 0.5 0.5"], verbose=False)
        with warnings.catch_warnings(record=True) as labels_only_warnings:
            warnings.simplefilter("always")
            labels_only = datasets.convert(
                self.source, files.join(self.root, "labels_only_output"),
                tree={"a": "missing/images", "b": "labels_only"}, classes={0: "item"},
            )
        self.assertTrue(files.is_dir(files.join(labels_only, "images")))
        self.assertTrue(any("without matching images" in str(item.message)
                            for item in labels_only_warnings))

    def test_rejects_unknown_label_ids_and_overlapping_paths(self) -> None:
        self.sample("batch", "bad", class_id=3)
        with self.assertRaisesRegex(ValueError, "unknown class ID"):
            datasets.convert(self.source, files.join(self.root, "bad"),
                             tree={"a": "batch/images", "b": "batch/labels"}, classes={0: "item"})
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            datasets.convert(self.source, files.join(self.source, "output"),
                             tree={"a": "batch/images", "b": "batch/labels"}, classes={0: "item"})


if __name__ == "__main__":
    unittest.main()
