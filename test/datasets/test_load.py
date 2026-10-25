"""Behavioral coverage for flattening and loading complete YOLO datasets."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from klygo import archive, config, datasets, files, media
from klygo.outputs.detect import Detections
from test.datasets.test_api import TextPath, make_dataset


class TestLoad(unittest.TestCase):
    def test_format_is_explicit_and_rejects_unsupported_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            flat = datasets.unpartition(source, root / "flat", verbose=False)
            self.assertEqual(len(datasets.load(flat, format="YOLO", verbose=False)), 4)
            with self.assertRaisesRegex(ValueError, "unsupported dataset format"):
                datasets.load(flat, format="coco", verbose=False)
            with self.assertRaisesRegex(TypeError, "format must be a string"):
                datasets.load(flat, format=None, verbose=False)

    def test_load_flattens_all_splits_and_preserves_duplicate_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            for index, split in enumerate(("train", "val", "test")):
                media.save(source / "images" / split / "same.png", Image.new("RGB", (40, 20)), verbose=False)
                files.save(
                    source / "labels" / split / "same.txt",
                    [f"{index % 2} 0.5 0.5 0.5 0.5"],
                    verbose=False,
                )
            with self.assertRaisesRegex(ValueError, "unpartition"):
                datasets.load(source, verbose=False)
            flat = datasets.unpartition(source, root / "flat", verbose=False)
            results = datasets.load(flat, verbose=False)
            self.assertIsInstance(results, Detections)
            self.assertEqual(results.metadata["source_type"], "images")
            self.assertEqual(len(results), 7)
            duplicates = [frame for frame in results if frame.metadata["relative_path"].endswith("same.png")]
            self.assertEqual(len(duplicates), 3)
            self.assertEqual({frame.metadata["split"] for frame in duplicates}, {"train", "val", "test"})
            self.assertEqual(len({frame.metadata["relative_path"] for frame in results}), 7)
            for frame in duplicates:
                self.assertEqual(frame.boxes, [(10.0, 5.0, 30.0, 15.0)])
                self.assertEqual(frame.scores, [1.0])
                self.assertTrue(frame[0].metadata["ground_truth"])
                self.assertEqual(frame[0].size, (20, 10))
                self.assertTrue((source / "images" / frame.metadata["relative_path"]).is_file())

    def test_archive_images_survive_source_and_temporary_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root / "nested" / "dataset")
            packed = root / "dataset.zip"
            archive.compress(root / "nested", packed, verbose=False)
            flat = datasets.unpartition(packed, root / "flat.zip", verbose=False)
            results = datasets.load(flat, verbose=False)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]._resolve_source().size, (32, 24))
        self.assertEqual(results[0]._resolve_source().size, (32, 24))
        self.assertEqual(sum(frame.count for frame in results[:1]), 1)

    def test_sparse_ids_missing_empty_labels_and_nested_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            config.save(source / "data.yaml", {"names": {3: "fruit", 8: "tree"}}, overwrite=True, verbose=False)
            (source / "labels" / "train" / "sample_0.txt").unlink()
            files.save(source / "labels" / "train" / "sample_1.txt", "", overwrite=True, verbose=False)
            files.save(source / "labels" / "val" / "sample_2.txt", ["", "3 0.5 0.5 0.5 0.5", "8 0.5 0.5 1 1"], overwrite=True, verbose=False)
            files.save(source / "labels" / "val" / "sample_3.txt", ["8 0.5 0.5 1 1"], overwrite=True, verbose=False)
            media.save(source / "images" / "train" / "nested" / "a%name.png", Image.new("L", (8, 8)), verbose=False)
            flat = datasets.unpartition(source, root / "flat", verbose=False)
            results = datasets.load(flat, verbose=False)
            by_path = {frame.metadata["relative_path"]: frame for frame in results}
            self.assertEqual(len(results), 5)
            self.assertEqual(results.total_objects, 3)
            self.assertTrue(by_path["train/sample_0.jpg"].empty)
            self.assertTrue(by_path["train/sample_1.jpg"].empty)
            self.assertEqual(by_path["val/sample_2.jpg"].labels, ["fruit", "tree"])
            self.assertEqual([box.metadata["class_id"] for box in by_path["val/sample_2.jpg"]], [3, 8])
            self.assertEqual([box.uid for box in by_path["val/sample_2.jpg"]], [0, 1])
            self.assertEqual(by_path["train/nested/a%name.png"]._resolve_source().mode, "RGB")

    def test_flat_and_pathlike_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            flat = datasets.unpartition(source, root / "flat", verbose=False)
            for value in (str(flat), flat, TextPath(flat)):
                with self.subTest(path_type=type(value)):
                    results = datasets.load(value, verbose=False)
                    self.assertEqual(len(results), 4)
                    self.assertEqual({frame.metadata["split"] for frame in results}, {"train", "val"})
                self.assertEqual(results[0].boxes, [(8, 6, 24, 18)])

    def test_invalid_labels_report_file_and_line_and_cleanup(self):
        invalid = (
            "0 0.5 0.5",
            "0 0.5 0.5 0.5 0.5 0.9",
            "1.5 0.5 0.5 0.5 0.5",
            "9 0.5 0.5 0.5 0.5",
            "0 nan 0.5 0.5 0.5",
            "0 0.5 inf 0.5 0.5",
            "0 0.5 0.5 -0.1 0.5",
            "0 0.5 0.5 0 0.5",
            "0 1.1 0.5 0.5 0.5",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            flat = datasets.unpartition(source, root / "flat", verbose=False)
            label = source / "labels" / "train" / "sample_0.txt"
            flat_label = next(path for path in (flat / "labels").glob("*.txt") if "sample_0" in path.name)
            for annotation in invalid:
                with self.subTest(annotation=annotation):
                    files.save(label, ["", annotation], overwrite=True, verbose=False)
                    files.save(flat_label, ["", annotation], overwrite=True, verbose=False)
                    with self.assertRaisesRegex(ValueError, r"sample_0\.txt:2:"):
                        datasets.load(flat, verbose=False)
                    self.assertEqual(files.load(label, as_lines=True, verbose=False), ["", annotation])

    def test_missing_empty_invalid_metadata_and_corrupt_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                datasets.load(root / "missing", verbose=False)
            source = make_dataset(root / "source")
            config.save(source / "data.yaml", {}, overwrite=True, verbose=False)
            with self.assertRaisesRegex(ValueError, "names"):
                datasets.load(source, verbose=False)
            source = make_dataset(root / "corrupt")
            flat = datasets.unpartition(source, root / "corrupt-flat", verbose=False)
            next((flat / "images").glob("*sample_0.jpg")).write_bytes(b"invalid image")
            with self.assertRaises(OSError):
                datasets.load(flat, verbose=False)
            empty = root / "empty"
            (empty / "images").mkdir(parents=True)
            config.save(empty / "data.yaml", {"names": ["object"]}, verbose=False)
            with self.assertRaisesRegex(ValueError, "no supported images"):
                datasets.load(empty, verbose=False)


if __name__ == "__main__":
    unittest.main()
