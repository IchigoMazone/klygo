"""Behavioral and architectural contracts for the public datasets API."""

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from klygo import archive, config, datasets, files, media
from klygo.outputs.detect import Box, Detection, Detections


def make_dataset(root: Path, names=("apple", "banana"), prefix="sample") -> Path:
    for index, class_id in enumerate((0, 1, 0, 1)):
        split = "train" if index < 2 else "val"
        image_path = root / "images" / split / f"{prefix}_{index}.jpg"
        label_path = root / "labels" / split / f"{prefix}_{index}.txt"
        media.save(image_path, Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(
            label_path,
            [f"{class_id} 0.5 0.5 0.5 0.5"],
            overwrite=True,
            verbose=False,
        )
    config.save(
        root / "data.yaml",
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": list(names),
        },
        overwrite=True,
        verbose=False,
    )
    return root


class StubDetector:
    def __call__(self, images, prompt=None, **kwargs):
        detections = []
        for image in images:
            width, height = image.size
            detections.append(
                Detection(
                    source_image=image,
                    objects=[
                        Box(
                            id=0,
                            label=prompt[0],
                            score=0.9,
                            box=[0, 0, width, height],
                            parent_image=image,
                        )
                    ],
                )
            )
        return Detections(detections, source_type="images")


class TextPath:
    def __init__(self, path: Path):
        self.path = path

    def __fspath__(self) -> str:
        return str(self.path)


class TestDatasetsAPI(unittest.TestCase):
    def test_public_api_is_function_oriented(self):
        expected = {
            "get_dataset_info",
            "partition",
            "repartition",
            "unpartition",
            "merge",
            "split",
            "remap_classes",
            "export",
        }
        self.assertEqual(set(datasets.__all__), expected)
        for name in expected:
            self.assertTrue(callable(getattr(datasets, name)))
            self.assertTrue(inspect.getdoc(getattr(datasets, name)))

    def test_info_partition_repartition_and_unpartition(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            info = datasets.get_dataset_info(source)
            self.assertEqual(info["classes"], ["apple", "banana"])
            self.assertEqual(info["total_images"], 4)

            partitioned = datasets.partition(
                source,
                root / "partitioned",
                (0.5, 0.25, 0.25),
                verbose=False,
            )
            self.assertTrue(files.is_dir(partitioned / "images" / "train"))
            self.assertEqual(datasets.get_dataset_info(partitioned)["total_images"], 4)

            repartitioned = datasets.repartition(
                partitioned,
                root / "repartitioned.zip",
                (0.5, 0.25),
                verbose=False,
            )
            self.assertTrue(archive.is_archive(repartitioned))

            flat = datasets.unpartition(
                repartitioned,
                root / "flat",
                verbose=False,
            )
            self.assertEqual(datasets.get_dataset_info(flat)["total_images"], 4)
            self.assertEqual(files.find(flat / "images", "*.jpg", recursive=True).__len__(), 4)

    def test_merge_split_and_remap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_dataset(root / "first", ("apple", "banana"), "first")
            second = make_dataset(root / "second", ("banana", "orange"), "second")

            merged = datasets.merge([first, second], root / "merged.zip", verbose=False)
            merged_info = datasets.get_dataset_info(merged)
            self.assertEqual(merged_info["classes"], ["apple", "banana", "orange"])
            self.assertEqual(merged_info["total_images"], 8)

            remapped = datasets.remap_classes(
                first,
                root / "remapped",
                {"apple": "fruit", 1: 3},
                verbose=False,
            )
            remapped_config = config.load(remapped / "data.yaml", verbose=False)
            self.assertEqual(remapped_config.names, {0: "fruit", 3: "banana"})
            self.assertEqual(remapped_config.nc, 4)

            class_parts = datasets.split(
                first,
                root / "class-parts",
                by_class=True,
                verbose=False,
            )
            ratio_parts = datasets.split(
                first,
                root / "ratio-parts",
                ratios=(0.5, 0.5),
                verbose=False,
            )
            self.assertEqual(len(class_parts), 2)
            self.assertEqual(len(ratio_parts), 2)
            self.assertTrue(all(archive.is_archive(path) for path in class_parts + ratio_parts))

    def test_export_detection_and_classification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = [Image.new("RGB", (20, 10), "white")]
            detector = StubDetector()

            detection = datasets.export(
                detector,
                root / "detection",
                "yolo",
                images,
                classes=["object"],
                verbose=False,
            )
            self.assertTrue(files.is_file(detection / "images" / "img_00001.jpg"))
            self.assertEqual(
                files.load(detection / "labels" / "img_00001.txt", verbose=False).strip(),
                "0 0.500000 0.500000 1.000000 1.000000",
            )
            self.assertEqual(config.load(detection / "data.yaml", verbose=False).names, ["object"])

            classification = datasets.export(
                detector,
                root / "classification",
                "classification",
                images,
                text_prompt=["object"],
                verbose=False,
            )
            self.assertEqual(
                len(files.find(classification / "object", "*.jpg", recursive=True)),
                1,
            )

    def test_export_keeps_existing_output_when_validation_or_prediction_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "generated"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("original", encoding="utf-8")
            images = [Image.new("RGB", (20, 10), "white")]

            with self.assertRaises(ValueError):
                datasets.export(StubDetector(), output, "invalid", images, classes=["object"], overwrite=True)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

            class IncompleteDetector:
                def __call__(self, images, **kwargs):
                    return iter(())

            with self.assertRaises(ValueError):
                datasets.export(IncompleteDetector(), output, "yolo", images, classes=["object"], overwrite=True)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_flattening_keeps_distinct_nested_image_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            for split in ("train", "val"):
                media.save(source / "images" / split / "duplicate.jpg", Image.new("RGB", (8, 8)), verbose=False)
                files.save(source / "labels" / split / "duplicate.txt", ["0 0.5 0.5 1 1"], verbose=False)

            flat = datasets.unpartition(source, root / "flat", verbose=False)
            self.assertEqual(datasets.get_dataset_info(flat)["total_images"], 6)
            self.assertEqual(len(files.find(flat / "images", "*.jpg", recursive=True)), 6)

            parts = datasets.split(source, root / "parts", by_class=True, verbose=False)
            apple_part = next(path for path in parts if path.name == "split_apple.zip")
            self.assertEqual(datasets.get_dataset_info(apple_part)["total_images"], 4)

    def test_archive_metadata_is_portable_and_remap_rejects_id_collisions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            other = make_dataset(root / "other", prefix="other")
            outputs = [
                datasets.partition(source, root / "partitioned.zip", (0.5, 0.25, 0.25), verbose=False),
                datasets.unpartition(source, root / "flat.zip", verbose=False),
                datasets.merge([source, other], root / "merged.zip", verbose=False),
                datasets.remap_classes(source, root / "remapped.zip", {"apple": "fruit"}, verbose=False),
                *datasets.split(source, root / "parts", ratios=(0.5, 0.5), verbose=False),
            ]
            for output in outputs:
                with self.subTest(output=output), TemporaryDirectory() as extracted:
                    archive.extract(output, extracted, verbose=False)
                    metadata = config.load(Path(extracted) / "data.yaml", verbose=False)
                    self.assertNotIn("path", metadata)

            with self.assertRaisesRegex(ValueError, "multiple class names"):
                datasets.remap_classes(source, root / "collision", {1: 0}, verbose=False)
            self.assertFalse((root / "collision").exists())

    def test_pathlike_inputs_and_invalid_split_ratios(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "source")
            self.assertEqual(datasets.get_dataset_info(TextPath(source))["total_images"], 4)
            partitioned = datasets.partition(
                TextPath(source), TextPath(root / "partitioned"), (0.5, 0.25, 0.25), verbose=False
            )
            self.assertTrue(partitioned.exists())
            outputs = datasets.split(TextPath(source), TextPath(root / "parts"), ratios=(0.5, 0.5), verbose=False)
            self.assertEqual(len(outputs), 2)
            for ratios in ((-0.2, 1.2), (float("nan"), 0.5)):
                with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                    datasets.split(source, root / "invalid", ratios=ratios, verbose=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                datasets.split(source, root / "conflict", by_class=True, ratios=(0.5, 0.5), verbose=False)

    def test_sparse_class_ids_survive_dataset_transforms(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_dataset(root / "sparse")
            metadata = config.load(source / "data.yaml", verbose=False).to_dict()
            metadata["names"] = {0: "apple", 3: "banana"}
            metadata["nc"] = 4
            config.save(source / "data.yaml", metadata, overwrite=True, verbose=False)
            for label in (source / "labels").rglob("*.txt"):
                label.write_text(label.read_text(encoding="utf-8").replace("1 ", "3 "), encoding="utf-8")

            sparse_info = datasets.get_dataset_info(source)
            self.assertEqual(sparse_info["nc"], 4)
            self.assertEqual(sparse_info["class_ids"], [0, 3])
            flat = datasets.unpartition(source, root / "flat", verbose=False)
            self.assertEqual(config.load(flat / "data.yaml", verbose=False).names, {0: "apple", 3: "banana"})

            ratio_parts = datasets.split(source, root / "ratio", ratios=(0.5, 0.5), verbose=False)
            self.assertEqual(datasets.get_dataset_info(ratio_parts[0])["nc"], 4)
            banana_part = next(
                path for path in datasets.split(source, root / "class", by_class=True, verbose=False)
                if path.name == "split_banana.zip"
            )
            self.assertEqual(datasets.get_dataset_info(banana_part)["total_images"], 2)

            remapped = datasets.remap_classes(source, root / "remapped", {3: 5}, verbose=False)
            self.assertEqual(config.load(remapped / "data.yaml", verbose=False).names, {0: "apple", 5: "banana"})

            other = make_dataset(root / "other", ("banana", "orange"), "other")
            merged = datasets.merge([source, other], root / "merged.zip", verbose=False)
            self.assertEqual(datasets.get_dataset_info(merged)["classes"], ["apple", "banana", "orange"])

    def test_dataset_sources_use_shared_klygo_modules(self):
        package = Path(datasets.__file__).parent
        public_and_internal = [
            path for path in package.glob("*.py") if path.name != "__pycache__"
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in public_and_internal)
        self.assertNotIn("import shutil", combined)
        self.assertNotIn("from zipfile", combined)


if __name__ == "__main__":
    unittest.main()
