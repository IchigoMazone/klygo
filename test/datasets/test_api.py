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
