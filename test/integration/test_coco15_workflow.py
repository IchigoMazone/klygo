"""Check the 15-class dataset, Colab notebooks, and portable source bundle."""

from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

try:
    import nbformat
except ImportError:
    nbformat = None

from klygo import datasets

from .build_coco15 import CLASSES, SOURCE_SHA256
from .package_coco15_colab import DATASET, DEFAULT_OUTPUT


NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "integration"


class TestCOCO15Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = (
            json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
            if (DATASET / "manifest.json").is_file() else None
        )

    def test_real_source_and_all_fifteen_classes(self):
        if self.manifest is None:
            self.skipTest("Build COCO15 from the pinned official archive first")
        self.assertEqual(self.manifest["source_sha256"], SOURCE_SHA256)
        self.assertEqual(self.manifest["image_count"], 128)
        self.assertEqual(len(self.manifest["classes"]), 15)
        self.assertEqual(
            [(item["source_coco_id"], item["name"]) for item in self.manifest["classes"]],
            list(CLASSES),
        )
        self.assertTrue(all(item["boxes"] > 0 and item["positive_images"] > 0 for item in self.manifest["classes"]))
        self.assertGreater(self.manifest["negative_image_count"], 0)

    def test_klygo_reads_all_images_and_class_ids(self):
        if self.manifest is None:
            self.skipTest("Build COCO15 from the pinned official archive first")
        results = datasets.load(DATASET, format="yolo", verbose=False)
        self.assertEqual(len(results), 128)
        self.assertEqual(
            sum(len(frame) for frame in results),
            sum(item["boxes"] for item in self.manifest["classes"]),
        )
        self.assertEqual(
            {box.label for frame in results for box in frame},
            {name for _, name in CLASSES},
        )
        self.assertEqual(
            {box.metadata["class_id"] for frame in results for box in frame},
            set(range(15)),
        )

    def test_one_valid_colab_notebook_per_class(self):
        if nbformat is None:
            self.skipTest("nbformat is not installed")
        paths = sorted(NOTEBOOK_DIR.glob("Klygo_COCO15_*.ipynb"))
        self.assertEqual(len(paths), 15)
        seen = set()
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            self.assertEqual(notebook.metadata["accelerator"], "GPU")
            sources = "\n".join(cell.source for cell in notebook.cells)
            for module in (
                "archive", "config", "datasets", "files", "media", "models", "outputs",
                "postprocessing", "preprocessing", "utils", "validators", "visual",
            ):
                self.assertIn(module, sources)
            target_id = next(index for index in range(15) if f"_{index:02d}_" in path.name)
            self.assertIn(f"TARGET_CLASS = {CLASSES[target_id][1]!r}", sources)
            seen.add(target_id)
        self.assertEqual(seen, set(range(15)))

    def test_bundle_is_complete_and_reproducible(self):
        if not DEFAULT_OUTPUT.is_file():
            self.skipTest("Package the generated COCO15 dataset first")
        with zipfile.ZipFile(DEFAULT_OUTPUT) as bundle:
            self.assertIsNone(bundle.testzip())
            names = set(bundle.namelist())
            self.assertIn("project/pyproject.toml", names)
            self.assertIn("project/klygo/models/models.json", names)
            self.assertIn("dataset/data.yaml", names)
            self.assertIn("dataset/manifest.json", names)
            self.assertIn("dataset/LICENSE", names)
            self.assertEqual(len([name for name in names if name.startswith("dataset/images/") and name.endswith(".jpg")]), 128)
            bundled_manifest = json.loads(bundle.read("dataset/manifest.json"))
            self.assertEqual(bundled_manifest["source_sha256"], SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()
