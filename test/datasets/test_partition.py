import unittest
import os
import shutil
import tempfile
import yaml
from pathlib import Path
from klygo.datasets import partition, repartition
from pydantic import ValidationError

try:
    from ._support import ensure_test_data_zip
except ImportError:
    from test.datasets._support import ensure_test_data_zip

class TestPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_test_data_zip()

    def test_partition_and_repartition(self):
        print("Testing partition and repartition success paths...")
        with tempfile.TemporaryDirectory() as td:
            temp_out = os.path.join(td, "test_partition_out")

            # 1. Partition ZIP to folder
            partition(
                source="test/data.zip",
                target=temp_out,
                ratios=(0.8, 0.1, 0.1),
                overwrite=True,
                verbose=False
            )

            assert os.path.exists(temp_out)
            assert os.path.exists(os.path.join(temp_out, "images", "train"))
            assert os.path.exists(os.path.join(temp_out, "images", "val"))
            assert os.path.exists(os.path.join(temp_out, "images", "test"))

            with open(os.path.join(temp_out, "data.yaml"), "r") as f:
                data = yaml.safe_load(f)
            assert data["train"] == "images/train"
            assert data["val"] == "images/val"
            assert data["test"] == "images/test"

            # 2. Repartition folder in-place
            repartition(
                source=temp_out,
                target=temp_out,
                ratios=(0.5, 0.5, 0.0), # 2 splits only
                overwrite=True,
                verbose=False
            )

            assert os.path.exists(os.path.join(temp_out, "images", "train"))
            assert os.path.exists(os.path.join(temp_out, "images", "val"))
            assert not os.path.exists(os.path.join(temp_out, "images", "test"))

            with open(os.path.join(temp_out, "data.yaml"), "r") as f:
                data2 = yaml.safe_load(f)
            assert data2["train"] == "images/train"
            assert data2["val"] == "images/val"
            assert "test" not in data2

    def test_partition_validation_errors(self):
        print("Testing partition validation errors...")
        # 1. Invalid ratio sum (must equal 1.0 or sum <= 1.0)
        try:
            partition(
                source="test/data.zip",
                target="dummy_out",
                ratios=(0.5, 0.6), # sum = 1.1 > 1.0
                overwrite=True
            )
            assert False, "Should raise ValidationError for ratios sum > 1.0"
        except (ValidationError, ValueError):
            pass

        # 2. Non-existent source directory/file
        try:
            partition(
                source="non_existent_zip_or_folder.zip",
                target="dummy_out",
                ratios=(0.8,),
                overwrite=True
            )
            assert False, "Should raise ValidationError/FileNotFoundError for non-existent source"
        except (ValidationError, FileNotFoundError):
            pass

    def test_partition_shuffle_and_seed_controls(self):
        with tempfile.TemporaryDirectory() as td:
            out_canonical = os.path.join(td, "canonical")
            partition(
                source="test/data.zip",
                target=out_canonical,
                ratios=(0.5, 0.5, 0.0),
                shuffle=False,
                overwrite=True,
                verbose=False,
            )
            canonical_train = sorted(os.listdir(os.path.join(out_canonical, "images", "train")))

            # With shuffle=True and seed=42
            out_seeded1 = os.path.join(td, "seeded1")
            partition(
                source="test/data.zip",
                target=out_seeded1,
                ratios=(0.5, 0.5, 0.0),
                shuffle=True,
                seed=42,
                overwrite=True,
                verbose=False,
            )
            seeded1_train = sorted(os.listdir(os.path.join(out_seeded1, "images", "train")))

            # Same seed produces same partition
            out_seeded2 = os.path.join(td, "seeded2")
            partition(
                source="test/data.zip",
                target=out_seeded2,
                ratios=(0.5, 0.5, 0.0),
                shuffle=True,
                seed=42,
                overwrite=True,
                verbose=False,
            )
            seeded2_train = sorted(os.listdir(os.path.join(out_seeded2, "images", "train")))
            self.assertEqual(seeded1_train, seeded2_train)

            # Different seed produces different partition
            out_seeded3 = os.path.join(td, "seeded3")
            partition(
                source="test/data.zip",
                target=out_seeded3,
                ratios=(0.5, 0.5, 0.0),
                shuffle=True,
                seed=999,
                overwrite=True,
                verbose=False,
            )
            seeded3_train = sorted(os.listdir(os.path.join(out_seeded3, "images", "train")))
            self.assertNotEqual(seeded1_train, seeded3_train)

    def test_partition_stratified_preserves_class_proportions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src")
            os.makedirs(os.path.join(src, "images"), exist_ok=True)
            os.makedirs(os.path.join(src, "labels"), exist_ok=True)
            with open(os.path.join(src, "data.yaml"), "w", encoding="utf-8") as fh:
                fh.write("names:\n  0: majority\n  1: rare\n")

            # 14 majority-only images, 4 rare-only images, 2 background images = 20 total
            for i in range(20):
                Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(os.path.join(src, "images", f"img_{i:02d}.png"))
                label_file = os.path.join(src, "labels", f"img_{i:02d}.txt")
                if i < 14:
                    with open(label_file, "w", encoding="utf-8") as fh:
                        fh.write("0 0.5 0.5 0.2 0.2\n")
                elif i < 18:
                    with open(label_file, "w", encoding="utf-8") as fh:
                        fh.write("1 0.5 0.5 0.2 0.2\n")
                else:
                    with open(label_file, "w", encoding="utf-8") as fh:
                        fh.write("")

            out = os.path.join(td, "out")
            partition(
                source=src,
                target=out,
                ratios=(0.5, 0.25, 0.25),
                stratify=True,
                shuffle=True,
                seed=42,
                overwrite=True,
                verbose=False,
            )

            def _split_class_counts(split: str):
                counts = {"0": 0, "1": 0, "bg": 0}
                lbl_dir = os.path.join(out, "labels", split)
                for name in os.listdir(lbl_dir):
                    with open(os.path.join(lbl_dir, name), encoding="utf-8") as fh:
                        lines = [ln.strip() for ln in fh if ln.strip()]
                    if not lines:
                        counts["bg"] += 1
                    else:
                        for cls in {ln.split()[0] for ln in lines}:
                            counts[cls] += 1
                return counts

            train_c = _split_class_counts("train")
            val_c = _split_class_counts("val")
            test_c = _split_class_counts("test")

            self.assertEqual(sum(train_c.values()), 10)
            self.assertEqual(sum(val_c.values()), 5)
            self.assertEqual(sum(test_c.values()), 5)
            # Rare class (4 images) split 2 / 1 / 1 across 0.5 / 0.25 / 0.25
            self.assertEqual((train_c["1"], val_c["1"], test_c["1"]), (2, 1, 1))
            # Majority class (14 images) split 7 / 3-4 / 3-4
            self.assertEqual(train_c["0"], 7)

            with self.assertRaises(TypeError):
                partition(source=src, target=out, ratios=(0.8, 0.2, 0.0), stratify="yes", overwrite=True)  # type: ignore[arg-type]

            from klygo.datasets import repartition
            repart_out = os.path.join(td, "repart_out")
            repartition(
                source=out,
                target=repart_out,
                ratios=(0.5, 0.25, 0.25),
                stratify=True,
                shuffle=True,
                seed=7,
                overwrite=True,
                verbose=False,
            )
            repart_rare = 0
            for name in os.listdir(os.path.join(repart_out, "labels", "train")):
                with open(os.path.join(repart_out, "labels", "train", name), encoding="utf-8") as fh:
                    if "1 0.5" in fh.read():
                        repart_rare += 1
            self.assertEqual(repart_rare, 2)
