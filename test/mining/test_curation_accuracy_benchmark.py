"""Unit tests for the dataset creation and mining curation accuracy benchmark."""

import tempfile
import unittest

import numpy as np
from PIL import Image

from benchmarks.mining.curation_accuracy_benchmark import (
    HEAD_CLASSES,
    RARE_CLASSES,
    TARGET_CLASSES,
    aggregate_benchmark_results,
    build_curation_strategies,
    build_realistic_splits,
    export_yolo_subset,
    index_coco_records,
    materialize_benchmark_datasets,
    render_markdown_report,
)
from klygo import datasets, files, media


class TestCurationAccuracyBenchmark(unittest.TestCase):
    """Verify dataset construction, mining pipelines, export, and reporting."""

    def _build_fake_coco_corpus(self, root: str) -> tuple[str, str]:
        images_dir = files.join(root, "val2017")
        files.mkdir(images_dir)

        name_to_coco_id = {
            "person": 1,
            "bicycle": 2,
            "car": 3,
            "bus": 6,
            "traffic light": 10,
            "dog": 18,
            "bottle": 44,
            "cup": 47,
            "chair": 62,
            "laptop": 73,
        }
        categories = [
            {"id": cid, "name": cname} for cname, cid in name_to_coco_id.items()
        ]
        images = []
        annotations = []
        ann_id = 1
        img_id = 1

        def _make_sample(class_names: list[str], seed_val: int) -> None:
            nonlocal img_id, ann_id
            rng = np.random.default_rng(seed_val)
            # Crisp textured pattern so quality metrics stay above uniform-flat floor
            grid = np.indices((64, 64))
            base = ((grid[0] * 11 + grid[1] * 17 + seed_val * 29) % 256).astype(np.uint8)
            noise = rng.integers(0, 60, (64, 64, 3), dtype=np.uint8)
            rgb = np.clip(np.stack([base, 255 - base, base // 2], axis=-1) + noise, 0, 255).astype(np.uint8)
            fname = f"{img_id:012d}.jpg"
            media.save(files.join(images_dir, fname), Image.fromarray(rgb), overwrite=True, verbose=False)
            images.append({"id": img_id, "file_name": fname, "width": 64, "height": 64})
            for offset, cls_name in enumerate(class_names):
                bx = 6 + (offset * 12) % 30
                by = 8 + (offset * 10) % 28
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_coco_id[cls_name],
                    "bbox": [bx, by, 20, 18],
                    "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1

        # 6 images per rare class (shared across val + train core)
        for r_idx, r_name in enumerate(RARE_CLASSES):
            for k in range(6):
                _make_sample([r_name, HEAD_CLASSES[k % len(HEAD_CLASSES)]], 100 + r_idx * 20 + k)

        # 40 head-only images across head classes
        for k in range(40):
            cls_pair = [HEAD_CLASSES[k % len(HEAD_CLASSES)], HEAD_CLASSES[(k + 1) % len(HEAD_CLASSES)]]
            _make_sample(cls_pair, 1000 + k)

        ann_path = files.join(root, "instances_val2017.json")
        files.save(
            ann_path,
            {"categories": categories, "images": images, "annotations": annotations},
            overwrite=True,
            verbose=False,
        )
        return str(images_dir), str(ann_path)

    def test_splits_materialization_curation_and_reporting(self) -> None:
        """End-to-end verification of zero leakage, noise synthesis, all strategies, and reports."""
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, ann_path = self._build_fake_coco_corpus(files.join(tmp, "coco"))
            records = index_coco_records(images_dir, ann_path)
            self.assertEqual(len(records), 70)

            val_recs, train_core = build_realistic_splits(
                records,
                val_per_rare=2,
                val_per_head=3,
                train_rare_cap=3,
                train_head_target=24,
                seed=42,
            )
            val_ids = {r["image_id"] for r in val_recs}
            train_ids = {r["image_id"] for r in train_core}
            self.assertTrue(val_ids.isdisjoint(train_ids))

            ds_meta = materialize_benchmark_datasets(
                val_recs,
                train_core,
                files.join(tmp, "ws"),
                n_exact_dupes=6,
                n_near_dupes=6,
                n_degraded=6,
                seed=42,
            )
            self.assertEqual(ds_meta["pool_composition"]["exact_dup"], 6)
            self.assertEqual(ds_meta["pool_composition"]["near_dup"], 6)
            self.assertEqual(ds_meta["pool_composition"]["degraded"], 6)

            pool = datasets.load(ds_meta["pool_dir"], format="yolo", verbose=False)
            telemetry, run_plans = build_curation_strategies(pool, budget=18, seeds=(42,))
            self.assertGreaterEqual(telemetry["duplicate_report"]["exact_groups"], 1)
            self.assertEqual(len(run_plans), 11)

            smart_plan = next(item for item in run_plans if item["method"] == "klygo_smart_b")
            self.assertLessEqual(len(smart_plan["subset"]), 18)

            stage_dir = files.join(tmp, "stage_smart")
            yaml_path = export_yolo_subset(smart_plan["subset"], ds_meta["val_dir"], stage_dir)
            yaml_data = files.load(yaml_path, verbose=False)
            self.assertEqual(yaml_data["nc"], len(TARGET_CLASSES))
            self.assertEqual(len(yaml_data["names"]), len(TARGET_CLASSES))

            fake_runs = []
            for idx, plan in enumerate(run_plans):
                per_cls = {name: round(0.30 + idx * 0.01 + cid * 0.002, 4) for cid, name in TARGET_CLASSES.items()}
                fake_runs.append({
                    "method": plan["method"],
                    "seed": plan["seed"],
                    "subset_stats": plan["subset_stats"],
                    "metrics": {
                        "mAP50_95": round(0.35 + idx * 0.01, 5),
                        "mAP50": round(0.55 + idx * 0.01, 5),
                        "mAP75": round(0.30 + idx * 0.01, 5),
                        "precision": 0.65,
                        "recall": 0.58,
                        "rare_mAP50_95": round(0.28 + idx * 0.01, 5),
                        "head_mAP50_95": round(0.42 + idx * 0.01, 5),
                        "per_class_mAP50_95": per_cls,
                        "per_class_mAP50": per_cls,
                        "train_seconds": 12.5,
                    },
                })

            leaderboard = aggregate_benchmark_results(fake_runs)
            self.assertEqual(len(leaderboard), 11)
            payload = {
                "environment": {"gpu_name": "UnitTestGPU", "gpu_vram_gb": 24.0, "torch": "2.x", "ultralytics": "8.x"},
                "config": {"weights": "yolov8n.pt", "epochs": 35, "imgsz": 640, "batch": 32, "budget": 18, "seeds": [42]},
                "dataset": ds_meta,
                "mining_telemetry": telemetry,
                "leaderboard": leaderboard,
            }
            md = render_markdown_report(payload)
            self.assertIn("Klygo Mining Curation Accuracy Benchmark", md)
            self.assertIn("`klygo_smart_b`", md)


if __name__ == "__main__":
    unittest.main()
