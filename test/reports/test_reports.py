"""Unit tests for klygo.reports."""

import tempfile
import unittest

import numpy as np
from PIL import Image

from klygo import files, reports, utils
from klygo.outputs import Box, Detection, Detections


class TestReports(unittest.TestCase):
    """Verify report loaders, aggregators, audits, and files.save compatibility."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = files.path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_run(self, name: str, seed: int, epochs: int, base_map: float, total_time: float) -> str:
        run_dir = self.root / name
        files.mkdir(run_dir)
        files.save(
            run_dir / "args.yaml",
            {"model": "yolo11n.pt", "epochs": 50, "batch": 16, "close_mosaic": 10, "seed": seed},
            overwrite=True,
            verbose=False,
        )
        rows = []
        for ep in range(1, epochs + 1):
            frac = ep / max(1, epochs)
            rows.append({
                "                  epoch": ep,
                "         train/box_loss": round(1.5 - 0.8 * frac, 4),
                "         train/cls_loss": round(2.0 - 1.4 * frac, 4),
                "         train/dfl_loss": round(1.3 - 0.4 * frac, 4),
                "   metrics/precision(B)": round(0.75 + 0.15 * frac, 4),
                "      metrics/recall(B)": round(0.65 + 0.18 * frac, 4),
                "       metrics/mAP50(B)": round(base_map + 0.10 * frac, 4),
                "    metrics/mAP50-95(B)": round(base_map + 0.08 * frac, 4),
                "                 time": round(total_time * frac, 2),
            })
        files.save(run_dir / "results.csv", rows, overwrite=True, verbose=False)
        return str(run_dir)

    def test_load_aggregate_and_hyperparams(self) -> None:
        s1_a = self._make_run("S1_42", 42, 5, 0.55, 150.0)
        s1_b = self._make_run("S1_100", 100, 4, 0.57, 140.0)  # uneven epoch length
        s4_a = self._make_run("S4_42", 42, 5, 0.60, 220.0)
        s4_b = self._make_run("S4_100", 100, 5, 0.62, 225.0)

        loaded = reports.load(s1_a, format="yolo")
        self.assertEqual(loaded["best_epoch"], 5)
        self.assertEqual(loaded["total_time_s"], 150.0)
        self.assertIn("f1", loaded["best_metrics"])

        hp = reports.extract_hyperparams([s1_a, s1_b, s4_a, s4_b])
        self.assertEqual(hp["diffs"], {})
        self.assertEqual(hp["common"]["close_mosaic"], 10)

        grouped = reports.aggregate_runs(
            {"S1": [s1_a, s1_b], "S4": [s4_a, s4_b]},
            compare_pair=("S4", "S1"),
        )
        self.assertEqual(len(grouped["table"]), 2)
        self.assertEqual(len(grouped["per_seed"]), 4)
        self.assertGreater(grouped["delta"]["mAP50-95"], 0.0)
        self.assertEqual(len(grouped["groups"]["S1"]["history_mean"]["train/box_loss"]), 5)

        # Compatible with utils.tabulate and files.save
        utils.tabulate(grouped["table"], title="Summary")
        out_json = self.root / "summary.json"
        files.save(out_json, grouped["table"], overwrite=True, verbose=False)
        self.assertEqual(len(files.load(out_json, verbose=False)), 2)

        with self.assertRaises(ValueError):
            reports.load(s1_a, format="keras")

    def test_partition_summary_threshold_mining_and_ranking(self) -> None:
        # 1. Partition summary on a partitioned YOLO directory
        part_dir = self.root / "part"
        for split, count, pos in (("train", 4, 3), ("val", 2, 1)):
            files.mkdir(part_dir / "images" / split)
            files.mkdir(part_dir / "labels" / split)
            for i in range(count):
                Image.new("RGB", (16, 16)).save(part_dir / "images" / split / f"img_{i}.png")
                txt = "0 0.5 0.5 0.2 0.2\n" if i < pos else ""
                files.save(part_dir / "labels" / split / f"img_{i}.txt", txt, overwrite=True, verbose=False)
        files.save(part_dir / "data.yaml", {"names": {0: "hat"}}, overwrite=True, verbose=False)

        ps = reports.partition_summary(part_dir)
        self.assertEqual(sum(r["images"] for r in ps), 6)
        self.assertAlmostEqual(sum(r["ratio_pct"] for r in ps), 100.0, places=1)

        # 2. Threshold audit & mining breakdown
        img_p = self.root / "sample.png"
        Image.new("RGB", (64, 64)).save(img_p)
        pred = Detections([
            Detection([
                Box((0, 0, 20, 20), label="hat", class_id=0, score=0.9),
                Box((30, 30, 50, 50), label="hat", class_id=0, score=0.4),
            ], source_id="a", source_path=str(img_p), image_size=(64, 64), names={0: "hat"}),
            Detection([], source_id="b", source_path=str(img_p), image_size=(64, 64), names={0: "hat"}),
        ], names={0: "hat"})
        target = Detections([
            Detection([Box((0, 0, 20, 20), label="hat", class_id=0)], source_id="a", source_path=str(img_p), image_size=(64, 64), names={0: "hat"}),
            Detection([], source_id="b", source_path=str(img_p), image_size=(64, 64), names={0: "hat"}),
        ], names={0: "hat"})

        audit = reports.threshold_audit(
            pred, target, thresholds=[0.3, 0.5, 0.8],
            pred_confirm=pred, target_confirm=target,
        )
        self.assertEqual(len(audit["rows"]), 3)
        self.assertIn(audit["best_threshold"], (0.5, 0.8))
        self.assertIn("delta", audit["comparison"])

        mb = reports.mining_breakdown({"S4": pred})
        self.assertEqual((mb[0]["positive_images"], mb[0]["background_images"], mb[0]["boxes"]), (1, 1, 2))

        ranked = reports.rank_evaluations(
            [
                {"code": "S1", "mAP50-95": 0.5696},
                {"code": "S4", "mAP50-95": 0.5955},
            ],
            baseline="S1",
        )
        self.assertEqual((ranked[0]["code"], ranked[0]["rank"]), ("S4", 1))
        self.assertAlmostEqual(ranked[0]["delta"], 0.0259, places=4)


if __name__ == "__main__":
    unittest.main()
