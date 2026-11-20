"""Unit tests for klygo.graphics layout (cell, compose, grid) and report charts."""

import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

from klygo import files, graphics
from klygo.outputs import Box, Detection


class TestGraphicsLayoutAndCharts(unittest.TestCase):
    """Exercise composable grid layout and report chart rendering."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = files.path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cell_compose_and_grid_2x3_mixed_sources(self) -> None:
        img_path = self.root / "raw.png"
        Image.new("RGB", (80, 60), (200, 100, 50)).save(img_path)

        det = Detection(
            [Box((5, 5, 40, 40), label="hat", class_id=0, score=0.92)],
            source_id="det",
            source_path=str(img_path),
            image_size=(80, 60),
            names={0: "hat"},
        )
        fig = graphics.plot_partition_distribution(
            [
                {"split": "train", "images": 300, "ratio_pct": 50.0},
                {"split": "val", "images": 300, "ratio_pct": 50.0},
            ],
            show=False,
            return_figure=True,
        )
        self.assertIsNotNone(fig)

        cells = graphics.compose([
            graphics.cell(img_path, width=160, title="(a) Path"),
            graphics.cell(det, height=120, align="top-left", offset=(6, 4), title="(b) Det"),
            graphics.cell(fig, width=180, height=110, align="bottom-right", title="(c) Fig"),
            graphics.cell(np.zeros((40, 60, 3), dtype=np.uint8), align=(10, 10), title="(d) Array"),
            graphics.cell(Image.new("RGB", (50, 50), (10, 120, 220)), align="center"),
            graphics.cell(Image.new("RGB", (50, 50), (30, 180, 60))),
        ])
        self.assertEqual(len(cells), 6)
        self.assertEqual(cells[0].source.size, (160, 120))
        self.assertEqual(cells[2].source.size, (180, 110))

        out_png = self.root / "grid_2x3.png"
        with patch("matplotlib.pyplot.show") as show_mock:
            res = graphics.grid(
                cells,
                rows=2,
                cols=3,
                cell_size=(200, 150),
                gap=(12, 16),
                padding=(28, 12, 12, 12),
                title="Mixed 2x3 Grid",
                save_path=out_png,
            )
            show_mock.assert_called_once()
        self.assertIsNone(res)
        self.assertTrue(files.is_file(out_png))

        with self.assertRaises(TypeError):
            graphics.compose([img_path])  # type: ignore[list-item]

    def test_all_report_charts_show_save_and_return_contract(self) -> None:
        grouped = {
            "table": [
                {"strategy": "S1", "mAP50-95_mean": 0.65, "mAP50-95_std": 0.01, "mAP50_mean": 0.87, "mAP50_std": 0.01},
                {"strategy": "S4", "mAP50-95_mean": 0.68, "mAP50-95_std": 0.006, "mAP50_mean": 0.89, "mAP50_std": 0.005},
            ],
            "groups": {
                "S1": {
                    "runs": [{"config": {"epochs": 5, "close_mosaic": 2, "seed": 42}, "history": {"epoch": [1, 2, 3, 4, 5], "train/box_loss": [1.5, 1.3, 1.1, 0.9, 0.8]}}],
                    "history_mean": {"epoch": [1, 2, 3, 4, 5], "train/box_loss": [1.5, 1.3, 1.1, 0.9, 0.8]},
                    "history_std": {"train/box_loss": [0.1, 0.1, 0.08, 0.05, 0.04]},
                },
            },
        }
        audit = {
            "best_threshold": 0.55,
            "rows": [
                {"threshold": 0.30, "tp": 94, "fp": 21, "precision": 0.81, "recall": 0.76, "f1": 0.78},
                {"threshold": 0.55, "tp": 88, "fp": 5, "precision": 0.94, "recall": 0.71, "f1": 0.81},
            ],
        }
        mining_rows = [
            {"strategy": "S3", "positive_images": 93, "background_images": 157, "boxes": 210, "mean_conf": 0.80, "median_conf": 0.82},
            {"strategy": "S4", "positive_images": 250, "background_images": 0, "boxes": 544, "mean_conf": 0.85, "median_conf": 0.86},
        ]

        out_pdf = self.root / "loss.pdf"
        with patch("matplotlib.pyplot.show") as show_mock:
            self.assertIsNone(graphics.plot_loss_curves(grouped, metric="box_loss", save_path=out_pdf))
            self.assertIsNone(graphics.plot_loss_curves(grouped, metric="box_loss", per_seed=True, show=False))
            self.assertIsNone(graphics.plot_threshold_sweep(audit, kind="curves", show=False))
            self.assertIsNone(graphics.plot_threshold_sweep(audit, kind="counts", show=False))
            self.assertIsNone(graphics.plot_mining_summary(mining_rows, kind="counts", show=False))
            self.assertIsNone(graphics.plot_mining_summary(mining_rows, kind="confidence", show=False))
            self.assertIsNone(graphics.plot_benchmark_bars(grouped["table"], metrics=("mAP50-95", "mAP50"), show=False))
            show_mock.assert_called_once()
        self.assertTrue(files.is_file(out_pdf))


if __name__ == "__main__":
    unittest.main()
