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
        fig = graphics.partitions(
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
            graphics.cell(np.zeros((40, 60, 1), dtype=np.uint8), align="center", title="(e) Gray1"),
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
                    "runs": [{"config": {"epochs": 5, "close_mosaic": 2, "seed": 42}, "history": {"epoch": [1, 2, 3, 4, 5], "train/box_loss": [1.5, 1.3, 1.1, 0.9, 0.8], "mAP50-95": [0.3, 0.45, 0.55, 0.62, 0.65]}}],
                    "history_mean": {"epoch": [1, 2, 3, 4, 5], "train/box_loss": [1.5, 1.3, 1.1, 0.9, 0.8], "mAP50-95": [0.3, 0.45, 0.55, 0.62, 0.65]},
                    "history_std": {"train/box_loss": [0.1, 0.1, 0.08, 0.05, 0.04], "mAP50-95": [0.02, 0.02, 0.01, 0.01, 0.01]},
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
            self.assertIsNone(graphics.curves(grouped, metric="box_loss", save_path=out_pdf))
            self.assertIsNone(graphics.curves(grouped, metric="box_loss", per_seed=True, show=False))
            map_fig = graphics.curves(grouped, metric="mAP50-95", show=False, return_figure=True)
            self.assertIsNotNone(map_fig)
            self.assertGreater(len(map_fig.axes[0].lines), 0)
            self.assertIsNone(graphics.thresholds(audit, kind="curves", show=False))
            self.assertIsNone(graphics.thresholds(audit, kind="counts", show=False))
            self.assertIsNone(graphics.mining(mining_rows, kind="counts", show=False))
            self.assertIsNone(graphics.mining(mining_rows, kind="confidence", show=False))
            self.assertIsNone(graphics.bars(grouped["table"], metrics=("mAP50-95", "mAP50"), show=False))
            show_mock.assert_called_once()
        self.assertTrue(files.is_file(out_pdf))

    def test_grid_accepts_media_frames_directly(self) -> None:
        from klygo import media

        img_dir = self.root / "grid_media"
        files.mkdir(img_dir)
        for idx, color in enumerate(("red", "green", "blue")):
            Image.new("RGB", (32, 24), color).save(img_dir / f"{idx}.png")
        frames = media.load(img_dir, indexed=True)
        out_png = self.root / "media_grid.png"
        graphics.grid(frames, cols=2, show=False, save_path=out_png)
        self.assertTrue(files.is_file(out_png))

    def test_error_handling_and_format_validation(self) -> None:
        # graphics.dataset_stats format validation
        with self.assertRaises(TypeError):
            graphics.dataset_stats(self.root, format=123, show=False)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graphics.dataset_stats(self.root, format="coco", show=False)

        # graphics.save validation
        fig = graphics.partitions(
            [{"split": "train", "images": 10, "ratio_pct": 100.0}],
            show=False,
            return_figure=True,
        )
        assert fig is not None
        with self.assertRaises(TypeError):
            graphics.save("not-a-fig", self.root / "out.png")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            graphics.save(fig, self.root / "out.png", dpi=True)
        with self.assertRaises(ValueError):
            graphics.save(fig, self.root / "out.png", dpi=0)
        with self.assertRaises(TypeError):
            graphics.save(fig, self.root / "out.png", tight="yes")  # type: ignore[arg-type]
        saved = graphics.save(fig, self.root / "valid.png", dpi=150)
        self.assertTrue(files.is_file(saved))

        # Chart input/type/value validation
        with self.assertRaises(TypeError):
            graphics.partitions("bad", show=False)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graphics.partitions([], show=False)
        with self.assertRaises(TypeError):
            graphics.curves([], show=False)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graphics.curves({}, show=False)
        with self.assertRaises(TypeError):
            graphics.curves({"S1": {}}, mosaic_marker=True, show=False)
        with self.assertRaises(TypeError):
            graphics.thresholds([], show=False)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graphics.thresholds({"rows": []}, show=False)
        with self.assertRaises(ValueError):
            graphics.thresholds({"rows": [{"threshold": 0.5}]}, kind="unknown", show=False)
        with self.assertRaises(ValueError):
            graphics.mining([], show=False)
        with self.assertRaises(ValueError):
            graphics.mining([{"strategy": "S1"}], kind="unknown", show=False)
        with self.assertRaises(ValueError):
            graphics.bars([], show=False)
        with self.assertRaises(ValueError):
            graphics.bars([{"strategy": "S1"}], metrics=(), show=False)

        # Layout cell validation
        with self.assertRaises(ValueError):
            graphics.cell(Image.new("RGB", (10, 10)), width=0)
        with self.assertRaises(ValueError):
            graphics.cell(Image.new("RGB", (10, 10)), height=-5)
        with self.assertRaises(ValueError):
            graphics.cell(Image.new("RGB", (10, 10)), offset=(1,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graphics.cell(Image.new("RGB", (10, 10)), letterbox=True, resize=True)

        c_letter = graphics.cell(Image.new("RGB", (100, 50)), width=100, height=100, letterbox=True)
        self.assertTrue(c_letter.letterbox)
        self.assertFalse(c_letter.resize)
        comp_letter = graphics.compose([c_letter])
        self.assertEqual(comp_letter[0].source.size, (100, 100))

        c_resize = graphics.cell(Image.new("RGB", (100, 50)), width=60, height=30, resize=True)
        self.assertFalse(c_resize.letterbox)
        self.assertTrue(c_resize.resize)
        comp_resize = graphics.compose([c_resize])
        self.assertEqual(comp_resize[0].source.size, (60, 30))


if __name__ == "__main__":
    unittest.main()
