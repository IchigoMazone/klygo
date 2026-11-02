"""Behavioral tests for public visualization helpers."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib
from PIL import Image

from klygo import files, visual

matplotlib.use("Agg", force=True)


class TestVisual(unittest.TestCase):
    def test_draw_bboxes_preserves_input_and_output_type(self):
        pixels = np.full((48, 64, 3), 255, dtype=np.uint8)
        result = visual.draw_bboxes(pixels, [[10, 10, 30, 30]],
                                    labels=["apple"], scores=[0.95],
                                    box_color=(255, 0, 0))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, pixels.shape)
        self.assertTrue(np.all(pixels == 255))
        self.assertFalse(np.array_equal(result, pixels))

        image = Image.fromarray(pixels)
        annotated = visual.draw_bboxes(image, [[10, 10, 30, 30]],
                                       show_labels=False, show_scores=False)
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))

    def test_draw_box_alias_and_annotator_integration(self):
        pixels = np.full((100, 100, 3), 200, dtype=np.uint8)
        self.assertTrue(callable(visual.draw_box))
        res = visual.draw_box(pixels, [[10, 10, 60, 60]], labels=["0 person"], scores=[0.88])
        self.assertIsInstance(res, np.ndarray)
        self.assertFalse(np.array_equal(res, pixels))

    def test_show_image_auto_uses_desktop_viewer_outside_notebook(self):
        image = Image.new("RGB", (20, 10))
        with patch("klygo.visual.show_image._is_notebook", return_value=False):
            with patch.object(Image.Image, "show") as shown:
                visual.show_image(image, title="preview", width=40)
        shown.assert_called_once_with(title="preview")

    def test_plot_dataset_stats_counts_label_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            files.mkdir(root / "images")
            files.mkdir(root / "labels")
            files.save(root / "data.yaml", "names: [cat, dog]\nnc: 2\n")
            Image.new("RGB", (8, 8)).save(root / "images" / "sample.png")
            files.save(root / "labels" / "sample.txt",
                       "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
            with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.bar") as bar:
                visual.plot_dataset_stats(root)
            self.assertEqual(list(bar.call_args.args[0]), ["cat", "dog"])
            self.assertEqual(list(bar.call_args.args[1]), [2, 1])

    def test_plot_dataset_stats_preserves_sparse_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            files.mkdir(root / "images")
            files.mkdir(root / "labels")
            files.save(root / "data.yaml", {"names": {2: "cat", 5: "dog"}}, verbose=False)
            Image.new("RGB", (8, 8)).save(root / "images" / "sample.png")
            files.save(root / "labels" / "sample.txt", "2 0.5 0.5 0.2 0.2\n5 0.3 0.3 0.1 0.1\n", verbose=False)
            with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.bar") as bar:
                visual.plot_dataset_stats(root)
            self.assertEqual(list(bar.call_args.args[0]), ["cat", "dog"])
            self.assertEqual(list(bar.call_args.args[1]), [1, 1])
