"""Smoke checks for the current visualization API."""

import numpy as np
from PIL import Image

from klygo.visual import draw_bboxes, plot_dataset_stats, show_image


def test_visual_exports_and_drawing():
    image = Image.fromarray(np.full((48, 64, 3), 255, dtype=np.uint8))
    boxes = [[10, 10, 30, 30]]
    labels = ["apple"]
    scores = [0.95]

    annotated = draw_bboxes(image, boxes, labels, scores)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == image.size
    assert callable(show_image)
    assert callable(plot_dataset_stats)
