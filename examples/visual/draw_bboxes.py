"""Annotate a generated image without modifying its pixels."""

import numpy as np

from klygo import visual


image = np.full((80, 120, 3), 255, dtype=np.uint8)
annotated = visual.draw_bboxes(
    image, [[10, 12, 80, 65]], labels=["sample"], scores=[0.92],
    box_color=(220, 30, 30),
)
print("Input unchanged:", bool(np.all(image == 255)))
print("Output shape:", annotated.shape)
