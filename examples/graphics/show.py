"""Display a generated image with the Matplotlib backend."""

import matplotlib
import numpy as np

from klygo import graphics


matplotlib.use("Agg", force=True)
image = np.full((40, 60, 3), (30, 100, 180), dtype=np.uint8)
graphics.show(image, title="Sample", width=120, backend="matplotlib")
print("Displayed a 120-pixel-wide image")
