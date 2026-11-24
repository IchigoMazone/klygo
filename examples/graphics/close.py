"""Close open Matplotlib figures using Klygo's graphics facade."""

import matplotlib
from klygo import graphics

matplotlib.use("Agg", force=True)
fig = graphics.bars([{"strategy": "random", "mAP50-95": 0.5}], show=False, return_figure=True)
graphics.close(fig)
print("Closed Matplotlib figure successfully")
