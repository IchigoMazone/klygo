# `klygo.outputs`

`outputs.detect` provides structured results for detection tasks: `Box`,
`Crops`, `Detection`, and `Detections`. A box stores class, score, pixel
coordinates, and optional crop context. `Detection` groups boxes for one image;
`Detections` groups images or streams.

| Implementation | API | Example | Tutorial |
| --- | --- | --- | --- |
| `detect.py` | [detect](api/detect.md) | [Run](../../examples/outputs/detect.py) | [Guide](../../tutorials/outputs/Document_Outputs_Detect.ipynb) |

Start with the [overview notebook](../../tutorials/outputs/Document_Outputs_Overview.ipynb).
