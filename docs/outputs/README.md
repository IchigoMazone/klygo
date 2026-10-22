# `klygo.outputs`

`klygo.outputs` converts normalized backend predictions and provides structured
results for detection tasks. A box keeps its frame-local `id` separate from
its vocabulary `class_id`. `Detection.names` is the sparse mapping used by one
frame, while `Detections.names` contains the complete result vocabulary.

| Implementation | API | Example | Tutorial |
| --- | --- | --- | --- |
| `detect.py` | [`build_detections`](api/build_detections.md) | [Run](../../examples/outputs/build_detections.py) | [Guide](../../tutorials/outputs/Document_Outputs_Detect.ipynb) |
| `detect.py` | [detect](api/detect.md) | [Run](../../examples/outputs/detect.py) | [Guide](../../tutorials/outputs/Document_Outputs_Detect.ipynb) |

Start with the [overview notebook](../../tutorials/outputs/Document_Outputs_Overview.ipynb).
