# `klygo.postprocessing`

Compose transformations for normalized object-detection results. Import with `from klygo import postprocessing as post`.

## Core

`core.py` defines operations and pipelines.

| API | Example |
| --- | --- |
| [`Operation`](api/Operation.md) | [Run](../../examples/postprocessing/Operation.py) |
| [`Pipeline`](api/Pipeline.md) | [Run](../../examples/postprocessing/Pipeline.py) |
| [`compose`](api/compose.md) | [Run](../../examples/postprocessing/compose.py) |
| [`apply`](api/apply.md) | [Run](../../examples/postprocessing/apply.py) |
| [`custom`](api/custom.md) | [Run](../../examples/postprocessing/custom.py) |

## Detection

`detect.py` filters and modifies boxes.

| API | Example |
| --- | --- |
| [`select`](api/select.md) | [Run](../../examples/postprocessing/select.py) |
| [`map_boxes`](api/map_boxes.md) | [Run](../../examples/postprocessing/map_boxes.py) |
| [`relabel`](api/relabel.md) | [Run](../../examples/postprocessing/relabel.py) |
| [`drop`](api/drop.md) | [Run](../../examples/postprocessing/drop.py) |
| [`filter_score`](api/filter_score.md) | [Run](../../examples/postprocessing/filter_score.py) |
| [`clip_boxes`](api/clip_boxes.md) | [Run](../../examples/postprocessing/clip_boxes.py) |
| [`remove_invalid_boxes`](api/remove_invalid_boxes.md) | [Run](../../examples/postprocessing/remove_invalid_boxes.py) |
| [`box_iou`](api/box_iou.md) | [Run](../../examples/postprocessing/box_iou.py) |
| [`suppress`](api/suppress.md) | [Run](../../examples/postprocessing/suppress.py) |
| [`top_boxes`](api/top_boxes.md) | [Run](../../examples/postprocessing/top_boxes.py) |
| [`threshold`](api/threshold.md) | [Run](../../examples/postprocessing/threshold.py) |
| [`clip`](api/clip.md) | [Run](../../examples/postprocessing/clip.py) |
| [`remove_invalid`](api/remove_invalid.md) | [Run](../../examples/postprocessing/remove_invalid.py) |
| [`nms`](api/nms.md) | [Run](../../examples/postprocessing/nms.py) |
| [`top`](api/top.md) | [Run](../../examples/postprocessing/top.py) |

Start with the [overview notebook](../../tutorials/postprocessing/Document_Postprocessing_Overview.ipynb), then open [Core](../../tutorials/postprocessing/Document_Postprocessing_Core.ipynb) or [Detect](../../tutorials/postprocessing/Document_Postprocessing_Detect.ipynb).
