# `postprocessing.nms`

```python
postprocessing.nms(iou=0.5, class_agnostic=False)
```

Creates a score-ordered non-maximum-suppression `Operation`.

## Returns

`Operation`.

## Errors

Raises `TypeError` for unsupported result or argument types. Numeric operations raise `ValueError` when a limit is outside its documented range. Mutation is validated before commit.

## Example

See [`nms.py`](../../../examples/postprocessing/nms.py).

```python
from klygo import outputs, postprocessing as post

frame = outputs.Detection(
    [
        outputs.Box([-2, 0, 30, 30], label="cat", score=0.9, class_id=0),
        outputs.Box([0, 0, 28, 28], label="cat", score=0.8, class_id=0),
        outputs.Box([32, 5, 48, 35], label="dog", score=0.4, class_id=1),
    ],
    source_id="example",
    image_size=(50, 40),
    names={0: "cat", 1: "dog"},
)

operation = post.nms(iou=0.5)
operation(frame)
print(frame.labels)
```

## Tests

See [`test_postprocessing.py`](../../../test/postprocessing/test_postprocessing.py).
