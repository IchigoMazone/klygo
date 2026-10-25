# `postprocessing.clip_boxes`

```python
postprocessing.clip_boxes(result, image_size=None)
```

Clamps coordinates to an image boundary and removes boxes wholly outside it.

## Returns

`None`.

## Errors

Raises `TypeError` for unsupported result or argument types. Numeric operations raise `ValueError` when a limit is outside its documented range. Mutation is validated before commit.

## Example

See [`clip_boxes.py`](../../../examples/postprocessing/clip_boxes.py).

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

post.clip_boxes(frame)
print(frame.labels)
```

## Tests

See [`test_postprocessing.py`](../../../test/postprocessing/test_postprocessing.py).
