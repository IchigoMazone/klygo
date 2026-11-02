# `postprocessing.clip`

```python
postprocessing.clip(image_size=None)
```

Creates an `Operation` that clips coordinates to the image boundary.

## Returns

`Operation`.

## Errors

Raises `TypeError` for unsupported result or argument types. Numeric operations raise `ValueError` when a limit is outside its documented range. Mutation is validated before commit.

## Example

See [`clip.py`](../../../examples/postprocessing/clip.py).

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

operation = post.clip()
operation(frame)
print(frame.labels)
```

## Tests

See [`test_postprocessing.py`](../../../test/postprocessing/test_postprocessing.py).
