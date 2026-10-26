# `postprocessing.select`

```python
postprocessing.select(*, where=None, labels=None, class_ids=None, uids=None)
```

Builds an operation that keeps boxes matching every supplied selector.
`labels`, `class_ids`, and `uids` accept either one scalar selector or an iterable.

## Returns

`Operation` — apply it with `result.transform(...)` or include it in `postprocessing.compose(...)`.

## Errors

Raises `TypeError` for unsupported result or argument types. Numeric operations raise `ValueError` when a limit is outside its documented range. Mutation is validated before commit.

## Example

See [`select.py`](../../../examples/postprocessing/select.py).

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

frame.transform(post.select(labels="cat", where=lambda box: box.score >= 0.5))
print(frame.labels)
```

## Tests

See [`test_postprocessing.py`](../../../test/postprocessing/test_postprocessing.py).
