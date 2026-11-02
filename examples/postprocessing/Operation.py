"""Executable example for klygo.postprocessing.Operation."""

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

operation = post.Operation("keep-confident", post.threshold(0.5))
operation(frame)
print(frame.labels)
