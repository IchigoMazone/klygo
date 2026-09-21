from PIL import Image

from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection, Detections


def _detection(frame_index=0):
    image = Image.new("RGB", (100, 80), "white")
    return Detection(
        image,
        [
            Box(0, "vehicle", 0.95, [-5, -5, 50, 50], image),
            Box(1, "vehicle", 0.80, [0, 0, 48, 48], image),
            Box(2, "person", 0.60, [60, 10, 90, 70], image),
            Box(3, "invalid", 0.99, [20, 20, 10, 30], image),
        ],
        image_frame_index=frame_index,
    )


def test_box_copy_operations():
    box = _detection()[0].with_metadata(reviewed=True)
    changed = box.with_label("car").with_score(0.9).translate(2, 3).scale(1.2).clip((100, 80))
    assert changed.uid == box.uid
    assert changed.metadata["reviewed"] is True
    assert changed.label == "car"
    assert changed.score == 0.9
    assert box.label == "vehicle"


def test_detection_pythonic_operations():
    result = _detection()

    selected = result.select(ids=[0, 2])
    assert [box.id for box in selected] == [0, 2]

    renamed = result.relabel("car", ids=[0, 1])
    assert renamed.labels == ["car", "car", "person", "invalid"]
    assert result.labels[0] == "vehicle"

    mapped = result.relabel({0: "truck", 2: "pedestrian"})
    assert mapped.labels == ["truck", "vehicle", "pedestrian", "invalid"]

    dropped = result.drop(where=lambda box: box.score < 0.7)
    assert dropped.labels == ["vehicle", "vehicle", "invalid"]

    transformed = result.map_boxes(
        lambda box: box.with_label("human") if box.label == "person" else box
    )
    assert "human" in transformed.labels

    cleaned = result.clip().remove_invalid(min_area=10).filter_score(0.7).nms(iou=0.5).top(1)
    assert len(cleaned) == 1
    assert cleaned[0].id == 0
    assert cleaned[0].box == [0.0, 0.0, 50.0, 50.0]


def test_pipeline_and_frame_operations():
    pipeline = post.compose(
        post.clip(),
        post.remove_invalid(min_area=10),
        post.threshold(0.7),
        post.nms(iou=0.5),
        post.top(1),
    )

    first = _detection(frame_index=10)
    second = _detection(frame_index=20)
    results = Detections([first, second], source_type="video", fps=25)
    cleaned = pipeline(results)
    assert [len(frame) for frame in cleaned] == [1, 1]
    assert len(post.compose([post.threshold(0.5), post.top(2)])) == 2

    assert results.get_frame(20) is second
    replacement = second.relabel("car")
    replaced = results.replace_frame(20, replacement)
    assert replaced.get_frame(20).labels == ["car"] * 4
    assert results.get_frame(20).labels[0] == "vehicle"

    updated = results.update_frame(10, lambda frame: frame.drop(labels=["invalid"]))
    assert "invalid" not in updated.get_frame(10).labels
    assert [frame.frame_index for frame in results.remove_frame(10)] == [20]


def test_stream_stays_lazy():
    consumed = []

    def source():
        for index in (0, 1, 2):
            consumed.append(index)
            yield _detection(frame_index=index)

    results = Detections(source(), stream=True, total_frames=3)
    processed = results.filter_score(0.9).relabel("object")
    assert consumed == []
    assert processed[0].labels == ["object", "object"]
    assert consumed == [0]


if __name__ == "__main__":
    test_box_copy_operations()
    test_detection_pythonic_operations()
    test_pipeline_and_frame_operations()
    test_stream_stays_lazy()
    print("ALL KLYGO.POSTPROCESSING TESTS PASSED SUCCESSFULLY!")
