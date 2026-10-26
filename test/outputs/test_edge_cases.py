"""Boundary and failure-path tests for all three output levels."""

import math
import random
import tempfile
import unittest

from PIL import Image

from klygo import files, outputs


def frame(source_id="source", frame_index=0, labels=("car", "person")):
    return outputs.Detection(
        [
            outputs.Box([index, index, index + 4, index + 5], label=label, class_id=index)
            for index, label in enumerate(labels)
        ],
        source_id=source_id,
        frame_index=frame_index,
        image_size=(100, 80),
        source=Image.new("RGB", (100, 80)),
    )


def assert_consistent(test, results):
    labels = {}
    for detection in results:
        test.assertEqual(len(detection.uids), len(set(detection.uids)))
        for box in detection:
            test.assertEqual(detection.names[box.class_id], box.label)
            test.assertEqual(results.names[box.class_id], box.label)
            if box.label in labels:
                test.assertEqual(labels[box.label], box.class_id)
            labels[box.label] = box.class_id
    test.assertEqual(set(results.names.values()), set(labels))


class TestBoxEdgeCases(unittest.TestCase):
    def test_rejects_non_finite_and_degenerate_values(self):
        for coordinates in (
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, math.inf, 1],
            [0, math.nan, 1, 2],
        ):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    outputs.Box(coordinates, label="car")
        for score in (-0.01, 1.01, math.inf, math.nan):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    outputs.Box([0, 0, 1, 1], label="car", score=score)

    def test_failed_mutations_leave_box_unchanged(self):
        item = outputs.Box([0, 0, 10, 10], label="car", score=.5)
        before = (item.box, item.score, item.track_id, item.metadata)
        for operation in (
            lambda: setattr(item, "box", [5, 5, 4, 8]),
            lambda: setattr(item, "score", 2),
            lambda: setattr(item, "track_id", True),
            lambda: item.translate(math.inf, 0),
            lambda: item.scale(0),
        ):
            with self.assertRaises((TypeError, ValueError)):
                operation()
            self.assertEqual((item.box, item.score, item.track_id, item.metadata), before)

    def test_constructor_copies_metadata(self):
        metadata = {"nested": {"value": 1}}
        item = outputs.Box([0, 0, 1, 1], label=" car ", metadata=metadata)
        metadata["nested"]["value"] = 2
        self.assertEqual(item.label, "car")
        self.assertEqual(item.metadata["nested"]["value"], 1)


class TestDetectionEdgeCases(unittest.TestCase):
    def test_same_box_twice_gets_distinct_uids(self):
        box = outputs.Box([0, 0, 3, 3], label="car")
        detection = outputs.Detection(
            [box, box], source_id="same", image_size=(10, 10)
        )
        self.assertEqual(detection.uids, [0, 1])
        self.assertIsNot(detection[0], detection[1])

    def test_box_owned_by_another_frame_is_cloned(self):
        first = frame("first")
        second = frame("second", labels=())
        second.append(first[0])
        second[0].score = .1
        self.assertEqual(first[0].score, 1.0)
        self.assertIsNot(first[0], second[0])

    def test_extend_and_filter_fail_atomically(self):
        detection = frame(); before = detection.to_dict()
        candidate = outputs.Box([10, 10, 20, 20], label="bus")
        with self.assertRaises(TypeError):
            detection.extend([candidate, object()])
        self.assertEqual(detection.to_dict(), before)
        with self.assertRaises(RuntimeError):
            detection.filter(
                lambda box: setattr(box, "score", .1) or (_ for _ in ()).throw(RuntimeError())
            )
        self.assertEqual(detection.to_dict(), before)

    def test_clear_does_not_reuse_old_uids(self):
        detection = frame(); previous = max(detection.uids)
        detection.clear()
        detection.add_box([0, 0, 2, 2], label="new")
        self.assertGreater(detection[0].uid, previous)


class EmptyCallModel:
    def __init__(self):
        self.called = False

    def predict(self, *_args, **_kwargs):
        self.called = True
        raise AssertionError("predict must not be called")


class TestDetectionsEdgeCases(unittest.TestCase):
    def test_child_frame_is_cloned_between_collections(self):
        shared = frame()
        first = outputs.Detections([shared])
        second = outputs.Detections([shared])
        second[0].drop("car")
        self.assertEqual(first[0].labels, ["car", "person"])
        self.assertEqual(second[0].labels, ["person"])

    def test_filter_failure_and_self_merge_are_atomic(self):
        results = outputs.Detections([frame(frame_index=0), frame(frame_index=1)])
        before = results.to_dict()
        def fail_after_mutation(item):
            item.drop("car")
            raise RuntimeError()
        with self.assertRaises(RuntimeError):
            results.filter(fail_after_mutation)
        self.assertEqual(results.to_dict(), before)
        with self.assertRaises(ValueError):
            results.merge(results)
        self.assertEqual(results.to_dict(), before)

    def test_empty_repredict_is_noop_and_boolean_index_is_rejected(self):
        results = outputs.Detections([frame()]); model = EmptyCallModel()
        before = results.to_dict()
        results.repredict(model, index=[])
        self.assertFalse(model.called)
        self.assertEqual(results.to_dict(), before)
        with self.assertRaises(TypeError):
            results.repredict(model, index=True)
        self.assertFalse(model.called)

    def test_invalid_insert_and_duplicate_extend_are_atomic(self):
        results = outputs.Detections([frame()]); before = results.to_dict()
        with self.assertRaises(TypeError):
            results.insert(0, object())
        with self.assertRaises(ValueError):
            results.extend([frame("new"), frame("new")])
        self.assertEqual(results.to_dict(), before)

    def test_empty_collection_exports_valid_annotations(self):
        with tempfile.TemporaryDirectory() as root:
            path = files.join(root, "empty")
            outputs.Detections().export(path)
            self.assertEqual(files.load(files.join(path, "data.yaml"), verbose=False), {"names": {}})
            self.assertEqual(files.load(files.join(path, "manifest.json"), verbose=False), [])

    def test_deterministic_random_mutations_keep_invariants(self):
        generator = random.Random(17)
        results = outputs.Detections(
            [frame("random", index, ("car", "person", "bus")) for index in range(8)]
        )
        labels = ["car", "person", "bus", "truck", "bicycle"]
        for step in range(250):
            detection = results[generator.randrange(len(results))]
            operation = generator.randrange(4)
            if operation == 0:
                start = generator.randrange(50)
                detection.add_box(
                    [start, start, start + 2, start + 3],
                    label=generator.choice(labels),
                )
            elif operation == 1 and detection:
                detection.remove_box(generator.choice(detection.uids))
            elif operation == 2:
                old, new = generator.sample(labels, 2)
                results.relabel({old: new})
            else:
                results.drop(generator.choice(labels))
            assert_consistent(self, results)


if __name__ == "__main__":
    unittest.main()
