"""Stress tests for label consistency across the three output levels."""

import unittest

from klygo import outputs


def make_frame(source_id: str, frame_index: int, labels: list[str]) -> outputs.Detection:
    boxes = [
        outputs.Box(
            [index, index, index + 10, index + 12],
            label=label,
            class_id=index % 7,
        )
        for index, label in enumerate(labels)
    ]
    return outputs.Detection(
        boxes,
        source_id=source_id,
        frame_index=frame_index,
        image_size=(1000, 1000),
    )


def assert_frame_consistent(test: unittest.TestCase, frame: outputs.Detection) -> None:
    test.assertEqual(len(frame.uids), len(set(frame.uids)))
    active = {}
    reverse = {}
    for box in frame:
        test.assertEqual(frame.names[box.class_id], box.label)
        active[box.class_id] = box.label
        if box.label in reverse:
            test.assertEqual(reverse[box.label], box.class_id)
        reverse[box.label] = box.class_id
    test.assertEqual(frame.names, active)


def assert_collection_consistent(
    test: unittest.TestCase, results: outputs.Detections
) -> None:
    identities = [frame.identity for frame in results]
    test.assertEqual(len(identities), len(set(identities)))
    reverse = {}
    active = {}
    for frame in results:
        assert_frame_consistent(test, frame)
        for box in frame:
            test.assertEqual(results.names[box.class_id], box.label)
            active[box.class_id] = box.label
            if box.label in reverse:
                test.assertEqual(reverse[box.label], box.class_id)
            reverse[box.label] = box.class_id
    test.assertEqual(results.names, active)


class TestLabelMutations(unittest.TestCase):
    def test_box_edits_preserve_identity_and_label(self):
        frame = make_frame("single", 0, ["car"])
        box = frame[0]
        uid, label, class_id = box.uid, box.label, box.class_id
        for step in range(100):
            box.score = (step % 100) / 100
            box.track_id = step
            box.metadata = {"revision": step}
            box.translate(1, 1)
        self.assertEqual((box.uid, box.label, box.class_id), (uid, label, class_id))
        assert_frame_consistent(self, frame)

    def test_detection_many_add_relabel_replace_and_delete_operations(self):
        frame = make_frame("one", 0, ["car", "person", "bus"] * 40)
        original_uids = set(frame.uids)

        frame.relabel({"car": "vehicle", "bus": "vehicle"})
        for index in range(50):
            frame.add_box([index, 200, index + 5, 210], label="bicycle")
        for index in range(0, 30, 3):
            frame.replace(index, outputs.Box([0, 0, 5, 5], label="truck"))
        frame.drop("person")
        for uid in list(frame.uids)[::4]:
            frame.remove_box(uid)

        assert_frame_consistent(self, frame)
        self.assertTrue(original_uids & set(frame.uids))
        self.assertTrue({"vehicle", "bicycle", "truck"}.issuperset(frame.unique_labels))
        self.assertGreater(max(frame.uids), 119)

    def test_collection_reconciles_direct_child_mutations(self):
        results = outputs.Detections(
            [
                make_frame("video", index, ["car", "person", "bus"] * 20)
                for index in range(10)
            ]
        )
        assert_collection_consistent(self, results)

        results[0].relabel({"car": "person"})
        results[1].add_box([1, 1, 4, 4], label="bicycle")
        results[2].replace(0, outputs.Box([2, 2, 8, 8], label="truck"))
        results[3].drop("bus")
        results[4].remove_box(results[4].uids[0])
        assert_collection_consistent(self, results)

        person_ids = {
            box.class_id
            for frame in results
            for box in frame
            if box.label == "person"
        }
        self.assertEqual(len(person_ids), 1)

    def test_collection_many_global_label_changes_remove_inactive_names(self):
        results = outputs.Detections(
            [make_frame("set", index, [f"class-{index % 8}"] * 25) for index in range(40)]
        )
        for cycle in range(20):
            results.relabel({f"class-{cycle % 8}": f"group-{cycle % 4}"})
            if cycle % 3 == 0:
                results.drop(f"group-{cycle % 4}")
            results[cycle % len(results)].add_box(
                [cycle, cycle, cycle + 4, cycle + 4],
                label=f"new-{cycle % 5}",
            )
            assert_collection_consistent(self, results)

        active_labels = {box.label for frame in results for box in frame}
        self.assertEqual(set(results.names.values()), active_labels)

    def test_collection_frame_crud_reconciles_conflicting_class_ids(self):
        results = outputs.Detections([make_frame("crud", 0, ["car", "person"])])

        results.append(make_frame("crud", 1, ["person", "bicycle", "car"]))
        results.insert(1, make_frame("inserted", 9, ["truck", "person"]))
        assert_collection_consistent(self, results)

        original_identity = results[0].identity
        results.replace(0, make_frame("replacement", 99, ["truck", "bus", "bus"]))
        self.assertEqual(results[0].identity, original_identity)
        assert_collection_consistent(self, results)

        incoming = outputs.Detections(
            [make_frame("merged", index, ["bus", "motorcycle", "person"]) for index in range(5)]
        )
        results.merge(incoming)
        assert_collection_consistent(self, results)

        while len(results) > 1:
            results.pop()
            assert_collection_consistent(self, results)
        results.clear()
        self.assertEqual(results.names, {})
        self.assertEqual(results.total_objects, 0)

    def test_failed_bulk_relabel_leaves_all_labels_unchanged(self):
        results = outputs.Detections([make_frame("atomic", 0, ["a", "b", "c"] * 20)])
        before = results.to_dict()
        with self.assertRaises(ValueError):
            results.relabel({"a": ""})
        self.assertEqual(results.to_dict(), before)
        assert_collection_consistent(self, results)

    def test_ten_thousand_boxes_keep_one_class_id_per_label(self):
        labels = [f"class-{index % 10}" for index in range(100)]
        results = outputs.Detections(
            [make_frame("large", frame_index, labels) for frame_index in range(100)]
        )
        results.relabel({"class-0": "vehicle"})
        results.drop("class-1")
        results[0].add_box([500, 500, 510, 510], label="new-class")

        self.assertEqual(results.total_objects, 9001)
        assert_collection_consistent(self, results)


if __name__ == "__main__":
    unittest.main()
