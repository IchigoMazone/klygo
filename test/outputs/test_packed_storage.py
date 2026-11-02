"""Compact shared-array detection storage tests."""

import gc
import random
import tempfile
import unittest

from klygo import files, outputs, postprocessing


def frame(index, boxes=2):
    items = [
        outputs.Box(
            [position, position, position + 10, position + 10],
            label="car" if position % 2 else "person",
            score=0.9 - position * 0.1,
        )
        for position in range(boxes)
    ]
    return outputs.Detection(
        items,
        source_id="packed-source",
        frame_index=index,
        image_size=(100, 100),
        metadata={"index": index},
    )


class TestPackedStorage(unittest.TestCase):
    def test_lazy_frame_facades_and_empty_frames(self):
        results = outputs.Detections(
            [frame(0), frame(1, 0), frame(2, 3)],
            packed=True,
        )
        gc.collect()

        self.assertTrue(results.is_packed)
        self.assertEqual(len(results), 3)
        self.assertEqual(results.total_objects, 5)
        self.assertEqual(len(results._packed_store.cache), 0)
        first = results[0]
        self.assertIs(first, results[0])
        self.assertEqual(first.metadata, {"index": 0})
        self.assertEqual(len(results[1]), 0)
        self.assertLessEqual(len(results._packed_store.cache), 2)
        self.assertIsNone(results._packed_store.identity_positions)
        self.assertIs(results.get_frame(2, source_id="packed-source"), results[2])
        self.assertIsNotNone(results._packed_store.identity_positions)

    def test_direct_edits_survive_facade_collection_and_gc(self):
        results = outputs.Detections([frame(0), frame(1)], packed=True)
        edited = results[0]
        uid = edited[0].uid
        edited[0].score = 0.42
        edited.append(outputs.Box([20, 20, 30, 30], label="bus", score=0.8))
        del edited
        gc.collect()

        self.assertAlmostEqual(results[0].get_box(uid).score, 0.42)
        self.assertEqual(results[0].labels[-1], "bus")
        self.assertIn("bus", results.names.values())
        self.assertIn(("packed-source", 0), results.modified_frames)

    def test_structural_crud_materializes_without_losing_data(self):
        results = outputs.Detections([frame(0), frame(1)], packed=True)
        results.append(frame(2, 1))
        self.assertFalse(results.is_packed)
        self.assertEqual([item.frame_index for item in results], [0, 1, 2])
        removed = results.pop(1)
        self.assertEqual(removed.frame_index, 1)
        results.replace(0, frame(9, 1))
        self.assertEqual(results[0].identity, ("packed-source", 0))
        self.assertEqual(len(results[0]), 1)

    def test_pack_and_unpack_existing_collection(self):
        results = outputs.Detections([frame(0), frame(1)])
        linked = results[0]
        self.assertIs(results.pack(), results)
        self.assertTrue(results.is_packed)
        self.assertEqual(results.history[-1]["operation"], "pack")
        linked[0].score = 0.33
        self.assertIs(results[0], linked)
        self.assertAlmostEqual(results[0][0].score, 0.33)
        self.assertIs(results.unpack(), results)
        self.assertFalse(results.is_packed)
        self.assertEqual(results.history[-1]["operation"], "unpack")

    def test_copy_filter_export_and_storage_conversion(self):
        results = outputs.Detections([frame(0), frame(1)], packed=True).astype(32)
        copied = results.copy()
        self.assertTrue(copied.is_packed)
        copied[0][0].score = 0.1
        self.assertNotEqual(copied[0][0].score, results[0][0].score)

        postprocessing.threshold(0.85)(results)
        self.assertTrue(results.is_packed)
        self.assertEqual([len(item) for item in results], [1, 1])
        with tempfile.TemporaryDirectory() as root:
            destination = files.join(root, "labels")
            results.export(destination, format="yolo")
            self.assertTrue(files.exists(files.join(destination, "data.yaml")))

    def test_collection_label_edits_use_compact_position_index(self):
        results = outputs.Detections([frame(index, 3) for index in range(20)], packed=True)
        self.assertIsNone(results._packed_store.identity_positions)
        results.drop("person").relabel({"car": "vehicle"})

        self.assertTrue(results.is_packed)
        self.assertIsNone(results._packed_store.identity_positions)
        self.assertEqual(results.names, {1: "vehicle"})
        self.assertTrue(all(item.labels == ["vehicle"] for item in results))

    def test_random_edits_match_ordinary_storage(self):
        randomizer = random.Random(19)
        source = []
        labels = ("person", "car", "bus")
        for frame_index in range(30):
            boxes = [
                outputs.Box(
                    [position, position, position + 5, position + 6],
                    label=randomizer.choice(labels),
                    score=randomizer.random(),
                )
                for position in range(randomizer.randrange(5))
            ]
            source.append(outputs.Detection(
                boxes,
                source_id="random",
                frame_index=frame_index,
                image_size=(50, 50),
            ))

        ordinary = outputs.Detections([item._clone() for item in source])
        packed = outputs.Detections([item._clone() for item in source], packed=True)
        for position in (0, 7, 12, 21):
            if len(ordinary[position]):
                score = randomizer.random()
                ordinary[position][0].score = score
                packed[position][0].score = score
            extra = outputs.Box([20, 20, 25, 25], label="truck", score=0.75)
            ordinary[position].append(extra)
            packed[position].append(extra._clone())

        ordinary.drop("bus").relabel({"car": "vehicle"}).compact()
        packed.drop("bus").relabel({"car": "vehicle"}).compact()
        self.assertEqual(ordinary.names, packed.names)
        self.assertEqual(
            [item.to_dict() for item in ordinary],
            [item.to_dict() for item in packed],
        )


if __name__ == "__main__":
    unittest.main()
