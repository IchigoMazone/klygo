"""Behavioral tests for the public detection result hierarchy."""

import unittest

from PIL import Image

from klygo import outputs


class TestDetectOutputs(unittest.TestCase):
    def test_build_detections_from_normalized_predictions(self):
        images = [Image.new("RGB", (40, 30), "white") for _ in range(2)]
        predictions = [
            {
                "boxes": [[1, 2, 10, 12], [4, 5, 20, 22]],
                "scores": [0.91234, 0.8],
                "labels": ["cat", "dog"],
            },
            {
                "boxes": [[2, 3, 11, 13]],
                "scores": [0.75],
                "labels": ["cat"],
            },
        ]

        frames = outputs.build_detections(
            images,
            predictions,
            metadata={"prompt": ["cat", "dog"]},
        )

        self.assertEqual([box.id for frame in frames for box in frame], [0, 1, 0])
        self.assertEqual(frames[0][0].score, 0.912)
        self.assertEqual(frames[0].metadata["prompt"], ["cat", "dog"])
        self.assertIs(frames[0][0].parent_image, images[0])

    def test_box_ids_and_class_ids_have_distinct_scopes(self):
        image = Image.new("RGB", (40, 30), "white")
        frames = outputs.build_detections(
            [image],
            [{
                "boxes": [[1, 2, 10, 12], [4, 5, 20, 22]],
                "scores": [0.9, 0.8],
                "labels": ["car", "car"],
                "class_ids": [7, 7],
            }],
            names={7: "car"},
        )

        self.assertEqual(frames[0].ids, [0, 1])
        self.assertEqual(frames[0].class_ids, [7, 7])
        self.assertEqual(frames[0].names, {7: "car"})
        self.assertEqual(frames[0][0].metadata["class_id"], 7)

    def test_result_names_include_prompt_and_generated_labels(self):
        images = [Image.new("RGB", (20, 20), "white") for _ in range(2)]
        frames = outputs.build_detections(
            images,
            [
                {
                    "boxes": [[0, 0, 5, 5], [5, 5, 10, 10], [10, 10, 15, 15]],
                    "scores": [0.9, 0.8, 0.7],
                    "labels": ["person", "person riding bicycle", "   "],
                },
                {
                    "boxes": [[0, 0, 5, 5], [5, 5, 10, 10]],
                    "scores": [0.6, 0.5],
                    "labels": ["car", "person riding bicycle"],
                },
            ],
            names={0: "person", 1: "car"},
        )
        results = outputs.detect.Detections(frames, names={0: "person", 1: "car"})

        self.assertEqual(
            results.names,
            {0: "person", 1: "car", 2: "person riding bicycle", 3: None},
        )
        self.assertEqual(frames[0].names, {0: "person", 2: "person riding bicycle", 3: None})
        self.assertEqual(frames[1].names, {1: "car", 2: "person riding bicycle"})
        self.assertEqual(frames[0].crops.names, frames[0].names)
        self.assertIsNone(frames[0][2].label)

    def test_names_reject_conflicting_explicit_class_ids(self):
        image = Image.new("RGB", (20, 20), "white")
        with self.assertRaisesRegex(ValueError, "maps to both"):
            outputs.build_detections(
                [image],
                [{
                    "boxes": [[0, 0, 5, 5]],
                    "scores": [0.9],
                    "labels": ["dog"],
                    "class_ids": [0],
                }],
                names={0: "cat"},
            )

    def test_text_labels_take_precedence_for_grounded_outputs(self):
        image = Image.new("RGB", (20, 20), "white")
        frame = outputs.build_detections(
            [image],
            [{
                "boxes": [[0, 0, 5, 5]],
                "scores": [0.9],
                "labels": [99],
                "text_labels": ["person riding bicycle"],
            }],
            names={0: "person"},
        )[0]

        self.assertEqual(frame[0].label, "person riding bicycle")
        self.assertEqual(frame[0].class_id, 1)

    def test_legacy_metadata_promotes_class_id(self):
        box = outputs.detect.Box(
            4,
            "cat",
            0.9,
            [0, 0, 5, 5],
            metadata={"class_id": 12},
        )
        self.assertEqual(box.id, 4)
        self.assertEqual(box.class_id, 12)
        self.assertEqual(box.to_dict()["class_id"], 12)

    def test_stream_names_grow_only_as_frames_are_consumed(self):
        image = Image.new("RGB", (20, 20), "white")

        def frames():
            yield outputs.build_detections(
                [image],
                [{"boxes": [[0, 0, 5, 5]], "scores": [0.9], "labels": ["combined label"]}],
                names={0: "person"},
            )[0]

        results = outputs.detect.Detections(
            frames(),
            stream=True,
            names={0: "person"},
        )

        self.assertEqual(results.names, {0: "person"})
        self.assertEqual(results[0].names, {1: "combined label"})
        self.assertEqual(results.names, {0: "person", 1: "combined label"})

    def test_build_detections_validates_aligned_inputs(self):
        image = Image.new("RGB", (40, 30), "white")
        with self.assertRaises(ValueError):
            outputs.build_detections([image], [])
        with self.assertRaises(ValueError):
            outputs.build_detections(
                [image],
                [{"boxes": [[1, 2, 3, 4]], "scores": [], "labels": ["cat"]}],
            )

    def test_build_detections_copies_nested_metadata_per_frame(self):
        images = [Image.new("RGB", (20, 20), "white") for _ in range(2)]
        predictions = [
            {"boxes": [], "scores": [], "labels": []},
            {"boxes": [], "scores": [], "labels": []},
        ]
        metadata = {"prompt": ["cat"], "post": {"threshold": 0.4}}

        frames = outputs.build_detections(images, predictions, metadata=metadata)
        frames[0].metadata["prompt"].append("dog")
        frames[0].metadata["post"]["threshold"] = 0.8

        self.assertEqual(frames[1].metadata, metadata)
        self.assertEqual(metadata, {"prompt": ["cat"], "post": {"threshold": 0.4}})

    def test_box_crop_and_immutable_transform(self):
        source = Image.new("RGB", (40, 30), "white")
        box = outputs.detect.Box(0, "cat", 0.9, [5, 6, 20, 16], source)
        self.assertEqual(box.image.size, (15, 10))
        moved = box.translate(dx=3)
        self.assertEqual(box.box, [5.0, 6.0, 20.0, 16.0])
        self.assertEqual(moved.box, [8.0, 6.0, 23.0, 16.0])
        with self.assertRaises(ValueError):
            box.with_box([1, 2, 3])

    def test_detection_hierarchy(self):
        source = Image.new("RGB", (40, 30), "white")
        box = outputs.detect.Box(0, "cat", 0.9, [5, 6, 20, 16])
        frame = outputs.detect.Detection(source, [box])
        results = outputs.detect.Detections([frame], source_type="image")
        self.assertIs(results[0], frame)
        self.assertIs(frame[0], box)
        self.assertEqual(results.total_frames, 1)

    def test_aliases_resolve_to_canonical_classes(self):
        self.assertIs(outputs.detect.Boxes, outputs.detect.Crops)
        self.assertIs(outputs.detect.DetectionResult, outputs.detect.Detection)

    def test_detections_class_ids_and_frame_deletion_sync(self):
        source = Image.new("RGB", (40, 30), "white")
        box1 = outputs.detect.Box(0, "car", 0.9, [0, 0, 10, 10], class_id=1)
        box2 = outputs.detect.Box(1, "person", 0.8, [10, 10, 20, 20], class_id=0)
        f1 = outputs.detect.Detection(source, [box1], names={0: "person", 1: "car"}, image_frame_index=0)
        f2 = outputs.detect.Detection(source, [box2], names={0: "person", 1: "car"}, image_frame_index=1)
        results = outputs.detect.Detections([f1, f2], names={0: "person", 1: "car"})

        self.assertEqual(results.class_ids, [[1], [0]])
        self.assertEqual(results.unique_class_ids, [1, 0])
        self.assertEqual(results.active_names, {0: "person", 1: "car"})
        self.assertEqual(results.unique_labels, ["car", "person"])

        # Delete frame 0 containing 'car' (class_id=1)
        del results[0]

        self.assertEqual(results.class_ids, [0])
        self.assertEqual(results.unique_class_ids, [0])
        self.assertEqual(results.active_names, {0: "person"})
        self.assertEqual(results.unique_labels, ["person"])
        self.assertEqual(results.names, {0: "person", 1: "car"})
        self.assertEqual(results[0][0].class_id, 0)
        self.assertEqual(results[0][0].label, "person")

    def test_detections_reindex_and_remove_frame_with_reindex(self):
        source = Image.new("RGB", (40, 30), "white")
        f0 = outputs.detect.Detection(source, [outputs.detect.Box(0, "car", 0.9, [0, 0, 10, 10])], image_frame_index=0)
        f10 = outputs.detect.Detection(source, [outputs.detect.Box(0, "person", 0.8, [10, 10, 20, 20])], image_frame_index=10)
        f20 = outputs.detect.Detection(source, [outputs.detect.Box(0, "dog", 0.7, [20, 20, 30, 30])], image_frame_index=20)
        results = outputs.detect.Detections([f0, f10, f20])

        # Test remove_frame with reindex=True
        cleaned = results.remove_frame(0, reindex=True)
        self.assertEqual(cleaned[0].frame_index, 0)
        self.assertEqual(cleaned[1].frame_index, 1)

        # Test explicit reindex with custom start
        results.reindex(start=5)
        self.assertEqual(results[0].frame_index, 5)
        self.assertEqual(results[1].frame_index, 6)
        self.assertEqual(results[2].frame_index, 7)

    def test_box_reindexing_and_drop_with_reindex(self):
        source = Image.new("RGB", (40, 30), "white")
        b0 = outputs.detect.Box(0, "car", 0.9, [0, 0, 10, 10], class_id=1)
        b1 = outputs.detect.Box(1, "person", 0.8, [10, 10, 20, 20], class_id=0)
        b2 = outputs.detect.Box(2, "dog", 0.7, [20, 20, 30, 30], class_id=2)
        frame = outputs.detect.Detection(source, [b0, b1, b2])

        # Drop 'car' with reindex=True -> person becomes id 0, dog becomes id 1
        dropped = frame.drop(labels=["car"], reindex=True)
        self.assertEqual(len(dropped), 2)
        self.assertEqual(dropped[0].id, 0)
        self.assertEqual(dropped[0].label, "person")
        self.assertEqual(dropped[1].id, 1)
        self.assertEqual(dropped[1].label, "dog")

        # In-place box reindex on Detection
        del frame[0]
        frame.reindex(start=10)
        self.assertEqual(frame[0].id, 10)
        self.assertEqual(frame[1].id, 11)

        # Batch box reindex across Detections
        results = outputs.detect.Detections([frame])
        results.reindex_boxes(start=0)
        self.assertEqual(results[0][0].id, 0)
        self.assertEqual(results[0][1].id, 1)
