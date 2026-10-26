"""Multi-class scale and stress tests for Box, Detection, and Detections.

Tests scenarios with 80 (COCO) up to 1,000 (LVIS/OpenImages) distinct classes,
including hierarchical many-to-one relabeling, cyclic relabeling, massive drops,
sparse ID compaction, disjoint collection merges, and YOLO export fidelity.
"""

import math
import random
import tempfile
import unittest
from PIL import Image
import numpy as np

from klygo import files, outputs, postprocessing

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def make_multi_class_frame(source_id="cam-0", frame_index=0, classes_subset=COCO_80, boxes_per_frame=10, rng=None):
    if rng is None:
        rng = random.Random(42 + frame_index)
    chosen_labels = [rng.choice(classes_subset) for _ in range(boxes_per_frame)]
    boxes = [
        outputs.Box(
            [i * 10, i * 10 + 5, i * 10 + 100, i * 10 + 120],
            label=lbl,
            score=round(rng.uniform(0.2, 0.98), 3),
            class_id=classes_subset.index(lbl) if lbl in classes_subset else None,
        )
        for i, lbl in enumerate(chosen_labels)
    ]
    return outputs.Detection(
        boxes,
        source_id=source_id,
        frame_index=frame_index,
        image_size=(1920, 1080),
        names={idx: name for idx, name in enumerate(classes_subset)},
    )


class TestManyClassesScaleAndStress(unittest.TestCase):
    """Stress testing the outputs subsystem with 80 to 1,000 distinct classes."""

    def test_thousand_classes_scale(self):
        """Scale test: 100 frames with 1,000 distinct classes (LVIS scale) and 5,000 boxes."""
        num_classes = 1000
        class_pool = [f"class_{i:04d}" for i in range(num_classes)]
        rng = random.Random(1337)

        frames = []
        for f_idx in range(100):
            # 50 boxes per frame = 5,000 total boxes
            chosen = [rng.choice(class_pool) for _ in range(50)]
            boxes = [
                outputs.Box(
                    [b * 5, b * 5, b * 5 + 40, b * 5 + 40],
                    label=lbl,
                    score=round(rng.uniform(0.1, 0.99), 3),
                )
                for b, lbl in enumerate(chosen)
            ]
            frames.append(outputs.Detection(boxes, source_id="stream", frame_index=f_idx, image_size=(1000, 1000)))

        dets = outputs.Detections(frames)
        self.assertEqual(len(dets), 100)
        self.assertEqual(dets.total_objects, 5000)

        # Verify registry integrity
        active_labels = {lbl for f in dets for lbl in f.labels}
        self.assertEqual(set(dets.names.values()), active_labels)

        # Verify class_ids in every frame correspond 100% to global names
        for f in dets:
            self.assertEqual(len(f.class_ids), len(f.labels))
            for cid, lbl in zip(f.class_ids, f.labels):
                self.assertEqual(dets.names[cid], lbl)

    def test_coco_hierarchical_many_to_one_relabel(self):
        """Relabel 80 COCO classes down to 10 supercategories."""
        rng = random.Random(999)
        frames = [make_multi_class_frame("coco", i, COCO_80, boxes_per_frame=20, rng=rng) for i in range(25)]
        dets = outputs.Detections(frames)

        # Map 80 classes to 10 supercategories
        mapping = {}
        for c in COCO_80:
            if c in {"person"}:
                mapping[c] = "human"
            elif c in {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}:
                mapping[c] = "vehicle"
            elif c in {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}:
                mapping[c] = "animal"
            elif c in {"backpack", "umbrella", "handbag", "tie", "suitcase"}:
                mapping[c] = "accessory"
            elif c in {"frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"}:
                mapping[c] = "sports"
            elif c in {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"}:
                mapping[c] = "kitchenware"
            elif c in {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}:
                mapping[c] = "food"
            elif c in {"chair", "couch", "potted plant", "bed", "dining table", "toilet"}:
                mapping[c] = "furniture"
            elif c in {"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"}:
                mapping[c] = "electronics"
            else:
                mapping[c] = "indoor_object"

        dets.relabel(mapping)

        # Invariant checks:
        # 1. Total objects must remain identical
        self.assertEqual(dets.total_objects, 25 * 20)
        # 2. Number of active classes must be at most 10
        self.assertLessEqual(len(dets.names), 10)
        # 3. None of the original granular classes can exist
        for old_c in ("cat", "dog", "pizza", "laptop", "car"):
            self.assertNotIn(old_c, dets.names.values())
        # 4. Check each frame's class_ids and boxes
        for f in dets:
            for b in f:
                self.assertIn(b.label, mapping.values())
                self.assertEqual(dets.names[b.class_id], b.label)

    def test_cyclic_class_swaps(self):
        """Cyclic class swapping: swap 'cat' <-> 'dog' without loss."""
        boxes = [
            outputs.Box([10, 10, 50, 50], label="cat", score=0.9),
            outputs.Box([60, 60, 100, 100], label="dog", score=0.8),
            outputs.Box([110, 110, 150, 150], label="cat", score=0.7),
        ]
        det = outputs.Detection(boxes, source_id="swap", image_size=(500, 500))
        self.assertEqual(det.labels, ["cat", "dog", "cat"])

        # Swap cat and dog
        det.relabel({"cat": "dog", "dog": "cat"})
        self.assertEqual(det.labels, ["dog", "cat", "dog"])
        self.assertEqual(det[0].label, "dog")
        self.assertEqual(det[1].label, "cat")
        self.assertEqual(det[2].label, "dog")

    def test_massive_drop_and_compaction(self):
        """Drop 50 of 80 classes, then compact sparse class IDs to 0..N-1."""
        rng = random.Random(123)
        frames = [make_multi_class_frame("drop_test", i, COCO_80, boxes_per_frame=40, rng=rng) for i in range(10)]
        dets = outputs.Detections(frames)

        classes_to_drop = COCO_80[:50]
        remaining_classes = set(COCO_80[50:])

        surviving_expected = sum(
            1 for f in dets for lbl in f.labels if lbl in remaining_classes
        )

        # Drop 50 classes
        dets.drop(classes_to_drop)
        self.assertEqual(dets.total_objects, surviving_expected)

        for f in dets:
            for lbl in f.labels:
                self.assertNotIn(lbl, classes_to_drop)
                self.assertIn(lbl, remaining_classes)

        # Class IDs may currently have gaps; now compact them
        dets.compact()

        # After compact: class IDs must be strictly 0, 1, ..., num_active - 1
        num_active = len(dets.names)
        self.assertEqual(sorted(dets.names.keys()), list(range(num_active)))

        for f in dets:
            for b in f:
                self.assertGreaterEqual(b.class_id, 0)
                self.assertLess(b.class_id, num_active)
                self.assertEqual(dets.names[b.class_id], b.label)

    def test_disjoint_class_collections_merge(self):
        """Merging two collections with conflicting class IDs (0..9) but completely different labels."""
        # Collection A: classes 0..9 are animals
        animals = ["ant", "bear", "cat", "dog", "elephant", "fox", "goat", "horse", "iguana", "jaguar"]
        frames_a = [
            outputs.Detection(
                [outputs.Box([1, 1, 20, 20], label=animals[i % len(animals)], class_id=i % len(animals))],
                source_id="coll_a",
                frame_index=i,
                image_size=(100, 100),
                names=dict(enumerate(animals)),
            )
            for i in range(10)
        ]
        dets_a = outputs.Detections(frames_a)

        # Collection B: classes 0..9 are vehicles (conflicting integer IDs with Collection A)
        vehicles = ["airplane", "bicycle", "car", "drone", "electric_scooter", "ferry", "glider", "helicopter", "icebreaker", "jet"]
        frames_b = [
            outputs.Detection(
                [outputs.Box([2, 2, 30, 30], label=vehicles[i % len(vehicles)], class_id=i % len(vehicles))],
                source_id="coll_b",
                frame_index=i,
                image_size=(100, 100),
                names=dict(enumerate(vehicles)),
            )
            for i in range(10)
        ]
        dets_b = outputs.Detections(frames_b)

        # Merge B into A
        dets_a.merge(dets_b)

        # Must have 20 frames and 20 distinct classes
        self.assertEqual(len(dets_a), 20)
        self.assertEqual(dets_a.total_objects, 20)
        self.assertEqual(len(dets_a.names), 20)

        # All 20 labels must exist in names
        all_expected_labels = set(animals + vehicles)
        self.assertEqual(set(dets_a.names.values()), all_expected_labels)

        # Every frame's box must map accurately
        for f in dets_a:
            for b in f:
                self.assertEqual(dets_a.names[b.class_id], b.label)

    def test_incremental_dynamic_new_classes(self):
        """Adding new boxes with completely unseen class labels dynamically updates collection registry."""
        f0 = outputs.Detection(
            [outputs.Box([0, 0, 10, 10], label="init_class", score=0.9)],
            source_id="dyn",
            frame_index=0,
            image_size=(100, 100),
        )
        dets = outputs.Detections([f0])
        self.assertEqual(dets.names, {0: "init_class"})

        # Add 50 new classes dynamically one by one
        for i in range(1, 51):
            new_label = f"dynamic_class_{i}"
            f0.add_box([i, i, i + 10, i + 10], label=new_label, score=0.8)
            self.assertIn(new_label, dets.names.values())
            self.assertEqual(len(f0), i + 1)

        self.assertEqual(len(dets.names), 51)
        self.assertEqual(dets.total_objects, 51)

    def test_yolo_export_fidelity_with_many_classes(self):
        """Export a collection with 80 classes to YOLO format and verify data.yaml & txt annotations."""
        rng = random.Random(777)
        frames = [make_multi_class_frame("yolo_test", i, COCO_80, boxes_per_frame=15, rng=rng) for i in range(5)]
        dets = outputs.Detections(frames)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = files.join(tmp_dir, "dataset")
            dets.export(out_dir, format="yolo")

            self.assertTrue(files.exists(files.join(out_dir, "data.yaml")))
            self.assertTrue(files.exists(files.join(out_dir, "manifest.json")))
            self.assertTrue(files.is_dir(files.join(out_dir, "labels")))

            data_yaml = files.load(files.join(out_dir, "data.yaml"), verbose=False)
            names_in_yaml = data_yaml["names"]
            # YAML names must match sorted collection active names
            active_sorted = sorted(dets.names.items())
            expected_names = {idx: lbl for idx, (_cid, lbl) in enumerate(active_sorted)}
            self.assertEqual(names_in_yaml, expected_names)

            # Check one of the label files
            label_files = files.find(files.join(out_dir, "labels"), pattern="*.txt")
            self.assertEqual(len(label_files), 5)
            for lf in label_files:
                content = files.load(lf, verbose=False)
                lines = content.strip().splitlines() if isinstance(content, str) else content
                for line in lines:
                    parts = line.strip().split()
                    class_idx = int(parts[0])
                    # class index in txt must be a valid key in data.yaml names
                    self.assertIn(class_idx, names_in_yaml)

    def test_intense_multiclass_crud_fuzzing(self):
        """Intense fuzzing: 200 random operations over 100 classes across 30 frames."""
        class_pool = [f"cls_{i:03d}" for i in range(100)]
        rng = random.Random(42)

        # Initialize 30 frames with random classes
        frames = []
        for f_idx in range(30):
            n_boxes = rng.randint(3, 15)
            boxes = [
                outputs.Box(
                    [b * 10, b * 10, b * 10 + 50, b * 10 + 50],
                    label=rng.choice(class_pool),
                    score=round(rng.uniform(0.1, 0.99), 3),
                )
                for b in range(n_boxes)
            ]
            frames.append(outputs.Detection(boxes, source_id="cam", frame_index=f_idx, image_size=(1000, 1000)))

        dets = outputs.Detections(frames)

        def assert_invariants(msg=""):
            self.assertEqual(dets.total_objects, sum(len(f) for f in dets), f"Object count mismatch: {msg}")
            active_labels = {b.label for f in dets for b in f}
            self.assertEqual(set(dets.names.values()), active_labels, f"Registry mismatch: {msg}")
            for f_idx, f in enumerate(dets):
                self.assertEqual(len(f.uids), len(set(f.uids)), f"Duplicate UIDs in frame {f_idx}: {msg}")
                self.assertEqual(len(f.class_ids), len(f.labels), f"Class IDs len mismatch in frame {f_idx}: {msg}")
                for b in f:
                    self.assertEqual(dets.names[b.class_id], b.label, f"Box class ID mismatch {b.label}: {msg}")
                # NumPy sync
                self.assertEqual(len(f._scores), len(f._labels))
                self.assertEqual(len(f._xyxy), len(f._labels))

        assert_invariants("initial")

        for step in range(200):
            op = rng.choice(["add_box", "pop_box", "relabel_batch", "drop_batch", "compact", "mutate_box", "replace_box"])

            if op == "add_box":
                target_f = rng.choice(list(dets))
                lbl = rng.choice(class_pool) if rng.random() > 0.1 else f"brand_new_cls_{step}"
                target_f.add_box([10, 10, 80, 80], label=lbl, score=round(rng.uniform(0.1, 0.99), 3))

            elif op == "pop_box":
                valid_frames = [f for f in dets if len(f) > 0]
                if valid_frames:
                    target_f = rng.choice(valid_frames)
                    idx = rng.randint(0, len(target_f) - 1)
                    target_f.pop(idx)

            elif op == "relabel_batch":
                active_classes = list(dets.names.values())
                if len(active_classes) >= 2:
                    k = min(len(active_classes), rng.randint(2, 5))
                    chosen = rng.sample(active_classes, k)
                    target = rng.choice(class_pool)
                    mapping = {src: target for src in chosen if src != target}
                    if mapping:
                        dets.relabel(mapping)

            elif op == "drop_batch":
                active_classes = list(dets.names.values())
                if len(active_classes) > 5:
                    k = rng.randint(1, min(5, len(active_classes) - 1))
                    to_drop = rng.sample(active_classes, k)
                    dets.drop(to_drop)

            elif op == "compact":
                dets.compact()

            elif op == "mutate_box":
                valid_frames = [f for f in dets if len(f) > 0]
                if valid_frames:
                    target_f = rng.choice(valid_frames)
                    b = target_f[rng.randint(0, len(target_f) - 1)]
                    b.score = round(rng.uniform(0.1, 0.99), 3)
                    b.box = [rng.randint(0, 100), rng.randint(0, 100), rng.randint(101, 300), rng.randint(101, 300)]

            elif op == "replace_box":
                valid_frames = [f for f in dets if len(f) > 0]
                if valid_frames:
                    target_f = rng.choice(valid_frames)
                    idx = rng.randint(0, len(target_f) - 1)
                    new_b = outputs.Box([5, 5, 50, 50], label=rng.choice(class_pool), score=0.88)
                    target_f.replace(idx, new_b)

            if step % 25 == 0 or step == 199:
                assert_invariants(f"step_{step}_{op}")


if __name__ == "__main__":
    unittest.main()
