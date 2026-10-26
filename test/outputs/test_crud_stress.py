"""Comprehensive and intense CRUD stress tests for Box, Detection, and Detections.

Tests all CRUD operations, boundary edge cases, high-churn mutations,
cross-frame box transfers, stream transformations, and Copy-On-Write baseline integrity.
"""

import copy
import math
import random
import tempfile
import unittest
from PIL import Image
import numpy as np

from klygo import files, outputs, postprocessing


def make_box(index=0, label="car", score=0.9):
    return outputs.Box(
        [index * 2, index * 2 + 1, index * 2 + 20, index * 2 + 25],
        label=label,
        score=score,
        class_id=index % 5,
        track_id=index if index % 2 == 0 else None,
        metadata={"orig_idx": index} if index % 3 == 0 else None,
    )


def make_frame(source_id="cam-0", frame_index=0, num_boxes=10, labels=("car", "person", "bus", "truck")):
    boxes = [
        make_box(i, label=labels[i % len(labels)], score=round(max(0.01, min(0.99, 0.5 + 0.05 * (i % 10))), 3))
        for i in range(num_boxes)
    ]
    return outputs.Detection(
        boxes,
        source_id=source_id,
        frame_index=frame_index,
        image_size=(1920, 1080),
        source=Image.new("RGB", (100, 100)),
        names={i: lbl for i, lbl in enumerate(labels)},
    )


class TestIntenseCRUDStress(unittest.TestCase):
    """Stress tests on single Detection CRUD operations."""

    def test_detection_empty_lifecycle_crud(self):
        """Test CRUD on an empty Detection: add, mutate, clear, re-add."""
        det = outputs.Detection(source_id="empty-test", frame_index=0, image_size=(640, 480))
        self.assertEqual(len(det), 0)
        self.assertTrue(det.empty)
        self.assertFalse(det.has_objects)
        self.assertEqual(det.boxes, [])
        self.assertEqual(det.scores, [])
        self.assertEqual(det.labels, [])
        self.assertEqual(det.class_ids, [])
        self.assertEqual(det.uids, [])

        # Add 1 box
        det.add_box([10, 20, 100, 200], label="dog", score=0.85)
        self.assertEqual(len(det), 1)
        self.assertFalse(det.empty)
        self.assertTrue(det.has_objects)
        self.assertEqual(det[0].label, "dog")
        self.assertEqual(det[0].box, (10.0, 20.0, 100.0, 200.0))
        self.assertAlmostEqual(det[0].score, 0.85, places=5)
        self.assertEqual(det.uids, [0])

        # Mutate the box
        det[0].score = 0.99
        det[0].box = [15, 25, 105, 205]
        self.assertAlmostEqual(det.scores[0], 0.99, places=5)
        self.assertEqual(det.boxes[0], (15.0, 25.0, 105.0, 205.0))

        # Clear
        det.clear()
        self.assertEqual(len(det), 0)
        self.assertTrue(det.empty)

        # Re-add after clear
        det.add_box([5, 5, 50, 50], label="cat", score=0.7)
        self.assertEqual(len(det), 1)
        self.assertEqual(det[0].label, "cat")
        # UID must be monotonically increasing
        self.assertEqual(det.uids, [1])

    def test_high_churn_mutations_and_lazy_materialization(self):
        """Build 200 boxes via build_detections, perform arbitrary CRUD, verify NumPy sync at every step."""
        raw_boxes = [[i * 2, i * 2 + 1, i * 2 + 20, i * 2 + 25] for i in range(200)]
        raw_scores = [round(max(0.01, min(0.99, 0.5 + 0.05 * (i % 10))), 3) for i in range(200)]
        raw_labels = [["car", "person", "bus", "truck"][i % 4] for i in range(200)]
        built = outputs.build_detections(
            [Image.new("RGB", (1920, 1080))],
            [{"boxes": raw_boxes, "scores": raw_scores, "labels": raw_labels}],
        )
        det = built[0]
        self.assertEqual(len(det), 200)
        self.assertIsNone(det._boxes_cache)  # 0 boxes materialized!

        # 1. Pop from front (unmaterialized)
        popped_front = det.pop(0)
        self.assertEqual(len(det), 199)
        self.assertEqual(popped_front.uid, 0)
        self.assertIsNone(popped_front._owner)

        # 2. Materialize a few boxes in the middle
        mid_box = det[50]
        self.assertEqual(mid_box.uid, 51)
        self.assertIsNotNone(det._boxes_cache)
        self.assertIsNotNone(det._boxes_cache[50])
        self.assertIsNone(det._boxes_cache[0])  # Still unmaterialized!

        # 3. Mutate mid_box in-place
        mid_box.score = 0.1234
        mid_box.box = [100.0, 200.0, 300.0, 400.0]
        self.assertAlmostEqual(det.scores[50], 0.1234, places=4)
        self.assertEqual(det.boxes[50], (100.0, 200.0, 300.0, 400.0))
        self.assertAlmostEqual(det._scores[50], 0.1234, places=4)
        self.assertEqual(tuple(det._xyxy[50].tolist()), (100.0, 200.0, 300.0, 400.0))

        # 4. Remove a box by UID before and after mid_box
        det.remove_box(10)  # Before mid_box
        self.assertEqual(len(det), 198)
        # mid_box index should have shifted from 50 to 49
        self.assertIs(det.get_box(51), mid_box)
        self.assertAlmostEqual(det.scores[49], 0.1234, places=4)

        # 5. Insert at index 0
        new_box = outputs.Box([1, 1, 20, 20], label="inserted_start", score=0.999)
        det.insert(0, new_box)
        self.assertEqual(len(det), 199)
        self.assertEqual(det[0].label, "inserted_start")
        # mid_box index shifted back to 50
        self.assertIs(det.get_box(51), mid_box)
        self.assertAlmostEqual(det.scores[50], 0.1234, places=4)

        # 6. Replace index 50 (which is mid_box)
        replacement = outputs.Box([2, 2, 22, 22], label="replacement_box", score=0.888)
        det.replace(50, replacement)
        self.assertEqual(len(det), 199)
        self.assertEqual(det[50].label, "replacement_box")
        self.assertAlmostEqual(det.scores[50], 0.888, places=3)
        self.assertIsNone(mid_box._owner)  # Old box detached!

        # 7. Slicing
        slice_boxes = det[10:20]
        self.assertIsInstance(slice_boxes, tuple)
        self.assertEqual(len(slice_boxes), 10)
        for b in slice_boxes:
            self.assertIsInstance(b, outputs.Box)

    def test_randomized_crud_fuzzing(self):
        """Fuzz 300 interleaved CRUD operations against an independent reference state."""
        labels_pool = ["car", "pedestrian", "bicycle", "traffic_light", "dog", "sign"]
        det = outputs.Detection(source_id="fuzz", frame_index=0, image_size=(1000, 1000))

        # Reference state: list of dicts [{'box': tuple, 'label': str, 'score': float, 'uid': int}]
        ref = []
        next_ref_uid = 0

        rng = random.Random(42)

        for step in range(300):
            available_ops = ["add"]
            if len(ref) > 0:
                available_ops.extend(["pop", "remove_uid", "replace", "mutate_score", "mutate_coords", "drop", "relabel"])
            if len(ref) > 1:
                available_ops.append("filter")

            op = rng.choice(available_ops)

            if op == "add":
                lbl = rng.choice(labels_pool)
                sc = round(rng.uniform(0.1, 0.99), 3)
                coords = (
                    rng.randint(0, 500),
                    rng.randint(0, 500),
                    rng.randint(501, 999),
                    rng.randint(501, 999),
                )
                det.add_box(coords, label=lbl, score=sc)
                ref.append({
                    "box": (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                    "label": lbl,
                    "score": sc,
                    "uid": next_ref_uid,
                })
                next_ref_uid += 1

            elif op == "pop":
                idx = rng.randint(0, len(ref) - 1)
                popped_det = det.pop(idx)
                popped_ref = ref.pop(idx)
                self.assertEqual(popped_det.uid, popped_ref["uid"])
                self.assertEqual(popped_det.label, popped_ref["label"])

            elif op == "remove_uid":
                chosen = rng.choice(ref)
                uid_to_rem = chosen["uid"]
                det.remove_box(uid_to_rem)
                ref = [item for item in ref if item["uid"] != uid_to_rem]

            elif op == "replace":
                idx = rng.randint(0, len(ref) - 1)
                lbl = rng.choice(labels_pool)
                sc = round(rng.uniform(0.1, 0.99), 3)
                coords = (
                    rng.randint(0, 500),
                    rng.randint(0, 500),
                    rng.randint(501, 999),
                    rng.randint(501, 999),
                )
                new_b = outputs.Box(coords, label=lbl, score=sc)
                preserved_uid = ref[idx]["uid"]
                det.replace(idx, new_b)
                ref[idx] = {
                    "box": (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                    "label": lbl,
                    "score": sc,
                    "uid": preserved_uid,
                }

            elif op == "mutate_score":
                idx = rng.randint(0, len(ref) - 1)
                new_sc = round(rng.uniform(0.01, 0.99), 3)
                det[idx].score = new_sc
                ref[idx]["score"] = new_sc

            elif op == "mutate_coords":
                idx = rng.randint(0, len(ref) - 1)
                new_coords = [
                    rng.randint(0, 400),
                    rng.randint(0, 400),
                    rng.randint(401, 800),
                    rng.randint(401, 800),
                ]
                det[idx].box = new_coords
                ref[idx]["box"] = (float(new_coords[0]), float(new_coords[1]), float(new_coords[2]), float(new_coords[3]))

            elif op == "drop":
                target_lbl = rng.choice(labels_pool)
                det.drop(target_lbl)
                ref = [item for item in ref if item["label"] != target_lbl]

            elif op == "relabel":
                src_lbl = rng.choice(labels_pool)
                dst_lbl = rng.choice(labels_pool)
                if src_lbl != dst_lbl:
                    det.relabel({src_lbl: dst_lbl})
                    for item in ref:
                        if item["label"] == src_lbl:
                            item["label"] = dst_lbl

            elif op == "filter":
                thresh = round(rng.uniform(0.3, 0.7), 2)
                det.filter(lambda b: b.score >= thresh)
                ref = [item for item in ref if item["score"] >= thresh]

            # Invariant check at every iteration
            self.assertEqual(len(det), len(ref), f"Length mismatch at step {step} ({op})")
            if ref:
                self.assertEqual(det.uids, [r["uid"] for r in ref], f"UIDs mismatch at step {step} ({op})")
                self.assertEqual(det.labels, [r["label"] for r in ref], f"Labels mismatch at step {step} ({op})")
                for r_idx, r_item in enumerate(ref):
                    self.assertAlmostEqual(det.scores[r_idx], r_item["score"], places=3)
                    self.assertEqual(det.boxes[r_idx], r_item["box"])

    def test_atomic_rejection_of_invalid_mutations(self):
        """Verify mutating a Box in a Detection with invalid values raises and keeps state intact."""
        det = make_frame("atomic", 0, num_boxes=3)
        box = det[1]
        orig_box = box.box
        orig_score = box.score

        # 1. Invalid coordinates: zero width, inverted, non-finite
        for bad_coords in ([10, 10, 10, 20], [20, 10, 10, 20], [0, 0, math.nan, 10], [0, 0, math.inf, 10]):
            with self.assertRaises(ValueError):
                box.box = bad_coords
            self.assertEqual(box.box, orig_box)
            self.assertEqual(det.boxes[1], orig_box)
            self.assertEqual(tuple(det._xyxy[1].tolist()), orig_box)

        # 2. Invalid scores: out of [0, 1] range, NaN, Inf
        for bad_score in (-0.1, 1.01, math.nan, math.inf, "invalid"):
            with self.assertRaises((ValueError, TypeError)):
                box.score = bad_score
            self.assertEqual(box.score, orig_score)
            self.assertEqual(det.scores[1], orig_score)
            self.assertEqual(det._scores[1], orig_score)

    def test_relabel_many_to_one_consolidation(self):
        """Relabel multiple distinct classes to one class and verify class IDs and names consolidate."""
        det = make_frame("consolidate", 0, num_boxes=8, labels=("car", "bus", "truck", "van"))
        self.assertEqual(set(det.labels), {"car", "bus", "truck", "van"})

        # Relabel all vehicles to single label 'vehicle'
        det.relabel({"car": "vehicle", "bus": "vehicle", "truck": "vehicle", "van": "vehicle"})
        self.assertEqual(set(det.labels), {"vehicle"})
        self.assertEqual(len(det.unique_labels), 1)
        self.assertEqual(len(det.unique_class_ids), 1)
        self.assertEqual(det.names, {det.class_ids[0]: "vehicle"})

        # Verify all boxes reflect this change
        for box in det:
            self.assertEqual(box.label, "vehicle")
            self.assertEqual(box.class_id, det.class_ids[0])

    def test_export_and_dict_serialization_fidelity(self):
        """Verify to_dict and reconstruction work with columnar representation."""
        det = make_frame("serial", 0, num_boxes=5)
        d = det.to_dict()
        self.assertEqual(d["source_id"], "serial")
        self.assertEqual(d["frame_index"], 0)
        self.assertEqual(len(d["boxes"]), 5)
        self.assertIn("score", d["boxes"][0])
        self.assertIn("box", d["boxes"][0])
        self.assertIn("label", d["boxes"][0])

        # Reconstructed from dict
        reconstructed = outputs.Detection(
            [
                outputs.Box(b["box"], label=b["label"], score=b["score"], class_id=b["class_id"])
                for b in d["boxes"]
            ],
            source_id=d["source_id"],
            frame_index=d["frame_index"],
            image_size=det.image_size,
        )
        self.assertEqual(len(reconstructed), len(det))
        self.assertEqual(reconstructed.boxes, det.boxes)
        self.assertEqual(reconstructed.labels, det.labels)
        self.assertEqual(reconstructed.class_ids, det.class_ids)


class TestDetectionsCollectionCRUD(unittest.TestCase):
    """Stress tests on Detections multi-frame CRUD and state tracking."""

    def test_detections_crud_and_cow_tracking(self):
        """Test append, insert, replace, pop, remove_frame, relabel, drop on Detections."""
        f0 = make_frame("video1", 0, num_boxes=5)
        f1 = make_frame("video1", 1, num_boxes=8)
        f2 = make_frame("video1", 2, num_boxes=4)

        dets = outputs.Detections([f0, f1, f2])
        self.assertEqual(len(dets), 3)
        self.assertEqual(dets.total_objects, 17)
        self.assertEqual(dets.modified_frames, ())
        self.assertEqual(dets.added_frames, ())
        self.assertEqual(dets.deleted_frames, ())

        # 1. Mutate a box in f1
        f1_box = dets[1][0]
        orig_score = f1_box.score
        f1_box.score = 0.05
        self.assertIn(("video1", 1), dets.modified_frames)

        # 2. Revert score back to original
        f1_box.score = orig_score
        self.assertNotIn(("video1", 1), dets.modified_frames)  # COW baseline revert!

        # 3. Add a frame via append
        f3 = make_frame("video1", 3, num_boxes=6)
        dets.append(f3)
        self.assertEqual(len(dets), 4)
        self.assertIn(("video1", 3), dets.added_frames)

        # 4. Remove a frame
        dets.remove_frame(0, source_id="video1")
        self.assertEqual(len(dets), 3)
        self.assertIn(("video1", 0), dets.deleted_frames)

        # 5. Insert frame in position 0
        f_new = make_frame("video2", 0, num_boxes=3)
        dets.insert(0, f_new)
        self.assertEqual(len(dets), 4)
        self.assertEqual(dets[0].identity, ("video2", 0))

        # 6. Relabel across collection
        dets.relabel({"car": "automobile"})
        for frame in dets:
            self.assertNotIn("car", frame.labels)

        # 7. Drop label across collection
        dets.drop(["bus", "truck"])
        for frame in dets:
            self.assertNotIn("bus", frame.labels)
            self.assertNotIn("truck", frame.labels)

    def test_cross_frame_box_transfer_isolation(self):
        """Moving a box from frame A to frame B must clone and keep them completely isolated."""
        frame_a = make_frame("a", 0, num_boxes=3)
        frame_b = make_frame("b", 0, num_boxes=0)

        box_a = frame_a[0]
        frame_b.append(box_a)

        self.assertEqual(len(frame_b), 1)
        self.assertIsNot(frame_b[0], box_a)

        # Mutate frame_b[0]
        frame_b[0].score = 0.111
        frame_b[0].box = [99, 99, 199, 199]

        # frame_a[0] must NOT be affected!
        self.assertNotEqual(frame_a[0].score, 0.111)
        self.assertNotEqual(frame_a[0].box, (99.0, 99.0, 199.0, 199.0))

    def test_stream_transformations_and_crud(self):
        """Verify streaming Detections can be filtered, dropped, and relabeled without materializing."""
        def stream_gen():
            for i in range(20):
                yield make_frame("stream", i, num_boxes=5)

        stream_dets = outputs.Detections(stream_gen(), stream=True)
        self.assertTrue(stream_dets.is_stream)
        self.assertFalse(stream_dets.stream_exhausted)

        # Stream relabel
        stream_dets.relabel({"car": "auto"})

        # Stream drop
        stream_dets.drop(["truck"])

        # Stream map
        def add_tag(frame):
            frame.metadata["stream_processed"] = True
            return frame

        stream_dets.map(add_tag)

        # Stream filter stays lazy and evaluates detached frame drafts.
        self.assertIs(stream_dets.filter(lambda f: f.frame_index % 2 == 0), stream_dets)

        # Consume the stream
        consumed = list(stream_dets)
        self.assertEqual(len(consumed), 10)
        self.assertTrue(stream_dets.stream_exhausted)
        for f in consumed:
            self.assertNotIn("car", f.labels)
            self.assertNotIn("truck", f.labels)
            self.assertTrue(f.metadata.get("stream_processed"))

    def test_vectorized_postprocessing_stress(self):
        """Run full postprocessing pipeline on 1000 frames with varying box counts and edge-case boxes."""
        frames = []
        for i in range(500):
            # Include empty frames, single-box frames, and dense frames (25 boxes)
            n_boxes = 0 if i % 10 == 0 else (1 if i % 7 == 0 else 25)
            frames.append(make_frame("vid", i, num_boxes=n_boxes))

        dets = outputs.Detections(frames)
        self.assertEqual(len(dets), 500)

        pipeline = postprocessing.compose(
            postprocessing.clip(image_size=(1920, 1080)),
            postprocessing.threshold(0.55),
            postprocessing.nms(iou=0.45),
            postprocessing.top(10),
        )

        pipeline(dets)
        self.assertEqual(len(dets), 500)
        for frame in dets:
            # Check maximum 10 boxes per frame
            self.assertLessEqual(len(frame), 10)
            # Check threshold
            for sc in frame.scores:
                self.assertGreaterEqual(sc, 0.55)
            # Check clip boundaries
            for b in frame.boxes:
                self.assertGreaterEqual(b[0], 0.0)
                self.assertGreaterEqual(b[1], 0.0)
                self.assertLessEqual(b[2], 1920.0)
                self.assertLessEqual(b[3], 1080.0)


if __name__ == "__main__":
    unittest.main()
