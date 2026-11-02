"""Tests for in-place postprocessing over both container levels."""
import unittest
from unittest.mock import patch
from PIL import Image
from klygo import outputs, postprocessing as post

def result():
    frame = outputs.Detection([
        outputs.Box([-5,0,50,50], label="cat", score=.9, class_id=0),
        outputs.Box([0,0,48,48], label="cat", score=.8, class_id=0),
        outputs.Box([60,10,90,60], label="dog", score=.4, class_id=1),
    ], source_id="a", image_size=(80,50), names={0:"cat",1:"dog"})
    return outputs.Detections([frame])

class TestPostprocessing(unittest.TestCase):
    def test_pipeline_mutates_collection(self):
        pipe = post.compose(post.clip(), post.threshold(.5), post.nms(.5), post.top(1))
        results = result()
        pipe(results)
        self.assertEqual(results.total_objects, 1)
        self.assertEqual(results[0][0].box, (0.,0.,50.,50.)); self.assertEqual(results[0][0].uid, 0)

        # Test .transform() method equivalence
        results2 = result()
        returned = results2.transform(pipe)
        self.assertIs(returned, results2)
        self.assertEqual(results2.total_objects, 1)

    def test_direct_frame_operations(self):
        frame = result()[0]; frame.transform(post.threshold(.85))
        self.assertEqual(frame.uids, [0]); frame.relabel({"cat":"animal"})
        self.assertEqual(frame.labels, ["animal"]); frame.drop(["animal"])
        self.assertTrue(frame.empty)

        # Test frame.transform()
        frame2 = result()[0]
        frame2.transform(post.threshold(0.85))
        self.assertEqual(frame2.uids, [0])

    def test_select_accepts_scalar_selectors(self):
        frame = result()[0]
        frame.transform(post.select(labels="cat"))
        self.assertEqual(frame.labels, ["cat", "cat"])

        frame = result()[0]
        frame.transform(post.select(labels="cat", class_ids=0, uids=0))
        self.assertEqual(frame.uids, [0])

    def test_invalid_numeric_arguments_fail_before_mutation(self):
        checks = (
            lambda frame: frame.transform(post.threshold(float("nan"))),
            lambda frame: frame.transform(post.threshold(float("inf"))),
            lambda frame: frame.transform(post.remove_invalid(min_area=float("nan"))),
            lambda frame: frame.transform(post.clip(image_size=(-1, 20))),
        )
        for check in checks:
            with self.subTest(check=check):
                frame = result()[0]
                before = frame.to_dict()
                with self.assertRaises(ValueError):
                    check(frame)
                self.assertEqual(frame.to_dict(), before)

        frame = result()[0]
        with self.assertRaises(TypeError):
            frame.transform(post.clip(image_size=(20.0, 20)))

    def test_noop_operations_do_not_allocate_copy_on_write_baselines(self):
        results = result()
        original = results[0]
        results.transform(post.threshold(0.1))
        self.assertEqual(results._baseline_signatures, {})
        self.assertIs(results[0], original)

    def test_iou(self):
        frame = result()[0]; self.assertGreater(post.box_iou(frame[0], frame[1]), .8)

    def test_pipeline_is_atomic_when_a_later_operation_fails(self):
        results = result(); before = results.to_dict()
        def fail(_result):
            raise RuntimeError("expected failure")
        with self.assertRaises(RuntimeError):
            results.transform(post.threshold(.5), fail)
        self.assertEqual(results.to_dict(), before)

    def test_trusted_pipeline_avoids_full_collection_clones(self):
        results = outputs.Detections([
            result()[0]._clone(("video", index)) for index in range(3)
        ])
        original_clone = outputs.Detection._clone
        clone_count = 0

        def counted_clone(frame, *args, **kwargs):
            nonlocal clone_count
            clone_count += 1
            return original_clone(frame, *args, **kwargs)

        with patch.object(outputs.Detection, "_clone", counted_clone):
            results.transform(post.threshold(.5), post.top(1))

        self.assertEqual(clone_count, 0)
        self.assertEqual(results.total_objects, len(results))

    def test_pipeline_stays_lazy_for_streaming_detections(self):
        consumed = []

        def frames():
            for index in range(3):
                consumed.append(index)
                yield result()[0]._clone(("stream", index))

        results = outputs.Detections(frames(), stream=True)
        pipe = post.compose(post.clip(), post.threshold(.5), post.nms(.5), post.top(1))
        results.transform(pipe)
        self.assertEqual(consumed, [])
        processed = list(results)
        self.assertEqual(consumed, [0, 1, 2])
        self.assertEqual([frame.count for frame in processed], [1, 1, 1])
        self.assertTrue(results.stream_exhausted)

    def test_vectorized_postprocessing_keeps_boxes_unmaterialized(self):
        built = outputs.build_detections(
            [Image.new("RGB", (80, 50))],
            [{
                "boxes": [[-5, 0, 50, 50], [0, 0, 48, 48], [60, 10, 90, 60]],
                "scores": [0.9, 0.8, 0.4],
                "labels": ["cat", "cat", "dog"],
                "class_ids": [0, 0, 1],
            }],
            names={0: "cat", 1: "dog"},
        )
        results = outputs.Detections(built, names={0: "cat", 1: "dog"})
        results.transform(
            post.clip(),
            post.remove_invalid(min_area=10),
            post.threshold(0.5),
            post.nms(0.5),
            post.top(1),
        )
        self.assertIsNone(results[0]._boxes_cache)
        self.assertEqual(results.total_objects, 1)
        self.assertEqual(results[0].boxes, [(0.0, 0.0, 50.0, 50.0)])
        self.assertIn(results[0].identity, results.modified_frames)

    def test_clip_boxes_updates_already_materialized_box_instances(self):
        frame = result()[0]
        first_box = frame[0]
        self.assertEqual(first_box.box, (-5.0, 0.0, 50.0, 50.0))
        frame.transform(post.clip())
        self.assertIs(frame[0], first_box)
        self.assertEqual(first_box.box, (0.0, 0.0, 50.0, 50.0))

if __name__ == "__main__": unittest.main()
