"""Tests for in-place postprocessing over both container levels."""
import unittest
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
        results = result(); returned = post.apply(results, post.clip(), post.threshold(.5), post.nms(.5), post.top(1))
        self.assertIsNone(returned); self.assertEqual(results.total_objects, 1)
        self.assertEqual(results[0][0].box, (0.,0.,50.,50.)); self.assertEqual(results[0][0].uid, 0)

    def test_direct_frame_operations(self):
        frame = result()[0]; post.filter_score(frame, .85)
        self.assertEqual(frame.uids, [0]); post.relabel(frame, {"cat":"animal"})
        self.assertEqual(frame.labels, ["animal"]); post.drop(frame, labels=["animal"])
        self.assertTrue(frame.empty)

    def test_iou(self):
        frame = result()[0]; self.assertGreater(post.box_iou(frame[0], frame[1]), .8)

    def test_pipeline_is_atomic_when_a_later_operation_fails(self):
        results = result(); before = results.to_dict()
        def fail(_result):
            raise RuntimeError("expected failure")
        with self.assertRaises(RuntimeError):
            post.apply(results, post.threshold(.5), fail)
        self.assertEqual(results.to_dict(), before)

if __name__ == "__main__": unittest.main()
