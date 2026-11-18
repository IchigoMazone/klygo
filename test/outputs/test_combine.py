"""Regression coverage for source-safe detection combination."""
import unittest
from klygo import outputs


def frame(identity="image", index=0, size=(100, 100), boxes=None):
    return outputs.Detection(
        [outputs.Box(coords, label=label, score=score) for coords, label, score in
         (boxes if boxes is not None else [((0, 0, 20, 20), "cat", .9)])],
        source_id=identity, frame_index=index, image_size=size,
    )


class TestCombine(unittest.TestCase):
    def test_all_methods_require_same_identity_and_size_before_mutating(self):
        for method in ("union", "intersection", "average", "weighted", "first", "second", "append"):
            for other in (frame(identity="other"), frame(index=1), frame(size=(200, 100))):
                with self.subTest(method=method, identity=other.identity, size=other.image_size):
                    first = frame()
                    before = first.to_dict(), other.to_dict()
                    with self.assertRaises(ValueError):
                        first.combine(other, method=method)
                    self.assertEqual(before, (first.to_dict(), other.to_dict()))

    def test_empty_frames_still_validate_source(self):
        with self.assertRaises(ValueError):
            frame(boxes=[]).combine(frame(identity="other", boxes=[]))
        first = frame(boxes=[])
        self.assertIs(first.combine(frame(boxes=[])), first)

    def test_collection_checks_later_frame_before_any_mutation(self):
        for packed in (False, True):
            first = outputs.Detections([frame(index=0), frame(index=1)], packed=packed)
            other = outputs.Detections([
                frame(index=0, boxes=[((2, 2, 22, 22), "cat", .8)]),
                frame(index=1, size=(200, 100)), frame(index=2),
            ])
            before = first.to_dict(), other.to_dict(), first.history
            with self.assertRaises(ValueError):
                first.combine(other)
            self.assertEqual(before, (first.to_dict(), other.to_dict(), first.history))
            self.assertEqual(len(first), 2)

    def test_same_label_only_and_unmatched_boxes_are_retained(self):
        first = frame()
        other = frame(boxes=[((0, 0, 20, 20), "dog", .8)])
        snapshot = other.to_dict()
        self.assertIs(first.combine(other), first)
        self.assertEqual(first.labels, ["cat", "dog"])
        self.assertEqual(other.to_dict(), snapshot)

    def test_greedy_matching_prefers_iou_over_confidence(self):
        first = frame(boxes=[((0, 0, 20, 20), "cat", .99), ((2, 2, 22, 22), "cat", .1)])
        first.combine(frame(boxes=[((2, 2, 22, 22), "cat", .8)]), method="second")
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0].box, (0., 0., 20., 20.))
        self.assertEqual(first[1].score, .8)

    def test_weighted_zero_scores_and_max_score_rule(self):
        first = frame(boxes=[((0, 0, 20, 20), "cat", 0)])
        first.combine(frame(boxes=[((2, 2, 22, 22), "cat", 0)]), "weighted")
        self.assertEqual(first[0].box, (1., 1., 21., 21.))
        for method in ("first", "second", "average", "weighted"):
            first = frame()
            first.combine(frame(boxes=[((2, 2, 22, 22), "cat", .8)]), method)
            self.assertAlmostEqual(first[0].score, .9)

    def test_collection_alignment_extras_and_other_unchanged(self):
        for packed in (False, True):
            first = outputs.Detections([frame(index=0), frame(index=1)], packed=packed)
            other = outputs.Detections([
                frame(index=2), frame(index=0, boxes=[((2, 2, 22, 22), "cat", .8)])
            ])
            snapshot = other.to_dict()
            self.assertIs(first.combine(other), first)
            self.assertEqual([f.frame_index for f in first], [0, 1, 2])
            self.assertEqual(first[0][0].box, (0., 0., 22., 22.))
            first[2].clear()
            self.assertEqual(other.to_dict(), snapshot)

    def test_streams_are_rejected(self):
        for first, other in (
            (outputs.Detections([], stream=True), outputs.Detections()),
            (outputs.Detections(), outputs.Detections([], stream=True)),
        ):
            with self.assertRaises(TypeError):
                first.combine(other)
            first.close()
            other.close()


if __name__ == "__main__":
    unittest.main()
