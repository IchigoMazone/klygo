"""Same-source group operations retain independent annotation collections."""
import unittest
from klygo import outputs


def collection(source="image", size=(100, 100), index=0, labels=("cat", "dog")):
    return outputs.Detections([outputs.Detection(
        [outputs.Box((i, i, i+10, i+10), label=label, score=.9) for i, label in enumerate(labels)],
        source_id=source, frame_index=index, image_size=size,
    )])


class TestDetectionGroup(unittest.TestCase):
    def test_references_and_chaining_preserve_individual_apis(self):
        a, b = collection(), collection(labels=("cat",))
        group = outputs.group(a, b)
        self.assertIs(group[0], a)
        self.assertEqual(tuple(group), (a, b))
        self.assertEqual(group.members, (a, b))
        self.assertEqual(len(group), 2)
        self.assertIs(group.astype(32).cpu().pack().unpack(), group)
        self.assertEqual((a.dtype, b.dtype), (32, 32))
        self.assertFalse(a.is_packed)
        a.astype(64)
        self.assertEqual(b.dtype, 32)

    def test_label_operations_apply_to_all_without_merging(self):
        a, b = collection(), collection()
        group = outputs.group(a, b)
        self.assertIs(group.keep((name for name in ("cat",))).relabel({"cat": "animal"}).compact(), group)
        self.assertEqual(a[0].labels, ["animal"])
        self.assertEqual(b[0].labels, ["animal"])
        group.drop((name for name in ("animal",)))
        self.assertEqual((len(a[0]), len(b[0])), (0, 0))

    def test_source_size_identity_and_membership_validation(self):
        a = collection()
        for other in (collection(source="other"), collection(size=(200, 100)), collection(index=1)):
            with self.assertRaises(ValueError): outputs.group(a, other)
        with self.assertRaises(ValueError): outputs.group()
        with self.assertRaises(ValueError): outputs.group(a, a)
        with self.assertRaises(TypeError): outputs.group(a[0])
        with self.assertRaises(ValueError): outputs.group(a, outputs.Detections([], stream=True))
        self.assertEqual(len(outputs.group(a)), 1)
        self.assertEqual(len(outputs.group(outputs.Detections(), outputs.Detections())), 2)

    def test_order_independent_sources_and_revalidation(self):
        a = outputs.Detections([collection(index=0)[0], collection(index=1)[0]])
        b = outputs.Detections([collection(index=1)[0], collection(index=0)[0]])
        group = outputs.group(a, b)
        group.astype(32)
        b.pop()
        with self.assertRaises(ValueError): group.astype(16)
        self.assertEqual(a.dtype, 32)

    def test_named_and_automatic_members(self):
        a, b, c = collection(), collection(), collection()
        pair = outputs.group(a, b)
        self.assertEqual(list(pair.keys()), ["pred", "truth"])
        self.assertIs(pair["pred"], a)
        self.assertIs(pair["truth"], b)
        self.assertEqual(tuple(pair), (a, b))
        many = outputs.group(a, b, c)
        self.assertEqual(list(many.keys()), ["set_1", "set_2", "set_3"])
        self.assertIs(many["set_3"], c)
        named = outputs.DetectionGroup({"model_a": a, "ground_truth": b})
        self.assertEqual(dict(named.items()), {"model_a": a, "ground_truth": b})
        self.assertEqual(tuple(named.values()), (a, b))
        self.assertEqual(dict(named), {"model_a": a, "ground_truth": b})
        self.assertIs(named.astype(32)[0], a)
        with self.assertRaises(KeyError): named["missing"]
        with self.assertRaises(ValueError): outputs.group({})
        with self.assertRaises(ValueError): outputs.group({" ": a})
        with self.assertRaises(TypeError): outputs.group({1: a})
        with self.assertRaises(ValueError): outputs.group({"a": a, "b": a})
        with self.assertRaises(ValueError): outputs.group({"a": a, "b": collection(source="other")})

    def test_rename_preserves_members_order_annotations_and_chaining(self):
        a, b, c = collection(), collection(), collection()
        group = outputs.group(a, b, c)
        before = [member.to_dict() for member in group]
        self.assertIs(group.rename({"set_1": "pred", "set_2": "truth"}), group)
        self.assertEqual(list(group.keys()), ["pred", "truth", "set_3"])
        self.assertEqual(tuple(group), (a, b, c))
        self.assertIs(group["pred"], a)
        self.assertIs(group[0], a)
        self.assertEqual([member.to_dict() for member in group], before)
        group.rename({"pred": "truth", "truth": "pred"})
        self.assertIs(group["truth"], a)
        self.assertIs(group["pred"], b)
        self.assertIs(group.rename({}).rename({"truth": "truth"}), group)

    def test_rename_failures_leave_names_unchanged(self):
        group = outputs.group(collection(), collection())
        before = dict(group.items())
        for mapping, error in (([("pred", "model")], TypeError),
                               ({1: "model"}, TypeError),
                               ({"pred": 1}, TypeError),
                               ({"missing": "model"}, KeyError),
                               ({"pred": " "}, ValueError),
                               ({"pred": "truth"}, ValueError),
                               ({"pred": "same", "truth": "same"}, ValueError),
                               ({"pred": "model", "missing": "other"}, KeyError)):
            with self.subTest(mapping=mapping):
                with self.assertRaises(error): group.rename(mapping)
                self.assertEqual(dict(group.items()), before)
        group[1].pop()
        with self.assertRaises(ValueError): group.rename({"pred": "model"})
        self.assertEqual(dict(group.items()), before)

    def test_array_exports_are_independent(self):
        frame = collection()[0]
        arrays = frame.to_arrays()
        arrays['boxes'][0, 0] = 99
        arrays['scores'][0] = 0
        arrays['labels'][0] = 'changed'
        self.assertEqual(frame[0].box[0], 0)
        self.assertAlmostEqual(frame[0].score, .9)
        self.assertEqual(frame[0].label, 'cat')


if __name__ == '__main__': unittest.main()
