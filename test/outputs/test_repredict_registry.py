"""Canonical class identity, atomic replacement and explicit stream mutation."""

from copy import deepcopy
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from klygo import files, outputs

Box = outputs.detect.Box
Detection = outputs.detect.Detection
Detections = outputs.detect.Detections


def frame(label="person", class_id=0, index=0, image=None):
    image = Image.new("RGB", (20, 20)) if image is None else image
    return Detection(image, [Box(0, label, .8, [1, 2, 5, 6], class_id=class_id)],
                     image_frame_index=index, names={class_id: label},
                     config={"prompt": [label], "model_id": "old", "threshold": .4})


class ReplacementModel:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def predict(self, images, **kwargs):
        self.calls.append((images, kwargs))
        results = []
        for index, image in enumerate(images):
            if index == self.fail_at:
                raise RuntimeError("third inference failed")
            result = frame("dog", 0, image=image)
            result.config.update(model_id="new", threshold=.7)
            results.append(result)
        return Detections(results)


class TestCanonicalRegistry(unittest.TestCase):
    def test_collisions_and_equal_labels_across_models(self):
        person, car, dog, second_person = frame(), frame("car", 1), frame("dog", 0), frame("person", 9)
        collection = Detections([person, car])
        collection.extend([dog, second_person])
        self.assertEqual(collection.names, {0: "person", 1: "car", 2: "dog"})
        self.assertEqual(collection[2].class_ids, [2])
        self.assertEqual(collection[2][0].metadata["native_class_id"], 0)
        self.assertEqual(collection[3].class_ids, [0])
        self.assertEqual(dog.class_ids, [0])
        self.assertEqual(second_person.class_ids, [9])
        self.assertEqual(collection.active_names, collection.names)

    def test_mutating_member_updates_active_names_without_id_reuse(self):
        collection = Detections([frame()])
        member = collection[0]
        member.append(Box(0, "dog", .8, [1, 2, 3, 4], class_id=0))
        self.assertEqual(member.class_ids, [0, 1])
        self.assertEqual(len(set(member.ids)), 2)
        self.assertEqual(collection.active_names, {0: "person", 1: "dog"})
        member.pop()
        member.clear()
        self.assertEqual(member.names, {})
        self.assertEqual(collection.active_names, {})
        self.assertEqual(collection.names, {0: "person", 1: "dog"})
        member.insert(0, Box(0, "cat", .8, [1, 2, 3, 4], class_id=0))
        self.assertEqual(collection.active_names, {2: "cat"})

    def test_all_box_mutations_are_validated_and_bound(self):
        member = frame()
        other = frame("dog", 0)
        member.extend(other)
        self.assertIs(other[0].parent_image, other.source_image)
        self.assertIs(member[1].parent_image, member.source_image)
        member[0:1] = [Box(0, "cat", .9, [0, 0, 3, 4], class_id=0)]
        self.assertEqual(set(member.labels), {"cat", "dog"})
        before = member.to_dict()
        for mutate in (lambda: member.append(None), lambda: member.extend([other[0], None]),
                       lambda: member.__setitem__(0, None), lambda: member.insert(0, None)):
            with self.assertRaises(TypeError):
                mutate()
            self.assertEqual(member.to_dict(), before)
        del member[:1]
        member.remove(member[0])
        self.assertEqual(member.names, {})

    def test_frame_mutations_preserve_identity_and_new_metadata(self):
        collection = Detections([frame(index=10), frame(index=25)], fps=12, source_type="video")
        original = collection[1]
        replacement = frame("dog", 0, index=0)
        replacement.config["model_id"] = "new"
        collection[1] = replacement
        self.assertEqual(collection[1].frame_index, 25)
        self.assertEqual(collection[1].image_frame_index, 25)
        self.assertIs(collection[1].source_image, original.source_image)
        self.assertIs(collection[1][0].parent_image, original.source_image)
        self.assertEqual(collection[1].config["model_id"], "new")
        self.assertEqual(replacement.frame_index, 0)
        collection[:2] = [frame("cat"), frame("dog")]
        self.assertEqual([item.frame_index for item in collection], [10, 25])
        with self.assertRaises(ValueError):
            collection[:2] = [frame()]
        collection.insert(1, frame("bird", index=40))
        self.assertEqual(collection[1].frame_index, 40)
        self.assertEqual(collection.total_frames, 3)
        collection.pop()
        del collection[:1]
        self.assertEqual(collection.total_frames, 1)
        self.assertEqual(list(collection.active_names.values()), ["bird"])
        collection.clear()
        self.assertEqual(collection.active_names, {})

    def test_atomic_invalid_frame_extend_and_slice(self):
        collection = Detections([frame(), frame(index=1)])
        old_frames = list(collection)
        names = collection.names
        for operation in (lambda: collection.extend([frame("dog"), None]),
                          lambda: collection.__setitem__(slice(0, 2), [frame("cat"), None])):
            with self.assertRaises(TypeError):
                operation()
            self.assertEqual(list(collection), old_frames)
            self.assertEqual(collection.names, names)

    def test_functional_replacements_do_not_mutate_original(self):
        collection = Detections([frame(index=20)])
        replacement = collection.replace_frame(20, frame("dog"))
        self.assertEqual(replacement[0].frame_index, 20)
        self.assertEqual(collection[0].labels, ["person"])
        def mutate(member):
            member.clear()
            member.config["prompt"].append("changed")
            return member
        updated = collection.update_frame(20, mutate)
        self.assertEqual(updated.active_names, {})
        self.assertEqual(collection[0].labels, ["person"])
        self.assertEqual(collection[0].text_prompt, ["person"])

    def test_independent_collections_and_concatenation(self):
        first = Detections([frame()])
        second = Detections([frame("dog")])
        combined = first + second
        self.assertEqual(combined.names, {0: "person", 1: "dog"})
        combined[0].clear()
        self.assertEqual(first[0].labels, ["person"])
        self.assertEqual(second[0].class_ids, [0])

    def test_stream_names_do_not_consume_and_mutation_is_explicit(self):
        consumed = []
        def generate():
            for i in range(4):
                consumed.append(i)
                yield frame("dog" if i else "person", index=i * 5)
        collection = Detections(generate(), stream=True)
        self.assertEqual(collection.names, {})
        self.assertEqual(collection.active_names, {})
        with self.assertRaisesRegex(ValueError, "to_list"):
            collection.repredict(ReplacementModel())
        with self.assertRaisesRegex(ValueError, "to_list"):
            collection.append(frame())
        self.assertEqual(consumed, [])
        subset = collection[1:4:2]
        self.assertEqual([item.frame_index for item in subset], [5, 15])
        self.assertEqual([item.frame_index for item in collection], [0, 5, 10, 15])
        collection.to_list()
        collection.repredict(ReplacementModel(), -1)
        self.assertEqual(collection[-1].frame_index, 15)

    def test_export_preserves_sparse_canonical_ids(self):
        collection = Detections([frame("dog", 7)])
        with tempfile.TemporaryDirectory() as root:
            collection.export(root)
            data = files.load(files.join(root, "data.yaml"), verbose=False)
            labels = files.load(files.join(root, "labels", "frame_00000.txt"), verbose=False)
            self.assertEqual(data["names"], {7: "dog"})
            self.assertTrue(labels.startswith("7 "))


class TestRepredict(unittest.TestCase):
    def setUp(self):
        self.results = Detections([frame(index=10), frame(index=20), frame(index=40)], fps=12)

    def test_single_selected_all_and_empty(self):
        model = ReplacementModel()
        images = [item.source_image for item in self.results]
        returned = self.results.repredict(model, [2, 0, 2], prompt=["dog"], threshold=.7)
        self.assertIs(returned, self.results)
        self.assertEqual([item.frame_index for item in self.results], [10, 20, 40])
        self.assertEqual([item.labels for item in self.results], [["dog"], ["person"], ["dog"]])
        self.assertEqual(model.calls[0][0], [images[2], images[0]])
        self.assertEqual(model.calls[0][1], {"prompt": ["dog"], "threshold": .7})
        for item, original_image in zip(self.results, images):
            self.assertIs(item.source_image, original_image)
            self.assertIs(item[0].parent_image, original_image)
        self.results.repredict(model, index=-2)
        self.assertEqual(list(self.results.active_names.values()), ["dog"])
        self.results.repredict(model)
        self.assertEqual(self.results.fps, 12)
        self.results.repredict(model, [])
        self.assertEqual(len(model.calls), 3)
        self.assertIs(Detections([]).repredict(model).__class__, Detections)

    def test_third_failure_is_atomic(self):
        originals = list(self.results)
        snapshot = deepcopy(self.results.to_dict())
        names = self.results.names
        with self.assertRaises(RuntimeError):
            self.results.repredict(ReplacementModel(fail_at=2))
        self.assertEqual(list(self.results), originals)
        self.assertEqual(self.results.to_dict(), snapshot)
        self.assertEqual(self.results.names, names)

    def test_invalid_outputs_leave_no_partial_changes(self):
        originals = list(self.results)
        for output in ([frame("dog")], [frame("dog"), frame("cat"), None],
                       Detection(None, []), Detections(iter([frame()]), stream=True)):
            model = SimpleNamespace(predict=lambda source, output=output: output)
            with self.subTest(output=type(output)), self.assertRaises((ValueError, TypeError)):
                self.results.repredict(model)
            self.assertEqual(list(self.results), originals)
            self.assertEqual(self.results.names, {0: "person"})

    def test_invalid_indices_are_rejected_before_inference(self):
        model = ReplacementModel()
        for index in (True, [False], "0", 1.0, [0, 5], -4, {0, 1}):
            with self.subTest(index=index), self.assertRaises((TypeError, IndexError)):
                self.results.repredict(model, index)
        self.assertEqual(model.calls, [])
        self.results[0].source_image = None
        with self.assertRaisesRegex(ValueError, "source image"):
            self.results.repredict(model, 0)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
