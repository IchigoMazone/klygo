"""Behavioral tests for the three-level outputs contract."""
import tempfile
import unittest
from PIL import Image
from klygo import files, outputs

def frame(source_id="image-a", index=0, labels=("car", "person")):
    boxes = [outputs.Box([i, i, i + 5, i + 6], label=label, score=.9-i*.1, class_id=i)
             for i, label in enumerate(labels)]
    return outputs.Detection(boxes, source_id=source_id, frame_index=index,
                             image_size=(40, 30), source=Image.new("RGB", (40, 30)),
                             names=dict(enumerate(labels)))

class TestOutputs(unittest.TestCase):
    def test_builder_and_geometry(self):
        built = outputs.build_detections([Image.new("RGB", (40, 30))], [{"boxes": [[1,2,10,12]], "scores": [.8], "labels": ["car"]}])
        self.assertEqual(built[0].uids, [0]); self.assertEqual(built[0][0].area, 90)
        built[0][0].translate(2, 3); self.assertEqual(built[0][0].box, (3., 5., 12., 15.))

    def test_box_validation_and_read_only_identity(self):
        box = outputs.Box([0,0,10,10], label="car")
        with self.assertRaises(ValueError): box.score = 2
        with self.assertRaises(ValueError): box.box = [1,2,1,4]
        with self.assertRaises(AttributeError): box.uid = 4
        with self.assertRaises(AttributeError): box.label = "bus"

    def test_box_crud_uid_is_monotonic(self):
        item = frame(labels=("a", "b", "c")); item.remove_box(1); item.add_box([1,1,3,3], label="d")
        self.assertEqual(item.uids, [0,2,3]); self.assertEqual(item.get_box(3).label, "d")
        replacement = outputs.Box([2,2,8,8], label="x"); item.replace(0, replacement)
        self.assertEqual(item[0].uid, 0)

    def test_detection_transforms_mutate_atomically(self):
        item = frame(); original = item.to_dict()
        def invalid(box):
            if box.uid == 1: box.score = 3
        with self.assertRaises(ValueError): item.map(invalid)
        self.assertEqual(item.to_dict(), original)
        item.drop(["person"]); self.assertEqual(item.labels, ["car"])
        item.relabel({"car": "vehicle"}); self.assertEqual(item.labels, ["vehicle"])

    def test_drop_accepts_one_label_string(self):
        item = frame(); item.drop("car")
        self.assertEqual(item.labels, ["person"])

    def test_collection_crud_identity_and_net_state(self):
        first, second = frame(index=0), frame(index=1)
        results = outputs.Detections([first, second]); replacement = frame("other", 9, ("dog",))
        results.replace(0, replacement)
        self.assertEqual(results[0].identity, ("image-a", 0)); self.assertIn(("image-a",0), results.modified_frames)
        results.remove_frame(1, source_id="image-a")
        self.assertEqual(results.deleted_frames, (("image-a",1),)); self.assertEqual(results.total_frames, 2)
        added = frame("new", 0); results.append(added); self.assertEqual(results.added_frames, (("new",0),))

    def test_extend_and_merge_are_atomic(self):
        results = outputs.Detections([frame()]); before = results.to_dict()
        with self.assertRaises(TypeError): results.extend([frame("b"), object()])
        self.assertEqual(results.to_dict(), before)
        with self.assertRaises(ValueError): results.merge(outputs.Detections([frame()]))
        self.assertEqual(results.to_dict(), before)

    def test_frame_lookup_requires_source_when_ambiguous(self):
        results = outputs.Detections([frame("a"), frame("b")])
        with self.assertRaises(ValueError): results.get_frame(0)
        self.assertEqual(results.get_frame(0, source_id="b").source_id, "b")

    def test_collection_drop_relabel_and_active_names(self):
        results = outputs.Detections([frame()]); results.drop(["person"])
        self.assertEqual(results.names, {0:"car"}); results.relabel({"car":"vehicle"})
        self.assertEqual(results.names, {0:"vehicle"}); self.assertEqual(results.total_objects, 1)

    def test_export_is_annotation_only(self):
        with tempfile.TemporaryDirectory() as root:
            target = files.join(root, "export"); outputs.Detections([frame()]).export(target)
            self.assertTrue(files.exists(files.join(target,"data.yaml")))
            self.assertTrue(files.exists(files.join(target,"manifest.json")))
            self.assertFalse(files.exists(files.join(target,"images")))

if __name__ == "__main__": unittest.main()
