"""Behavioral tests for the three-level outputs contract."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from klygo import files, outputs, postprocessing

def frame(source_id="image-a", index=0, labels=("car", "person")):
    boxes = [outputs.Box([i, i, i + 5, i + 6], label=label, score=.9-i*.1, class_id=i)
             for i, label in enumerate(labels)]
    return outputs.Detection(boxes, source_id=source_id, frame_index=index,
                             image_size=(40, 30), source=Image.new("RGB", (40, 30)),
                             names=dict(enumerate(labels)))

def traced_frame(index=0, *, precision=64):
    image = Image.new("RGB", (50, 25))
    image._preprocessing_trace = [{
        "name": "resize", "before": (100, 50), "after": (50, 25),
        "scale": (0.5, 0.5),
    }]
    item = outputs.Detection(
        [outputs.Box([5, 5, 20, 15], label="car", class_id=0)],
        source_id="video", frame_index=index, image_size=image.size,
        source=image, names={0: "car"},
    )
    item.astype(precision)
    return item

class TestOutputs(unittest.TestCase):
    def test_builder_and_geometry(self):
        built = outputs.build_detections([Image.new("RGB", (40, 30))], [{"boxes": [[1,2,10,12]], "scores": [.8], "labels": ["car"]}])
        self.assertEqual(built[0].uids, [0]); self.assertEqual(built[0][0].area, 90)
        self.assertIs(built[0].identity, built[0].identity)
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

    def test_detection_fancy_indexing_preserves_context_and_is_independent(self):
        item = frame(labels=("a", "b", "c", "d"))
        item.metadata["nested"] = {"value": 1}
        selected = item[[2, 0, -1]]
        self.assertIsInstance(selected, outputs.Detection)
        self.assertEqual(selected.labels, ["c", "a", "d"])
        self.assertEqual(selected.uids, [2, 0, 3])
        self.assertEqual(selected.identity, item.identity)
        self.assertIs(selected._source, item._source)
        selected.metadata["nested"]["value"] = 2
        selected[0].score = 0.1
        self.assertEqual(item.metadata["nested"]["value"], 1)
        self.assertNotEqual(selected[0].score, item[2].score)
        with self.assertRaises(ValueError):
            item[[0, 0]]
        with self.assertRaises(TypeError):
            item[[True]]

    def test_output_mutations_support_method_chaining(self):
        item = frame(labels=("car", "person", "bus"))
        self.assertIs(
            item.filter(lambda box: box.score > 0.7).drop("person").relabel({"car": "vehicle"}),
            item,
        )
        self.assertEqual(item.labels, ["vehicle"])
        self.assertIs(item.clear(), item)

        results = outputs.Detections([frame(index=0), frame(index=1)])
        chained = results.drop("person").relabel({"car": "vehicle"}).compact()
        self.assertIs(chained, results)
        self.assertEqual(results.names, {0: "vehicle"})
        self.assertIs(results.filter(lambda candidate: candidate.has_objects), results)
        self.assertIs(results.clear(), results)

    def test_visual_exports_save_image_sequence_and_streaming_video(self):
        with tempfile.TemporaryDirectory() as root:
            item = frame(labels=("car",))
            image_path = files.join(root, "single.png")
            self.assertEqual(item.save(image_path, mode="uid"), files.path(image_path))
            self.assertTrue(files.exists(image_path))

            results = outputs.Detections([frame(index=0), frame(index=1)])
            image_dir = files.join(root, "frames")
            paths = results.save_images(image_dir, prefix="det", mode="index", extension=".png")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(files.exists(path) for path in paths))

            def generate():
                yield frame("stream", 0, ("car",))
                yield frame("stream", 1, ("car",))
            stream = outputs.Detections(generate(), stream=True)
            video_path = files.join(root, "result.mp4")
            self.assertEqual(stream.save_video(video_path, fps=5.0), files.path(video_path))
            self.assertTrue(files.exists(video_path))

    def test_detection_transforms_mutate_atomically(self):
        item = frame(); original = item.to_dict()
        def invalid(box):
            if box.uid == 1: box.score = 3
        with self.assertRaises(ValueError): item.map(invalid)
        self.assertEqual(item.to_dict(), original)
        item.drop(["person"]); self.assertEqual(item.labels, ["car"])
        item.relabel({"car": "vehicle"}); self.assertEqual(item.labels, ["vehicle"])

    def test_detection_restore_is_in_place_cached_and_idempotent(self):
        item = traced_frame(precision=32)
        cached = item[0]
        result = item.restore()
        self.assertIsNone(result)
        self.assertEqual(item.boxes, [(10.0, 10.0, 40.0, 30.0)])
        self.assertEqual(cached.box, (10.0, 10.0, 40.0, 30.0))
        self.assertEqual(item.image_size, (100, 50))
        self.assertEqual(item.dtype, 32)
        item.restore()
        self.assertEqual(item.boxes, [(10.0, 10.0, 40.0, 30.0)])

    def test_restore_resets_lazy_source_to_original_pixel_space(self):
        with tempfile.TemporaryDirectory() as root:
            source_path = files.join(root, "source.png")
            media_image = Image.new("RGB", (100, 50), "white")
            media_image.save(source_path)
            from klygo import media, preprocessing as p

            processed = p.resize((20, 10))(media.load(source_path)[0])
            p.materialize(processed, cache=False)
            item = outputs.Detection(
                [outputs.Box([1, 1, 5, 5], label="car")],
                source=processed, image_size=(20, 10), names={0: "car"},
            )
            item.restore()
            self.assertEqual(item.image_size, (100, 50))
            self.assertEqual(item._resolve_source().size, (100, 50))

    def test_stream_filter_is_lazy_and_keeps_matching_frames(self):
        consumed = []

        def generate():
            for index in range(4):
                consumed.append(index)
                labels = ("car",) if index % 2 == 0 else ()
                yield frame("stream", index, labels)

        results = outputs.Detections(generate(), stream=True)
        self.assertIs(results.filter(lambda candidate: candidate.has_objects), results)
        self.assertEqual(consumed, [])
        self.assertEqual([item.frame_index for item in results], [0, 2])

    def test_detection_restore_accepts_explicit_trace_and_missing_trace_is_noop(self):
        item = frame(labels=("car",))
        before = item.to_dict()
        item.restore()
        self.assertEqual(item.to_dict(), before)
        item.restore([{
            "name": "resize", "before": (80, 60), "after": (40, 30),
            "scale": (0.5, 0.5),
        }])
        self.assertEqual(item.boxes, [(0.0, 0.0, 10.0, 12.0)])
        self.assertEqual(item.image_size, (80, 60))

    def test_detections_restore_selected_or_all_frames(self):
        results = outputs.Detections([traced_frame(i) for i in range(4)])
        results.restore(index=[1, 3, 1])
        self.assertEqual(results[0].boxes[0], (5.0, 5.0, 20.0, 15.0))
        self.assertEqual(results[1].boxes[0], (10.0, 10.0, 40.0, 30.0))
        self.assertEqual(results[2].boxes[0], (5.0, 5.0, 20.0, 15.0))
        self.assertEqual(results[3].boxes[0], (10.0, 10.0, 40.0, 30.0))
        results.restore()
        self.assertTrue(all(item.boxes[0] == (10.0, 10.0, 40.0, 30.0) for item in results))

    def test_detections_restore_validates_indices_before_mutation(self):
        results = outputs.Detections([traced_frame(i) for i in range(2)])
        with self.assertRaises(IndexError):
            results.restore(index=[0, 3])
        self.assertEqual(results[0].boxes[0], (5.0, 5.0, 20.0, 15.0))
        with self.assertRaises(TypeError):
            results.restore(index=True)

    def test_detections_restore_streams_lazily(self):
        consumed = []
        def generate():
            consumed.append(True)
            yield traced_frame(0)
        results = outputs.Detections(generate(), stream=True)
        results.restore()
        self.assertEqual(consumed, [])
        restored = list(results)
        self.assertEqual(consumed, [True])
        self.assertEqual(restored[0].boxes[0], (10.0, 10.0, 40.0, 30.0))

    def test_drop_accepts_one_label_string(self):
        item = frame(); item.drop("car")
        self.assertEqual(item.labels, ["person"])

    def test_preview_box_draws_candidate_without_adding_it(self):
        item = frame(); before = item.to_dict()
        with patch("klygo.visual.show_image") as display:
            preview = item.preview_box(
                [20, 15, 30, 25], label="bicycle", score=.75,
                preview_color=(255, 0, 0),
            )
        self.assertEqual(item.to_dict(), before)
        self.assertEqual(preview.size, item.image_size)
        self.assertEqual(preview.getpixel((20, 15)), (255, 0, 0))
        display.assert_called_once()

    def test_preview_box_validates_before_display(self):
        item = frame(); before = item.to_dict()
        with patch("klygo.visual.show_image") as display:
            with self.assertRaises(ValueError):
                item.preview_box([20, 20, 10, 10], label="invalid")
        self.assertEqual(item.to_dict(), before)
        display.assert_not_called()

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

    def test_compact_rewrites_class_ids_without_changing_object_identity(self):
        item = frame(labels=("a", "b", "c", "d", "e", "f", "g"))
        results = outputs.Detections([item])
        results.drop(("b", "g"))
        results[0].add_box([20, 10, 25, 16], label="h")
        results[0].add_box([21, 11, 26, 17], label="i")
        before = [(box.uid, box.label) for box in results[0]]
        stored_ids = [box._class_id for box in results[0]]

        self.assertEqual(results.names, {0: "a", 2: "c", 3: "d", 4: "e", 5: "f", 7: "h", 8: "i"})
        results.compact()

        self.assertEqual(results.names, {0: "a", 1: "c", 2: "d", 3: "e", 4: "f", 5: "h", 6: "i"})
        self.assertEqual([(box.uid, box.label) for box in results[0]], before)
        self.assertEqual([box._class_id for box in results[0]], stored_ids)
        self.assertEqual(results[0].class_ids, list(range(7)))
        self.assertEqual(results.history[-1]["operation"], "compact")
        self.assertEqual(
            results.history[-1]["changed_fields"]["class_ids"],
            {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6},
        )

    def test_deleted_class_ids_are_reused_only_after_compact(self):
        results = outputs.Detections([frame(labels=("a", "b", "c"))])
        results.drop("c")
        results[0].add_box([10, 10, 15, 16], label="d")
        self.assertEqual(results.names, {0: "a", 1: "b", 3: "d"})

        results.compact()
        results.drop("d")
        results[0].add_box([10, 10, 15, 16], label="e")
        self.assertEqual(results.names, {0: "a", 1: "b", 3: "e"})

    def test_child_mutations_use_incremental_registry_updates(self):
        results = outputs.Detections([
            frame("video", index, ("a", "b", "c"))
            for index in range(20)
        ])
        with patch.object(
            results,
            "_synchronize_registry",
            side_effect=AssertionError("full registry scan"),
        ):
            results[10].add_box([10, 10, 15, 16], label="d")
            results[10].drop("a")
            results[10].relabel({"b": "vehicle"})
        self.assertEqual(
            results.names,
            {0: "a", 1: "b", 2: "c", 3: "d", 4: "vehicle"},
        )

    def test_single_frame_crud_does_not_rebuild_the_collection(self):
        results = outputs.Detections([frame("base", index) for index in range(20)])
        with patch.object(
            results,
            "_synchronize_registry",
            side_effect=AssertionError("full registry scan"),
        ):
            results.append(frame("append", 0, ("dog",)))
            results.insert(0, frame("insert", 0, ("cat",)))
            results.extend([frame("extend", 0, ("bird",))])
            results.replace(0, frame("ignored", 9, ("horse",)))
            results.pop()
            results.remove_frame(0, source_id="append")
        self.assertEqual(results[0].identity, ("insert", 0))
        self.assertEqual(results[0].labels, ["horse"])

    def test_sparse_bulk_operations_touch_only_indexed_frames(self):
        common = [frame("video", index, ("common",)) for index in range(20)]
        rare = frame("video", 20, ("common", "rare"))
        results = outputs.Detections([*common, rare])
        with patch.object(common[0], "drop", side_effect=AssertionError("unrelated frame")):
            results.drop("rare")
        self.assertNotIn("rare", results.names.values())

        rare.add_box([10, 10, 15, 16], label="special")
        with patch.object(common[0], "relabel", side_effect=AssertionError("unrelated frame")):
            results.relabel({"special": "vehicle"})
        self.assertIn("vehicle", results.names.values())

    def test_bulk_drop_batches_registry_updates(self):
        results = outputs.Detections([
            frame("video", index, ("car", "rare"))
            for index in range(3)
        ])
        with patch.object(
            results,
            "_refresh_frames",
            wraps=results._refresh_frames,
        ) as refresh:
            results.drop("rare")
        refresh.assert_called_once()
        self.assertEqual(len(refresh.call_args.args[0]), 3)
        self.assertEqual(results.names, {0: "car"})
        self.assertEqual(results.history[-1]["operation"], "drop_labels")

    def test_collection_copy_has_independent_annotations_and_metadata(self):
        original = outputs.Detections([frame()], metadata={"stage": {"value": "raw"}})
        working = original.copy()

        working[0][0].score = .2
        working[0][0].metadata = {"edited": True}
        working[0].relabel({"car": "vehicle"})
        working.drop("person")
        working.metadata["stage"]["value"] = "cleaned"

        self.assertEqual(original[0].labels, ["car", "person"])
        self.assertEqual(original[0][0].score, .9)
        self.assertEqual(original[0][0].metadata, {})
        self.assertEqual(original.metadata, {"stage": {"value": "raw"}})
        self.assertEqual(working[0].labels, ["vehicle"])
        self.assertEqual(working.metadata, {"stage": {"value": "cleaned"}})
        self.assertIs(original[0]._source, working[0]._source)

    def test_stream_is_lazy_one_pass_and_bounded(self):
        consumed = []

        def generate():
            for index in range(3):
                consumed.append(index)
                yield frame("stream", index, ("car",))

        results = outputs.Detections(generate(), stream=True, names={0: "car"})
        self.assertTrue(results.is_stream)
        self.assertEqual(consumed, [])
        self.assertEqual([item.frame_index for item in results], [0, 1, 2])
        self.assertEqual(consumed, [0, 1, 2])
        self.assertTrue(results.stream_exhausted)
        self.assertEqual(list(results), [])

    def test_stream_reports_requested_total_without_consumption(self):
        results = outputs.Detections(
            (frame("stream", index, ("car",)) for index in range(3)),
            stream=True,
            metadata={"requested_total_frames": 3},
        )
        self.assertEqual(results.total_frames, 3)
        self.assertTrue(results.is_stream)

    def test_stream_rejects_random_access(self):
        results = outputs.Detections(
            (frame("stream", index) for index in range(2)),
            stream=True,
        )
        with self.assertRaisesRegex(TypeError, "stream=False"):
            _ = results[0]
        iterator = iter(results)
        self.assertEqual(next(iterator).frame_index, 0)
        results.close()
        self.assertTrue(results.stream_exhausted)

    def test_stream_drop_and_relabel_are_lazy(self):
        consumed = []

        def generate():
            consumed.append(True)
            yield frame("stream", 0)

        results = outputs.Detections(generate(), stream=True)
        results.drop("person")
        results.relabel({"car": "vehicle"})
        self.assertEqual(consumed, [])
        self.assertEqual(next(iter(results)).labels, ["vehicle"])
        self.assertEqual(consumed, [True])

    def test_export_is_annotation_only(self):
        with tempfile.TemporaryDirectory() as root:
            target = files.join(root, "export"); outputs.Detections([frame()]).export(target)
            self.assertTrue(files.exists(files.join(target,"data.yaml")))
            self.assertTrue(files.exists(files.join(target,"manifest.json")))
            self.assertFalse(files.exists(files.join(target,"images")))

    def test_yolo_export_compacts_ids_without_mutating_results(self):
        with tempfile.TemporaryDirectory() as root:
            results = outputs.Detections([frame(labels=("a", "b", "c"))])
            results.drop("b")
            before = results.to_dict()
            target = files.join(root, "compact-export")
            results.export(target)

            self.assertEqual(
                files.load(files.join(target, "data.yaml"), verbose=False),
                {"names": {0: "a", 1: "c"}},
            )
            label_file = files.load(
                files.join(target, "labels", files.name(files.load(
                    files.join(target, "manifest.json"), verbose=False
                )[0]["label_file"])),
                verbose=False,
            )
            lines = label_file.splitlines() if isinstance(label_file, str) else label_file
            self.assertEqual(
                [line.split()[0] for line in lines if line.strip()],
                ["0", "1"],
            )
            self.assertEqual(results.to_dict(), before)

    def test_columnar_lazy_box_materialization_and_slots(self):
        built = outputs.build_detections(
            [Image.new("RGB", (64, 48))],
            [{"boxes": [[1, 2, 10, 12], [5, 6, 20, 24]], "scores": [.9, .7], "labels": ["car", "person"]}],
        )
        det = built[0]
        self.assertIsNone(det._boxes_cache)
        self.assertEqual(det.boxes, [(1.0, 2.0, 10.0, 12.0), (5.0, 6.0, 20.0, 24.0)])
        self.assertEqual(det.scores, [0.9, 0.7])
        self.assertEqual(det.labels, ["car", "person"])
        self.assertEqual(det.class_ids, [0, 1])
        self.assertEqual(det.uids, [0, 1])
        self.assertIsNone(det._boxes_cache)

        results = outputs.Detections(built)
        self.assertEqual(results._baseline_signatures, {})
        results.drop("person")
        self.assertIsNone(results[0]._boxes_cache)
        self.assertIn(results[0].identity, results.modified_frames)

        box = results[0][0]
        self.assertFalse(hasattr(box, "__dict__"))
        self.assertIsNone(box._metadata)
        self.assertEqual(box.metadata, {})
        box.metadata["tag"] = "tracked"
        self.assertEqual(results[0].to_dict()["boxes"][0]["metadata"], {"tag": "tracked"})

    def test_astype_preserves_precision_through_crud_copy_and_pipeline(self):
        item = frame(labels=("car", "person"))
        cached = item[0]
        item.astype(32)
        self.assertEqual(item.dtype, 32)
        self.assertEqual(item._xyxy.dtype, np.dtype("float32"))
        self.assertEqual(item._scores.dtype, np.dtype("float32"))
        self.assertEqual(item._class_ids.dtype, np.dtype("int32"))
        self.assertEqual(item._uids.dtype, np.dtype("int32"))
        self.assertEqual(cached.box, tuple(float(value) for value in item._xyxy[0]))

        item.add_box([10, 10, 20, 20], label="bus", score=.75)
        item.relabel({"person": "human"})
        self.assertEqual(item.dtype, 32)
        self.assertTrue(all(array.dtype.itemsize == 4 for array in (
            item._xyxy, item._scores, item._class_ids, item._uids,
        )))

        results = outputs.Detections([item])
        copied = results.copy()
        self.assertEqual(copied.modified_frames, ())
        copied.astype(64)
        self.assertEqual(copied.modified_frames, ())
        self.assertEqual(copied.history[-1]["operation"], "astype")
        copied.astype(32)
        copied.transform(postprocessing.threshold(.5))
        self.assertEqual(copied[0].dtype, 32)
        copied.astype(64)
        self.assertEqual(copied[0].dtype, 64)
        self.assertTrue(all(array.dtype.itemsize == 8 for array in (
            copied[0]._xyxy, copied[0]._scores, copied[0]._class_ids, copied[0]._uids,
        )))

    def test_astype_validates_before_collection_mutation_and_streams_lazily(self):
        normal = frame("video", 0, ("car",))
        huge = outputs.Detection([
            outputs.Box([0, 0, 1e39, 10], label="car", class_id=0)
        ], source_id="video", frame_index=1, image_size=(40, 30), names={0: "car"})
        results = outputs.Detections([normal, huge])
        with self.assertRaises(OverflowError):
            results.astype(32)
        self.assertEqual([item.dtype for item in results], [64, 64])
        with self.assertRaises(ValueError):
            normal.astype(16)

        consumed = []
        def generate():
            consumed.append(True)
            yield frame("stream", 0, ("car",))
        stream = outputs.Detections(generate(), stream=True)
        stream.astype(32)
        self.assertEqual(consumed, [])
        streamed = list(stream)
        self.assertEqual(consumed, [True])
        self.assertEqual(streamed[0].dtype, 32)

    def test_cow_baseline_revert_clears_modified_flag(self):
        results = outputs.Detections([frame()])
        self.assertEqual(results.modified_frames, ())
        original_score = results[0][0].score
        results[0][0].score = 0.15
        self.assertEqual(results.modified_frames, (("image-a", 0),))
        results[0][0].score = original_score
        self.assertEqual(results.modified_frames, ())

    def test_ten_thousand_frames_memory_and_stream_efficiency(self):
        import tracemalloc

        img = Image.new("RGB", (64, 64))
        raw_boxes = [[i * 2, i * 2, i * 2 + 10, i * 2 + 10] for i in range(10)]
        raw_scores = [0.95 - i * 0.05 for i in range(10)]
        raw_labels = ["car" if i % 2 == 0 else "person" for i in range(10)]
        names = {0: "car", 1: "person"}

        tracemalloc.start()
        frames = [
            outputs.Detection._from_arrays(
                outputs.detect.np.array(raw_boxes, dtype=outputs.detect.np.float64),
                outputs.detect.np.array(raw_scores, dtype=outputs.detect.np.float64),
                outputs.detect.np.array([0 if i % 2 == 0 else 1 for i in range(10)], dtype=outputs.detect.np.int64),
                outputs.detect.np.arange(10, dtype=outputs.detect.np.int64),
                list(raw_labels),
                source_id="video",
                frame_index=idx,
                image_size=(64, 64),
                source=img,
                registry=names,
            )
            for idx in range(10_000)
        ]
        results = outputs.Detections(frames, names=names)
        current_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(len(results), 10_000)
        self.assertEqual(results.total_objects, 100_000)
        self.assertLess(current_bytes, 35 * 1024 * 1024)

if __name__ == "__main__": unittest.main()
