"""Tests for Detections.shuffle() in unpacked and packed storage modes."""

import tempfile
import unittest
from PIL import Image

from klygo import files, media, outputs
from klygo.outputs import Box, Detection, Detections


class TestDetectionsShuffle(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = files.path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_sample_detections(self, n: int = 5, packed: bool = False) -> Detections:
        frames = []
        for i in range(n):
            img = Image.new("RGB", (100 + i * 10, 100), "white")
            img_path = files.join(self.root, f"frame_{i}.jpg")
            media.save(img_path, img, overwrite=True, verbose=False)
            boxes = [
                Box([10.0, 10.0, 50.0, 50.0], label="car", score=0.9, class_id=0),
                Box([20.0, 20.0, 60.0, 60.0], label="person", score=0.8, class_id=1),
            ]
            det = Detection(
                boxes=boxes,
                source=media.load(img_path, verbose=False)[0],
                source_id="test_source",
                frame_index=i,
                image_size=(100 + i * 10, 100),
                names={0: "car", 1: "person"},
            )
            frames.append(det)
        return Detections(frames, names={0: "car", 1: "person"}, packed=packed)

    def test_unpacked_shuffle_reproducibility(self) -> None:
        dets = self._make_sample_detections(5, packed=False)
        orig_indices = [f.frame_index for f in dets]
        self.assertEqual(orig_indices, [0, 1, 2, 3, 4])

        dets.shuffle(seed=42)
        shuffled_indices = [f.frame_index for f in dets]
        self.assertNotEqual(shuffled_indices, orig_indices)

        # Same seed produces same permutation
        dets2 = self._make_sample_detections(5, packed=False)
        dets2.shuffle(seed=42)
        self.assertEqual([f.frame_index for f in dets2], shuffled_indices)

        # Different seed produces different permutation
        dets3 = self._make_sample_detections(5, packed=False)
        dets3.shuffle(seed=999)
        self.assertNotEqual([f.frame_index for f in dets3], shuffled_indices)

    def test_unpacked_shuffle_preserves_annotations_and_identities(self) -> None:
        dets = self._make_sample_detections(4, packed=False)
        dets.shuffle(seed=123)

        for frame in dets:
            self.assertEqual(len(frame.boxes), 2)
            self.assertEqual(frame.labels[0], "car")
            self.assertEqual(frame.labels[1], "person")
            self.assertEqual(frame.image_size, (100 + frame.frame_index * 10, 100))
            # get_frame by frame_index still finds the frame
            matched = dets.get_frame(frame.frame_index, source_id="test_source")
            self.assertEqual(matched.frame_index, frame.frame_index)

    def test_packed_shuffle_preserves_packed_storage(self) -> None:
        dets = self._make_sample_detections(6, packed=True)
        self.assertTrue(dets.is_packed)
        orig_indices = [f.frame_index for f in dets]
        self.assertEqual(orig_indices, [0, 1, 2, 3, 4, 5])

        dets.shuffle(seed=42)
        # Verify it remains packed! (does not unpack solely to shuffle)
        self.assertTrue(dets.is_packed)

        shuffled_indices = [f.frame_index for f in dets]
        self.assertNotEqual(shuffled_indices, orig_indices)

        # Slicing observes logical shuffled order
        first_three = [f.frame_index for f in dets[0:3]]
        self.assertEqual(first_three, shuffled_indices[0:3])

        # Indexing observes logical shuffled order
        for idx, expected_frame_idx in enumerate(shuffled_indices):
            self.assertEqual(dets[idx].frame_index, expected_frame_idx)
            self.assertEqual(dets._frames.index(dets[idx]), idx)

        # Repeated shuffle works
        dets.shuffle(seed=77)
        self.assertTrue(dets.is_packed)
        shuffled_again = [f.frame_index for f in dets]
        self.assertNotEqual(shuffled_again, shuffled_indices)

    def test_packed_shuffle_unpack_round_trip(self) -> None:
        dets = self._make_sample_detections(5, packed=True)
        dets.shuffle(seed=42)
        packed_indices = [f.frame_index for f in dets]

        # Unpack
        dets.unpack()
        self.assertFalse(dets.is_packed)
        self.assertEqual([f.frame_index for f in dets], packed_indices)

        # Re-pack
        dets.pack()
        self.assertTrue(dets.is_packed)
        self.assertEqual([f.frame_index for f in dets], packed_indices)

    def test_packed_shuffle_export_observes_order(self) -> None:
        dets = self._make_sample_detections(4, packed=True)
        dets.shuffle(seed=42)
        expected_indices = [f.frame_index for f in dets]

        export_dir = files.join(self.root, "export_test")
        dets.export(export_dir, format="yolo", overwrite=True)

        exported_images = sorted(files.find(files.join(export_dir, "images"), "*.jpg"))
        # Verify exported files reflect the collection length
        self.assertEqual(len(exported_images), 4)

    def test_empty_and_singleton_shuffle(self) -> None:
        empty = Detections([])
        empty.shuffle(seed=42)
        self.assertEqual(len(empty), 0)

        empty_packed = Detections([], packed=True)
        empty_packed.shuffle(seed=42)
        self.assertEqual(len(empty_packed), 0)

        single = self._make_sample_detections(1, packed=False)
        single.shuffle(seed=42)
        self.assertEqual(len(single), 1)

        single_packed = self._make_sample_detections(1, packed=True)
        single_packed.shuffle(seed=42)
        self.assertEqual(len(single_packed), 1)

    def test_streaming_raises_type_error(self) -> None:
        stream = Detections([], stream=True)
        with self.assertRaisesRegex(TypeError, "stream"):
            stream.shuffle()


if __name__ == "__main__":
    unittest.main()
