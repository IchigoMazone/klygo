"""Tests for MediaFrames.shuffle()."""

import tempfile
import unittest
from PIL import Image

from klygo import files, media
from klygo.media import MediaFrames


class TestMediaFramesShuffle(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = files.path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_sample_images(self, n: int = 5) -> list[str]:
        paths = []
        for i in range(n):
            path = files.join(self.root, f"img_{i:02d}.jpg")
            img = Image.new("RGB", (50, 50), color=(i * 40, i * 40, i * 40))
            media.save(path, img, overwrite=True, verbose=False)
            paths.append(path)
        return paths

    def test_indexed_shuffle_reproducibility(self) -> None:
        self._create_sample_images(6)
        frames = media.load(self.root, indexed=True, verbose=False)
        self.assertTrue(frames.is_indexed)

        orig_paths = [files.name(f.path) for f in frames]
        self.assertEqual(orig_paths, [f"img_{i:02d}.jpg" for i in range(6)])

        frames.shuffle(seed=42)
        shuffled_paths = [files.name(f.path) for f in frames]
        self.assertNotEqual(shuffled_paths, orig_paths)

        # Same seed gives same permutation
        frames2 = media.load(self.root, indexed=True, verbose=False)
        frames2.shuffle(seed=42)
        self.assertEqual([files.name(f.path) for f in frames2], shuffled_paths)

        # Different seed gives different permutation
        frames3 = media.load(self.root, indexed=True, verbose=False)
        frames3.shuffle(seed=999)
        self.assertNotEqual([files.name(f.path) for f in frames3], shuffled_paths)

    def test_materialized_list_shuffle(self) -> None:
        paths = self._create_sample_images(5)
        # Materialize directly into list-backed MediaFrames
        raw_items = [media.LazyImage(p) for p in paths]
        frames = MediaFrames._from_items(raw_items)
        self.assertFalse(frames.is_stream)
        self.assertFalse(frames.is_indexed)

        orig_names = [files.name(f.path) for f in frames]
        frames.shuffle(seed=42)
        shuffled_names = [files.name(f.path) for f in frames]
        self.assertNotEqual(shuffled_names, orig_names)
        self.assertEqual(set(shuffled_names), set(orig_names))

    def test_stream_shuffle_raises_type_error(self) -> None:
        self._create_sample_images(3)
        stream_frames = media.load(self.root, stream=True, verbose=False)
        self.assertTrue(stream_frames.is_stream)
        with self.assertRaisesRegex(TypeError, "stream"):
            stream_frames.shuffle()

    def test_empty_and_singleton(self) -> None:
        empty = MediaFrames._from_items([])
        empty.shuffle(seed=42)
        self.assertEqual(len(empty), 0)

        single = MediaFrames._from_items([media.LazyImage(self._create_sample_images(1)[0])])
        single.shuffle(seed=42)
        self.assertEqual(len(single), 1)


if __name__ == "__main__":
    unittest.main()
