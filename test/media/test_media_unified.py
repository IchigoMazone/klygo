import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from klygo import files, media, preprocessing


class TestMediaUnified(unittest.TestCase):
    """Test unified streaming and materialized media container behavior."""

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.tmp_path = files.path(temporary_directory.name)

        # Create 5 test frames
        self.img_dir = self.tmp_path / "test_frames"
        files.mkdir(self.img_dir)
        for i in range(5):
            img = Image.new("RGB", (32, 24), color=(i * 40, i * 40, i * 40))
            media.save(self.img_dir / f"frame_{i:02d}.png", img)

        # Create a video file
        self.video_path = self.tmp_path / "test_video.mp4"
        writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (32, 24),
        )
        for i in range(6):
            writer.write(np.full((24, 32, 3), i * 30, dtype=np.uint8))
        writer.release()

    def test_type_hierarchy_and_mode_flags(self):
        loaded = media.load(self.img_dir, stream=False)
        streamed = media.load(self.img_dir, stream=True)

        self.assertFalse(loaded.is_stream)
        self.assertTrue(streamed.is_stream)

        # One public container type represents both modes.
        self.assertIs(type(loaded), media.MediaFrames)
        self.assertIsInstance(loaded, list)

        self.assertIs(type(streamed), media.MediaFrames)

    def test_indexed_mode_creates_proxies_on_demand(self):
        indexed = media.load(self.img_dir, indexed=True)

        self.assertFalse(indexed.is_stream)
        self.assertTrue(indexed.is_indexed)
        self.assertEqual(len(indexed), 5)
        self.assertEqual(list.__len__(indexed), 0)
        self.assertEqual(indexed.loaded_count, 0)

        first = indexed[0]
        self.assertIs(first, indexed[0])
        self.assertIsInstance(first, media.LazyImage)
        self.assertFalse(first.is_loaded)
        first.load()
        self.assertEqual(indexed.loaded_count, 1)
        indexed.unload()
        self.assertEqual(indexed.loaded_count, 0)

        subset = indexed[1:5:2]
        self.assertTrue(subset.is_indexed)
        self.assertEqual(len(subset), 2)
        self.assertEqual([item.path.name for item in subset], ["frame_01.png", "frame_03.png"])

    def test_indexed_mode_crud_and_lazy_transform(self):
        indexed = media.load(self.img_dir, indexed=True)
        transformed = indexed.transform(preprocessing.resize((10, 8)))
        self.assertTrue(transformed.is_indexed)
        self.assertEqual(list.__len__(transformed), 0)
        self.assertEqual(transformed[0].size, (10, 8))

        replacement = Image.new("RGB", (7, 7), "red")
        indexed[0] = replacement
        indexed.insert(1, replacement)
        indexed.append(replacement)
        self.assertIs(indexed[0], replacement)
        self.assertIn(replacement, indexed)
        self.assertEqual(indexed.index(replacement), 0)
        self.assertEqual(indexed.count(replacement), 3)
        self.assertIs(indexed.pop(1), replacement)
        del indexed[-1]
        self.assertEqual(len(indexed), 5)
        copied = indexed.copy()
        self.assertEqual(copied, indexed)
        copied.reverse()
        self.assertEqual(list(reversed(copied)), list(indexed))

    def test_indexed_video_keeps_frame_numbers_without_proxies(self):
        indexed = media.load(self.video_path, indexed=True)
        self.addCleanup(indexed.close)

        self.assertTrue(indexed.is_indexed)
        self.assertEqual(len(indexed), 6)
        self.assertEqual(list.__len__(indexed), 0)
        self.assertEqual(indexed[3].frame_index, 3)
        self.assertIs(indexed[3], indexed[3])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            media.MediaFrames(self.video_path, stream=True, indexed=True)

    def test_require_materialized_guards_on_stream(self):
        streamed = media.load(self.img_dir, stream=True)

        with self.assertRaisesRegex(TypeError, "indexing and slicing is unavailable for a stream"):
            _ = streamed[0]

        with self.assertRaisesRegex(TypeError, "indexing and slicing is unavailable for a stream"):
            _ = streamed[0:2]

        with self.assertRaisesRegex(TypeError, "len\\(\\) is unavailable for a stream"):
            _ = len(streamed)

        with self.assertRaisesRegex(TypeError, "loaded_count is unavailable for a stream"):
            _ = streamed.loaded_count

        with self.assertRaisesRegex(TypeError, "unload\\(\\) is unavailable for a stream"):
            streamed.unload()

        with self.assertRaisesRegex(TypeError, "to_list\\(\\) is unavailable for a stream"):
            streamed.to_list()

        with self.assertRaisesRegex(TypeError, "append\\(\\) is unavailable for a stream"):
            streamed.append(None)

        with self.assertRaisesRegex(TypeError, "pop\\(\\) is unavailable for a stream"):
            streamed.pop()

        with self.assertRaisesRegex(TypeError, "clear\\(\\) is unavailable for a stream"):
            streamed.clear()

    def test_unified_filtering_and_quality_on_materialized_frames(self):
        loaded = media.load(self.img_dir, stream=False)
        self.assertEqual(len(loaded), 5)

        # where
        filtered = loaded.where(lambda f: "02" not in str(f.path))
        self.assertEqual(len(filtered), 4)
        self.assertIsInstance(filtered, media.MediaFrames)

        # filter (alias)
        filtered2 = loaded.filter(lambda f: "02" not in str(f.path))
        self.assertEqual(len(filtered2), 4)

        # where_quality on materialized frames
        quality_filtered = loaded.where_quality(min_brightness=0.3)
        self.assertIsInstance(quality_filtered, media.MediaFrames)
        self.assertTrue(len(quality_filtered) < len(loaded))

        # where_motion on materialized frames
        motion_filtered = loaded.where_motion(threshold=0.01)
        self.assertIsInstance(motion_filtered, media.MediaFrames)

        # deduplicate on materialized frames
        deduped = loaded.deduplicate(threshold=0.999)
        self.assertIsInstance(deduped, media.MediaFrames)

    def test_unified_stream_and_frames_constructor_switching(self):
        streamed = media.MediaFrames(self.video_path, stream=True)
        self.assertIs(type(streamed), media.MediaFrames)
        self.assertTrue(streamed.is_stream)

        materialized = media.MediaFrames(self.video_path, stream=False)
        self.assertIs(type(materialized), media.MediaFrames)
        self.assertIsInstance(materialized, list)
        self.assertFalse(materialized.is_stream)

        default_mode = media.MediaFrames(self.video_path)
        self.assertIs(type(default_mode), media.MediaFrames)
        self.assertFalse(default_mode.is_stream)

    def test_save_video_and_save_images_on_both_modes(self):
        loaded = media.load(self.img_dir, stream=False)
        streamed = media.load(self.img_dir, stream=True)

        # save_video from materialized
        out_vid1 = self.tmp_path / "from_frames.mp4"
        saved1 = loaded.save_video(out_vid1, fps=15)
        self.assertTrue(files.exists(saved1))

        # save_video from stream
        out_vid2 = self.tmp_path / "from_stream.mp4"
        saved2 = streamed.save_video(out_vid2, fps=15)
        self.assertTrue(files.exists(saved2))

        # save_images from stream
        out_imgs = self.tmp_path / "stream_out_imgs"
        saved_imgs = streamed.save_images(out_imgs, prefix="st_")
        self.assertTrue(files.exists(out_imgs))
        self.assertEqual(len(saved_imgs), 5)

    def test_media_compose_and_chaining_pipeline(self):
        # 1. media.compose with varargs
        pipe1 = media.compose(
            media.compose.quality(min_brightness=0.1),
            media.compose.sample(1),
            preprocessing.resize((16, 16)),
        )
        stream_out = pipe1(media.load(self.img_dir, stream=True))
        self.assertTrue(stream_out.is_stream)
        stream_frames = list(stream_out)
        self.assertTrue(len(stream_frames) > 0)
        self.assertEqual(stream_frames[0].size, (16, 16))

        # 2. media.compose with list
        pipe2 = media.compose([
            media.compose.where_quality(min_brightness=0.1),
            media.compose.deduplicate(0.99),
            media.compose.limit(3),
            preprocessing.resize((12, 12)),
        ])
        frames_out = pipe2(media.load(self.img_dir, stream=False))
        self.assertFalse(frames_out.is_stream)
        self.assertIsInstance(frames_out, media.MediaFrames)
        self.assertTrue(len(frames_out) <= 3)
        self.assertEqual(frames_out[0].size, (12, 12))

        # 3. Apply compose directly to file path string
        from_path = pipe2(str(self.img_dir))
        self.assertIsInstance(from_path, media.MediaFrames)

    def test_transform_updates_collection_dimensions_without_decoding(self):
        materialized = media.load(self.img_dir, stream=False).transform(preprocessing.resize((10, 8)))
        streamed = media.load(self.video_path, stream=True).transform(preprocessing.resize((10, 8)))
        self.assertEqual((materialized.width, materialized.height), (10, 8))
        self.assertEqual((streamed.width, streamed.height), (10, 8))
        self.assertTrue(all(not frame.is_loaded for frame in materialized))

    def test_save_video_rejects_mixed_sizes_without_partial_output(self):
        output = self.tmp_path / "mixed.mp4"
        frames = [Image.new("RGB", (20, 20)), Image.new("RGB", (30, 20))]
        with self.assertRaisesRegex(ValueError, "all video frames must have size"):
            media.save_video(output, frames)
        self.assertFalse(files.exists(output))
        self.assertEqual(list(files.iter_find(self.tmp_path, pattern="*.tmp.mp4")), [])

    def test_sample_limit_and_transform_methods(self):
        # Streaming mode
        stream = media.load(self.img_dir, stream=True)
        sampled_stream = stream.sample(2)
        limited_stream = sampled_stream.limit(2)
        self.assertTrue(limited_stream.is_stream)
        frames = list(limited_stream)
        self.assertTrue(len(frames) <= 2)

        # Materialized mode
        m_frames = media.load(self.img_dir, stream=False)
        sampled_frames = m_frames.sample(2)
        limited_frames = sampled_frames.limit(2)
        self.assertFalse(limited_frames.is_stream)
        self.assertEqual(len(limited_frames), 2)

        # transform method on both
        p_res1 = stream.transform(lambda s: s.where_quality(min_brightness=0.1))
        self.assertTrue(p_res1.is_stream)
        p_res2 = m_frames.transform(lambda f: f.limit(1))
        self.assertEqual(len(p_res2), 1)

        # transform on both accepting media.compose and preprocessing.compose
        med_pipe = media.compose(media.quality(min_brightness=0.1), media.sample(1))
        prep_pipe = preprocessing.resize((10, 10))
        applied_stream = stream.transform(med_pipe).transform(prep_pipe)
        self.assertTrue(applied_stream.is_stream)
        self.assertEqual(list(applied_stream)[0].size, (10, 10))

        applied_frames = m_frames.transform(med_pipe).transform(prep_pipe)
        self.assertFalse(applied_frames.is_stream)
        self.assertEqual(applied_frames[0].size, (10, 10))

    def test_stream_sampling_updates_fps_and_none_limit_is_a_noop(self):
        stream = media.load(self.video_path, stream=True)
        sampled = stream.sample(2)
        self.assertEqual(sampled.fps, stream.fps / 2)

        limited = sampled.limit(2)
        unchanged = limited.limit(None)
        self.assertEqual(unchanged.max_frames, 2)
        self.assertEqual(len(list(unchanged)), 2)

    def test_transform_propagates_collection_callable_errors_once(self):
        stream = media.load(self.img_dir, stream=True)
        calls = []

        def fail(value):
            calls.append(type(value).__name__)
            raise RuntimeError("expected failure")

        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            stream.transform(fail)
        self.assertEqual(calls, ["MediaFrames"])

    def test_context_manager_on_both_modes(self):
        with media.load(self.video_path, stream=False) as frames:
            self.assertEqual(len(frames), 6)
            self.assertFalse(frames.is_stream)

        with media.load(self.video_path, stream=True) as stream:
            self.assertTrue(stream.is_stream)
            first = next(iter(stream))
            self.assertIsNotNone(first)


if __name__ == "__main__":
    unittest.main()
