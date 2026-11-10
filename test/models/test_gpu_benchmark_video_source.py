"""A caller-provided video must be consumed unchanged and hashed separately."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from klygo import files
from benchmarks.gpu_suite.run import run_model


class TestVideoSource(unittest.TestCase):
    def test_existing_input_is_not_replaced_by_generated_bus_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = files.path(temporary)
            video = root / 'input.mp4'
            video.write_bytes(b'fixed external video fixture')
            source = root / 'bus.jpg'
            source.write_bytes(b'placeholder not used for video inference')
            args = SimpleNamespace(model='yolo', device='cpu', output=str(root / 'results'),
                                   profile='video', batch=1, source_video=str(video),
                                   iterations=1, repeats=2, imgsz=640, video_frames=120)
            detector = SimpleNamespace(model=SimpleNamespace(config=None), model_id='fixture',
                                       dtype='float32', device='cpu')
            captured = [{'boxes': [], 'scores': [], 'labels': []}]
            def measure(*positional, **kwargs):
                self.assertEqual(positional[6], str(video.resolve()))
                return dict(seconds=1, fps=1, frames=1, boxes=0, first_frame_host_ms=1,
                            delivery_intervals_ms=[1], pack_latencies_ms=[1]), captured
            with patch('benchmarks.gpu_suite.run.prepare', return_value=(source, [object()])), \
                 patch('benchmarks.gpu_suite.run.build', return_value=(detector, None)), \
                 patch('benchmarks.gpu_suite.run.measure', side_effect=measure), \
                 patch('benchmarks.gpu_suite.run.media.save_video') as generated:
                run_model(args)
            generated.assert_not_called()
            report = files.load(root / 'results/yolo.json', verbose=False)
            self.assertEqual(report['input_sha256'], files.hash(video))
            self.assertIn('caller-provided video', report['input_scope'])
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['device_at_factory_return'], 'cpu')
            self.assertEqual(report['device_observation_stage'], 'after complete warmup, before measured repeats')


if __name__ == '__main__':
    unittest.main()
