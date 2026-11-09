"""Benchmark paths must not inherit persistent upstream stream configuration."""

from types import SimpleNamespace
import unittest

from benchmarks.gpu_suite.run import execute


class Recorder:
    def __init__(self):
        self.calls = []
        self.options = {}

    def predict(self, source, **options):
        self.options.update(options)
        self.calls.append(dict(self.options))
        return []


class TestGpuBenchmarkModes(unittest.TestCase):
    def test_native_and_managed_paths_reset_the_upstream_stream_flag(self):
        detector = Recorder()
        args = SimpleNamespace(batch=1)
        for mode in ("native-stream", "native-non-stream", "managed-stream", "managed-non-stream"):
            execute(detector,"yolo",None,mode,["image"],args)
        self.assertEqual([call["model_stream"] for call in detector.calls],[True,False,False,False])
        self.assertEqual([call.get("stream",False) for call in detector.calls],[False,False,True,False])

    def test_non_ultralytics_paths_do_not_receive_an_unsupported_stream_option(self):
        detector = Recorder()
        execute(detector,"grounding-dino",["person"],"native-non-stream",["image"],SimpleNamespace(batch=1))
        self.assertNotIn("model_stream",detector.calls[0])


if __name__=="__main__":
    unittest.main()
