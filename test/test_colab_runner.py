"""Check Colab command orchestration without running benchmark workloads."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

from klygo import files


class ColabRunnerTest(unittest.TestCase):
    def test_device_forwarding_to_all_six_jobs(self):
        path = files.resolve(__file__).parent.parent / 'benchmarks' / 'colab_runner.py'
        spec = importlib.util.spec_from_file_location('colab_runner_test_target', path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for device, expected in (('2', 'cuda:2'), ('cpu', 'cpu')):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as tmp:
                with patch.object(sys, 'argv', ['runner', '--workdir', tmp, '--device', device]), \
                     patch.object(runner, 'prepare_voc2007', return_value=files.path(tmp) / 'voc'), \
                     patch.object(runner, 'prepare_val2017', return_value=files.path(tmp) / 'coco'), \
                     patch.object(runner, 'run_cmd') as command, \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                self.assertEqual(command.call_count, 6)
                for call in command.call_args_list:
                    args = call.args[0]
                    self.assertEqual(args[args.index('--device') + 1],
                                     expected if '/metrics/' in args[1] else device)
