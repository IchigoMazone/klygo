"""Validate isolated dependency repair and prevent incomplete release charts."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from klygo import files
from benchmarks.gpu_suite import evaluate_release as release
from benchmarks.gpu_suite.locate_runtime import LocateRuntime, result_errors, SUPPORTED, UNSUPPORTED


def report(dtype='float32', device='cpu', repeats=2):
    return {'model': 'locate-anything', 'status': 'complete',
            'settings': {'dtype': dtype, 'device': device, 'repeats': repeats},
            'actual_model_dtype': f'torch.{dtype}', 'actual_model_device': device,
            'cases': [{'mode': m, 'status': 'N/A'} for m in sorted(UNSUPPORTED)] + [
                {'mode': m, 'status': 'ok', 'fps_median': 2., 'ready_interval_p50_ms': 500.,
                 'runs': [{'fps': 2.} for _ in range(repeats)],
                 'parity': [{'pass': True} for _ in range(repeats)], 'all_parity_pass': True}
                for m in sorted(SUPPORTED)]}


class TestLocateRelease(unittest.TestCase):
    def test_result_gate_requires_real_complete_precision_and_parity(self):
        valid = report()
        self.assertEqual(result_errors(valid, 'float32', 'cpu', 2), [])
        edits = [lambda r: r.update(status='error'),
                 lambda r: r.update(actual_model_dtype='torch.float16'),
                 lambda r: r.update(actual_model_device='cuda:0'),
                 lambda r: r['cases'].pop(),
                 lambda r: r['cases'][-1].update(all_parity_pass=False),
                 lambda r: r['cases'][-1].update(fps_median=float('nan')),
                 lambda r: r['cases'][-1].update(runs=[]),
                 lambda r: r['cases'][-1]['parity'][0].update({'pass': False})]
        for edit in edits:
            data = deepcopy(valid)
            edit(data)
            self.assertTrue(result_errors(data, 'float32', 'cpu', 2))

    def test_missing_precision_blocks_all_final_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            output = files.path(root)
            summary = {'models': []}
            with patch.object(release, '_generate_charts') as charts, self.assertRaisesRegex(RuntimeError, 'blocked'):
                release._finalize(summary, output, require_locate=True,
                                  dtypes=['float32', 'float16'], device='cpu', repeats=2)
            charts.assert_not_called()
            self.assertFalse(files.exists(output / 'summary.json'))
            self.assertFalse(files.exists(output / 'REPORT.md'))
            self.assertTrue(files.exists(output / 'summary.partial.json'))
            self.assertEqual(files.load(output / 'completion.json', verbose=False)['status'], 'incomplete')

    def test_cv2_repair_installs_into_managed_environment(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = LocateRuntime(Path(root) / 'run', venv_dir=Path(root) / 'env')
            ok = subprocess.CompletedProcess([], 0, 'probe metadata', '')
            missing = subprocess.CompletedProcess([], 1, '', "ModuleNotFoundError: No module named 'cv2'")
            with patch.object(runtime, '_call', side_effect=[missing, ok, ok, ok, ok]) as calls:
                runtime.prepare()
            commands = [c.args[0] for c in calls.call_args_list]
            installs = [cmd for cmd in commands if 'pip' in cmd]
            self.assertEqual(len(installs), 2)
            self.assertTrue(all(cmd[0] == str(Path(root) / 'env/bin/python') for cmd in installs))
            self.assertTrue(any('opencv-python-headless>=4.10' in cmd for cmd in installs))
            self.assertNotEqual(runtime.python, sys.executable)

    def test_unknown_errors_are_not_retried_or_hidden(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = LocateRuntime(root, retries=2)
            runtime.ready = True
            failed = subprocess.CompletedProcess([], 1, '', 'RuntimeError: CUDA out of memory')
            with patch.object(runtime, '_call', return_value=failed) as calls, patch.object(runtime, '_install') as install:
                result = runtime.run(['python', 'runner'], Path(root) / 'case', 'float32')
            self.assertEqual(result.returncode, 1)
            self.assertEqual(calls.call_count, 1)
            install.assert_not_called()

    def test_managed_repair_cannot_modify_supplied_environment(self):
        with tempfile.TemporaryDirectory() as root:
            environment = Path(root) / "supplied"
            runtime = LocateRuntime(Path(root) / "run", python=str(environment / "bin/python"), venv_dir=environment)
            probe = subprocess.CompletedProcess([], 0, str(environment), "")
            with patch.object(runtime, "_call", return_value=probe) as calls:
                with self.assertRaisesRegex(RuntimeError, "supplied interpreter"):
                    runtime._create_managed()
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(calls.call_args.args[0][1], "-c")

    def test_known_dependency_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = LocateRuntime(root, retries=2)
            failed = subprocess.CompletedProcess([], 1, "", "No module named 'cv2'")
            with patch.object(runtime, "prepare"), patch.object(runtime, "_call", return_value=failed) as calls, patch.object(runtime, "_install") as install:
                result = runtime.run(["python", "runner"], Path(root) / "case", "float32")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(calls.call_count, 3)
            self.assertEqual(install.call_count, 2)

    def test_hardware_or_package_change_prevents_merge(self):
        with self.assertRaisesRegex(RuntimeError, "gpu_uuid"):
            release._check_resume_environment({"gpu_uuid": "GPU-a"}, {"gpu_uuid": "GPU-b"})
        with self.assertRaisesRegex(RuntimeError, "torch"):
            release._check_resume_environment({"versions": {"torch": "a"}}, {"versions": {"torch": "b"}})

    def test_failed_locate_then_only_retry_merges_existing_other_models(self):
        with tempfile.TemporaryDirectory() as root:
            output = files.path(root) / 'release'
            base = ['evaluate_release.py', '--output', str(output), '--include-models',
                    '--models', 'yolo,locate-anything', '--device', 'cpu', '--repeats', '2',
                    '--allow-remote-code']
            def launch(cmd, **kwargs):
                model = cmd[cmd.index('--model') + 1]
                dtype = cmd[cmd.index('--dtype') + 1]
                dest = files.path(cmd[cmd.index('--output') + 1])
                files.mkdir(dest)
                data = report(dtype)
                data['model'] = model
                files.save(dest / f'{model}.json', data, overwrite=True, verbose=False)
                return subprocess.CompletedProcess(cmd, 0, '', '')
            def broken_locate(cmd, dest, dtype):
                if dtype == 'float16':
                    return subprocess.CompletedProcess(cmd, 1, '', "No module named 'cv2'")
                return launch(cmd)
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(patch.object(release.utils, "tabulate"))
                for name in ('benchmark_metrics', 'benchmark_groups_and_postprocessing', 'benchmark_mining'):
                    stack.enter_context(patch.object(release, name, return_value=[]))
                stack.enter_context(patch.object(release, 'benchmark_datasets_media_visual', return_value=([], 'synthetic')))
                stack.enter_context(patch.object(release, 'collect_environment', return_value={'versions': {}}))
                calls = stack.enter_context(patch.object(release.subprocess, 'run', side_effect=launch))
                located = stack.enter_context(patch.object(LocateRuntime, 'run', side_effect=broken_locate))
                charts = stack.enter_context(patch.object(release, '_generate_charts', return_value=[]))
                stack.enter_context(patch.object(sys, 'argv', base))
                with self.assertRaisesRegex(RuntimeError, 'blocked'):
                    release.main()
                charts.assert_not_called()
                self.assertEqual(calls.call_count, 2)  # Two YOLO precisions only.
                files.save(output / 'scaling_runs.json', [{'model': 'yolo', 'fps_median': 99.}], verbose=False)
                calls.reset_mock()
                located.side_effect = lambda cmd, dest, dtype: launch(cmd)
                sys.argv[:] = ['evaluate_release.py', '--locate-only', '--output', str(output), '--allow-remote-code']
                for name in ('benchmark_metrics', 'benchmark_groups_and_postprocessing', 'benchmark_mining'):
                    getattr(release, name).reset_mock()
                release.main()
                calls.assert_not_called()
                for name in ('benchmark_metrics', 'benchmark_groups_and_postprocessing', 'benchmark_mining'):
                    getattr(release, name).assert_not_called()
                charts.assert_called_once()
                rendered = charts.call_args.args[0]
                self.assertEqual({row['dtype'] for row in rendered['model_modes']
                                  if row['model'] == 'locate-anything'}, {'float32', 'float16'})
            final = files.load(output / 'summary.json', verbose=False)
            self.assertEqual(len(final['models']), 4)
            self.assertEqual(len([r for r in final['model_modes'] if r['model'] == 'locate-anything']), 12)
            self.assertEqual(final['scaling'][0]['fps_median'], 99.)
            self.assertFalse(files.exists(output / 'summary.partial.json'))
            self.assertTrue(files.exists(output / 'REPORT.md'))
