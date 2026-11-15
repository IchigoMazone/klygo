"""Keep the repository skill wired to the canonical release benchmark suite."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.gpu_suite import evaluate_release


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / '.agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py'


class TestReleaseBenchmarkSkill(unittest.TestCase):
    def test_launcher_help_is_canonical_from_another_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(LAUNCHER), '--help'], cwd=cwd,
                capture_output=True, text=True, check=True,
            )
        self.assertIn('--include-models', result.stdout)
        self.assertIn('--locate-python', result.stdout)
        self.assertIn('--dataset-dir', result.stdout)
        self.assertIn('--include-scaling', result.stdout)
        self.assertNotIn('--modes', result.stdout)

    def test_launcher_preserves_arguments_and_restores_process_state_on_error(self):
        spec = importlib.util.spec_from_file_location('release_skill_launcher', LAUNCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_cwd = Path.cwd()
        original_path = sys.path[:]
        arguments = [str(LAUNCHER), '--repeats', '2', '--dataset', 'synthetic']

        def execute(path, *, run_name):
            self.assertEqual(Path(path), ROOT / 'benchmarks/gpu_suite/evaluate_release.py')
            self.assertEqual(Path.cwd(), ROOT)
            self.assertEqual(sys.argv[1:], arguments[1:])
            self.assertEqual(run_name, '__main__')
            raise RuntimeError('fixture failure')

        with patch.object(sys, 'argv', arguments[:]), patch.object(module.runpy, 'run_path', side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                module.main()
            self.assertEqual(sys.argv, arguments)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(sys.path, original_path)

    def test_coco128_download_does_not_require_a_returned_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def extract(source, destination, **kwargs):
                self.assertEqual(source, root / 'artifacts/datasets/coco128.zip')
                self.assertEqual(destination, root / 'artifacts/datasets')
                (destination / 'coco128/images/train2017').mkdir(parents=True)

            with patch.object(evaluate_release.files, 'path', side_effect=lambda value: root / value), \
                 patch.object(evaluate_release.files, 'download', return_value=None) as download, \
                 patch.object(evaluate_release.archive, 'extract', side_effect=extract):
                dataset_root, images, labels, tag = evaluate_release._prepare_real_dataset('coco128', '')
            download.assert_called_once()
            self.assertEqual(tag, 'coco128')
            self.assertEqual(dataset_root, root / 'artifacts/datasets/coco128')

    def test_annotation_mining_fixture_runs_without_image_sources(self):
        with patch('klygo.mining._common._image', side_effect=AssertionError('unexpected image read')):
            rows = evaluate_release.benchmark_mining({'cupy_cuda': False}, repeats=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'passed')
        self.assertEqual(rows[0]['score_weights'], {'rarity': .5, 'coverage': .3, 'boxes': .2})
        self.assertTrue(rows[0]['select_includes_scoring'])
