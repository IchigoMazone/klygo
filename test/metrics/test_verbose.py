"""Progress must remain optional, result-neutral, and independent of configs."""
from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch
from klygo import metrics, outputs
from test.metrics.test_metrics import _make_pair


class TestVerbose(unittest.TestCase):
    def test_all_metrics_preserve_results_and_default_is_silent(self):
        pred, truth = _make_pair()
        for name in metrics.__all__:
            if name in {'compose', 'evaluate'}:
                continue
            operation = getattr(metrics, name)
            with self.subTest(metric=name):
                with patch('klygo.utils.progress.tqdm') as bar:
                    expected = operation(pred, truth)
                    bar.assert_not_called()
                with redirect_stderr(StringIO()) as output:
                    actual = operation(pred, truth, verbose=True)
                self.assertEqual(actual, expected)
                self.assertIn('100%', output.getvalue())
                with patch('klygo.utils.progress.tqdm') as bar:
                    self.assertEqual(operation(verbose=True), operation())
                    bar.assert_not_called()

    def test_composition_uses_one_bar(self):
        pred, truth = _make_pair()
        config = metrics.compose(metrics.map_score(), metrics.confusion_matrix())
        import klygo.utils.progress as progress
        with patch.object(progress, 'tqdm', wraps=progress.tqdm) as bars, redirect_stderr(StringIO()):
            actual = metrics.evaluate(pred, truth, config, verbose=True)
        self.assertEqual(bars.call_count, 1)
        self.assertEqual(actual, metrics.evaluate(pred, truth, config))

    def test_invalid_flag_and_error_cleanup(self):
        pred, truth = _make_pair()
        with self.assertRaises(TypeError):
            metrics.evaluate(pred, truth, verbose='yes')
        with patch('klygo.utils.progress.tqdm') as bars:
            with self.assertRaises(TypeError):
                metrics.evaluate(object(), truth, verbose=True)
            bars.return_value.close.assert_called_once()

    def test_combine_regular_and_packed(self):
        for packed in (False, True):
            pred, truth = _make_pair()
            if packed:
                pred.pack()
                truth.pack()
            expected = pred.copy().combine(truth).to_dict()
            with patch('klygo.utils.progress.tqdm') as bars:
                pred.copy().combine(truth)
                bars.assert_not_called()
            with redirect_stderr(StringIO()) as output:
                actual = pred.copy().combine(truth, verbose=True).to_dict()
            self.assertEqual(actual, expected)
            self.assertIn('100%', output.getvalue())
