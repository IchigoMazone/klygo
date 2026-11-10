"""Validate raw-repeat dispersion and pairing without a GPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.gpu_suite.raw_statistics import describe, summarize


class TestRawStatistics(unittest.TestCase):
    def test_dispersion_uses_individual_repeats(self):
        result = describe([10, 20, 30], resamples=100)
        self.assertEqual(result['median'], 20)
        self.assertEqual(result['sample_sd'], 10)
        self.assertEqual(result['cv_percent'], 50)
        self.assertEqual(result['minimum'], 10)
        self.assertEqual(result['maximum'], 30)

    def test_bootstrap_is_reproducible_and_covers_constant_samples(self):
        self.assertEqual(describe([2, 2, 2], 100)['median_bootstrap_95'], [2, 2])
        self.assertEqual(describe([1, 3, 6], 100), describe([1, 3, 6], 100))

    def test_paired_elapsed_overhead_is_not_ratio_of_medians(self):
        def case(mode, seconds):
            return dict(mode=mode, status='ok', all_parity_pass=True,
                        runs=[dict(repeat=i, seconds=t, fps=10/t, frames=10, first_frame_host_ms=1)
                              for i, t in enumerate(seconds)])
        bundle = {'reports': [dict(model='fixture', settings=dict(profile='images', batch=1), cases=[
            case('raw-non-stream', [1, 2, 10]), case('managed-stream', [2, 3, 11])])]}
        data = summarize(bundle, resamples=100)
        self.assertEqual(data['rows'][0]['paired_elapsed_overhead_percent']['median'], 0)
        self.assertEqual(data['rows'][1]['paired_elapsed_overhead_percent']['median'], 50)

    def test_nvml_uses_active_cuda_uuid(self):
        from benchmarks.gpu_suite.run import Telemetry
        seen = []
        nvml = SimpleNamespace(nvmlInit=lambda: None,
                               nvmlDeviceGetHandleByUUID=lambda uuid: seen.append(uuid) or 'handle')
        with patch.dict('sys.modules', pynvml=nvml), \
             patch('torch.cuda.current_device', return_value=1), \
             patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(uuid='GPU-selected')):
            telemetry = Telemetry()
        self.assertEqual(seen, ['GPU-selected'])
        self.assertEqual(telemetry.handle, 'handle')

    def test_missing_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Raw repeat samples'):
            describe([])


if __name__ == '__main__':
    unittest.main()
