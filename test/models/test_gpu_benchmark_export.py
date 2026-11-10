"""Raw benchmark exports must reject incomplete or inconsistent evidence."""

import copy
import unittest

from benchmarks.gpu_suite.export_bundle import MODES, validate
from benchmarks.gpu_suite.statistics_report import compute


class TestBenchmarkExport(unittest.TestCase):
    def report(self):
        return dict(model='fixture', status='complete', settings=dict(profile='images', batch=1),
                    cases=[dict(mode=mode, status='ok', fps_median=2.0, all_parity_pass=True,
                                runs=[dict(repeat=i, seconds=1.0, fps=2.0, frames=2) for i in range(7)],
                                parity=[{'pass': True} for _ in range(7)],
                                ready_latency_diagnostic={'seconds': 1}, first_repeat_output=[{'boxes': []}])
                           for mode in MODES])

    def test_complete_reports_and_explicit_unsupported_modes(self):
        report = self.report()
        report['cases'][1] = dict(mode='raw-stream', status='N/A', reason='No upstream stream')
        self.assertEqual(validate([report], 7), 1)

    def test_missing_raw_repeat_is_rejected(self):
        report = self.report()
        report['cases'][0]['runs'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing raw repeats'):
            validate([report], 7)

    def test_raw_non_stream_cannot_be_labeled_unsupported(self):
        report = self.report()
        report['cases'][0] = dict(mode='raw-non-stream', status='N/A', reason='incorrect capability claim')
        with self.assertRaisesRegex(ValueError, 'measured raw non-stream baseline'):
            validate([report], 7)

    def test_rounded_summary_cannot_be_promoted_to_raw_evidence(self):
        with self.assertRaisesRegex(ValueError, 'full raw runner report'):
            validate([{'model': 'fixture', 'profile': 'images', 'batch': 1}], 7)

    def test_inconsistent_fps_or_parity_is_rejected(self):
        report = self.report()
        report['cases'][0]['runs'][0]['fps'] = 3
        with self.assertRaisesRegex(ValueError, 'FPS disagrees'):
            validate([report], 7)
        report = self.report()
        report['cases'][0]['parity'][0]['pass'] = False
        with self.assertRaisesRegex(ValueError, 'parity flag disagrees'):
            validate([report], 7)

    def test_duplicate_configuration_is_rejected(self):
        report = self.report()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate([report, copy.deepcopy(report)], 7)

    def test_ratio_direction_and_whole_check_throughput(self):
        summary = {'reports': [dict(model='fixture', profile='video', batch=1, cases=[
            dict(mode='raw-non-stream', status='ok', fps_median=100, all_parity_pass=True),
            dict(mode='managed-stream', status='ok', fps_median=80, all_parity_pass=True)])]}
        samples = [dict(fps=v, rss_mib=100, vram_allocated_mib=10,
                        vram_reserved_mib=20, open_fds=5) for v in (60, 80, 100)]
        data = compute(summary, dict(total_frames=600, elapsed_seconds=10, samples=samples))
        row = data['comparisons'][1]
        self.assertEqual(row['raw_throughput_percent'], 80)
        self.assertEqual(row['elapsed_overhead_percent'], 25)
        self.assertEqual(data['stability_whole_check_fps'], 60)
        self.assertEqual(data['stability_distribution']['fps']['median'], 80)


if __name__ == '__main__':
    unittest.main()
