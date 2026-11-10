"""Validate full raw GPU reports and export a self-contained benchmark ZIP."""

import argparse
import math
import statistics
import tempfile

from klygo import archive, files

from benchmarks.gpu_suite.analyze import MODES, render


def validate(reports, expected_repeats):
    if not reports:
        raise ValueError('At least one complete raw report is required')
    identities = set()
    for report in reports:
        if 'settings' not in report:
            raise ValueError('A full raw runner report is required; rounded summaries cannot be exported as raw evidence')
        settings = report['settings']
        identity = (report['model'], settings['profile'], settings['batch'])
        if identity in identities:
            raise ValueError(f'Duplicate selected configuration: {identity}')
        identities.add(identity)
        if report.get('status') != 'complete':
            raise ValueError(f'Selected report is incomplete: {identity}')
        cases = report['cases']
        if len(cases) != len(MODES) or {c['mode'] for c in cases} != set(MODES):
            raise ValueError(f'All six explicit mode statuses are required: {identity}')
        for case in cases:
            if case['status'] == 'N/A':
                if not case.get('reason'):
                    raise ValueError('N/A requires a capability reason')
                continue
            runs = case.get('runs', [])
            parity = case.get('parity', [])
            if case['status'] != 'ok' or len(runs) != expected_repeats or len(parity) != expected_repeats:
                raise ValueError(f'Missing raw repeats/parity: {identity}, {case["mode"]}')
            if [run.get('repeat') for run in runs] != list(range(expected_repeats)):
                raise ValueError('Repeat indices must be consecutive and ordered')
            for run in runs:
                if not all(math.isfinite(run[k]) and run[k] > 0 for k in ('seconds', 'fps', 'frames')):
                    raise ValueError('Timing, FPS and frame counts must be finite and positive')
                if not math.isclose(run['fps'], run['frames'] / run['seconds'], rel_tol=1e-8):
                    raise ValueError('FPS disagrees with frame count and elapsed time')
            if not math.isclose(case['fps_median'], statistics.median(r['fps'] for r in runs), rel_tol=1e-8):
                raise ValueError('Stored median disagrees with raw samples')
            if case.get('all_parity_pass') != all(p['pass'] for p in parity):
                raise ValueError('Stored parity flag disagrees with individual records')
            if not case.get('ready_latency_diagnostic') or not case.get('first_repeat_output'):
                raise ValueError('Latency diagnostic and first-repeat output must be retained')
    return len(identities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, help='Explicit selected raw JSON; repeat for each configuration')
    parser.add_argument('--failed-report', action='append', default=[], help='Unselected failed attempt retained for audit')
    parser.add_argument('--stability')
    parser.add_argument('--expected-repeats', type=int, default=7)
    parser.add_argument('--output', required=True, help='New ZIP file under artifacts/benchmarks/')
    args = parser.parse_args()
    if args.expected_repeats < 2:
        parser.error('expected-repeats must be at least two')
    output = files.path(args.output)
    if files.exists(output):
        parser.error('Output exists; choose a new ZIP filename')
    reports = [files.load(path, verbose=False) for path in args.report]
    configurations = validate(reports, args.expected_repeats)
    bundle = dict(reports=reports, failed_attempts=[files.load(p, verbose=False) for p in args.failed_report])
    if args.stability:
        bundle['stability'] = files.load(args.stability, verbose=False)
    with tempfile.TemporaryDirectory() as temporary:
        folder = files.path(temporary)
        bundle_path = folder / 'bundle.json'
        report_path = folder / 'REPORT.md'
        files.save(bundle_path, bundle, verbose=False)
        files.save(report_path, render(bundle), verbose=False)
        manifest = dict(schema='klygo.benchmark_export.v1', configurations=configurations,
                        expected_repeats=args.expected_repeats,
                        measured_cases=sum(c['status'] == 'ok' for r in reports for c in r['cases']),
                        files={p.name: files.hash(p) for p in (bundle_path, report_path)})
        files.save(folder / 'manifest.json', manifest, verbose=False)
        files.mkdir(output.parent)
        archive.compress(folder, output, include_root=False, verbose=False)
    print(f'Exported {configurations} configurations to {output}')


if __name__ == '__main__':
    main()
