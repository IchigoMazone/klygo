"""Compute descriptive comparisons from saved median FPS and stability samples."""

import argparse
import math
import statistics

from klygo import files


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compute(summary, stability):
    comparisons = []
    for report in summary['reports']:
        cases = {case['mode']: case for case in report['cases']}
        base = cases['raw-non-stream']['fps_median']
        for case in report['cases']:
            if case['status'] != 'ok':
                continue
            fps = case['fps_median']
            comparisons.append(dict(model=report['model'], profile=report['profile'],
                                    batch=report['batch'], mode=case['mode'], fps=fps,
                                    raw_fps=base, raw_throughput_percent=100 * fps / base,
                                    elapsed_overhead_percent=100 * (base / fps - 1),
                                    parity=case['all_parity_pass']))
    scaling = []
    lookup = {(r['model'], r['profile'], r['batch']): r for r in summary['reports']}
    for (model, profile, batch), report in lookup.items():
        other = lookup.get((model, profile, 4))
        if batch != 1 or other is None:
            continue
        right = {c['mode']: c for c in other['cases']}
        for case in report['cases']:
            mode = case['mode']
            if case['status'] == 'ok' and right[mode]['status'] == 'ok':
                scaling.append(dict(model=model, profile=profile, mode=mode,
                                    batch1_fps=case['fps_median'], batch4_fps=right[mode]['fps_median'],
                                    throughput_ratio=right[mode]['fps_median'] / case['fps_median']))
    distribution = {}
    for key in ('fps', 'rss_mib', 'vram_allocated_mib', 'vram_reserved_mib', 'open_fds'):
        values = [s[key] for s in stability['samples']]
        distribution[key] = dict(samples=len(values), minimum=min(values),
                                 p05=percentile(values, .05), median=statistics.median(values),
                                 p95=percentile(values, .95), maximum=max(values),
                                 mean=statistics.mean(values), sample_sd=statistics.stdev(values))
    return dict(schema='klygo.descriptive_statistics.v1', comparisons=comparisons,
                batch_scaling=scaling, stability_distribution=distribution,
                stability_whole_check_fps=stability['total_frames'] / stability['elapsed_seconds'],
                matrix_uncertainty='Unavailable: individual matrix repeat timings were not saved in the notebook export.',
                inference_limit='Ratios use rounded medians, not paired individual runs. Stability loops are serial, correlated observations; no independent-sample confidence interval is claimed.')


def render(data):
    lines = ['# Descriptive benchmark statistics', '',
             'Derived from the companion saved median-FPS summary and complete stability samples. No additional GPU run was performed.', '',
             '**Statistical limit:** per-case matrix SD, confidence intervals and significance cannot be recovered from medians. Ratios below are ratios of rounded medians, not medians of paired ratios. Differences near the rounding precision are not meaningful.', '',
             'Throughput retained = mode FPS / raw non-stream FPS. Elapsed overhead = raw FPS / mode FPS − 1 for equal frame counts. Negative overhead means lower measured elapsed time. These are end-to-end comparisons, not isolated wrapper cost.', '']
    rows = data['comparisons']
    for profile, batch in sorted({(r['profile'], r['batch']) for r in rows}):
        lines += [f'## Against raw non-stream: {profile}, batch {batch}', '',
                  '| Model | Mode | FPS | Raw throughput retained % | Elapsed overhead % |',
                  '|---|---|---:|---:|---:|']
        for row in rows:
            if (row['profile'], row['batch']) == (profile, batch) and row['mode'] != 'raw-non-stream':
                lines.append(f"|{row['model']}|{row['mode']}|{row['fps']:.3f}|{row['raw_throughput_percent']:.2f}|{row['elapsed_overhead_percent']:+.2f}|")
        lines.append('')
    lines += ['## Batch scaling', '',
              'Batch 4 / batch 1 is a throughput comparison across different batch settings. It is not a same-workload optimization gain or a latency speedup. Image packs use eight calls in both settings, so batch 4 also processes four times as many images.', '',
              '| Model | Profile | Mode | Batch 1 FPS | Batch 4 FPS | Throughput ratio |',
              '|---|---|---|---:|---:|---:|']
    for row in data['batch_scaling']:
        lines.append(f"|{row['model']}|{row['profile']}|{row['mode']}|{row['batch1_fps']:.3f}|{row['batch4_fps']:.3f}|{row['throughput_ratio']:.3f}|")
    lines += ['', '## Stability distribution', '',
              '312 consecutive loops from one loaded model. Quantiles describe observed loop samples; samples are correlated. They are not frame-level latency percentiles or production tail guarantees. SD is descriptive sample SD, not uncertainty of the matrix medians.', '',
              '| Metric | Min | P05 | Median | P95 | Max | Mean | Sample SD |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, row in data['stability_distribution'].items():
        lines.append('|' + key + '|' + '|'.join(f"{row[k]:.3f}" for k in ('minimum','p05','median','p95','maximum','mean','sample_sd')) + '|')
    lines += ['', f"Whole-check throughput including loop cleanup: **{data['stability_whole_check_fps']:.3f} FPS**. Loop FPS above excludes cleanup. First/last-minute memory and FD gates are retained in the original report; min/max here additionally expose intermediate variation.", '',
              '## Evidence completeness', '',
              '| Evidence | Availability |', '|---|---|',
              '| Median FPS / printed parity | 124 supported cases, available |',
              '| Unsupported modes | 26 N/A cells, recorded capability reasons |',
              '| Seven repeat markers per selected job | Available in archived stdout |',
              '| 868 individual matrix timing / parity records | Missing from saved output |',
              '| Matrix SD / confidence intervals / TTFF / P95 / P99 | Cannot independently reconstruct |',
              '| Stability telemetry | Complete: 312 samples, hash verified on import |',
              '| Ground-truth accuracy / diverse real video | Not evaluated |', '',
              'The existing experiment supports descriptive performance comparison and a ten-minute memory/FD check. Full statistical review requires the original matrix bundle or a fresh run with file-based export. Use `export_bundle.py` to reject incomplete raw reports and preserve the next complete run as a ZIP.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--stability', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = compute(files.load(args.summary, verbose=False), files.load(args.stability, verbose=False))
    files.mkdir(args.output)
    files.save(files.join(args.output, 'statistics.json'), data, overwrite=True, verbose=False)
    files.save(files.join(args.output, 'STATISTICS.md'), render(data), overwrite=True, verbose=False)


if __name__ == '__main__':
    main()
