"""Summarize full per-repeat GPU evidence without reconstructing missing samples."""

import argparse
import random
import statistics

from klygo import files


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values, resamples=2000):
    """Return sample dispersion and an exploratory within-session bootstrap interval."""
    if not values:
        raise ValueError('Raw repeat samples are required')
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    rng = random.Random(20261110)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)] if len(values) > 1 else []
    return dict(n=len(values), median=statistics.median(values), mean=mean,
                sample_sd=sd, cv_percent=100 * sd / abs(mean) if sd is not None and mean else None,
                minimum=min(values), maximum=max(values),
                median_bootstrap_95=[quantile(medians, .025), quantile(medians, .975)] if medians else None)


def summarize(bundle, resamples=2000):
    rows = []
    for report in bundle['reports']:
        cases = {c['mode']: c for c in report['cases']}
        raw = cases.get('raw-non-stream', {}).get('runs', [])
        for case in report['cases']:
            row = dict(model=report['model'], profile=report['settings']['profile'],
                       batch=report['settings']['batch'], mode=case['mode'], status=case['status'])
            if case['status'] == 'N/A':
                row['reason'] = case['reason']
            else:
                runs = case.get('runs', [])
                if not runs:
                    raise ValueError('A median-only summary cannot supply raw statistics')
                row.update(fps=describe([r['fps'] for r in runs], resamples),
                           seconds=describe([r['seconds'] for r in runs], resamples),
                           parity=case['all_parity_pass'],
                           measured_frames=[r['frames'] for r in runs],
                           first_host_frame_ms=[r['first_frame_host_ms'] for r in runs],
                           ready_pack_p50_p95_p99_ms=[case.get(f'ready_pack_p{p}_ms') for p in (50,95,99)],
                           rss_peak_sampled_mib=max((r.get('rss_peak_sampled_mib') or 0) for r in runs),
                           allocated_vram_peak_mib=max((r.get('vram_allocated_peak_mib') or 0) for r in runs),
                           reserved_vram_peak_mib=max((r.get('vram_reserved_peak_mib') or 0) for r in runs))
                if len(raw) == len(runs) and [r.get('repeat') for r in raw] == [r.get('repeat') for r in runs]:
                    row['paired_elapsed_overhead_percent'] = describe(
                        [100 * (r['seconds'] / b['seconds'] - 1) for r, b in zip(runs, raw)], resamples)
            rows.append(row)
    return dict(schema='klygo.raw_repeat_statistics.v1', bootstrap_resamples=resamples,
                bootstrap_seed=20261110, rows=rows,
                uncertainty_scope='Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.',
                timing_scope='End-to-end timed passes. Host first delivery is not isolated GPU kernel latency. Ready latency is a separate synchronized diagnostic.')


def number(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def render(data):
    lines = ['# Full raw-repeat benchmark statistics', '', data['uncertainty_scope'], '',
             'All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.', '']
    profiles = sorted({(r['profile'], r['batch']) for r in data['rows']})
    for profile, batch in profiles:
        selected = [r for r in data['rows'] if (r['profile'], r['batch']) == (profile, batch)]
        lines += [f'## {profile}, batch {batch}', '',
                  '| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for row in selected:
            if row['status'] == 'N/A':
                lines.append(f"|{row['model']}|{row['mode']}|N/A|N/A|N/A|N/A|N/A|N/A|N/A|{row['reason']}|")
                continue
            fps = row['fps']
            interval = fps['median_bootstrap_95']
            overhead = row.get('paired_elapsed_overhead_percent', {}).get('median')
            lines.append(f"|{row['model']}|{row['mode']}|{fps['n']}|{fps['median']:.3f}|{fps['mean']:.3f} ± {number(fps['sample_sd'])}|{fps['minimum']:.3f}–{fps['maximum']:.3f}|{number(fps['cv_percent'],2)}|{number(interval[0])+'–'+number(interval[1]) if interval else 'N/A'}|{number(overhead,2)}|{'PASS' if row['parity'] else 'FAIL'}|")
        lines += ['', '| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for row in selected:
            if row['status'] != 'ok':
                continue
            ready = '/'.join(number(v) for v in row['ready_pack_p50_p95_p99_ms'])
            lines.append(f"|{row['model']}|{row['mode']}|{row['seconds']['median']:.6f}|{number(statistics.median(row['first_host_frame_ms']))}|{ready}|{row['rss_peak_sampled_mib']:.2f}|{row['allocated_vram_peak_mib']:.2f}/{row['reserved_vram_peak_mib']:.2f}|")
        lines.append('')
    lines += ['## Interpretation limits', '',
              '- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.',
              '- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.',
              '- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.',
              '- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.',
              '- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = summarize(files.load(args.bundle, verbose=False))
    files.mkdir(args.output)
    files.save(files.join(args.output, 'raw_statistics.json'), data, overwrite=True, verbose=False)
    files.save(files.join(args.output, 'STATISTICS.md'), render(data), overwrite=True, verbose=False)


if __name__ == '__main__':
    main()
