"""Run the fixed cross-GPU matrix sequentially and retain all raw job reports."""

import argparse
import os
import subprocess
import sys

from klygo import files

MODELS = ('yolo', 'yolo11', 'yolo26', 'yolo-world', 'grounding-dino',
          'grounding-dino-base', 'faster-rcnn', 'd-fine', 'locate-anything')


def jobs_for(selected):
    return ([(m, 'images', 1) for m in selected]
            + [(m, 'images', 4) for m in selected if m != 'locate-anything']
            + [(m, 'video', batch) for m in selected if m in MODELS[:4] for batch in (1, 4)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--allow-remote-code', action='store_true')
    parser.add_argument('--stability-seconds', type=float, default=600)
    parser.add_argument('--output', required=True, help='New empty run directory under artifacts/benchmarks/')
    args = parser.parse_args()
    if not args.device.startswith('cuda:'):
        parser.error('This matrix targets an explicit CUDA device; no silent CPU fallback')
    if args.repeats < 2 or args.stability_seconds < 0:
        parser.error('Use at least two repeats and nonnegative stability seconds')
    if len(set(args.models)) != len(args.models):
        parser.error('Model keys must be unique')
    if args.stability_seconds and ('yolo' not in args.models or args.device != 'cuda:0'):
        parser.error('Stability requires yolo on logical cuda:0; otherwise use --stability-seconds 0')
    if 'locate-anything' in args.models and not args.allow_remote_code:
        parser.error('The full matrix includes LocateAnything; explicitly provide --allow-remote-code or select other --models')
    root = files.path(args.output).resolve()
    if files.exists(root):
        parser.error('Choose a new output directory to preserve existing measurements')
    files.mkdir(root)
    suite = files.path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(suite)
    env['KERAS_BACKEND'] = 'torch'
    records = []
    plan = jobs_for(args.models)
    files.save(root / 'plan.json', [dict(model=m, profile=p, batch=b) for m, p, b in plan], verbose=False)
    for index, (model, profile, batch) in enumerate(plan):
        folder = root / f'{profile}_b{batch}'
        files.mkdir(folder)
        iterations = 64 if model in MODELS[:4] else 32 if model == 'faster-rcnn' else 1 if model == 'locate-anything' else 8
        command = [sys.executable, '-u', str(suite / 'benchmarks/gpu_suite/run.py'),
                   '--model', model, '--device', args.device, '--profile', profile,
                   '--batch', str(batch), '--iterations', str(iterations),
                   '--imgsz', '640', '--video-frames', '120', '--repeats', str(args.repeats),
                   '--output', str(folder)]
        if args.allow_remote_code:
            command.append('--allow-remote-code')
        print(f'JOB {index + 1}/{len(plan)} {model} {profile} batch={batch}', flush=True)
        process = subprocess.run(command, cwd=suite, env=env, capture_output=True, text=True)
        files.save(folder / f'{model}.log', process.stdout + process.stderr, overwrite=True, verbose=False)
        records.append(dict(model=model, profile=profile, batch=batch, returncode=process.returncode,
                            report=str(folder / f'{model}.json')))
        files.save(root / 'jobs.json', records, overwrite=True, verbose=False)
        print(f'FINISHED returncode={process.returncode}', flush=True)
    stability_path = root / 'stability.json'
    stability_returncode = None
    if args.stability_seconds and files.exists(root / 'video_b1/bus_motion_h264.mp4'):
        command = [sys.executable, str(suite / 'benchmarks/gpu_suite/stability.py'),
                   str(root / 'video_b1/bus_motion_h264.mp4'), '--seconds', str(args.stability_seconds),
                   '--output', str(stability_path)]
        process = subprocess.run(command, cwd=suite, env=env, capture_output=True, text=True)
        stability_returncode = process.returncode
        files.save(root / 'stability.log', process.stdout + process.stderr, overwrite=True, verbose=False)
    selected = [r for r in records if r['returncode'] == 0]
    command = [sys.executable, str(suite / 'benchmarks/gpu_suite/export_bundle.py'),
               '--expected-repeats', str(args.repeats), '--output', str(root / 'complete.zip')]
    for record in records:
        if files.exists(record['report']):
            command.extend(['--report' if record['returncode'] == 0 else '--failed-report', record['report']])
    if files.exists(stability_path):
        command.extend(['--stability', str(stability_path)])
    process = subprocess.run(command, cwd=suite, env=env) if selected else None
    result = dict(configurations=len(records), complete=len(selected), failed=len(records) - len(selected),
                  export_returncode=process.returncode if process else None,
                  stability_returncode=stability_returncode,
                  stability_pass=files.load(stability_path, verbose=False).get('pass') if files.exists(stability_path) else None)
    files.save(root / 'completion.json', result, verbose=False)
    print(result, flush=True)
    if result['failed'] or result['export_returncode'] != 0 or (args.stability_seconds and
            (result['stability_returncode'] != 0 or result['stability_pass'] is not True)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
