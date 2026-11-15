"""Isolated Locate Anything execution and strict release-result validation."""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from typing import Any

from klygo import files

SUPPORTED = {'raw-non-stream', 'native-non-stream', 'managed-non-stream', 'managed-stream'}
UNSUPPORTED = {'raw-stream', 'native-stream'}
# Only known dependency fixes are automated; model errors keep their full logs.
PACKAGES = {
    'cv2': 'opencv-python-headless>=4.10', 'einops': 'einops',
    'decord': 'decord==0.6.0', 'lmdb': 'lmdb>=1.7.5', 'timm': 'timm>=1.0.11',
    'peft': 'peft', 'accelerate': 'accelerate>=1.5.2',
    'transformers': 'transformers==4.57.1', 'sentencepiece': 'sentencepiece',
}
PROBE = '''import importlib, json, sys
names = ('cv2', 'torch', 'torchvision', 'transformers', 'accelerate', 'peft', 'decord', 'lmdb', 'timm', 'klygo')
versions = {}
for name in names:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, '__version__', 'unknown'))
if versions['transformers'] != '4.57.1':
    raise RuntimeError('Locate Anything requires transformers==4.57.1')
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix, 'versions': versions, 'klygo_file': importlib.import_module('klygo').__file__}))
'''


def result_errors(data: Any, dtype: str, device: str, repeats: int) -> list[str]:
    """Reject partial, stale, or parity-failing Locate Anything reports."""
    if not isinstance(data, dict):
        return ['Missing or invalid raw report']
    errors = []
    if data.get('model') != 'locate-anything' or data.get('status') != 'complete':
        errors.append('Locate Anything report is not complete')
    settings = data.get('settings', {})
    if any(settings.get(key) != value for key, value in
           [('dtype', dtype), ('device', device), ('repeats', repeats)]):
        errors.append('Report settings do not match this release run')
    if data.get('actual_model_dtype', '').split('.')[-1] != dtype:
        errors.append('Actual precision differs from requested precision')
    if data.get('actual_model_device') != device:
        errors.append('Actual device differs from requested device')
    rows = data.get('cases', [])
    cases = {row.get('mode'): row for row in rows}
    if len(cases) != len(rows) or set(cases) != SUPPORTED | UNSUPPORTED:
        errors.append('Expected all six distinct mode records')
    for mode in sorted(SUPPORTED):
        row = cases.get(mode, {})
        fps = row.get('fps_median')
        if (row.get('status') != 'ok' or row.get('all_parity_pass') is not True
                or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0):
            errors.append(f'{mode}: missing measurement, invalid FPS, or failed parity')
        if len(row.get('runs', [])) != repeats or len(row.get('parity', [])) != repeats:
            errors.append(f'{mode}: incomplete repeat measurements')
        elif any(item.get('pass') is not True for item in row['parity']):
            errors.append(f'{mode}: a repeat failed parity')
    for mode in sorted(UNSUPPORTED):
        if cases.get(mode, {}).get('status') != 'N/A':
            errors.append(f'{mode}: upstream streaming must be explicitly N/A')
    return errors


class LocateRuntime:
    """Repair dependencies only inside a managed venv; preserve every attempt."""

    def __init__(self, output, *, python='', isolated=False, auto_fix=True,
                 retries=2, venv_dir='', repo=None):
        self.repo = files.path(repo or '.').resolve()
        self.logs = files.path(output) / 'locate_runtime' / f'run_{time.time_ns()}'
        files.mkdir(self.logs)
        self.supplied_python = python
        self.python = python or sys.executable
        self.isolated = isolated
        self.auto_fix = auto_fix
        self.retries = retries
        self.venv_dir = files.path(venv_dir or f'artifacts/benchmark-envs/locate-py{sys.version_info.major}{sys.version_info.minor}').resolve()
        self.managed = False
        self.ready = False
        self.events = []

    def _call(self, command, name):
        try:
            env = dict(os.environ)
            env['PYTHONPATH'] = str(self.repo) + os.pathsep + env.get('PYTHONPATH', '')
            proc = subprocess.run(command, cwd=self.repo, env=env, capture_output=True, text=True)
        except OSError as exc:
            proc = subprocess.CompletedProcess(command, 1, '', f'{type(exc).__name__}: {exc}')
        files.save(self.logs / f'{name}.stdout.txt', proc.stdout or '', overwrite=True, verbose=False)
        files.save(self.logs / f'{name}.stderr.txt', proc.stderr or '', overwrite=True, verbose=False)
        self.events.append({'stage': name, 'command': [str(x) for x in command], 'returncode': proc.returncode})
        files.save(self.logs / 'attempts.json', self.events, overwrite=True, verbose=False)
        return proc

    def _create_managed(self):
        if self.venv_dir == files.path(sys.prefix).resolve():
            raise RuntimeError('Locate venv must differ from the main Python environment')
        if self.supplied_python:
            probe = self._call([self.supplied_python, '-c', 'import sys; print(sys.prefix)'], 'supplied_prefix')
            if probe.returncode == 0 and probe.stdout.strip():
                supplied_prefix = files.path(probe.stdout.strip().splitlines()[-1]).resolve()
                if self.venv_dir == supplied_prefix:
                    raise RuntimeError('Managed Locate venv must differ from the supplied interpreter environment')
        files.mkdir(self.venv_dir.parent)
        proc = self._call([sys.executable, '-m', 'venv', '--system-site-packages', str(self.venv_dir)], 'create_venv')
        if proc.returncode:
            raise RuntimeError(f'Could not create Locate venv: {proc.stderr}')
        self.python = str(self.venv_dir / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python'))
        self.managed = True

    def _install(self, packages, name):
        if not self.managed:
            self._create_managed()
        proc = self._call([self.python, '-m', 'pip', 'install', *packages], name)
        if proc.returncode:
            raise RuntimeError(f'Locate dependency repair failed; see {self.logs / (name + ".stderr.txt")}')

    def prepare(self):
        if self.ready:
            return
        if self.isolated and not self.managed:
            self._create_managed()
        proc = self._call([self.python, '-c', PROBE], 'preflight')
        if proc.returncode and self.auto_fix:
            self._install(['-e', f'{self.repo}[locate-anything]'], 'install_locate_extra')
            text = (proc.stdout or '') + (proc.stderr or '')
            # Headless OpenCV also handles servers without libGL.
            if 'cv2' in text or 'libGL' in text:
                self._install([PACKAGES['cv2']], 'install_opencv_headless')
            proc = self._call([self.python, '-c', PROBE], 'preflight_after_repair')
        if proc.returncode:
            raise RuntimeError(f'Locate preflight failed: {proc.stderr or proc.stdout}')
        self.ready = True
        files.save(self.logs / 'runtime.json', {
            'python': self.python, 'managed': self.managed, 'venv': str(self.venv_dir) if self.managed else None,
            'preflight': proc.stdout, 'system_site_packages': self.managed,
        }, overwrite=True, verbose=False)

    def run(self, command, case_out, dtype):
        try:
            self.prepare()
            for attempt in range(self.retries + 1):
                command = [self.python, *command[1:]]
                raw = files.path(case_out) / 'locate-anything.json'
                if files.exists(raw):
                    files.move(raw, self.logs / f'{dtype}_before_attempt_{attempt + 1}.json')
                proc = self._call(command, f'{dtype}_attempt_{attempt + 1}')
                raw = files.path(case_out) / 'locate-anything.json'
                if files.exists(raw):
                    files.copy(raw, self.logs / f'{dtype}_attempt_{attempt + 1}.json', overwrite=True)
                if proc.returncode == 0 or not self.auto_fix or attempt == self.retries:
                    return proc
                text = (proc.stdout or '') + (proc.stderr or '')
                missing = re.search(r"No module named ['\"]([^'\"]+)", text)
                package = PACKAGES.get(missing.group(1).split('.')[0]) if missing else None
                if not package and 'libGL' in text:
                    package = PACKAGES['cv2']
                if not package:
                    return proc
                self._install(['-e', f'{self.repo}[locate-anything]', package], f'{dtype}_repair_{attempt + 1}')
                self.ready = False
                self.prepare()
            return proc
        except RuntimeError as exc:
            return subprocess.CompletedProcess(command, 1, '', str(exc))


def require_results(output, dtypes, device, repeats, runs, *, expected_env=None):
    """Block final charts/report until all requested Locate precisions pass."""
    root = files.path(output)
    errors = []
    for dtype in dtypes:
        matching = [row for row in runs if row.get('model') == 'locate-anything' and row.get('dtype') == dtype]
        path = root / 'inference' / f'locate-anything_{dtype}' / 'locate-anything.json'
        if len(matching) != 1 or matching[0].get('status') != 'passed':
            errors.append(f'{dtype}: Locate Anything subprocess failed or is missing')
        try:
            data = files.load(path, verbose=False) if files.exists(path) else None
        except (ValueError, OSError, TypeError) as exc:
            errors.append(f'{dtype}: unreadable raw report: {exc}')
            data = None
        errors.extend(f'{dtype}: {error}' for error in result_errors(data, dtype, device, repeats))
        if isinstance(data, dict):
            for key in ('gpu', 'gpu_uuid'):
                expected = (expected_env or {}).get(key)
                if expected not in (None, 'none') and data.get('environment', {}).get(key) != expected:
                    errors.append(f'{dtype}: Locate {key} differs from the release environment')
    return errors
