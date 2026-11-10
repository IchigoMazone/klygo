"""Pinned, checkpointed T4 comparison of the two YOLO preparation revisions."""

import importlib.util
import os
import statistics
import subprocess
import sys
import time

import torch
from klygo import archive, files

root = files.path(sys.argv[1])
revisions = {
    "before": "5ed34c82fa6db1f700d0e0a591e69e9259538640",
    "after": "237efc7537f65f6ff1016aac60acd9b67c6f2ad4",
}
assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4"
evidence = root / "evidence"
files.mkdir(evidence)
records = []
selected = {label: [] for label in revisions}
configs = [(model, profile, batch) for model in ("yolo", "yolo26")
           for profile in ("images", "video") for batch in (1, 4)]
files.save(evidence / "setup.json", files.load(root / "setup.json", verbose=False), overwrite=True, verbose=False)
files.copy(root / "driver.py", evidence / "executed_driver.txt", overwrite=True)

for index, (model, profile, batch) in enumerate(configs):
    order = ("before", "after") if index % 2 == 0 else ("after", "before")
    for label in order:
        checkout = root / label
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == revisions[label]
        folder = root / "results" / label / f"{profile}_b{batch}" / model
        files.mkdir(folder)
        files.copy(root / "inputs/bus.jpg", folder / "bus.jpg", overwrite=True)
        destination = folder / f"{model}.json"
        args = [sys.executable, "-u", str(checkout / "benchmarks/gpu_suite/run.py"),
                "--model", model, "--device", "cuda:0", "--imgsz", "640",
                "--batch", str(batch), "--iterations", "64", "--repeats", "7",
                "--profile", profile, "--video-frames", "120", "--output", str(folder)]
        if profile == "video":
            args += ["--source-video", str(root / "inputs/bus_motion_h264.mp4")]
        env = dict(os.environ, PYTHONPATH=str(checkout), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        record = dict(version=label, model=model, profile=profile, batch=batch,
                      commit=revisions[label], command=args, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        records.append(record)
        files.save(evidence / "jobs.json", records, overwrite=True, verbose=False)
        print("START", len(records), 16, label, model, profile, batch, flush=True)
        started = time.perf_counter()
        log_path = evidence / f"{label}_{model}_{profile}_b{batch}.log"
        with open(log_path, "w") as log:
            result = subprocess.run(args, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, elapsed_seconds=time.perf_counter() - started)
        files.save(evidence / "jobs.json", records, overwrite=True, verbose=False)
        if result.returncode:
            raise RuntimeError(f"Benchmark job failed; retained log: {log_path}")
        report = files.load(destination, verbose=False)
        assert report["status"] == "complete"
        assert report["environment"]["tested_commit"] == revisions[label]
        assert report["actual_model_device"] == "cuda:0"
        for case in report["cases"]:
            if case["status"] == "ok":
                assert len(case["runs"]) == 7 and case["all_parity_pass"]
                assert all(run["execution_state"]["device"] == "cuda:0" for run in case["runs"])
        selected[label].append((destination, report))
        print("DONE", label, model, profile, batch,
              [(c["mode"], round(c["fps_median"], 3)) for c in report["cases"] if c["status"] == "ok"], flush=True)

spec = importlib.util.spec_from_file_location("t4_runner", root / "after/benchmarks/gpu_suite/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
comparison = []
identity = lambda report: (report["model"], report["settings"]["profile"], report["settings"]["batch"])
before = {identity(report): report for _, report in selected["before"]}
for _, after in selected["after"]:
    previous = before[identity(after)]
    for key in ("input_sha256", "weights_sha256", "runner_sha256"):
        assert previous[key] == after[key], (identity(after), key)
    previous_cases = {case["mode"]: case for case in previous["cases"]}
    for case in after["cases"]:
        old = previous_cases[case["mode"]]
        assert old["status"] == case["status"]
        if case["status"] != "ok":
            continue
        parity = runner.parity(old["first_repeat_output"], case["first_repeat_output"])
        assert parity["pass"], (identity(after), case["mode"], parity)
        comparison.append(dict(model=after["model"], profile=after["settings"]["profile"],
                               batch=after["settings"]["batch"], mode=case["mode"],
                               before_fps=old["fps_median"], after_fps=case["fps_median"],
                               fps_change_percent=100 * (case["fps_median"] / old["fps_median"] - 1),
                               cross_revision_parity=parity,
                               before_cv_percent=100 * statistics.stdev(r["fps"] for r in old["runs"]) / statistics.mean(r["fps"] for r in old["runs"]),
                               after_cv_percent=100 * statistics.stdev(r["fps"] for r in case["runs"]) / statistics.mean(r["fps"] for r in case["runs"])))

assert len(comparison) == 44
for label, reports in selected.items():
    checkout = root / label
    target = evidence / f"{label}_raw.zip"
    args = [sys.executable, str(checkout / "benchmarks/gpu_suite/export_bundle.py"), "--output", str(target)]
    for path, _ in reports:
        args += ["--report", str(path)]
    subprocess.run(args, cwd=checkout, env=dict(os.environ, PYTHONPATH=str(checkout)), check=True)
    expanded = root / "exports" / label
    archive.extract(target, expanded, verbose=False)
    subprocess.run([sys.executable, str(checkout / "benchmarks/gpu_suite/raw_statistics.py"),
                    str(expanded / "bundle.json"), "--output", str(evidence / label)],
                   cwd=checkout, env=dict(os.environ, PYTHONPATH=str(checkout)), check=True)

files.save(evidence / "comparison.json", comparison, overwrite=True, verbose=False)
lines = ["# Tesla T4: YOLO preparation before/after", "",
         "Pinned revisions, same GPU, input hashes and weights. Each revision has eight configurations, 44 supported mode cases and seven repeats (308 timed passes). Cross-revision first-repeat outputs also pass parity.", "",
         "Version order alternates between adjacent configuration pairs. Modes alternate within each isolated job. Versions run in separate processes; do not treat repeat indices across revisions as statistically paired samples.", "",
         "| Model | Profile | Batch | Mode | Before FPS | After FPS | Change % | CV before/after % |",
         "|---|---|---:|---|---:|---:|---:|---:|"]
for row in comparison:
    lines.append(f"|{row['model']}|{row['profile']}|{row['batch']}|{row['mode']}|{row['before_fps']:.3f}|{row['after_fps']:.3f}|{row['fps_change_percent']:.2f}|{row['before_cv_percent']:.2f}/{row['after_cv_percent']:.2f}|")
files.save(evidence / "COMPARISON.md", "\n".join(lines), overwrite=True, verbose=False)
manifest = {str(path.relative_to(evidence)): files.hash(path) for path in evidence.rglob("*") if path.is_file()}
files.save(evidence / "manifest.json", manifest, overwrite=True, verbose=False)
package = root / "t4_before_after_evidence.zip"
archive.compress(evidence, package, include_root=False, verbose=False)
files.save(root / "completion.json", dict(status="complete", configurations=16, supported_cases=88,
           measured_passes=616, cross_revision_parity=True, package=str(package),
           sha256=files.hash(package)), overwrite=True, verbose=False)
print("ALL_T4_EVIDENCE_READY", files.size(package), files.hash(package), flush=True)

