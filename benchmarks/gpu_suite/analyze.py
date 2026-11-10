"""Render all measured samples and explicit N/A/errors from the GPU suite."""

import argparse
import statistics

from klygo import files

MODES = ("raw-non-stream", "raw-stream", "native-non-stream", "native-stream",
         "managed-non-stream", "managed-stream")


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def maximum(runs, key):
    values = [run[key] for run in runs if run.get(key) is not None]
    return max(values) if values else None


def render(bundle):
    reports = bundle["reports"]
    lines = ["# Colab GPU benchmark: six execution modes", "",
             "All values below are actual measurements. Unsupported modes are N/A; failed jobs remain errors.", "",
             "## Scope and interpretation", "",
             "- Image-pack results exclude decoding uniformly and include a separate public API call per pack. They are not video throughput.",
             "- Video results include source loading/decoding. Compare paths within a model, profile and batch, not across different workloads.",
             "- Seven alternating measured repeats follow complete warmup. Conversion, hashing and parity checks are outside throughput timing.",
             "- Native-stream currently buffers upstream results into a materialized Klygo result. TTFF therefore includes that buffering.",
             "- Native video batch is locked to one by the tested API; native rows at video batch four are N/A.",
             "- Most detectors use FP32. LocateAnything uses FP16, one query, fast generation and a 512-token limit; it is not directly comparable to FP32 detector precision.",
             "- First-frame figures in throughput runs are host-visible delivery observations. The separate ready-latency pass synchronizes each result and is not included in FPS.",
             "- Batched/non-stream output arrives in bursts. Delivery intervals are not isolated per-frame inference time; ready pack latency includes the entire call.",
             "- Image diagnostic sample counts depend on the recorded iteration and batch settings. Their interpolated P95/P99 are descriptive samples, not tail-latency SLA evidence.",
             "- Throughput passes retain yielded results for untimed parity; peak memory includes retention. The separate stability probe consumes without retaining frames.",
             "- RSS is sampled at 50 ms; short peaks can be missed. CPU percentages are approximate, expressed relative to one core and include probe setup. NVML uses device-window statistics.",
             "- Reserved VRAM includes allocator caches from preceding modes. Allocated peaks describe live tensor allocation; neither is an isolated minimum-memory requirement.",
             "- Factory load time excludes process/hardware initialization and includes checkpoint lookup/download when needed; cache states differ between jobs.",
             "- These recorded workloads do not qualify production accuracy or live-camera behavior. A separate stability result, when present, covers one managed pipeline, not all models or 30-minute production use.", ""]
    if reports:
        env = reports[0].get("environment",{})
        lines += ["## Environment", "", f"GPU: **{env.get('gpu')}**, VRAM {number(env.get('vram_mib'))} MiB. CUDA {env.get('cuda')}.",
                  f"Python: `{env.get('python','').splitlines()[0]}`. OS: `{env.get('os')}`.",
                  f"CPU logical cores: {env.get('cpu_count')}; Torch threads: {env.get('torch_threads')}.", "",
                  "| Package | Version |", "| --- | --- |"]
        lines += [f"| {name} | {version} |" for name,version in env.get("versions",{}).items()]
        lines += ["", "Tested commit(s): "+", ".join(f"`{value}`" for value in sorted({r.get('environment',{}).get('tested_commit','unknown') for r in reports})), ""]
    if bundle.get("hardware_details"):
        lines += ["Additional hardware/runtime details: `"+str(bundle["hardware_details"])+"`.", ""]
    if bundle.get("failed_attempts") or bundle.get("rejected_control_probe_attempts"):
        lines += ["## Attempt audit", "",
                  "The original model groups are frozen: native overrides are transient and are not remembered by managed prediction. Real Detector API tests confirm that the original measurements used independent mode settings.",
                  f"Rejected control probes: {len(bundle.get('rejected_control_probe_attempts',[]))}. These attempted an explicit managed-time override of the frozen model group, were rejected before final measurements, and are excluded from tables.",
                  "Setup/execution failures are retained separately from the selected final attempts:", ""]
        for report in bundle.get("failed_attempts",[]):
            lines.append(f"- {report.get('model')}: {report.get('error_type')} — {report.get('error')}")
        lines.append("")
    profiles = sorted({(r["settings"]["profile"],r["settings"]["batch"]) for r in reports})
    for profile,batch in profiles:
        selected = [r for r in reports if r["settings"]["profile"]==profile and r["settings"]["batch"]==batch]
        lines += [f"## FPS matrix — {profile}, batch {batch}", "",
                  "| Model | "+" | ".join(MODES)+" |", "| --- |"+" ---: |"*len(MODES)]
        for report in selected:
            cases = {case["mode"]:case for case in report.get("cases",[])}
            cells = []
            for mode in MODES:
                case = cases.get(mode,{})
                if case.get("fps_median") is not None:
                    cells.append(number(case["fps_median"])+(" ⚠" if not case.get("all_parity_pass") else ""))
                else:
                    cells.append("N/A" if case.get("status")=="N/A" else "ERROR" if report.get("status")=="error" else "—")
            lines.append("| "+report["model"]+" | "+" | ".join(cells)+" |")
        lines += ["", "⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.", ""]
    for report in reports:
        settings = report["settings"]
        lines += [f"## Detail — {report['model']}, {settings['profile']}, batch {settings['batch']}", "",
                  f"Status: **{report.get('status')}**. Load {number(report.get('load_seconds'))} s. Device `{report.get('actual_model_device')}`, dtype `{report.get('actual_model_dtype')}`.",
                  f"Input scope: {report.get('input_scope','not available')}. Input SHA256: `{report.get('input_sha256')}`.",
                  f"Runner SHA256: `{report.get('runner_sha256')}`. Prompt: `{report.get('prompt')}`.", ""]
        if report.get("status")=="error":
            lines += [f"**{report.get('error_type')}:** `{report.get('error')}`", ""]
        lines += ["| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        cases = {case["mode"]:case for case in report.get("cases",[])}
        for mode in MODES:
            case = cases.get(mode)
            if not case:
                continue
            runs = case.get("runs",[])
            if not runs:
                lines += [f"| {mode} | {case.get('status')} | — | — | — | — | — | {case.get('reason','not measured')} |"]
                continue
            first = runs[0]
            latency = "/".join(number(case.get(f"ready_pack_p{p}_ms")) for p in (50,95,99))
            lines.append(f"| {mode} | {number(case.get('fps_median'))} ± {number(case.get('fps_std'))} | {number(case.get('first_frame_host_ms_median'))} | {latency} | {number(maximum(runs,'rss_peak_sampled_mib'))} | {number(maximum(runs,'vram_allocated_peak_mib'))}/{number(maximum(runs,'vram_reserved_peak_mib'))} | {first['frames']}/{first['boxes']} | {'PASS' if case.get('all_parity_pass') else 'FAIL'} |")
        for mode,case in cases.items():
            runs = case.get("runs",[])
            if not runs:
                continue
            baseline = cases.get("raw-non-stream",{}).get("runs",[])
            if len(baseline)==len(runs):
                paired = [100*(test["seconds"]/base["seconds"]-1) for base,test in zip(baseline,runs)]
                paired_text = number(statistics.median(paired))+"% median paired elapsed-time overhead vs raw non-stream"
            else:
                paired_text = "paired overhead unavailable"
            lines += ["", f"### {mode}", "", paired_text+". Negative means lower measured elapsed time.", "",
                      "| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for index,run in enumerate(runs):
                gpu = run.get("gpu_samples",[])
                def average(key):
                    return statistics.mean(sample[key] for sample in gpu) if gpu else None
                values = [str(index+1),number(run["seconds"],6),number(run["fps"]),number(run.get("first_frame_host_ms")),
                          number(run.get("rss_peak_sampled_mib")),number(run.get("vram_allocated_peak_mib")),
                          number(run.get("cpu_percent_one_core")),number(average("utilization")),number(average("power_w")),
                          number(average("sm_clock_mhz")),number(average("temperature_c"))]
                lines.append("| "+" | ".join(values)+" |")
            diagnostics = case.get("ready_latency_diagnostic",{})
            lines += ["", "Ready result delivery-interval P50/P95/P99 ms: "+"/".join(number(case.get(f"ready_interval_p{p}_ms")) for p in (50,95,99))+".",
                      f"Separate latency-pass duration: {number(diagnostics.get('seconds'))} s; not used for throughput.", "",
                      "Parity for every measured repeat:", ""]
            for index,result in enumerate(case.get("parity",[])):
                lines.append(f"- Repeat {index+1}: `{result}`")
            backend = [frame.get("backend_speed_ms") for frame in case.get("first_repeat_output",[]) if frame.get("backend_speed_ms")]
            if backend:
                means = {stage:statistics.mean(frame[stage] for frame in backend) for stage in backend[0]}
                lines += ["", "Upstream-reported first-repeat stage averages (ms/frame): "+str(means)+". These are upstream timers, not an additive decomposition of total wall time.", ""]
            captures = case.get("first_repeat_output",[])
            if captures and all("uids" in frame for frame in captures):
                unique = all(len(set(frame["uids"]))==len(frame["uids"]) for frame in captures)
                sources = len({frame["source_id"] for frame in captures})
                lines += ["", f"First-repeat metadata: per-frame UID uniqueness={unique}; distinct source identities={sources}."]
                if settings["profile"]=="video":
                    ordered = [frame["frame_index"] for frame in captures]==list(range(len(captures)))
                    lines += [f"Sequential video frame indices={ordered}."]
                else:
                    lines += ["Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings."]
                lines.append("")
    stable = bundle.get("stability")
    if stable:
        lines += ["## Ten-minute managed-stream stability", "",
                  f"GPU: {stable.get('gpu')}; model {stable.get('model')}; elapsed {number(stable.get('elapsed_seconds'))} s; frames {stable.get('total_frames')}.",
                  f"Pass: **{stable.get('pass')}**. Exceptions: `{stable.get('exceptions')}`. Failed gates: `{stable.get('failed_gates')}`.",
                  "Each loop verifies frame count and order, consumes without retaining frames, closes the stream and collects unreachable Python objects. The model remains loaded; allocator caches are not emptied between loops.", "",
                  "| Metric | First 60-second median | Last 60-second median | Change % |",
                  "| --- | ---: | ---: | ---: |"]
        for key in ("rss_mib","vram_allocated_mib","fps","open_fds"):
            lines.append(f"| {key} | {number(stable.get(key+'_first_window'))} | {number(stable.get(key+'_last_window'))} | {number(stable.get(key+'_growth_percent'))} |")
        elapsed = stable.get("elapsed_seconds",0)
        if elapsed:
            lines += ["",f"Whole-check throughput including between-loop cleanup: {number(stable.get('total_frames',0)/elapsed)} FPS. Per-loop FPS above excludes between-loop cleanup.",""]
    lines += ["## Reproduction and limits", "",
              "Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.",
              "No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.",
              "Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--markdown",required=True)
    args = parser.parse_args()
    bundle = files.load(args.bundle,verbose=False)
    files.save(args.markdown,render(bundle),overwrite=True,verbose=False)


if __name__=="__main__":
    main()
