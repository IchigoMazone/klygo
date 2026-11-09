"""Ten-minute single-model managed-stream stability check on CUDA."""

import argparse
import gc
import statistics
import subprocess
import time

import cv2
import psutil
import torch

from klygo import files, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--seconds",type=float,default=600)
    parser.add_argument("--output",default="artifacts/benchmarks/stability.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the stability run")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    detector = models.load("yolov8n.pt",device="cuda:0",imgsz=640,threshold=0.25,iou=0.7)
    process = psutil.Process()

    def consume():
        frames = 0
        result = detector.predict(args.source,stream=True,verbose=False)
        try:
            for frame in result:
                if frame.frame_index != frames:
                    raise AssertionError("Frame order mismatch")
                frames += 1
        finally:
            result.close()
        return frames

    for _ in range(10):
        expected_frames = consume()
    gc.collect()
    torch.cuda.synchronize()
    started = time.perf_counter()
    samples = []
    report = {"model":"yolov8n.pt","source_sha256":files.hash(args.source),
              "requested_seconds":args.seconds,"expected_frames_per_loop":expected_frames,
              "gpu":torch.cuda.get_device_name(0),"runner_sha256":files.hash(__file__),
              "tested_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "samples":samples,"exceptions":[]}
    try:
        while time.perf_counter()-started < args.seconds:
            torch.cuda.synchronize()
            before = time.perf_counter()
            frames = consume()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-before
            if frames != expected_frames:
                raise AssertionError("Dropped file-input frames")
            gc.collect()
            sample = {"elapsed_seconds":time.perf_counter()-started,"frames":frames,
                      "seconds":elapsed,"fps":frames/elapsed,"rss_mib":process.memory_info().rss/2**20,
                      "vram_allocated_mib":torch.cuda.memory_allocated()/2**20,
                      "vram_reserved_mib":torch.cuda.memory_reserved()/2**20,"open_fds":process.num_fds()}
            samples.append(sample)
            files.save(args.output,report,overwrite=True,verbose=False)
            if len(samples)%20==0:
                print(f"STABILITY {sample['elapsed_seconds']:.0f}s: {sample['fps']:.2f} FPS; RSS={sample['rss_mib']:.1f} MiB; fds={sample['open_fds']}",flush=True)
    except Exception as error:
        report["exceptions"].append({"type":type(error).__name__,"message":str(error)})
    elapsed = time.perf_counter()-started
    first = [x for x in samples if x["elapsed_seconds"]<=60]
    last = [x for x in samples if x["elapsed_seconds"]>=elapsed-60]
    report["elapsed_seconds"] = elapsed
    report["total_frames"] = sum(x["frames"] for x in samples)
    for key in ("rss_mib","vram_allocated_mib","fps","open_fds"):
        if first and last:
            a,b = statistics.median(x[key] for x in first),statistics.median(x[key] for x in last)
            report[key+"_first_window"] = a
            report[key+"_last_window"] = b
            report[key+"_growth_percent"] = 100*(b/a-1) if a else None
    report["pass"] = not report["exceptions"] and elapsed>=args.seconds
    report["failed_gates"] = []
    for key in ("rss_mib", "vram_allocated_mib"):
        if (report.get(key+"_growth_percent") or 0)>5:
            report["failed_gates"].append(key+" growth >5%")
    if report.get("open_fds_last_window",0)>report.get("open_fds_first_window",0):
        report["failed_gates"].append("open file descriptors grew")
    report["pass"] = report["pass"] and not report["failed_gates"]
    files.save(args.output,report,overwrite=True,verbose=False)
    print("STABILITY FINISHED",report["pass"],report.get("rss_mib_growth_percent"),flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
