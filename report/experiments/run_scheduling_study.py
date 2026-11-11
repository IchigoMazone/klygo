"""Fresh six-mode and single-variable worker/prefetch study on fixed COCO inputs."""
from extension_common import *
import subprocess
from datetime import datetime, timezone

SIX = "execution_six_20261107"
SCHEDULE = "scheduling_20261107"
MODES = ["raw", "raw_stream", "native", "native_backend_stream", "managed", "managed_stream"]
CONFIGS = [(0,0),(1,1),(2,1),(4,1),(2,2),(2,4)]

if __name__ == "__main__":
    protocol = {"revision":REVISION,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "models":["yolo","world"],"six_modes":MODES,"scheduling_configs":CONFIGS,
        "batch":1,"precision":"FP32","imgsz":640,"confidence":.25,"nms_iou":.7,"max_det":100,
        "images":100,"warmup_full_passes":1,"timed_repeats":3,"parity_passes":1,
        "scope":"JPEG decode through public results; independent processes; no concurrent GPU work",
        "design":"worker sweep at prefetch=1; prefetch sweep at workers=2; serial reference (0,0)",
        "limitation":"fixed subsets, one machine, fixed process order, short finite stream; no optimality claim"}
    target = OUT / SCHEDULE / "protocol.json"
    if not target.exists(): save(target, protocol)
    script = ROOT / "report/experiments/benchmark_execution.py"
    for model in ["yolo","world"]:
        for mode in MODES:
            target = OUT / SIX / f"{model}_{mode}.json"
            if not target.exists():
                subprocess.run([sys.executable,"-X","utf8","-u",str(script),"--model",model,"--mode",mode,"--output-tag",SIX],check=True)
        for workers,prefetch in CONFIGS:
            # The (1,1) reference is the managed_stream run from the same six-mode session.
            if (workers,prefetch)==(1,1): continue
            target=OUT / SCHEDULE / f"{model}_managed_stream_w{workers}_p{prefetch}.json"
            if not target.exists():
                subprocess.run([sys.executable,"-X","utf8","-u",str(script),"--model",model,"--mode","managed_stream",
                    "--workers",str(workers),"--prefetch",str(prefetch),"--output-tag",SCHEDULE],check=True)
