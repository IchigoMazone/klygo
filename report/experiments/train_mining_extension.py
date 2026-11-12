"""Small fixed-budget fine-tuning study; not an official COCO benchmark."""
from extension_common import *
import argparse
import subprocess
import time
import yaml
from ultralytics import YOLO
import torch
TRAIN = OUT / "training_canonical"

def config(method):
    base = OUT / "mining/training_datasets"
    names = yaml.safe_load((base / method / "data.yaml").read_text(encoding="utf-8"))["names"]
    data = {"path": str(base), "train": f"{method}/images", "val": "val/images", "test": "test/images", "names": names}
    target = TRAIN / f"{method}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return target

def run(method, seed):
    data = config(method)
    model = YOLO(str(WEIGHTS / "yolov8n.pt"))
    if method == "initial":
        checkpoint = WEIGHTS / "yolov8n.pt"
        seconds = 0
    else:
        start = time.perf_counter()
        model.train(data=str(data), epochs=5, imgsz=320, batch=8, device=0, workers=0,
            seed=seed, deterministic=True, amp=False, optimizer="AdamW", lr0=0.001, lrf=0.1,
            warmup_epochs=0, patience=0, mosaic=0, close_mosaic=0, mixup=0,
            project=str(TRAIN / "runs"), name=f"{method}_{seed}", exist_ok=False,
            plots=False, save=True, val=False, verbose=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        checkpoint = TRAIN / "runs" / f"{method}_{seed}" / "weights/last.pt"
        model = YOLO(str(checkpoint))
    metrics = model.val(data=str(data), split="test", imgsz=320, batch=8, device=0, workers=0,
                        conf=0.001, iou=0.7, max_det=100, half=False, plots=False, save_json=True,
                        project=str(TRAIN / "evaluation"), name=f"{method}_{seed}", verbose=False)
    save(TRAIN / f"{method}_{seed}.json", {"method": method, "seed": seed,
        "revision": REVISION, "epochs": 0 if method == "initial" else 5,
        "training_wall_s": seconds, "AP": float(metrics.box.map), "AP50": float(metrics.box.map50),
        "AP75": float(metrics.box.map75), "metrics": metrics.results_dict,
        "checkpoint": str(checkpoint), "evaluation": "Ultralytics bbox metrics on fixed 100-image held-out test; imgsz320; last epoch; no test-based selection"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.method:
        if args.method == "initial":
            # Same data configuration, zero fine-tuning.
            base = OUT / "mining/datasets"
            # config() reads names from the full export for this case.
            config_path = config("full")
            original_config = config
            config = lambda method: config_path
        run(args.method, args.seed)
    else:
        protocol = {"revision": REVISION, "methods": ["full", "random", "weighted", "diversity"],
            "training_seeds": [42,43,44], "selection_seed": 42, "epochs": 5, "imgsz": 320,
            "batch": 8, "amp": False, "optimizer": "AdamW", "lr0": 0.001,
            "last_checkpoint": True, "test_images": 100,
            "limitation": "Short fine-tuning from COCO-pretrained weights; fixed subsets across seeds; equal epochs, unequal optimizer updates; not from-scratch sample efficiency."}
        save(TRAIN / "protocol.json", protocol)
        jobs = [("initial",42)] + [(method,seed) for seed in [42,43,44] for method in protocol["methods"]]
        for method, seed in jobs:
            if not (TRAIN / f"{method}_{seed}.json").exists():
                subprocess.run([sys.executable,"-X","utf8","-u",__file__,"--method",method,"--seed",str(seed)], check=True)
