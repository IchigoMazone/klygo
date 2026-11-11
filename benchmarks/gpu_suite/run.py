"""Six-path GPU measurements with explicit capabilities and untimed parity.

Use PYTHONPATH=. from the repository root. Runtime output stays in artifacts/.
Image packs compare equal inference batches; the additional Ultralytics video
profile includes source decoding and measures actual long-stream delivery.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
import platform
import re
import statistics
import subprocess
import sys
from threading import Event, Thread
import time
import traceback

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
from PIL import Image
import psutil
import torch

from klygo import files, media, models, outputs

MODES = ("raw-non-stream", "raw-stream", "native-non-stream", "native-stream",
         "managed-non-stream", "managed-stream")
MODEL_IDS = ("yolo", "yolo11", "yolo26", "yolo-world", "grounding-dino",
             "grounding-dino-base", "faster-rcnn", "d-fine", "locate-anything")


class KerasDFine(models.Detector):
    def __init__(self, device):
        import keras_hub

        super().__init__(backend="Keras", model_id="dfine_nano_coco")
        self.model = keras_hub.models.ObjectDetector.from_preset(self.model_id)
        self.model.to(device)
        self.names = {index: str(index) for index in range(80)}

    def forward(self, images, prompt=None):
        raw = self.run_inference(self.cast_inputs(images))
        formatted = self.format_results(raw)
        for item in formatted:
            keep = [i for i, score in enumerate(item["scores"]) if score >= 0.25]
            for key in ("boxes", "scores", "class_ids", "labels"):
                item[key] = [item[key][i] for i in keep]
        return outputs.build_detections(images, formatted, names=self.names)


def build(key, args):
    if key in ("yolo", "yolo11", "yolo26", "yolo-world"):
        name = {"yolo": "yolov8n.pt", "yolo11": "yolo11n.pt", "yolo26": "yolo26n.pt",
                "yolo-world": "yolov8s-worldv2.pt"}[key]
        detector = models.load(name, device=args.device, imgsz=args.imgsz,
                               half=False, threshold=0.25, iou=0.7, model_stream=False)
        prompt = ["person", "bus", "car"] if key == "yolo-world" else None
        if prompt:
            detector.model.set_classes(prompt)
            detector.names = dict(detector.model.names)
        return detector, prompt
    if key.startswith("grounding-dino"):
        name = "grounding-dino-base" if key.endswith("base") else "grounding-dino-tiny"
        detector = models.load(name, size={"shortest_edge": args.imgsz, "longest_edge": args.imgsz*2})
        detector.model.to(args.device).eval()
        return detector, ["person", "bus", "car"]
    if key == "faster-rcnn":
        detector = models.load("fasterrcnn_mobilenet_v3_large_fpn", device=args.device, threshold=0.25)
        # Keep the declared resize policy equal in raw/native/managed execution.
        detector.model.transform.min_size = (args.imgsz,)
        detector.model.transform.max_size = args.imgsz*2
        return detector, None
    if key == "d-fine":
        return KerasDFine(args.device), None
    if not args.allow_remote_code:
        raise PermissionError("LocateAnything requires --allow-remote-code for its pinned NVIDIA checkpoint")
    detector = models.load("locate-anything", model_trust_remote_code=True,
                           model_device=args.device, model_torch_dtype=torch.float16,
                           generation_mode="fast", max_new_tokens=512)
    return detector, ["person"]


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def execution_state(detector, key):
    """Observe the executing engine, independently of a wrapper's template model."""
    if key in ("yolo", "yolo11", "yolo26", "yolo-world"):
        predictor = getattr(detector.model, "predictor", None)
        engine = getattr(predictor, "model", None)
        if engine is None:
            raise RuntimeError("Ultralytics predictor was not initialized by warmup")
        # Recent Ultralytics versions retain the original CPU model and execute
        # a separate AutoBackend. Inspect its parameters, not YOLO.device.
        native = getattr(engine, "model", engine)
        parameter = next(native.parameters())
        device = str(parameter.device)
        if str(predictor.device) != device or str(engine.device) != device:
            raise RuntimeError("Predictor device and executing parameter device disagree")
        return {"device": device, "dtype": str(parameter.dtype),
                "source": "detector.model.predictor.model.model.parameters",
                "wrapper_device": str(detector.device)}
    return {"device": str(detector.device), "dtype": str(detector.dtype),
            "source": "detector runtime", "wrapper_device": str(detector.device)}


def numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def raw(detector, key, images, prompt):
    """Native framework calls without constructing Klygo Detection objects."""
    if key.startswith("grounding-dino"):
        text = ". ".join(prompt)+"."
        inputs = detector.processor(images=images, text=[text]*len(images), return_tensors="pt",
                                    **dict(detector.metadata.get("processor", {})))
        inputs = inputs.to(detector.device)
        prediction = detector.model(**inputs)
        return detector.processor.post_process_grounded_object_detection(
            outputs=prediction, input_ids=inputs["input_ids"], threshold=0.25,
            text_threshold=0.3, target_sizes=[image.size[::-1] for image in images])
    if key == "faster-rcnn":
        tensors = [media.to_tensor(image).to(detector.device) for image in images]
        predictions = detector.model(tensors)
        result = []
        for item in predictions:
            keep = item["scores"] >= 0.25
            ids = item["labels"][keep].tolist()
            result.append({"boxes": item["boxes"][keep], "scores": item["scores"][keep],
                           "labels": [detector.names[index] for index in ids]})
        return result
    if key == "d-fine":
        import keras

        inputs = keras.ops.convert_to_tensor(np.stack([np.asarray(image) for image in images]), dtype="float32")
        converter = detector.model.preprocessor.image_converter
        prepared = detector.model.preprocessor(keras.ops.image.resize(inputs, converter.image_size))
        prediction = detector.model(prepared, training=False)
        decoded = detector.model.decode_predictions(prediction, inputs)
        # Retain raw tensors until timing has ended. Output conversion is untimed.
        return [(decoded, index) for index in range(len(images))]
    result = []
    for image in images:
        question = f"Locate all the instances that matches the following description: {prompt[0]}."
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": question}]}]
        text = detector.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pictures, videos = detector.processor.process_vision_info(messages)
        inputs = detector.processor(text=[text], images=pictures, videos=videos, return_tensors="pt")
        inputs = detector.cast_inputs(inputs)
        answer = detector.model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=detector.tokenizer, use_cache=True, verbose=False, **dict(detector.metadata["generation"]))
        result.append((answer[0] if isinstance(answer, tuple) else answer, image.size))
    return result


def canonical(item, key, is_raw):
    if not is_raw:
        return {"boxes": np.asarray(item.boxes).reshape(-1,4).tolist(),
                "scores": list(item.scores), "labels": list(item.labels),
                "frame_index": item.frame_index, "source_id": str(item.source_id),
                "uids": [box.uid for box in item]}
    if key in ("yolo", "yolo11", "yolo26", "yolo-world"):
        ids = item.boxes.cls.detach().cpu().tolist()
        return {"boxes": item.boxes.xyxy.detach().cpu().tolist(),
                "scores": item.boxes.conf.detach().cpu().tolist(),
                "labels": [item.names[int(index)] for index in ids],
                "backend_speed_ms": dict(item.speed)}
    if key == "d-fine":
        import keras

        decoded, index = item
        boxes = keras.ops.convert_to_numpy(decoded["boxes"])[index]
        scores = keras.ops.convert_to_numpy(decoded.get("confidence", decoded.get("scores")))[index]
        ids = keras.ops.convert_to_numpy(decoded.get("classes", decoded.get("labels")))[index]
        keep = (scores >= 0.25) & (ids >= 0)
        boxes = boxes[keep]
        return {"boxes": boxes[:,[1,0,3,2]].tolist(), "scores": scores[keep].tolist(),
                "labels": [str(int(x)) for x in ids[keep]]}
    if key == "locate-anything":
        answer, (width,height) = item
        found = re.findall(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>",answer)
        return {"boxes": [[int(a)*width/1000,int(b)*height/1000,int(c)*width/1000,int(d)*height/1000]
                           for a,b,c,d in found], "scores": [1.0]*len(found), "labels": ["person"]*len(found),
                "raw_answer": answer}
    labels = item.get("labels", [])
    keep = [i for i,label in enumerate(labels) if label is not None and str(label).strip()]
    return {"boxes": numpy(item["boxes"])[keep].reshape(-1,4).tolist(),
            "scores": numpy(item["scores"])[keep].tolist(), "labels": [str(labels[i]) for i in keep]}


def parity(reference, candidate):
    if len(reference) != len(candidate):
        return {"pass": False, "reason": "frame count mismatch"}
    box_error = score_error = 0.0
    for left,right in zip(reference,candidate):
        for item in (left,right):
            if len(item["boxes"]) != len(item["labels"]) or len(item["scores"]) != len(item["labels"]):
                return {"pass": False, "reason": "inconsistent detection fields"}
            if not np.isfinite(np.asarray(item["boxes"])).all() or not np.isfinite(np.asarray(item["scores"])).all():
                return {"pass": False, "reason": "non-finite detection values"}
        if sorted(left["labels"]) != sorted(right["labels"]):
            return {"pass": False, "reason": "per-frame label/count mismatch"}
        # Greedy match within each label so NMS output ordering cannot hide a mismatch.
        available = set(range(len(right["labels"])))
        for i,label in enumerate(left["labels"]):
            choices = [j for j in available if right["labels"][j] == label]
            j = min(choices,key=lambda j: float(np.max(np.abs(np.asarray(left["boxes"][i])-right["boxes"][j]))))
            available.remove(j)
            box_error = max(box_error,float(np.max(np.abs(np.asarray(left["boxes"][i])-right["boxes"][j]))))
            score_error = max(score_error,abs(left["scores"][i]-right["scores"][j]))
    return {"pass": box_error <= 0.01 and score_error <= 0.0001,
            "max_box_error_px": box_error, "max_score_error": score_error,
            "frame_label_parity": True}


class Telemetry:
    def __init__(self):
        try:
            self.process = psutil.Process()
        except psutil.Error:
            self.process = None
        self.stop = Event()
        self.rss = []
        self.gpu = []
        self.cpu_start = time.process_time()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            # CUDA_VISIBLE_DEVICES can remap logical cuda:0 to a different
            # physical device. Never silently sample physical GPU zero.
            device_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
        except Exception:
            self.nvml = None

    def sample(self):
        while not self.stop.is_set():
            if self.process is not None:
                self.rss.append(self.process.memory_info().rss/2**20)
            if self.nvml:
                try:
                    n = self.nvml
                    self.gpu.append({"utilization": n.nvmlDeviceGetUtilizationRates(self.handle).gpu,
                                     "power_w": n.nvmlDeviceGetPowerUsage(self.handle)/1000,
                                     "temperature_c": n.nvmlDeviceGetTemperature(self.handle,0),
                                     "sm_clock_mhz": n.nvmlDeviceGetClockInfo(self.handle,0),
                                     "vram_used_mib": n.nvmlDeviceGetMemoryInfo(self.handle).used/2**20})
                except Exception:
                    pass
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread = Thread(target=self.sample,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join()
        self.cpu_seconds = time.process_time()-self.cpu_start


def execute(detector,key,prompt,mode,images,args,video=None):
    if mode.startswith("raw"):
        if key in ("yolo","yolo11","yolo26","yolo-world"):
            return detector.model.predict(source=video or images, stream=mode.endswith("-stream") and mode!="raw-non-stream",
                device=args.device,imgsz=args.imgsz,conf=0.25,iou=0.7,half=False,batch=args.batch,
                rect=True,verbose=False)
        return raw(detector,key,images,prompt)
    if mode.startswith("native"):
        # Native calls may override a frozen model group transiently. Frozen
        # options are not remembered by Detector's runtime configuration.
        kwargs = {"model_stream": mode=="native-stream"} if key in ("yolo","yolo11","yolo26","yolo-world") else {}
        return detector.predict(video or images,prompt=prompt,native=True,verbose=False,**kwargs)
    return detector.predict(video or images,prompt=prompt,batch=args.batch,workers=1,prefetch=1,
                            stream=mode=="managed-stream",verbose=False)


def measure(detector,key,prompt,mode,packs,args,video=None,ready_latency=False):
    gc.collect()
    sync(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    delivered = []
    retained = []
    pack_latencies = []
    with Telemetry() as telemetry, torch.inference_mode():
        started = time.perf_counter()
        for pack in packs:
            pack_started = time.perf_counter()
            result = execute(detector,key,prompt,mode,pack,args,video)
            try:
                for item in result:
                    if ready_latency:
                        sync(args.device)
                    delivered.append((time.perf_counter()-started)*1000)
                    retained.append(item)
            finally:
                close = getattr(result,"close",None)
                if callable(close):
                    close()
            if ready_latency:
                sync(args.device)
            pack_latencies.append((time.perf_counter()-pack_started)*1000)
        sync(args.device)
        elapsed = time.perf_counter()-started
    # All conversion, hashing and parity comparisons are outside the timed region.
    captured = [canonical(item,key,mode.startswith("raw")) for item in retained]
    intervals = np.diff([0.0]+delivered).tolist()
    result = {"seconds": elapsed, "fps": len(captured)/elapsed, "frames": len(captured),
              "boxes": sum(len(item["boxes"]) for item in captured),
              "first_frame_host_ms": delivered[0] if delivered else None,
              "delivery_intervals_ms": intervals, "pack_latencies_ms": pack_latencies,
              "rss_peak_sampled_mib": max(telemetry.rss,default=None),
              "cpu_percent_one_core": 100*telemetry.cpu_seconds/elapsed,
              "gpu_samples": telemetry.gpu}
    if args.device.startswith("cuda"):
        result.update(vram_allocated_peak_mib=torch.cuda.max_memory_allocated(args.device)/2**20,
                      vram_reserved_peak_mib=torch.cuda.max_memory_reserved(args.device)/2**20)
    return result,captured


def environment():
    versions = {}
    for name in ("klygo","torch","torchvision","ultralytics","transformers","keras","keras-hub","opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    return {"versions":versions,"python":sys.version,"os":platform.platform(),
            "cpu_count":os.cpu_count(),"cpu":platform.processor(),"cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            "gpu_uuid":str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),"uuid",None)) if torch.cuda.is_available() else None,
            "vram_mib":torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory/2**20 if torch.cuda.is_available() else None,
            "float32_matmul_precision":torch.get_float32_matmul_precision(),
            "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark":torch.backends.cudnn.benchmark,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "torch_threads":torch.get_num_threads(),"tested_commit":commit}


def prepare(args):
    source = files.join(args.output,"bus.jpg")
    if not files.is_file(source):
        files.download("https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",args.output,verbose=False)
    image = media.to_pil(media.load(source,verbose=False)[0]).resize((640,480))
    images = [Image.fromarray(np.roll(np.asarray(image),i*3,axis=1).copy()) for i in range(max(args.batch,1))]
    return source,images


def run_model(args):
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    files.mkdir(args.output)
    report = {"model":args.model,"environment":environment(),"settings":vars(args),"cases":[]}
    destination = files.join(args.output,args.model+".json")
    try:
        source,images = prepare(args)
        report["input_sha256"] = files.hash(source)
        started = time.perf_counter()
        detector,prompt = build(args.model,args)
        sync(args.device)
        report["load_seconds"] = time.perf_counter()-started
        report["prompt"] = prompt
        report["actual_model_dtype"] = str(detector.dtype)
        report["actual_model_device"] = str(detector.device)
        report["device_at_factory_return"] = str(detector.device)
        report["detector_class"] = type(detector).__module__+"."+type(detector).__qualname__
        report["weights_revision"] = getattr(getattr(detector.model,"config",None),"_commit_hash",None)
        report["runner_sha256"] = files.hash(__file__)
        if files.is_file(detector.model_id):
            report["weights_sha256"] = files.hash(detector.model_id)
        report["input_scope"] = "predecoded 640x480 PIL packs; same effective inference batch in all supported modes"
        # Framework internals without an upstream stream are explicitly N/A.
        upstream_stream = args.model in ("yolo","yolo11","yolo26","yolo-world")
        active = [mode for mode in MODES if upstream_stream or mode not in ("raw-stream","native-stream")]
        if args.profile=="video" and args.batch!=1:
            active = [mode for mode in active if not mode.startswith("native")]
        for mode in MODES:
            if mode not in active:
                reason = "Native video batch is locked to 1 by the public API" if mode.startswith("native") and args.profile=="video" else "No upstream native source-stream engine"
                report["cases"].append({"mode":mode,"status":"N/A","reason":reason})
        if args.model=="locate-anything" and args.batch != 1:
            raise ValueError("LocateAnything is measured only with batch=1; generation is sequential")
        video = None
        if args.profile == "video":
            if not upstream_stream:
                raise ValueError("The video profile requires an upstream video decoder for fair raw/native comparison")
            temporary = files.join(args.output,"bus_motion_mpeg4.mp4")
            external_video = getattr(args,"source_video",None)
            video = str(files.path(external_video).resolve()) if external_video else str(files.join(args.output,"bus_motion_h264.mp4"))
            if external_video and not files.is_file(video):
                raise FileNotFoundError(f"Video input does not exist: {video}")
            if not files.is_file(video):
                base = np.asarray(media.to_pil(media.load(source,verbose=False)[0]).resize((1280,720)))
                frames = (Image.fromarray(np.roll(base,int(20*np.sin(index/15)),axis=1).copy())
                          for index in range(args.video_frames))
                media.save_video(temporary,frames,fps=30,overwrite=True,verbose=False)
                media.convert(temporary,video,codec="h264",gpu=False,preset="fast",overwrite=True,verbose=False)
            report["input_sha256"] = files.hash(video)
            report["input_scope"] = ("end-to-end caller-provided video; includes source decoding" if external_video else
                                     "end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding")
        packs = [images] if video else [images]*args.iterations
        cases = {mode:{"mode":mode,"status":"ok","runs":[],"parity":[]} for mode in active}
        references = {}
        print(f"Loaded {args.model}: {report['load_seconds']:.2f}s, dtype={detector.dtype}",flush=True)
        report["warmup_execution_states"] = {}
        for mode in active:
            measure(detector,args.model,prompt,mode,packs,args,video)
            state = execution_state(detector,args.model)
            report["warmup_execution_states"][mode] = state
            if args.device.startswith("cuda") and state["device"] != str(torch.device(args.device)):
                raise RuntimeError(f"Warmup execution did not use the requested CUDA device: {mode}")
        report["actual_model_device"] = state["device"]
        report["actual_model_dtype"] = state["dtype"]
        report["device_observation_source"] = state["source"]
        report["device_observation_stage"] = "after complete warmup, before measured repeats"
        report["execution_policy_after_warmup"] = {
            "float32_matmul_precision":torch.get_float32_matmul_precision(),
            "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark":torch.backends.cudnn.benchmark,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        }
        if args.device.startswith("cuda") and not report["actual_model_device"].startswith("cuda"):
            raise RuntimeError("Model did not remain on CUDA after warmup")
        for repeat in range(args.repeats):
            order = active if repeat%2==0 else list(reversed(active))
            print(f"{args.model} repeat {repeat+1}/{args.repeats}: {','.join(order)}",flush=True)
            captures = {}
            for mode in order:
                sample,captured = measure(detector,args.model,prompt,mode,packs,args,video)
                sample["execution_state"] = execution_state(detector,args.model)
                if args.device.startswith("cuda") and sample["execution_state"]["device"] != str(torch.device(args.device)):
                    raise RuntimeError(f"Measured execution did not use the requested CUDA device: {mode}")
                sample["repeat"] = repeat
                cases[mode]["runs"].append(sample)
                captures[mode] = captured
            reference = captures["raw-non-stream"]
            for mode in active:
                cases[mode]["parity"].append(parity(reference,captures[mode]))
            if repeat==0:
                references = captures
            report["cases"] = [case for case in report["cases"] if case.get("status")=="N/A"]+list(cases.values())
            files.save(destination,report,overwrite=True,verbose=False)
        for mode,case in cases.items():
            diagnostic,captured = measure(detector,args.model,prompt,mode,packs,args,video,ready_latency=True)
            intervals = diagnostic["delivery_intervals_ms"]
            case["ready_latency_diagnostic"] = diagnostic
            case["ready_interval_p50_ms"],case["ready_interval_p95_ms"],case["ready_interval_p99_ms"] = np.percentile(intervals,[50,95,99]).tolist()
            case["ready_pack_p50_ms"],case["ready_pack_p95_ms"],case["ready_pack_p99_ms"] = np.percentile(diagnostic["pack_latencies_ms"],[50,95,99]).tolist()
            case["fps_median"] = statistics.median(run["fps"] for run in case["runs"])
            case["fps_std"] = statistics.stdev(run["fps"] for run in case["runs"]) if args.repeats>1 else 0
            case["first_frame_host_ms_median"] = statistics.median(run["first_frame_host_ms"] for run in case["runs"])
            case["all_parity_pass"] = all(item["pass"] for item in case["parity"])
            case["first_repeat_output"] = references[mode]
            case["delivery_semantics"] = "caller-visible delivery"
            print(f"RESULT {args.model} {mode}: {case['fps_median']:.3f} FPS; parity={case['all_parity_pass']}",flush=True)
        report["status"] = "complete"
        report["cases"] = [case for case in report["cases"] if case.get("status")=="N/A"]+list(cases.values())
    except Exception as error:
        report.update(status="error",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    files.save(destination,report,overwrite=True,verbose=False)
    if report["status"]!="complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=MODEL_IDS,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--iterations",type=int,default=8)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--profile",choices=("images","video"),default="images")
    parser.add_argument("--video-frames",type=int,default=120)
    parser.add_argument("--source-video",help="Existing fixed video for the video profile; use a separately hashed run per source")
    parser.add_argument("--allow-remote-code",action="store_true")
    parser.add_argument("--output",default="artifacts/benchmarks/six-mode")
    args = parser.parse_args()
    if args.source_video and args.profile != "video":
        parser.error("source-video requires --profile video")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required; no silent CPU fallback")
    if min(args.batch,args.iterations,args.repeats) < 1:
        parser.error("batch, iterations and repeats must be positive")
    run_model(args)


if __name__=="__main__":
    main()
