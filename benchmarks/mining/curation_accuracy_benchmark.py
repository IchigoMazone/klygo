"""Real-world dataset creation & Klygo mining curation accuracy benchmark.

Constructs a realistic, long-tail, noisy 10-class object-detection dataset
from real COCO ``val2017`` images using ``klygo.datasets``, ``klygo.files``,
``klygo.archive``, ``klygo.media``, and ``klygo.preprocessing``, applies every
core ``klygo.mining`` curation strategy under controlled budgets, trains a
``YOLOv8n`` detector per curated subset on GPU across multiple seeds, and
evaluates on a strictly disjoint clean held-out validation split.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import random
from statistics import mean, stdev
import sys
from time import perf_counter
from typing import Any

import numpy as np

from klygo import (
    archive,
    config,
    datasets,
    files,
    media,
    mining,
    outputs,
    preprocessing,
)


COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

TARGET_CLASSES: dict[int, str] = {
    0: "person",
    1: "car",
    2: "chair",
    3: "bottle",
    4: "cup",
    5: "bicycle",
    6: "bus",
    7: "traffic light",
    8: "dog",
    9: "laptop",
}
HEAD_CLASSES: tuple[str, ...] = ("person", "car", "chair", "bottle", "cup")
RARE_CLASSES: tuple[str, ...] = ("bicycle", "bus", "traffic light", "dog", "laptop")

DEFAULT_BUDGET = 350
DEFAULT_EPOCHS = 35
DEFAULT_SEEDS: tuple[int, ...] = (42, 123)


def _coco_bbox_to_yolo(
    bbox: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Convert COCO ``[x, y, w, h]`` pixels into clipped normalized YOLO ``xc, yc, w, h``."""
    x, y, bw, bh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if width <= 0 or height <= 0 or bw <= 1.0 or bh <= 1.0:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + bw))
    y2 = max(0.0, min(float(height), y + bh))
    cw = x2 - x1
    ch = y2 - y1
    if cw <= 1.0 or ch <= 1.0:
        return None
    return (
        ((x1 + x2) * 0.5) / float(width),
        ((y1 + y2) * 0.5) / float(height),
        cw / float(width),
        ch / float(height),
    )


def _format_yolo_lines(boxes: list[tuple[int, float, float, float, float]]) -> str:
    """Format class-id and normalized YOLO coordinates into label text."""
    return "".join(
        f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        for cls_id, xc, yc, w, h in boxes
    )


def ensure_coco_val2017(cache_dir: str) -> tuple[str, str]:
    """Download and extract COCO ``val2017`` images and annotations if absent."""
    files.mkdir(cache_dir)
    images_dir = files.join(cache_dir, "val2017")
    ann_file = files.join(cache_dir, "annotations", "instances_val2017.json")

    if not files.is_dir(images_dir) or not files.find(images_dir, "*.jpg"):
        zip_path = files.join(cache_dir, "val2017.zip")
        if not files.is_file(zip_path):
            print(f"[coco] Downloading {COCO_VAL_IMAGES_URL} ...")
            files.download(
                COCO_VAL_IMAGES_URL,
                cache_dir,
                overwrite=False,
                verbose=True,
                timeout=600.0,
                retries=2,
            )
        print(f"[coco] Extracting {zip_path} ...")
        archive.extract(zip_path, cache_dir, overwrite=True, verbose=False)

    if not files.is_file(ann_file):
        ann_zip = files.join(cache_dir, "annotations_trainval2017.zip")
        if not files.is_file(ann_zip):
            print(f"[coco] Downloading {COCO_ANNOTATIONS_URL} ...")
            files.download(
                COCO_ANNOTATIONS_URL,
                cache_dir,
                overwrite=False,
                verbose=True,
                timeout=600.0,
                retries=2,
            )
        print(f"[coco] Extracting {ann_zip} ...")
        archive.extract(
            ann_zip,
            cache_dir,
            overwrite=True,
            verbose=False,
            include=["annotations/instances_val2017.json"],
        )

    return str(images_dir), str(ann_file)


def index_coco_records(
    images_dir: str,
    ann_file: str,
    target_classes: dict[int, str] = TARGET_CLASSES,
) -> list[dict[str, Any]]:
    """Load COCO annotations and map non-crowd target boxes to YOLO format."""
    payload = files.load(ann_file, verbose=False)
    name_to_local = {name: cls_id for cls_id, name in target_classes.items()}
    coco_id_to_local = {
        int(cat["id"]): name_to_local[cat["name"]]
        for cat in payload.get("categories", [])
        if cat.get("name") in name_to_local
    }

    images_by_id: dict[int, dict[str, Any]] = {}
    for img in payload.get("images", []):
        img_id = int(img["id"])
        img_path = files.join(images_dir, img["file_name"])
        if files.is_file(img_path):
            images_by_id[img_id] = {
                "image_id": img_id,
                "file_name": str(img["file_name"]),
                "path": str(img_path),
                "width": int(img["width"]),
                "height": int(img["height"]),
                "boxes": [],
            }

    for ann in payload.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        local_id = coco_id_to_local.get(int(ann["category_id"]))
        if local_id is None:
            continue
        record = images_by_id.get(int(ann["image_id"]))
        if record is None:
            continue
        yolo_box = _coco_bbox_to_yolo(ann["bbox"], record["width"], record["height"])
        if yolo_box is not None:
            record["boxes"].append((local_id, *yolo_box))

    records: list[dict[str, Any]] = []
    for record in images_by_id.values():
        if not record["boxes"]:
            continue
        record["labels"] = sorted({TARGET_CLASSES[box[0]] for box in record["boxes"]})
        record["has_rare"] = any(label in RARE_CLASSES for label in record["labels"])
        records.append(record)

    records.sort(key=lambda item: int(item["image_id"]))
    return records


def build_realistic_splits(
    records: list[dict[str, Any]],
    *,
    val_per_rare: int = 36,
    val_per_head: int = 50,
    train_rare_cap: int = 24,
    train_head_target: int = 520,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition indexed records into disjoint held-out val and long-tail train core."""
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    used_ids: set[int] = set()
    val_records: list[dict[str, Any]] = []
    val_counts: Counter[str] = Counter()

    def _add_val(rec: dict[str, Any]) -> None:
        used_ids.add(rec["image_id"])
        val_records.append(rec)
        val_counts.update(rec["labels"])

    # 1. Held-out val: ensure every rare class has enough clean evaluation images
    for rare_cls in RARE_CLASSES:
        for rec in pool:
            if val_counts[rare_cls] >= val_per_rare:
                break
            if rec["image_id"] in used_ids or rare_cls not in rec["labels"]:
                continue
            _add_val(rec)

    # Top up head classes on val using head-only images first to preserve rare train data
    for head_cls in HEAD_CLASSES:
        for head_only in (True, False):
            for rec in pool:
                if val_counts[head_cls] >= val_per_head:
                    break
                if rec["image_id"] in used_ids or head_cls not in rec["labels"]:
                    continue
                if head_only and rec["has_rare"]:
                    continue
                _add_val(rec)

    # 2. Disjoint clean long-tail training core
    remaining = [rec for rec in pool if rec["image_id"] not in used_ids]
    train_core: list[dict[str, Any]] = []
    train_used: set[int] = set()
    rare_img_counts: Counter[str] = Counter()

    for rare_cls in RARE_CLASSES:
        for rec in remaining:
            if rare_img_counts[rare_cls] >= train_rare_cap:
                break
            if rec["image_id"] in train_used or rare_cls not in rec["labels"]:
                continue
            # Keep rare counts balanced and genuinely scarce
            if any(
                rare_img_counts[other] >= train_rare_cap + 6
                for other in rec["labels"]
                if other in RARE_CLASSES
            ):
                continue
            train_used.add(rec["image_id"])
            train_core.append(rec)
            rare_img_counts.update(rec["labels"])

    # Abundant head-only images (strictly excluding rare classes to maintain long-tail ratio)
    head_only_candidates = [
        rec for rec in remaining
        if rec["image_id"] not in train_used and not rec["has_rare"]
    ]
    for rec in head_only_candidates[:train_head_target]:
        train_used.add(rec["image_id"])
        train_core.append(rec)

    assert used_ids.isdisjoint(train_used), "Train/val split overlap detected"
    return val_records, train_core


def materialize_benchmark_datasets(
    val_records: list[dict[str, Any]],
    train_core: list[dict[str, Any]],
    workspace_dir: str,
    *,
    n_exact_dupes: int = 110,
    n_near_dupes: int = 110,
    n_degraded: int = 120,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create flat YOLO ``val_fixed`` and noisy/imbalanced ``train_pool`` via Klygo APIs."""
    files.mkdir(workspace_dir)
    val_stage = files.join(workspace_dir, "_stage_val")
    pool_stage = files.join(workspace_dir, "_stage_pool")
    val_dir = files.join(workspace_dir, "val_fixed")
    pool_dir = files.join(workspace_dir, "train_pool")

    for stage in (val_stage, pool_stage):
        files.mkdir(files.join(stage, "images"))
        files.mkdir(files.join(stage, "labels"))

    # Stage clean held-out validation split
    for idx, rec in enumerate(val_records):
        stem = f"val_{idx:05d}_{rec['image_id']}"
        ext = files.extension(rec["path"]) or ".jpg"
        files.copy(rec["path"], files.join(val_stage, "images", f"{stem}{ext}"), overwrite=True)
        files.save(
            files.join(val_stage, "labels", f"{stem}.txt"),
            _format_yolo_lines(rec["boxes"]),
            overwrite=True,
            verbose=False,
        )

    datasets.convert(
        val_stage,
        val_dir,
        format="yolo",
        tree={"images": "images", "labels": "labels"},
        classes=dict(TARGET_CLASSES),
        overwrite=True,
    )

    # Prepare Klygo preprocessing corruption/near-duplicate transforms
    degrade_transforms = [
        preprocessing.gaussian_blur(kernel=17, sigma=6.0),
        preprocessing.compose(preprocessing.brightness(0.24), preprocessing.contrast(0.45)),
        preprocessing.compose(preprocessing.contrast(0.22), preprocessing.brightness(1.35)),
    ]
    near_dupe_transforms = [
        preprocessing.compose(preprocessing.brightness(1.04), preprocessing.contrast(0.97)),
        preprocessing.compose(preprocessing.gaussian_blur(kernel=3, sigma=0.5), preprocessing.brightness(0.97)),
    ]

    rng = random.Random(seed)
    head_only_indices = [idx for idx, rec in enumerate(train_core) if not rec["has_rare"]]
    rng.shuffle(head_only_indices)
    degrade_set = set(head_only_indices[: min(n_degraded, len(head_only_indices))])

    # Reserve clean head-only donors for exact & near duplicate generation
    clean_head_indices = [
        idx for idx in head_only_indices if idx not in degrade_set
    ] or list(head_only_indices)

    entries: list[tuple[str, dict[str, Any], str]] = []
    for idx, rec in enumerate(train_core):
        mode = "degraded" if idx in degrade_set else "clean"
        entries.append((f"core_{idx:05d}_{rec['image_id']}", rec, mode))

    if clean_head_indices:
        # Concentrate duplicates on a subset of donor images (4-8x redundancy bursts)
        donor_pool = clean_head_indices[: max(1, min(28, len(clean_head_indices)))]
        for dup_i in range(n_exact_dupes):
            donor = train_core[donor_pool[dup_i % len(donor_pool)]]
            entries.append((f"exactdup_{dup_i:04d}_{donor['image_id']}", donor, "exact_dup"))
        for near_i in range(n_near_dupes):
            donor = train_core[donor_pool[(near_i * 3 + 1) % len(donor_pool)]]
            entries.append((f"neardup_{near_i:04d}_{donor['image_id']}", donor, "near_dup"))

    rng.shuffle(entries)

    origin_counts: Counter[str] = Counter()
    for order_idx, (stem, rec, mode) in enumerate(entries):
        origin_counts[mode] += 1
        ext = files.extension(rec["path"]) or ".jpg"
        dst_img = files.join(pool_stage, "images", f"{stem}{ext}")
        dst_lbl = files.join(pool_stage, "labels", f"{stem}.txt")

        if mode in {"clean", "exact_dup"}:
            files.copy(rec["path"], dst_img, overwrite=True)
        elif mode == "near_dup":
            loaded = media.load(rec["path"], verbose=False)[0]
            op = near_dupe_transforms[order_idx % len(near_dupe_transforms)]
            media.save(dst_img, op(loaded).to_pil(), overwrite=True, verbose=False)
        else:
            loaded = media.load(rec["path"], verbose=False)[0]
            op = degrade_transforms[order_idx % len(degrade_transforms)]
            media.save(dst_img, op(loaded).to_pil(), overwrite=True, verbose=False)

        files.save(dst_lbl, _format_yolo_lines(rec["boxes"]), overwrite=True, verbose=False)

    datasets.convert(
        pool_stage,
        pool_dir,
        format="yolo",
        tree={"images": "images", "labels": "labels"},
        classes=dict(TARGET_CLASSES),
        overwrite=True,
    )

    files.remove(val_stage, missing_ok=True)
    files.remove(pool_stage, missing_ok=True)

    return {
        "val_dir": str(val_dir),
        "pool_dir": str(pool_dir),
        "val_images": len(val_records),
        "pool_images": len(entries),
        "pool_composition": dict(origin_counts),
    }


def export_yolo_subset(
    subset: outputs.Detections,
    val_dir: str,
    target_dir: str,
    class_names: dict[int, str] = TARGET_CLASSES,
) -> str:
    """Materialize a curated ``Detections`` subset as a flat YOLO training dataset."""
    files.remove(target_dir, missing_ok=True)
    images_out = files.join(target_dir, "images")
    labels_out = files.join(target_dir, "labels")
    files.mkdir(images_out)
    files.mkdir(labels_out)

    name_to_id = {name: int(cls_id) for cls_id, name in class_names.items()}
    for idx, frame in enumerate(subset):
        src_path = str(frame.source_path)
        ext = files.extension(src_path) or ".jpg"
        stem = f"{idx:05d}_{files.stem(src_path)}"
        files.copy(src_path, files.join(images_out, f"{stem}{ext}"), overwrite=True)

        width, height = frame.image_size
        yolo_boxes: list[tuple[int, float, float, float, float]] = []
        for box in frame:
            cls_id = name_to_id.get(box.label, int(box.class_id))
            x1, y1, x2, y2 = float(box.xmin), float(box.ymin), float(box.xmax), float(box.ymax)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if width <= 0 or height <= 0 or bw <= 0.0 or bh <= 0.0:
                continue
            yolo_boxes.append((
                cls_id,
                min(1.0, max(0.0, ((x1 + x2) * 0.5) / float(width))),
                min(1.0, max(0.0, ((y1 + y2) * 0.5) / float(height))),
                min(1.0, bw / float(width)),
                min(1.0, bh / float(height)),
            ))

        files.save(
            files.join(labels_out, f"{stem}.txt"),
            _format_yolo_lines(yolo_boxes),
            overwrite=True,
            verbose=False,
        )

    yaml_path = files.join(target_dir, "data.yaml")
    val_images_abs = str(files.resolve(files.join(val_dir, "images")))
    train_images_abs = str(files.resolve(images_out))
    config.save(
        yaml_path,
        {
            "path": str(files.resolve(target_dir)),
            "train": train_images_abs,
            "val": val_images_abs,
            "nc": len(class_names),
            "names": {int(k): v for k, v in sorted(class_names.items())},
        },
        overwrite=True,
        verbose=False,
    )
    return str(yaml_path)


def _summarize_subset(
    subset: outputs.Detections,
    baseline_stats: mining.MiningStats,
    mining_seconds: float,
    note: str = "",
) -> dict[str, Any]:
    """Extract comparable mining statistics and rarity telemetry for one subset."""
    cur_stats = mining.stats(subset, quality=True, cooccurrence=False)
    return {
        "images": int(cur_stats.images),
        "boxes": int(cur_stats.boxes),
        "classes": int(cur_stats.classes),
        "imbalance_ratio": round(float(cur_stats.imbalance_ratio), 3),
        "delta_imbalance_ratio": round(
            float(cur_stats.imbalance_ratio - baseline_stats.imbalance_ratio), 3
        ),
        "images_per_class": dict(cur_stats.images_per_class),
        "boxes_per_class": dict(cur_stats.boxes_per_class),
        "rare_images_sum": sum(cur_stats.images_per_class.get(cls, 0) for cls in RARE_CLASSES),
        "rare_boxes_sum": sum(cur_stats.boxes_per_class.get(cls, 0) for cls in RARE_CLASSES),
        "quality_summary": cur_stats.quality,
        "mining_seconds": round(float(mining_seconds), 3),
        "note": note,
    }


def build_curation_strategies(
    pool: outputs.Detections,
    *,
    budget: int = DEFAULT_BUDGET,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run all Klygo mining strategies and return telemetry plus per-run subset plans."""
    t0 = perf_counter()
    pool_stats = mining.stats(pool, quality=True, cooccurrence=True)
    stats_seconds = perf_counter() - t0

    t0 = perf_counter()
    dup_report = mining.find_duplicates(pool, exact=True, method="phash", threshold=0.92)
    find_dup_seconds = perf_counter() - t0

    t0 = perf_counter()
    dedup = mining.deduplicate(pool, keep="best", threshold=0.92)
    dedup_seconds = perf_counter() - t0

    t0 = perf_counter()
    pool_q_scores = mining.score(pool, weights={"quality": 1.0, "rarity": 0.0, "coverage": 0.0, "boxes": 0.0})
    pool_b_scores = mining.score(pool, weights={"quality": 0.35, "rarity": 0.35, "coverage": 0.20, "boxes": 0.10})
    score_seconds = perf_counter() - t0

    t0 = perf_counter()
    n_clusters = min(len(pool), max(2, budget))
    pool_clusters = mining.cluster(pool, clusters=n_clusters, seed=seeds[0])
    cluster_seconds = perf_counter() - t0

    # Shared deterministic subsets (independent of training RNG seed)
    t0 = perf_counter()
    sub_quality = mining.select(pool, method="quality", threshold=0.0, limit=budget, scores=pool_q_scores)
    t_quality = score_seconds + (perf_counter() - t0)

    t0 = perf_counter()
    sub_rare = mining.select(pool, method="rare_first", limit=budget, scores=pool_b_scores)
    t_rare = score_seconds + (perf_counter() - t0)

    t0 = perf_counter()
    sub_weighted = mining.select(pool, method="weighted", limit=budget, scores=pool_b_scores)
    t_weighted = score_seconds + (perf_counter() - t0)

    t0 = perf_counter()
    sub_diversity = mining.select(
        pool,
        method="diversity",
        limit=budget,
        cluster_result=pool_clusters,
        per_cluster=1,
        seed=seeds[0],
    )
    t_diversity = cluster_seconds + (perf_counter() - t0)

    t0 = perf_counter()
    sub_coverage = mining.select(pool, method="coverage", per_class=budget, limit=budget)
    t_coverage = perf_counter() - t0

    t0 = perf_counter()
    sub_balance = mining.balance(dedup, target="median", strategy="undersample", count_by="images", seed=seeds[0])
    t_balance = dedup_seconds + (perf_counter() - t0)

    t0 = perf_counter()
    dedup_b_scores = mining.score(
        dedup,
        weights={"quality": 0.35, "rarity": 0.35, "coverage": 0.20, "boxes": 0.10},
    )
    sub_dedup_weighted = mining.select(dedup, method="weighted", limit=budget, scores=dedup_b_scores)
    t_dedup_weighted = dedup_seconds + (perf_counter() - t0)

    # Smart 3-stage Klygo pipeline: deduplicate -> filter bottom-15% quality -> rarity+quality weighted select
    t0 = perf_counter()
    dedup_q_scores = mining.score(
        dedup,
        weights={"quality": 1.0, "rarity": 0.0, "coverage": 0.0, "boxes": 0.0},
    )
    q_floor = float(np.percentile(dedup_q_scores.values, 15)) if dedup_q_scores.values else 0.0
    clean_dedup = mining.select(dedup, method="quality", threshold=q_floor, scores=dedup_q_scores)
    clean_b_scores = mining.score(
        clean_dedup,
        weights={"quality": 0.35, "rarity": 0.45, "coverage": 0.15, "boxes": 0.05},
    )
    sub_smart = mining.select(clean_dedup, method="weighted", limit=budget, scores=clean_b_scores)
    t_smart = dedup_seconds + (perf_counter() - t0)

    deterministic_arms: list[tuple[str, outputs.Detections, float, str]] = [
        ("full_raw", pool, 0.0, "Uncurated noisy long-tail ingestion pool"),
        ("dedup_only", dedup, dedup_seconds, f"mining.deduplicate(keep='best', threshold=0.92)"),
        ("quality_b", sub_quality, t_quality, f"mining.select(method='quality', limit={budget})"),
        ("rare_first_b", sub_rare, t_rare, f"mining.select(method='rare_first', limit={budget})"),
        ("diversity_b", sub_diversity, t_diversity, f"mining.select(method='diversity', limit={budget})"),
        ("weighted_b", sub_weighted, t_weighted, f"mining.select(method='weighted', limit={budget})"),
        ("coverage_b", sub_coverage, t_coverage, f"mining.select(method='coverage', per_class={budget}, limit={budget})"),
        ("balance_median", sub_balance, t_balance, "mining.balance(dedup, target='median', strategy='undersample')"),
        ("dedup_weighted_b", sub_dedup_weighted, t_dedup_weighted, f"deduplicate -> weighted select(limit={budget})"),
        ("klygo_smart_b", sub_smart, t_smart, f"deduplicate -> quality>=p15 ({q_floor:.3f}) -> weighted(rarity=0.45, quality=0.35)"),
    ]

    run_plans: list[dict[str, Any]] = []
    strategy_summaries: dict[str, Any] = {}

    for method_name, subset, m_sec, desc in deterministic_arms:
        summary = _summarize_subset(subset, pool_stats, m_sec, note=desc)
        strategy_summaries[method_name] = summary
        for seed in seeds:
            run_plans.append({
                "method": method_name,
                "seed": int(seed),
                "subset": subset,
                "subset_stats": summary,
            })

    # Random baseline draws an independent random sample per seed
    rand_stats_list: list[dict[str, Any]] = []
    for seed in seeds:
        t0 = perf_counter()
        sub_rand = mining.select(pool, method="random", limit=budget, seed=int(seed))
        t_rand = perf_counter() - t0
        r_summary = _summarize_subset(
            sub_rand,
            pool_stats,
            t_rand,
            note=f"mining.select(method='random', limit={budget}, seed={seed})",
        )
        rand_stats_list.append(r_summary)
        run_plans.append({
            "method": "random_b",
            "seed": int(seed),
            "subset": sub_rand,
            "subset_stats": r_summary,
        })
    strategy_summaries["random_b"] = rand_stats_list[0]

    telemetry = {
        "pool_stats": pool_stats.to_dict(),
        "duplicate_report": {
            "total_groups": len(dup_report.groups),
            "exact_groups": len(dup_report.exact_groups),
            "duplicate_images": dup_report.duplicate_images,
            "find_duplicates_seconds": round(find_dup_seconds, 3),
            "deduplicate_seconds": round(dedup_seconds, 3),
            "retained_after_dedup": len(dedup),
        },
        "op_timings_seconds": {
            "stats": round(stats_seconds, 3),
            "find_duplicates": round(find_dup_seconds, 3),
            "deduplicate": round(dedup_seconds, 3),
            "score": round(score_seconds, 3),
            "cluster": round(cluster_seconds, 3),
        },
        "strategies": strategy_summaries,
    }
    return telemetry, run_plans


def _reset_torch_memory() -> None:
    """Release CUDA cached allocations between detector training runs."""
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def train_and_evaluate_yolo(
    data_yaml: str,
    run_dir: str,
    *,
    weights: str = "yolov8n.pt",
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = 640,
    batch: int = 32,
    seed: int = 42,
    device: str | int = 0,
    workers: int = 4,
) -> dict[str, Any]:
    """Train ``YOLOv8n`` on ``data_yaml`` and evaluate best checkpoint on ``val_fixed``."""
    from ultralytics import YOLO

    files.remove(run_dir, missing_ok=True)
    model = YOLO(weights)

    t0 = perf_counter()
    model.train(
        data=data_yaml,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        seed=int(seed),
        deterministic=True,
        device=device,
        workers=int(workers),
        cache=False,
        amp=True,
        project=str(files.parent(run_dir)),
        name=files.name(run_dir),
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    train_seconds = perf_counter() - t0

    best_weights = files.join(run_dir, "weights", "best.pt")
    eval_model = YOLO(str(best_weights)) if files.is_file(best_weights) else model
    val_res = eval_model.val(
        data=data_yaml,
        split="val",
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        workers=int(workers),
        verbose=False,
        plots=False,
    )

    box = val_res.box
    per_class_map50_95 = {
        TARGET_CLASSES[cls_id]: round(float(box.maps[cls_id]), 5)
        for cls_id in sorted(TARGET_CLASSES)
        if cls_id < len(box.maps)
    }
    ap50_arr = box.ap50
    ap_idx = getattr(box, "ap_class_index", [])
    per_class_map50 = {TARGET_CLASSES[cls_id]: 0.0 for cls_id in sorted(TARGET_CLASSES)}
    for pos, cls_id in enumerate(ap_idx):
        if int(cls_id) in TARGET_CLASSES and pos < len(ap50_arr):
            per_class_map50[TARGET_CLASSES[int(cls_id)]] = round(float(ap50_arr[pos]), 5)

    rare_map50_95 = float(mean(per_class_map50_95[c] for c in RARE_CLASSES))
    head_map50_95 = float(mean(per_class_map50_95[c] for c in HEAD_CLASSES))

    metrics = {
        "mAP50_95": round(float(box.map), 5),
        "mAP50": round(float(box.map50), 5),
        "mAP75": round(float(box.map75), 5),
        "precision": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
        "rare_mAP50_95": round(rare_map50_95, 5),
        "head_mAP50_95": round(head_map50_95, 5),
        "per_class_mAP50_95": per_class_map50_95,
        "per_class_mAP50": per_class_map50,
        "train_seconds": round(float(train_seconds), 2),
    }

    del model, eval_model
    _reset_torch_memory()
    return metrics


def aggregate_benchmark_results(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-seed training outcomes by strategy into a sorted leaderboard."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(row["method"], []).append(row)

    leaderboard: list[dict[str, Any]] = []
    for method, items in grouped.items():
        map_vals = [item["metrics"]["mAP50_95"] for item in items]
        map50_vals = [item["metrics"]["mAP50"] for item in items]
        prec_vals = [item["metrics"]["precision"] for item in items]
        rec_vals = [item["metrics"]["recall"] for item in items]
        rare_vals = [item["metrics"]["rare_mAP50_95"] for item in items]
        head_vals = [item["metrics"]["head_mAP50_95"] for item in items]
        train_secs = [item["metrics"]["train_seconds"] for item in items]

        avg_per_class = {
            cls_name: round(
                mean(item["metrics"]["per_class_mAP50_95"][cls_name] for item in items),
                4,
            )
            for cls_name in TARGET_CLASSES.values()
        }
        ref_stats = items[0]["subset_stats"]
        leaderboard.append({
            "method": method,
            "runs": len(items),
            "images": ref_stats["images"],
            "boxes": ref_stats["boxes"],
            "imbalance_ratio": ref_stats["imbalance_ratio"],
            "rare_images_sum": ref_stats["rare_images_sum"],
            "mining_seconds": ref_stats["mining_seconds"],
            "mean_train_seconds": round(float(mean(train_secs)), 2),
            "mAP50_95_mean": round(float(mean(map_vals)), 5),
            "mAP50_95_std": round(float(stdev(map_vals)), 5) if len(map_vals) > 1 else 0.0,
            "mAP50_mean": round(float(mean(map50_vals)), 5),
            "precision_mean": round(float(mean(prec_vals)), 5),
            "recall_mean": round(float(mean(rec_vals)), 5),
            "rare_mAP50_95_mean": round(float(mean(rare_vals)), 5),
            "head_mAP50_95_mean": round(float(mean(head_vals)), 5),
            "per_class_mAP50_95_mean": avg_per_class,
        })

    leaderboard.sort(key=lambda row: row["mAP50_95_mean"], reverse=True)
    return leaderboard


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Format complete benchmark metadata, dataset telemetry, and leaderboard as Markdown."""
    env = payload["environment"]
    ds = payload["dataset"]
    dup = payload["mining_telemetry"]["duplicate_report"]
    lb = payload["leaderboard"]

    lines = [
        "# Klygo Mining Curation Accuracy Benchmark",
        "",
        "## Environment & Protocol",
        "",
        f"- **GPU**: {env['gpu_name']} ({env['gpu_vram_gb']} GB)",
        f"- **PyTorch / Ultralytics**: `{env['torch']}` / `{env['ultralytics']}`",
        f"- **Dataset**: Real COCO `val2017` ({len(TARGET_CLASSES)} classes: 5 head + 5 rare, zero train/val leakage)",
        f"- **Validation (`val_fixed`)**: {ds['val_images']} clean images",
        f"- **Raw Train Pool (`train_pool`)**: {ds['pool_images']} images (`{ds['pool_composition']}`)",
        f"- **Duplicate Audit**: {dup['exact_groups']} exact groups, {dup['total_groups']} total groups, {dup['duplicate_images']} flagged duplicate frames -> {dup['retained_after_dedup']} retained",
        f"- **Training Config**: `{payload['config']['weights']}`, `epochs={payload['config']['epochs']}`, `imgsz={payload['config']['imgsz']}`, `budget={payload['config']['budget']}`, `seeds={payload['config']['seeds']}`",
        "",
        "## Curation Accuracy Leaderboard (Evaluated on Held-Out `val_fixed`)",
        "",
        "| Strategy | Images | Imbalance | Rare Imgs | Mining (s) | Train (s) | mAP@50-95 | mAP@50 | Rare mAP | Head mAP | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in lb:
        std_txt = f" ± {row['mAP50_95_std']:.4f}" if row["runs"] > 1 else ""
        lines.append(
            f"| `{row['method']}` | {row['images']} | {row['imbalance_ratio']:.2f} | "
            f"{row['rare_images_sum']} | {row['mining_seconds']:.2f} | {row['mean_train_seconds']:.1f} | "
            f"**{row['mAP50_95_mean']:.4f}**{std_txt} | {row['mAP50_mean']:.4f} | "
            f"{row['rare_mAP50_95_mean']:.4f} | {row['head_mAP50_95_mean']:.4f} | "
            f"{row['precision_mean']:.4f} | {row['recall_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## Per-Class Mean `AP@50-95`",
        "",
        "| Strategy | " + " | ".join(TARGET_CLASSES.values()) + " |",
        "| --- | " + " | ".join("---:" for _ in TARGET_CLASSES) + " |",
    ])
    for row in lb:
        vals = " | ".join(f"{row['per_class_mAP50_95_mean'][c]:.4f}" for c in TARGET_CLASSES.values())
        lines.append(f"| `{row['method']}` | {vals} |")

    lines.append("")
    return "\n".join(lines)


def _detect_environment(weights: str) -> dict[str, Any]:
    """Capture hardware and package versions for reproducibility."""
    import torch
    import ultralytics

    gpu_name = "CPU"
    vram_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = str(props.name)
        vram_gb = round(float(props.total_memory) / (1024.0 ** 3), 2)

    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__),
        "gpu_name": gpu_name,
        "gpu_vram_gb": vram_gb,
        "weights": weights,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute end-to-end COCO preparation, mining curation, YOLO training, and reporting."""
    output_dir = str(files.resolve(args.output))
    cache_dir = str(files.resolve(args.cache_dir))
    files.mkdir(output_dir)

    images_dir, ann_file = ensure_coco_val2017(cache_dir)
    records = index_coco_records(images_dir, ann_file)
    val_records, train_core = build_realistic_splits(records)

    datasets_dir = files.join(output_dir, "datasets")
    val_marker = files.join(datasets_dir, "val_fixed", "images")
    pool_marker = files.join(datasets_dir, "train_pool", "images")
    manifest_path = files.join(output_dir, "dataset_manifest.json")

    if files.is_file(manifest_path) and files.is_dir(val_marker) and files.is_dir(pool_marker):
        ds_meta = files.load(manifest_path, verbose=False)
        print("[dataset] Reusing cached val_fixed and train_pool from output directory.")
    else:
        ds_meta = materialize_benchmark_datasets(
            val_records,
            train_core,
            datasets_dir,
            n_exact_dupes=args.exact_dupes,
            n_near_dupes=args.near_dupes,
            n_degraded=args.degraded,
        )
        files.save(manifest_path, ds_meta, overwrite=True, verbose=False)

    print(f"[dataset] val_fixed={ds_meta['val_images']} imgs | train_pool={ds_meta['pool_images']} imgs | {ds_meta['pool_composition']}")

    pool_detections = datasets.load(ds_meta["pool_dir"], format="yolo", verbose=False)
    print(f"[mining] Loaded pool Detections ({len(pool_detections)} frames). Running curation strategies ...")
    telemetry, run_plans = build_curation_strategies(
        pool_detections,
        budget=args.budget,
        seeds=tuple(args.seeds),
    )

    checkpoint_path = files.join(output_dir, "results_checkpoint.json")
    completed_runs: list[dict[str, Any]] = []
    if files.is_file(checkpoint_path):
        saved_ckpt = files.load(checkpoint_path, verbose=False)
        completed_runs = list(saved_ckpt.get("runs", []))
    done_keys = {(row["method"], int(row["seed"])) for row in completed_runs}

    stages_dir = files.join(output_dir, "stages")
    yolo_runs_dir = files.join(output_dir, "yolo_runs")
    files.mkdir(stages_dir)
    files.mkdir(yolo_runs_dir)

    env = _detect_environment(args.weights)
    device: str | int = 0 if env["gpu_name"] != "CPU" else "cpu"
    total_runs = len(run_plans)

    for run_idx, plan in enumerate(run_plans, start=1):
        method = plan["method"]
        seed = int(plan["seed"])
        if (method, seed) in done_keys:
            print(f"[train {run_idx}/{total_runs}] Skipping already-completed {method} (seed={seed})")
            continue

        tag = f"{method}_seed{seed}"
        stage_dir = files.join(stages_dir, tag)
        run_dir = files.join(yolo_runs_dir, tag)

        print(f"[train {run_idx}/{total_runs}] Staging {tag} ({len(plan['subset'])} images) ...")
        data_yaml = export_yolo_subset(plan["subset"], ds_meta["val_dir"], stage_dir)

        metrics = train_and_evaluate_yolo(
            data_yaml,
            run_dir,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=seed,
            device=device,
            workers=args.workers,
        )
        print(
            f"[done {run_idx}/{total_runs}] {tag}: "
            f"mAP50-95={metrics['mAP50_95']:.4f}, mAP50={metrics['mAP50']:.4f}, "
            f"rare={metrics['rare_mAP50_95']:.4f}, time={metrics['train_seconds']:.1f}s"
        )

        completed_runs.append({
            "method": method,
            "seed": seed,
            "subset_stats": plan["subset_stats"],
            "metrics": metrics,
        })
        done_keys.add((method, seed))
        files.save(
            checkpoint_path,
            {"runs": completed_runs},
            overwrite=True,
            verbose=False,
        )
        # Clean heavy run weights once evaluated to preserve disk space
        files.remove(stage_dir, missing_ok=True)
        files.remove(run_dir, missing_ok=True)

    leaderboard = aggregate_benchmark_results(completed_runs)
    payload = {
        "environment": env,
        "config": {
            "weights": args.weights,
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "budget": int(args.budget),
            "seeds": [int(s) for s in args.seeds],
        },
        "dataset": ds_meta,
        "mining_telemetry": telemetry,
        "runs": completed_runs,
        "leaderboard": leaderboard,
    }

    report_dir = files.join(output_dir, "report")
    files.mkdir(report_dir)
    md_text = render_markdown_report(payload)
    files.save(files.join(report_dir, "summary.json"), payload, overwrite=True, verbose=False)
    files.save(files.join(report_dir, "summary.md"), md_text, overwrite=True, verbose=False)

    zip_path = files.join(output_dir, "mining_curation_results.zip")
    archive.compress(report_dir, zip_path, include_root=False, overwrite=True, verbose=False)
    payload["report_zip"] = str(zip_path)
    payload["report_zip_sha256"] = files.hash(zip_path)

    print("\n" + md_text)
    print(f"[report] ZIP: {zip_path} (sha256={payload['report_zip_sha256']})")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Klygo mining curation accuracy benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="artifacts/coco_val2017_cache")
    parser.add_argument("--output", default="artifacts/mining_curation_run")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exact-dupes", type=int, default=110)
    parser.add_argument("--near-dupes", type=int, default=110)
    parser.add_argument("--degraded", type=int, default=120)
    return parser.parse_args(argv)


if __name__ == "__main__":
    os.environ.setdefault("KLYGO_SUPPRESS_BANNER", "1")
    run_benchmark(parse_args())
