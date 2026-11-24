"""Standalone full-class curation accuracy & intrinsic quality benchmark runner for klygo.mining.

Evaluates 15 `klygo.mining` selection, deduplication, clustering, and balancing
methods across 100% of dataset classes (`coco128`, `voc2007`, `val2017`, or
custom `--dataset-dir`) on two complementary axes (no speed/timing benchmarks):

Axis 1 — Intrinsic Multi-Class Curation Metrics (`mining.stats`, `mining.score`, `mining.compare`, `mining.cluster`, `mining.find_duplicates`):
    - `active_classes` & `class_coverage_pct` (% of pool classes preserved)
    - `rare_tail_coverage_pct` (% of bottom-quartile rarest classes preserved)
    - `class_entropy_norm` (normalized Shannon entropy of class box distribution in [0, 1])
    - `imbalance_ratio` (max/min class count ratio)
    - `cluster_coverage_pct` (% of K-Means visual clusters covered)
    - `mean_quality_score` & `mean_composite_score` (from `mining.score`)
    - `boxes_per_image` & `multilabel_rate_pct`

Axis 2 — Downstream Detector Training Accuracy (`yolov8n.pt` fine-tuned on each subset, evaluated on a fixed clean validation split across all classes):
    - `mAP50_95`, `mAP50`, `precision`, `recall`, `f1`, `delta_map50_95_vs_random`

15 Evaluated Methods:
    1. `full_pool`                   : 100% of training pool (upper-bound reference)
    2. `random`                      : `mining.select(method="random")`
    3. `weighted`                    : `mining.select(method="weighted")`
    4. `quality`                     : `mining.select(method="quality")`
    5. `rare_first`                  : `mining.select(method="rare_first")`
    6. `diversity`                   : `mining.cluster()` + `mining.select(method="diversity")`
    7. `coverage`                    : `mining.select(method="coverage")`
    8. `quota`                       : `mining.select(method="quota")`
    9. `boxes`                       : `mining.select(method="boxes")`
   10. `balance_select_median`       : `mining.balance(target="median", strategy="select", count_by="images")`
   11. `balance_undersample_median`  : `mining.balance(target="median", strategy="undersample", count_by="images")`
   12. `balance_boxes_mean`          : `mining.balance(target="mean", strategy="select", count_by="boxes")`
   13. `dedup_best_weighted`         : `mining.deduplicate(keep="best")` + `mining.select(method="weighted")`
   14. `dedup_quality_rare`          : `mining.deduplicate(keep="quality")` + `mining.select(method="rare_first")`
   15. `dedup_annotations_diversity` : `mining.deduplicate(keep="annotations")` + `mining.select(method="diversity")`

Outputs the standardized benchmark file set:
    - `environment.json`
    - `curation_leaderboard.json`
    - `curation_runs.json`
    - `pool_analysis.json`
    - `summary.json`
    - `REPORT.md`
    - `charts/mining_map_comparison.png`
    - `charts/mining_f1_precision_recall.png`
    - `charts/mining_class_coverage_diversity.png`
"""

from __future__ import annotations

import argparse
import math
import random
import tempfile
from typing import Any

import numpy as np

from benchmarks.gpu_suite.evaluate_release import _md_table, collect_environment
from klygo import archive, datasets, files, mining, outputs, utils


def _split_train_val(
    dets: outputs.Detections,
    val_ratio: float = 0.2,
    seed: int = 42,
    max_pool: int = 0,
    max_val: int = 0,
) -> tuple[outputs.Detections, outputs.Detections]:
    frames = list(dets)
    rng = random.Random(seed)
    indices = list(range(len(frames)))
    rng.shuffle(indices)
    val_count = max(16, int(len(frames) * val_ratio))
    if max_val > 0:
        val_count = min(val_count, max_val)
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    if max_pool > 0 and len(train_indices) > max_pool:
        train_indices = sorted(rng.sample(train_indices, k=max_pool))
    train_frames = [frames[i] for i in train_indices]
    val_frames = [frames[i] for i in val_indices]
    return (
        outputs.Detections(train_frames, names=dets.names),
        outputs.Detections(val_frames, names=dets.names),
    )


def _pad_or_trim_to_budget(
    sub: outputs.Detections,
    pool: outputs.Detections,
    budget: int,
    fallback_scores: mining.MiningScores,
) -> outputs.Detections:
    if len(sub) > budget:
        return mining.select(sub, method="weighted", limit=budget)
    if len(sub) == budget or len(pool) <= len(sub):
        return sub
    chosen_ids = {(f.source, f.frame_index) for f in sub}
    ranked_pool = sorted(
        range(len(pool)),
        key=lambda idx: (-fallback_scores.values[idx], idx),
    )
    frames = list(sub)
    for idx in ranked_pool:
        if len(frames) >= budget:
            break
        cand = pool[idx]
        ident = (cand.source, cand.frame_index)
        if ident not in chosen_ids:
            frames.append(cand)
            chosen_ids.add(ident)
    return outputs.Detections(frames, names=pool.names)


def _build_all_methods(
    pool: outputs.Detections,
    budget: int,
    seed: int,
) -> tuple[dict[str, outputs.Detections], dict[str, Any]]:
    q_scores = mining.score(pool, weights={"quality": 1.0, "rarity": 0.0, "coverage": 0.0, "boxes": 0.0})
    b_scores = mining.score(pool, weights={"quality": 0.35, "rarity": 0.35, "coverage": 0.20, "boxes": 0.10})
    n_clusters = min(len(pool), max(4, budget))
    clusters = mining.cluster(pool, clusters=n_clusters, seed=seed)
    dup_report = mining.find_duplicates(pool, threshold=0.92)

    dedup_best = mining.deduplicate(pool, keep="best", threshold=0.92)
    dedup_qual = mining.deduplicate(pool, keep="quality", threshold=0.92)
    dedup_anno = mining.deduplicate(pool, keep="annotations", threshold=0.92)

    num_classes = max(1, len(pool.names))
    per_cls_cov = max(1, budget // num_classes)
    per_cls_quota = max(2, (budget * 2) // num_classes)

    boxes_raw = mining.boxes(pool, min_boxes=2, min_area_ratio=0.002, limit=budget)
    boxes_sub = _pad_or_trim_to_budget(boxes_raw, pool, budget, b_scores)

    bal_sel_med = _pad_or_trim_to_budget(
        mining.balance(pool, target="median", strategy="select", count_by="images", seed=seed),
        pool,
        budget,
        b_scores,
    )
    bal_und_med = _pad_or_trim_to_budget(
        mining.balance(pool, target="median", strategy="undersample", count_by="images", seed=seed),
        pool,
        budget,
        b_scores,
    )
    bal_box_mean = _pad_or_trim_to_budget(
        mining.balance(pool, target="mean", strategy="select", count_by="boxes", seed=seed),
        pool,
        budget,
        b_scores,
    )

    methods = {
        "full_pool": pool,
        "random": mining.select(pool, method="random", limit=budget, seed=seed),
        "weighted": mining.select(pool, method="weighted", limit=budget, scores=b_scores),
        "quality": mining.select(pool, method="quality", threshold=0.0, limit=budget, scores=q_scores),
        "rare_first": mining.select(pool, method="rare_first", limit=budget, scores=b_scores),
        "diversity": mining.select(pool, method="diversity", limit=budget, cluster_result=clusters, per_cluster=1, seed=seed),
        "coverage": _pad_or_trim_to_budget(
            mining.select(pool, method="coverage", per_class=per_cls_cov, limit=budget),
            pool,
            budget,
            b_scores,
        ),
        "quota": _pad_or_trim_to_budget(
            mining.select(pool, method="quota", per_class=per_cls_quota, limit=budget),
            pool,
            budget,
            b_scores,
        ),
        "boxes": boxes_sub,
        "balance_select_median": bal_sel_med,
        "balance_undersample_median": bal_und_med,
        "balance_boxes_mean": bal_box_mean,
        "dedup_best_weighted": mining.select(dedup_best, method="weighted", limit=min(budget, len(dedup_best))),
        "dedup_quality_rare": mining.select(dedup_qual, method="rare_first", limit=min(budget, len(dedup_qual))),
        "dedup_annotations_diversity": mining.select(dedup_anno, method="diversity", limit=min(budget, len(dedup_anno)), seed=seed),
    }

    aux = {
        "q_scores": q_scores,
        "b_scores": b_scores,
        "clusters": clusters,
        "dup_report": dup_report,
    }
    return methods, aux


def _compute_intrinsic_metrics(
    sub: outputs.Detections,
    pool: outputs.Detections,
    pool_stats: mining.MiningStats,
    rare_tail_classes: set[str],
    ident_to_idx: dict[tuple[Any, Any], int],
    q_scores: mining.MiningScores,
    b_scores: mining.MiningScores,
    clusters: mining.ClusterResult,
) -> dict[str, Any]:
    st = mining.stats(sub, quality=False, cooccurrence=False)
    cmp_rep = mining.compare(pool, sub)

    pool_active = [c for c, cnt in pool_stats.boxes_per_class.items() if cnt > 0]
    sub_active = [c for c, cnt in st.boxes_per_class.items() if cnt > 0]
    active_set = set(sub_active)

    class_cov_pct = (100.0 * len(sub_active) / max(1, len(pool_active)))
    tail_cov_pct = (
        100.0 * len(active_set & rare_tail_classes) / max(1, len(rare_tail_classes))
        if rare_tail_classes
        else 100.0
    )

    counts = np.array([cnt for cnt in st.boxes_per_class.values() if cnt > 0], dtype=np.float64)
    if len(counts) > 1 and counts.sum() > 0:
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        norm_entropy = entropy / math.log(len(pool_active) if len(pool_active) > 1 else 2)
    else:
        norm_entropy = 0.0

    sub_indices = [
        ident_to_idx[(f.source, f.frame_index)]
        for f in sub
        if (f.source, f.frame_index) in ident_to_idx
    ]
    mean_qual = float(np.mean([q_scores.values[i] for i in sub_indices])) if sub_indices else 0.0
    mean_comp = float(np.mean([b_scores.values[i] for i in sub_indices])) if sub_indices else 0.0
    covered_clusters = len({clusters.labels[i] for i in sub_indices}) if sub_indices else 0
    cluster_cov_pct = 100.0 * covered_clusters / max(1, clusters.clusters)

    return {
        "images_selected": len(sub),
        "boxes_selected": int(st.boxes),
        "boxes_per_image": round(float(st.boxes) / max(1, len(sub)), 2),
        "multilabel_rate_pct": round(100.0 * float(st.multilabel_images) / max(1, len(sub)), 2),
        "active_classes": len(sub_active),
        "pool_active_classes": len(pool_active),
        "class_coverage_pct": round(class_cov_pct, 2),
        "rare_tail_coverage_pct": round(tail_cov_pct, 2),
        "class_entropy_norm": round(norm_entropy, 4),
        "imbalance_ratio": round(float(st.imbalance_ratio), 2),
        "cluster_coverage_pct": round(cluster_cov_pct, 2),
        "mean_quality_score": round(mean_qual, 4),
        "mean_composite_score": round(mean_comp, 4),
        "image_retention_pct": round(100.0 * float(cmp_rep.retained_ratio), 2),
    }


def _train_and_eval_subset(
    train_sub: outputs.Detections,
    val_sub: outputs.Detections,
    *,
    rare_tail_classes: set[str],
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        root = files.path(tmp)
        train_dir = root / "train"
        val_dir = root / "val"
        train_sub.export(train_dir, format="yolo", overwrite=True)
        val_sub.export(val_dir, format="yolo", overwrite=True)

        names_map = {int(k): str(v) for k, v in sorted(train_sub.names.items())}
        data_yaml = root / "data.yaml"
        files.save(
            data_yaml,
            {
                "path": str(root.resolve()),
                "train": "train/images",
                "val": "val/images",
                "nc": len(names_map),
                "names": names_map,
            },
            overwrite=True,
            verbose=False,
        )

        model = YOLO(weights)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=seed,
            device=device,
            project=str(root / "runs"),
            name="exp",
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        val_res = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            plots=False,
        )
        box_m = val_res.box
        p_val = float(box_m.mp)
        r_val = float(box_m.mr)
        f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

        maps_arr = getattr(box_m, "maps", None)
        ap_cls_idx = [int(x) for x in getattr(box_m, "ap_class_index", [])]
        tail_aps: list[float] = []
        if maps_arr is not None and len(maps_arr) > 0:
            active_val_ids = set(ap_cls_idx) if ap_cls_idx else set(names_map.keys())
            for cid in active_val_ids:
                if cid < len(maps_arr) and names_map.get(cid) in rare_tail_classes:
                    tail_aps.append(float(maps_arr[cid]))
            if not tail_aps:
                for cid, cname in names_map.items():
                    if cid < len(maps_arr) and cname in rare_tail_classes:
                        tail_aps.append(float(maps_arr[cid]))
        rare_tail_map = float(np.mean(tail_aps)) if tail_aps else float(box_m.map)

        return {
            "mAP50_95": round(float(box_m.map), 4),
            "mAP50": round(float(box_m.map50), 4),
            "rare_tail_mAP50_95": round(rare_tail_map, 4),
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1": round(f1_val, 4),
        }


def _generate_mining_charts(leaderboard: list[dict[str, Any]], out_dir: Any) -> list[tuple[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    if not leaderboard:
        return []

    charts_dir = out_dir / "charts"
    files.mkdir(charts_dir)
    created: list[tuple[str, str]] = []

    methods = [r["method"] for r in leaderboard]
    x = np.arange(len(methods))
    width3 = 0.26

    # 1. mAP@50, mAP@50-95, and Rare-Tail mAP@50-95 by Curation Method
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar(x - width3, [float(r["mAP50"]) for r in leaderboard], width3, label="mAP@50")
    ax.bar(x, [float(r["mAP50_95"]) for r in leaderboard], width3, label="mAP@50-95")
    ax.bar(x + width3, [float(r["rare_tail_mAP50_95"]) for r in leaderboard], width3, label="Rare-Tail mAP@50-95")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=32, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Curation Method Downstream Accuracy Leaderboard (mAP@50, mAP@50-95 & Rare-Tail mAP@50-95)")
    ax.legend()
    fig.tight_layout()
    p1 = charts_dir / "mining_map_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    created.append(("Curation Method mAP Comparison", "charts/mining_map_comparison.png"))

    # 2. F1, Precision, Recall by Curation Method
    width3 = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.bar(x - width3, [float(r["precision"]) for r in leaderboard], width3, label="Precision")
    ax.bar(x, [float(r["recall"]) for r in leaderboard], width3, label="Recall")
    ax.bar(x + width3, [float(r["f1"]) for r in leaderboard], width3, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=32, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall & F1 by Curation Method")
    ax.legend()
    fig.tight_layout()
    p2 = charts_dir / "mining_f1_precision_recall.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    created.append(("Precision, Recall & F1 by Curation Method", "charts/mining_f1_precision_recall.png"))

    # 3. Intrinsic Class Coverage, Rare-Tail Retention & Visual Cluster Coverage
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.bar(x - width3, [float(r["class_coverage_pct"]) for r in leaderboard], width3, label="Class Coverage (%)")
    ax.bar(x, [float(r["rare_tail_coverage_pct"]) for r in leaderboard], width3, label="Rare-Tail Class Coverage (%)")
    ax.bar(x + width3, [float(r["cluster_coverage_pct"]) for r in leaderboard], width3, label="Visual Cluster Coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=32, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Intrinsic Multi-Class Coverage, Rare-Tail Retention & Visual Diversity")
    ax.legend()
    fig.tight_layout()
    p3 = charts_dir / "mining_class_coverage_diversity.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    created.append(("Intrinsic Class Coverage & Visual Diversity", "charts/mining_class_coverage_diversity.png"))

    return created


def _render_mining_report(summary: dict[str, Any], charts: list[tuple[str, str]]) -> str:
    env = summary["environment"]
    git_info = env.get("git", {})
    ds = summary.get("dataset", {})
    pa = summary.get("pool_analysis", {})
    lines = [
        "# Klygo Mining Full-Class Curation & Downstream Accuracy Report",
        "",
        f"- **GPU**: `{env.get('gpu', 'none')}` ({env.get('vram_mb', 0.0)} MB VRAM)",
        f"- **Dataset**: `{ds.get('source')}` (`{ds.get('total_classes')}` defined classes, `{pa.get('pool_active_classes')}` active in pool, train pool `{ds.get('train_pool_images')}` imgs, clean val `{ds.get('val_images')}` imgs, budget `{ds.get('budget')}` imgs)",
        f"- **Duplicate Analysis (`mining.find_duplicates`)**: `{pa.get('duplicate_groups', 0)}` duplicate groups (`{pa.get('duplicate_images', 0)}` redundant images at threshold `0.92`)",
        f"- **Model / Epochs**: `{summary.get('config', {}).get('weights')}` / `{summary.get('config', {}).get('epochs')}` epochs",
        f"- **Git**: `{git_info.get('branch', 'unknown')}` @ `{git_info.get('commit', 'unknown')}`",
        "",
    ]
    lines.extend(_md_table(summary.get("curation_leaderboard", []), "Curation Method Leaderboard (Sorted by Downstream mAP@50-95)"))
    lines.extend(_md_table(summary.get("curation_runs", []), "Detailed Intrinsic & Downstream Evaluation Metrics"))
    if charts:
        lines.extend(["## Charts", ""])
        for title, rel in charts:
            lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])
    return "\n".join(lines).rstrip() + "\n"


COCO80_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


def _resolve_dataset(dataset: str, dataset_dir: str) -> tuple[Any, str, str]:
    if dataset_dir:
        root = files.path(dataset_dir)
        if not files.is_dir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        img_sub = "images/train2017" if files.is_dir(root / "images" / "train2017") else "images"
        lbl_sub = "labels/train2017" if files.is_dir(root / "labels" / "train2017") else "labels"
        return root, img_sub, lbl_sub

    coco128_root = files.path("artifacts/datasets/coco128")
    if not files.is_dir(coco128_root / "images" / "train2017"):
        dl_dir = files.path("artifacts/datasets")
        files.mkdir(dl_dir)
        zip_path = files.download("https://ultralytics.com/assets/coco128.zip", dl_dir, overwrite=True, verbose=False)
        archive.extract(zip_path, dl_dir, overwrite=True, verbose=False)
    if not files.is_file(coco128_root / "data.yaml"):
        files.save(coco128_root / "data.yaml", {"names": COCO80_NAMES}, overwrite=True, verbose=False)
    return coco128_root, "images/train2017", "labels/train2017"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="coco128", help="Dataset preset (coco128) if --dataset-dir is omitted")
    parser.add_argument("--dataset-dir", default="", help="Flat YOLO dataset directory (e.g. coco128, voc2007, val2017)")
    parser.add_argument("--output", default="artifacts/benchmarks/mining_curation")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--budget-ratio", type=float, default=0.35, help="Fraction of train pool to select per curation method")
    parser.add_argument("--budget", type=int, default=0, help="Explicit image budget override (0 uses budget-ratio)")
    parser.add_argument("--max-pool", type=int, default=0, help="Optional cap on candidate training pool images (0 = use all)")
    parser.add_argument("--max-val", type=int, default=0, help="Optional cap on validation split images (0 = use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--zip", default="", help="Optional output .zip path")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    root, img_sub, lbl_sub = _resolve_dataset(args.dataset, args.dataset_dir)
    all_dets = datasets.load(root, images_dir=img_sub, labels_dir=lbl_sub, yaml_name="data.yaml", verbose=False)

    train_pool, val_clean = _split_train_val(
        all_dets,
        val_ratio=0.2,
        seed=args.seed,
        max_pool=args.max_pool,
        max_val=args.max_val,
    )
    budget = args.budget if args.budget > 0 else max(16, int(len(train_pool) * args.budget_ratio))

    env = collect_environment()
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)

    pool_stats = mining.stats(train_pool, quality=True, cooccurrence=True)
    active_sorted = sorted(
        [(cls_name, cnt) for cls_name, cnt in pool_stats.boxes_per_class.items() if cnt > 0],
        key=lambda item: (item[1], item[0]),
    )
    tail_k = max(1, len(active_sorted) // 4)
    rare_tail_classes = {cls_name for cls_name, _ in active_sorted[:tail_k]}

    ident_to_idx = {(f.source, f.frame_index): idx for idx, f in enumerate(train_pool)}
    subsets, aux = _build_all_methods(train_pool, budget=budget, seed=args.seed)

    pool_analysis = {
        "pool_images": int(pool_stats.images),
        "pool_boxes": int(pool_stats.boxes),
        "defined_classes": int(pool_stats.classes),
        "pool_active_classes": len(active_sorted),
        "rare_tail_classes": sorted(rare_tail_classes),
        "pool_imbalance_ratio": round(float(pool_stats.imbalance_ratio), 2),
        "duplicate_groups": len(aux["dup_report"].groups),
        "duplicate_images": int(aux["dup_report"].duplicate_images),
        "clusters": int(aux["clusters"].clusters),
        "quality_summary": pool_stats.quality,
    }
    files.save(out_dir / "pool_analysis.json", pool_analysis, overwrite=True, verbose=False)

    runs: list[dict[str, Any]] = []
    for idx, (method_name, sub) in enumerate(subsets.items(), 1):
        print(f"[{idx}/{len(subsets)}] Evaluating method '{method_name}' ({len(sub)} images)...")
        intrinsic = _compute_intrinsic_metrics(
            sub,
            train_pool,
            pool_stats,
            rare_tail_classes,
            ident_to_idx,
            aux["q_scores"],
            aux["b_scores"],
            aux["clusters"],
        )
        downstream = _train_and_eval_subset(
            sub,
            val_clean,
            rare_tail_classes=rare_tail_classes,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            device=args.device,
        )
        runs.append({
            "method": method_name,
            **downstream,
            **intrinsic,
        })

    random_map = next((r["mAP50_95"] for r in runs if r["method"] == "random"), 0.0)
    for r in runs:
        r["delta_map50_95_vs_random"] = round(float(r["mAP50_95"]) - float(random_map), 4)

    leaderboard_rows = sorted(
        runs,
        key=lambda r: (r["mAP50_95"], r["mAP50"], r["rare_tail_mAP50_95"], r["f1"], r["class_coverage_pct"]),
        reverse=True,
    )
    leaderboard: list[dict[str, Any]] = []
    for rank, row in enumerate(leaderboard_rows, 1):
        leaderboard.append({
            "rank": rank,
            "method": row["method"],
            "images_selected": row["images_selected"],
            "mAP50_95": row["mAP50_95"],
            "mAP50": row["mAP50"],
            "rare_tail_mAP50_95": row["rare_tail_mAP50_95"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "delta_map50_95_vs_random": row["delta_map50_95_vs_random"],
            "active_classes": f"{row['active_classes']}/{row['pool_active_classes']}",
            "class_coverage_pct": row["class_coverage_pct"],
            "rare_tail_coverage_pct": row["rare_tail_coverage_pct"],
            "cluster_coverage_pct": row["cluster_coverage_pct"],
            "class_entropy_norm": row["class_entropy_norm"],
            "mean_quality_score": row["mean_quality_score"],
        })

    summary = {
        "environment": env,
        "dataset": {
            "source": str(root),
            "total_images": len(all_dets),
            "total_classes": len(all_dets.names),
            "train_pool_images": len(train_pool),
            "val_images": len(val_clean),
            "budget": budget,
        },
        "config": {
            "weights": args.weights,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "seed": args.seed,
        },
        "pool_analysis": pool_analysis,
        "curation_leaderboard": leaderboard,
        "curation_runs": runs,
    }

    files.save(out_dir / "curation_leaderboard.json", leaderboard, overwrite=True, verbose=False)
    files.save(out_dir / "curation_runs.json", runs, overwrite=True, verbose=False)
    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)
    charts = _generate_mining_charts(leaderboard, out_dir)
    files.save(out_dir / "REPORT.md", _render_mining_report(summary, charts), overwrite=True, verbose=False)
    utils.tabulate(leaderboard, title="Mining 15-Method Curation Leaderboard")

    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(out_dir, zip_path, include_root=True, overwrite=True)
        print(f"Packaged ZIP: {zip_path}")


if __name__ == "__main__":
    main()
