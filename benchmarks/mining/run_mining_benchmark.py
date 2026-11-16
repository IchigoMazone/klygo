"""Standalone full-class curation accuracy benchmark runner for klygo.mining.

Evaluates ALL core `klygo.mining` selection, deduplication, clustering, and
balancing methods across ALL classes of a dataset (`coco128`, `voc2007`,
`val2017`, or custom `--dataset-dir`) without timing/speed benchmarks:
    1. `full_pool`           : 100% of training pool (upper-bound reference)
    2. `random`              : Random uniform subset of size `budget`
    3. `quality`             : `mining.select(method="quality")`
    4. `rare_first`          : `mining.select(method="rare_first")`
    5. `weighted`            : `mining.select(method="weighted")`
    6. `diversity`           : `mining.cluster()` + `mining.select(method="diversity")`
    7. `coverage`            : `mining.select(method="coverage")`
    8. `balance_select`      : `mining.balance(strategy="select")`
    9. `balance_undersample` : `mining.balance(strategy="undersample")`
   10. `dedup_weighted`      : `mining.deduplicate(keep="best")` + `mining.select(method="weighted")`

Outputs the standardized benchmark file set:
    - `environment.json`
    - `curation_leaderboard.json`
    - `curation_runs.json`
    - `summary.json`
    - `REPORT.md`
    - `charts/mining_map_comparison.png`
    - `charts/mining_f1_precision_recall.png`

Usage:
    PYTHONPATH=. python benchmarks/mining/run_mining_benchmark.py --dataset-dir /path/to/yolo --output artifacts/benchmarks/mining
"""

from __future__ import annotations

import argparse
import random
import tempfile
from typing import Any

import numpy as np

from benchmarks.gpu_suite.evaluate_release import _md_table, collect_environment
from klygo import archive, datasets, files, mining, outputs, utils


def _split_train_val(dets: outputs.Detections, val_ratio: float = 0.2, seed: int = 42) -> tuple[outputs.Detections, outputs.Detections]:
    frames = list(dets)
    rng = random.Random(seed)
    indices = list(range(len(frames)))
    rng.shuffle(indices)
    val_count = max(8, int(len(frames) * val_ratio))
    val_set = set(indices[:val_count])
    train_frames = [frames[i] for i in range(len(frames)) if i not in val_set]
    val_frames = [frames[i] for i in range(len(frames)) if i in val_set]
    return (
        outputs.Detections(train_frames, names=dets.names),
        outputs.Detections(val_frames, names=dets.names),
    )


def _build_all_methods(pool: outputs.Detections, budget: int, seed: int) -> dict[str, outputs.Detections]:
    rng = random.Random(seed)
    frames = list(pool)
    rand_indices = rng.sample(range(len(frames)), k=min(budget, len(frames)))
    random_sub = outputs.Detections([frames[i] for i in sorted(rand_indices)], names=pool.names)

    q_scores = mining.score(pool, weights={"quality": 1.0, "rarity": 0.0, "coverage": 0.0, "boxes": 0.0})
    b_scores = mining.score(pool, weights={"quality": 0.35, "rarity": 0.35, "coverage": 0.20, "boxes": 0.10})
    n_clusters = min(len(pool), max(2, budget))
    clusters = mining.cluster(pool, clusters=n_clusters, seed=seed)
    dedup = mining.deduplicate(pool, keep="best", threshold=0.95)
    dedup_scores = mining.score(dedup, weights={"quality": 0.35, "rarity": 0.35, "coverage": 0.20, "boxes": 0.10})

    num_classes = max(1, len(pool.names))
    per_cls = max(1, budget // num_classes)

    bal_select = mining.balance(pool, target="median", strategy="select", seed=seed)
    if len(bal_select) > budget:
        bal_select = mining.select(bal_select, method="weighted", limit=budget)

    bal_under = mining.balance(pool, target="median", strategy="undersample", seed=seed)
    if len(bal_under) > budget:
        bal_under = mining.select(bal_under, method="weighted", limit=budget)

    return {
        "full_pool": pool,
        "random": random_sub,
        "quality": mining.select(pool, method="quality", threshold=0.0, limit=budget, scores=q_scores),
        "rare_first": mining.select(pool, method="rare_first", limit=budget, scores=b_scores),
        "weighted": mining.select(pool, method="weighted", limit=budget, scores=b_scores),
        "diversity": mining.select(pool, method="diversity", limit=budget, cluster_result=clusters, per_cluster=1, seed=seed),
        "coverage": mining.select(pool, method="coverage", per_class=per_cls, limit=budget),
        "balance_select": bal_select,
        "balance_undersample": bal_under,
        "dedup_weighted": mining.select(dedup, method="weighted", limit=min(budget, len(dedup)), scores=dedup_scores),
    }


def _train_and_eval_subset(
    train_sub: outputs.Detections,
    val_sub: outputs.Detections,
    *,
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
        val_res = model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, device=device, verbose=False, plots=False)
        box_m = val_res.box
        p_val = float(box_m.mp)
        r_val = float(box_m.mr)
        f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
        return {
            "mAP50_95": round(float(box_m.map), 4),
            "mAP50": round(float(box_m.map50), 4),
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
    width = 0.35

    # 1. mAP@50 and mAP@50-95 by Curation Method
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width / 2, [float(r["mAP50"]) for r in leaderboard], width, label="mAP@50")
    ax.bar(x + width / 2, [float(r["mAP50_95"]) for r in leaderboard], width, label="mAP@50-95")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Curation Method Accuracy Leaderboard (mAP@50 & mAP@50-95)")
    ax.legend()
    fig.tight_layout()
    p1 = charts_dir / "mining_map_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    created.append(("Curation Method mAP Comparison", "charts/mining_map_comparison.png"))

    # 2. F1, Precision, Recall by Curation Method
    width3 = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - width3, [float(r["precision"]) for r in leaderboard], width3, label="Precision")
    ax.bar(x, [float(r["recall"]) for r in leaderboard], width3, label="Recall")
    ax.bar(x + width3, [float(r["f1"]) for r in leaderboard], width3, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall & F1 by Curation Method")
    ax.legend()
    fig.tight_layout()
    p2 = charts_dir / "mining_f1_precision_recall.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    created.append(("Precision, Recall & F1 by Curation Method", "charts/mining_f1_precision_recall.png"))

    return created


def _render_mining_report(summary: dict[str, Any], charts: list[tuple[str, str]]) -> str:
    env = summary["environment"]
    git_info = env.get("git", {})
    ds = summary.get("dataset", {})
    lines = [
        "# Klygo Mining Curation Accuracy Report (Full-Class Leaderboard)",
        "",
        f"- **GPU**: `{env.get('gpu', 'none')}` ({env.get('vram_mb', 0.0)} MB VRAM)",
        f"- **Dataset**: `{ds.get('source')}` (`{ds.get('total_classes')}` classes, train pool `{ds.get('train_pool_images')}` imgs, clean val `{ds.get('val_images')}` imgs, budget `{ds.get('budget')}` imgs)",
        f"- **Model / Epochs**: `{summary.get('config', {}).get('weights')}` / `{summary.get('config', {}).get('epochs')}` epochs",
        f"- **Git**: `{git_info.get('branch', 'unknown')}` @ `{git_info.get('commit', 'unknown')}`",
        "",
    ]
    lines.extend(_md_table(summary.get("curation_leaderboard", []), "Curation Method Leaderboard (Sorted by mAP@50-95)"))
    lines.extend(_md_table(summary.get("curation_runs", []), "Detailed Per-Method Evaluation Runs"))
    if charts:
        lines.extend(["## Charts", ""])
        for title, rel in charts:
            lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Flat YOLO dataset directory (e.g. coco128, voc2007, val2017)")
    parser.add_argument("--output", default="artifacts/benchmarks/mining_curation")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--budget-ratio", type=float, default=0.35, help="Fraction of train pool to select per curation method")
    parser.add_argument("--budget", type=int, default=0, help="Explicit image budget override (0 uses budget-ratio)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--zip", default="", help="Optional output .zip path")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    root = files.path(args.dataset_dir)
    img_sub = "images/train2017" if files.is_dir(root / "images" / "train2017") else "images"
    lbl_sub = "labels/train2017" if files.is_dir(root / "labels" / "train2017") else "labels"
    all_dets = datasets.load(root, images_dir=img_sub, labels_dir=lbl_sub, yaml_name="data.yaml", verbose=False)

    train_pool, val_clean = _split_train_val(all_dets, val_ratio=0.2, seed=args.seed)
    budget = args.budget if args.budget > 0 else max(16, int(len(train_pool) * args.budget_ratio))

    env = collect_environment()
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)

    subsets = _build_all_methods(train_pool, budget=budget, seed=args.seed)
    runs: list[dict[str, Any]] = []

    for method_name, sub in subsets.items():
        st = mining.stats(sub, quality=False, cooccurrence=False)
        metrics_res = _train_and_eval_subset(
            sub,
            val_clean,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            device=args.device,
        )
        runs.append({
            "method": method_name,
            "images_selected": len(sub),
            "boxes_selected": int(st.boxes),
            "classes_covered": int(st.classes),
            **metrics_res,
        })

    leaderboard = sorted(
        runs,
        key=lambda r: (r["mAP50_95"], r["mAP50"], r["f1"]),
        reverse=True,
    )
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank

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
        "curation_leaderboard": leaderboard,
        "curation_runs": runs,
    }

    files.save(out_dir / "curation_leaderboard.json", leaderboard, overwrite=True, verbose=False)
    files.save(out_dir / "curation_runs.json", runs, overwrite=True, verbose=False)
    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)
    charts = _generate_mining_charts(leaderboard, out_dir)
    files.save(out_dir / "REPORT.md", _render_mining_report(summary, charts), overwrite=True, verbose=False)
    utils.tabulate(leaderboard, title="Mining Curation Accuracy Leaderboard")

    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(out_dir, zip_path, include_root=True, overwrite=True)
        print(f"Packaged ZIP: {zip_path}")


if __name__ == "__main__":
    main()
