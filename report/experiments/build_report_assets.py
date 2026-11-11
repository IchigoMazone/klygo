"""Generate report tables, numeric macros and figures exclusively from saved evidence."""
from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

from prepare_coco import ROOT, DATA, write_json

REPORT = ROOT / "report"
GENERATED = REPORT / "generated"
FIGURES = REPORT / "images/klygo"
EVIDENCE = ROOT / "klygo/artifacts/report"
NAMES = {"yolo": "YOLOv8n", "world": "YOLO-World v2 small", "faster": "Faster R-CNN MobileNetV3"}
SHORT = ["YOLOv8n", "YOLO-World", "Faster R-CNN"]
COLORS = ["#167d8d", "#dd8a27", "#6563a5"]
MODELS = list(NAMES)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 220})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def vn(value, digits=2):
    return f"{value:.{digits}f}".replace(".", "{,}")


def table(filename, columns, header, rows):
    body = "\\begin{tabular}{" + columns + "}\n\\toprule\n"
    body += " & ".join(header) + " \\\\\n\\midrule\n"
    body += "".join(" & ".join(map(str, row)) + " \\\\\n" for row in rows)
    body += "\\bottomrule\n\\end{tabular}\n"
    (GENERATED / filename).write_text(body, encoding="utf-8")


def save(name, fig):
    fig.tight_layout()
    fig.savefig(FIGURES / (name + ".png"), bbox_inches="tight")
    fig.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def build():
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = read(DATA / "manifest.json")
    gt = read(DATA / "instances_sample.json")
    metrics = {name: read(EVIDENCE / "main" / name / "metrics.json") for name in MODELS}
    timing = {name: read(EVIDENCE / "main" / name / "timing_controlled/timing_summary.json") for name in MODELS}
    diagnostics = {name: read(EVIDENCE / "main" / name / "diagnostics.json") for name in MODELS}
    assert all(not result["mismatches"] for result in timing.values())
    categories = {item["id"]: item["name"] for item in gt["categories"]}
    count = Counter(item["category_id"] for item in gt["annotations"] if not item["iscrowd"])
    crowds = sum(item["iscrowd"] for item in gt["annotations"])
    macros = {"AnnotationCount": str(len(gt["annotations"])), "ValidGtCount": str(sum(count.values())),
              "CrowdCount": str(crowds), "SupportedClasses": str(len(count))}
    for name, suffix in zip(MODELS, ("Yolo", "World", "Faster")):
        macros["AP" + suffix] = vn(100 * metrics[name]["AP"]) + r"\%"
        macros["AP" + suffix + "EN"] = f"{metrics[name]['AP']:.4f}"
        macros["Latency" + suffix] = vn(timing[name]["predict"]["mean_ms"])
    (GENERATED / "numbers.tex").write_text("\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()) + "\n", encoding="utf-8")
    table("quality.tex", "lrrr", ["Mô hình", "AP (\\%)", "AP50 (\\%)", "AP75 (\\%)"],
          [[SHORT[i]] + [vn(100 * metrics[name][key]) for key in ("AP", "AP50", "AP75")] for i, name in enumerate(MODELS)])
    table("checksums.tex", "lp{9cm}", ["Checkpoint", "SHA-256 (hai dòng liên tiếp)"],
          [[SHORT[i], r"\shortstack[l]{\texttt{" + read(EVIDENCE / "main" / name / "configuration.json")["checkpoint_sha256"][:32] + r"}\\\texttt{" + read(EVIDENCE / "main" / name / "configuration.json")["checkpoint_sha256"][32:] + "}}"] for i, name in enumerate(MODELS)])
    table("operating.tex", "lrrrrrr", ["Mô hình", "TP", "FP", "FN", "P", "R", "F1"],
          [[SHORT[i]] + [str(metrics[name]["operating_point"][key]) for key in ("tp", "fp", "fn")]
           + [vn(metrics[name]["operating_point"][key], 4) for key in ("precision", "recall", "f1")] for i, name in enumerate(MODELS)])
    table("latency.tex", "lrrrr", ["Mô hình", "TB (ms)", "Trung vị", "p90", "FPS tuần tự"],
          [[SHORT[i]] + [vn(timing[name]["predict"][key]) for key in ("mean_ms", "median_ms", "p90_ms", "sequential_fps")] for i, name in enumerate(MODELS)])
    table("overhead.tex", "lrrrr", ["Mô hình", "forward (ms)", "predict (ms)", "$\\Delta$ (ms)", "$\\Delta$ (\\%)"],
          [[SHORT[i], vn(timing[name]["forward"]["mean_ms"]), vn(timing[name]["predict"]["mean_ms"]),
            vn(timing[name]["paired_difference"]["mean_ms"]), vn(timing[name]["paired_difference"]["relative_mean_percent"])] for i, name in enumerate(MODELS)])
    table("repeat_latency.tex", "lrrr", ["Mô hình", "Lượt 1 (ms)", "Lượt 2 (ms)", "Lượt 3 (ms)"],
          [[SHORT[i]] + [vn(v) for v in timing[name]["predict"]["repeat_mean_ms"]] for i, name in enumerate(MODELS)])
    table("size_ap.tex", "lrrr", ["Mô hình", "AP small (\\%)", "AP medium (\\%)", "AP large (\\%)"],
          [[SHORT[i]] + [vn(100 * metrics[name]["coco_stats"][index]) for index in (3, 4, 5)] for i, name in enumerate(MODELS)])
    byclass = {name: {item["category_id"]: item for item in diagnostics[name]["classes"]} for name in MODELS}
    selected = sorted(count, key=lambda category: (-count[category], category))[:12]
    table("classes12.tex", "lrrrr", ["Lớp", "GT hợp lệ", "YOLOv8n", "YOLO-World", "Faster R-CNN"],
          [[categories[category], str(count[category])] + [vn(100 * byclass[name][category]["AP"]) for name in MODELS] for category in selected])
    class_rows = [[categories[category], str(count[category])] + [vn(100 * byclass[name][category]["AP"]) if byclass[name][category]["AP"] is not None else "--" for name in MODELS]
                  for category in sorted(categories)]
    all_class_table = r"\small\begin{longtable}{lrrrr}" + "\n" + r"\caption{AP@[0.50:0.95] theo lớp trên mẫu, tính bằng phần trăm; GT không gồm crowd}\label{tab:allclasses}\\" + "\n"
    all_class_table += r"\toprule Lớp & GT & YOLOv8n & YOLO-World & Faster R-CNN \\\midrule\endfirsthead" + "\n"
    all_class_table += r"\toprule Lớp & GT & YOLOv8n & YOLO-World & Faster R-CNN \\\midrule\endhead" + "\n"
    all_class_table += "".join(" & ".join(row) + " \\\\\n" for row in class_rows)
    all_class_table += r"\bottomrule\end{longtable}\normalsize" + "\n"
    (GENERATED / "classes_all.tex").write_text(all_class_table, encoding="utf-8")
    tests = read(EVIDENCE / "logs/unittest_summary.json")
    passed = Counter(item.split(".")[1] for item in tests["passed_ids"])
    skipped = Counter(item["test"].split(".")[1] for item in tests["skipped"])
    table("test_groups.tex", "lrrr", ["Nhóm", "Thực hiện", "Đạt", "Bỏ qua"],
          [[group.replace("_", r"\_"), passed[group] + skipped[group], passed[group], skipped[group]] for group in sorted(passed.keys() | skipped.keys())])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(3)
    for j, metric in enumerate(("AP", "AP50", "AP75")):
        ax.bar(x + (j - 1) * .24, [100 * metrics[n][metric] for n in MODELS], .24, label=metric)
    ax.set_xticks(x, SHORT); ax.set_ylabel("Điểm (%)"); ax.set_ylim(0, 65); ax.legend(ncol=3); ax.grid(axis="y", alpha=.18)
    save("quality", fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    for name, label, color in zip(MODELS, SHORT, COLORS):
        ax.plot(diagnostics[name]["recall_thresholds"], diagnostics[name]["macro_precision_at_iou50"], label=label, color=color, linewidth=2)
    ax.set(xlabel="Recall", ylabel="Precision nội suy trung bình theo lớp", xlim=(0, 1), ylim=(0, 1)); ax.legend(); ax.grid(alpha=.2)
    save("pr", fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [[row["predict_ms"] for row in read(EVIDENCE / "main" / name / "timing_controlled/paired_timing.json")] for name in MODELS]
    ax.boxplot(values, tick_labels=SHORT, showfliers=False); ax.set_ylabel("Độ trễ predict (ms)"); ax.grid(axis="y", alpha=.2)
    save("latency", fig)
    fig, ax = plt.subplots(figsize=(8, 3.7))
    for name, label, color in zip(MODELS, SHORT, COLORS):
        rows = read(EVIDENCE / "main" / name / "timing_controlled/paired_timing.json")
        delta = [row["predict_ms"] - row["forward_ms"] for row in rows]
        ax.hist(delta, bins=45, alpha=.45, label=label, color=color)
    ax.set(xlabel="Chênh lệch predict − forward trong từng cặp (ms)", ylabel="Số cặp"); ax.legend(); ax.axvline(0, color="gray", linewidth=1)
    save("overhead", fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for j, size in enumerate(("small", "medium", "large")):
        ax.bar(x + (j - 1) * .24, [100 * metrics[n]["coco_stats"][3 + j] for n in MODELS], .24, label=size)
    ax.set_xticks(x, SHORT); ax.set_ylabel("AP theo kích thước (%)"); ax.legend(ncol=3); ax.grid(axis="y", alpha=.2)
    save("size_ap", fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    top20 = sorted(count, key=lambda category: count[category], reverse=True)[:20][::-1]
    ax.barh([categories[c] for c in top20], [count[c] for c in top20], color=COLORS[0]); ax.set_xlabel("Số annotation không phải crowd"); ax.grid(axis="x", alpha=.2)
    save("distribution", fig)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for name, label, color in zip(MODELS, SHORT, COLORS):
        t, ap = timing[name]["predict"]["mean_ms"], 100 * metrics[name]["AP"]
        ax.scatter(t, ap, s=100, color=color); ax.annotate(label, (t, ap), xytext=(6, 7), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Độ trễ trung bình predict (ms)", ylabel="AP@[0.50:0.95] (%)", xlim=(15, 90), ylim=(15, 47)); ax.grid(alpha=.2)
    save("tradeoff", fig)
    case_metadata = make_cases(gt, diagnostics)
    write_json(GENERATED / "summary.json", {"metrics": metrics, "timing": timing,
        "dataset": {"annotations": len(gt["annotations"]), "crowd": crowds, "valid_gt": sum(count.values()), "supported_classes": len(count)},
        "cases": case_metadata})
    print("Generated numeric tables, seven plots, and three qualitative panels.")


def make_cases(gt, diagnostics):
    images = {item["id"]: item for item in gt["images"]}
    annotations = defaultdict(list)
    for item in gt["annotations"]:
        if not item["iscrowd"]:
            annotations[item["image_id"]].append(item)
    category_names = {item["id"]: item["name"] for item in gt["categories"]}
    stats = diagnostics["yolo"]["per_image_at_conf025"]
    eligible = [i for i in sorted(images) if stats[str(i)]["gt"] >= 3]
    score = lambda i: 2 * stats[str(i)]["tp"] / max(1, stats[str(i)]["gt"] + stats[str(i)]["tp"] + stats[str(i)]["fp"])
    good = max(eligible, key=lambda i: (score(i), -i))
    hard = max(eligible, key=lambda i: (stats[str(i)]["gt"] - stats[str(i)]["tp"] + stats[str(i)]["fp"], -i))
    small = max([i for i in eligible if i not in (good, hard)], key=lambda i: (sum(a["area"] < 32**2 for a in annotations[i]), -i))
    predictions = {}
    for name in MODELS:
        grouped = defaultdict(list)
        for item in read(EVIDENCE / "main" / name / "predictions.json"):
            if item["score"] >= .25:
                grouped[item["image_id"]].append(item)
        predictions[name] = grouped
    output, tex = [], []
    for index, (case, image_id) in enumerate((("good", good), ("hard", hard), ("small", small))):
        with Image.open(DATA / "images" / images[image_id]["file_name"]) as image:
            pixels = image.convert("RGB")
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, name in zip(axes.flat, ["gt"] + MODELS):
            boxes = annotations[image_id] if name == "gt" else predictions[name][image_id]
            ax.imshow(pixels); ax.axis("off")
            for item in sorted(boxes, key=lambda a: a.get("score", 1), reverse=True)[:20]:
                x, y, w, h = item["bbox"]
                color = "#22c55e" if name == "gt" else "#ffb320"
                ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))
            ax.set_title(("Ground truth (không gồm crowd)" if name == "gt" else NAMES[name]) + f" | {len(boxes)} box", fontsize=9)
        fig.suptitle(f"COCO val2017 | image_id={image_id} | tối đa 20 box được vẽ mỗi ô", fontsize=10)
        save("case_" + case, fig)
        criterion = {"good": "F1 cao nhất của YOLOv8n trong các ảnh có ít nhất ba GT hợp lệ",
                     "hard": "tổng FP và FN lớn nhất của YOLOv8n trong các ảnh có ít nhất ba GT hợp lệ",
                     "small": "số annotation small nhiều nhất, sau khi loại hai ảnh đã chọn"}[case]
        output.append({"case": case, "image_id": image_id, "criterion": criterion,
                       "stats": {name: diagnostics[name]["per_image_at_conf025"][str(image_id)] for name in MODELS}})
        tex.append(r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{klygo/case_" + case + ".png}\n" +
                   r"\caption[So sánh kết quả trên ảnh COCO ID " + str(image_id) + r"]{Ảnh COCO ID " + str(image_id) + "; tiêu chí chọn: " + criterion +
                   r". Dự đoán hiển thị tại confidence 0,25; chỉ vẽ tối đa 20 box và lược nhãn trên ảnh để tránh che khuất. Số box trong tiêu đề là số trước giới hạn hiển thị. Metric dùng toàn bộ dự đoán theo cấu hình.}" +
                   r"\label{fig:case-" + case + "}\n\\end{figure}\n")
    (GENERATED / "cases.tex").write_text("\n".join(tex), encoding="utf-8")
    return output


if __name__ == "__main__":
    build()
