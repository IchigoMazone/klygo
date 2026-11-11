"""Generate report figures and tables directly from extension evidence."""
from extension_common import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEN = ROOT / "report/generated"
IMG = ROOT / "report/images/klygo"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def table(name, header, rows, layout):
    text = "\\begin{tabular}{"+layout+"}\n\\toprule\n"+" & ".join(header)+" \\\\\n\\midrule\n"
    text += "\n".join(" & ".join(str(v) for v in row)+" \\\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    (GEN / name).write_text(text, encoding="utf-8")

def main():
    mining = load(OUT / "mining/summary.json")
    methods = mining["methods"]
    rows = []
    for key, val in methods.items():
        stats = val["stats"]
        rows.append([key.replace("_", "-"), stats["images"], stats["boxes"], len(stats["images_per_class"]),
                     f'{stats["imbalance_ratio"]:.1f}', f'{val["median_ms"]:.1f}'])
    table("mining_results.tex", ["Cách chọn","Ảnh","Box","Lớp","IR","Chọn (ms)"],rows,"lrrrrr")
    labels = sorted(mining["before"]["images_per_class"], key=mining["before"]["images_per_class"].get, reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(10,4.2))
    x = np.arange(len(labels))
    for offset, key in enumerate(["full", "random", "weighted", "diversity", "balance"]):
        stats = methods[key]["stats"]
        vals = [stats["images_per_class"].get(c,0)/stats["images"]*100 for c in labels]
        ax.bar(x+(offset-2)*0.16, vals, width=.16, label=key)
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_ylabel("Tỷ lệ ảnh chứa lớp (%)")
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "mining_distribution.pdf"); plt.close(fig)
    cluster = load(OUT / "mining/clusters12.json")
    matrix = np.array(cluster["feature_vectors"])
    _, _, vectors = np.linalg.svd(matrix-matrix.mean(0), full_matrices=False)
    projected = (matrix-matrix.mean(0)) @ vectors[:2].T
    fig, axes = plt.subplots(1,2,figsize=(10,3.7))
    cmap=plt.get_cmap("tab20",12)
    scatter=axes[0].scatter(projected[:,0],projected[:,1],c=cluster["labels"],cmap=cmap,vmin=-.5,vmax=11.5,s=13,alpha=.8)
    fig.colorbar(scatter,ax=axes[0],ticks=range(12),label="Cụm",fraction=.045,pad=.02)
    axes[0].set(xlabel="Thành phần chính 1",ylabel="Thành phần chính 2",title="Chiếu PCA để minh họa 12 cụm")
    axes[1].bar(range(12),[len(cluster["groups"].get(str(i),[])) for i in range(12)],color=[cmap(i) for i in range(12)])
    axes[1].set(xlabel="Chỉ số cụm",ylabel="Số ảnh",title="Kích thước các cụm")
    fig.tight_layout(); fig.savefig(IMG / "mining_clusters.pdf"); plt.close(fig)
    execution = {}
    for model in ["yolo","world"]:
        paths = [OUT / "execution_six_20261107" / f"{model}_{mode}.json" for mode in ["raw","raw_stream","native","native_backend_stream","managed","managed_stream"]]
        if not all(path.exists() for path in paths): continue
        records = [load(path) for path in paths]
        reference = records[0]["predictions"]
        rows=[]
        summary=[]
        for record in records:
            assert len(record["predictions"]) == len(reference) == 100
            mismatches=[]; max_box=0.; max_score=0.
            for index,(left,right) in enumerate(zip(reference,record["predictions"])):
                a=np.array(left).reshape(-1,6); b=np.array(right).reshape(-1,6)
                if a.shape != b.shape:
                    mismatches.append(index); continue
                if not len(a): continue
                # Use descending score and class for deterministic alignment.
                a=a[np.lexsort((a[:,0],-a[:,4],a[:,5]))]; b=b[np.lexsort((b[:,0],-b[:,4],b[:,5]))]
                max_box=max(max_box,float(np.max(np.abs(a[:,:4]-b[:,:4]))))
                max_score=max(max_score,float(np.max(np.abs(a[:,4]-b[:,4]))))
                if not np.array_equal(a[:,5],b[:,5]) or not np.allclose(a[:,:4],b[:,:4],atol=.01,rtol=0) or not np.allclose(a[:,4],b[:,4],atol=.001,rtol=0):
                    mismatches.append(index)
            med={key:float(np.median([r[key] for r in record["rows"]])) for key in record["rows"][0]}
            summary.append({"mode":record["mode"],**med,"mismatching_frames":mismatches,"max_box_error":max_box,"max_score_error":max_score})
            label=record["mode"].replace("_", " ")
            rows.append([label,f'{med["total_s"]:.2f}',f'{med["fps"]:.2f}',f'{med["first_s"]*1000:.1f}',f'{med["peak_rss_mib"]:.0f}',f'{med["peak_vram_mib"]:.0f}'])
        table(f"execution_{model}.tex",["Chế độ","Giây","Ảnh/s","Đầu (ms)","RAM","VRAM"],rows,"lrrrrr")
        execution[model]=summary
    if execution:
        save(OUT / "execution_six_20261107/summary.json",execution)
        paragraphs=[]
        for model, records in execution.items():
            raw=records[0]
            pairs=[]
            for row in records[2:]:
                change=(row["total_s"]/raw["total_s"]-1)*100
                pairs.append(row["mode"].replace("_"," ")+f' {change:+.1f}\\%')
            name="YOLOv8n" if model=="yolo" else "YOLO-World"
            paragraphs.append(name+": chênh lệch thời gian toàn tập so với Raw là "+", ".join(pairs)+". Dấu dương biểu thị chậm hơn, dấu âm biểu thị nhanh hơn trong phép đo này.")
            mismatches=sum(len(r["mismatching_frames"]) for r in records[1:])
            maxbox=max(r["max_box_error"] for r in records)
            maxscore=max(r["max_score_error"] for r in records)
            compared=(len(records)-1)*100
            paragraphs.append(f"Đối chiếu {name} có {mismatches}/{compared} cặp frame vượt tiêu chí tương đương: cùng số box, class và sai số tọa độ không quá 0,01 pixel, score không quá 0,001. Sai số lớn nhất theo cách ghép class/score là {maxbox:.6f} pixel và {maxscore:.6f} điểm score. Phép kiểm tra này dùng dự đoán framework làm tham chiếu, không gọi đó là ground truth.")
        (GEN / "execution_discussion.tex").write_text("\n\n".join(paragraphs)+"\n",encoding="utf-8")
        fig,axes=plt.subplots(1,2,figsize=(10,4.8))
        for ax,(name,rows) in zip(axes,execution.items()):
            ax.barh([row["mode"].replace("_"," ") for row in rows],[row["fps"] for row in rows],color="#087f8c")
            ax.set_title("YOLOv8n" if name=="yolo" else "YOLO-World")
            ax.set_xlabel("Throughput (ảnh/giây)")
            ax.tick_params(axis="y", labelsize=10)
        fig.tight_layout();fig.savefig(IMG / "execution_throughput.pdf");plt.close(fig)
    paths=[OUT / "training_canonical" / f"{method}_{seed}.json" for method in ["full","random","weighted","diversity"] for seed in [42,43,44]]
    if all(p.exists() for p in paths):
        rows=[];summary={}
        for method in ["full","random","weighted","diversity"]:
            records=[load(OUT / "training_canonical" / f"{method}_{seed}.json") for seed in [42,43,44]]
            values=np.array([v["AP"] for v in records])*100
            ap50=np.mean([v["AP50"] for v in records])*100
            seconds=np.mean([v["training_wall_s"] for v in records])
            summary[method]={"AP_mean":float(values.mean()),"AP_sd":float(values.std(ddof=1)),"AP50":float(ap50),"train_s":float(seconds)}
            rows.append([method,methods[method]["stats"]["images"],f'{values.mean():.2f} $\\pm$ {values.std(ddof=1):.2f}',f'{ap50:.2f}',f'{seconds:.1f}'])
        table("training_results.tex",["Tập train","Ảnh","AP (\\%)","AP50 (\\%)","Train (s)"],rows,"lrrrr")
        save(OUT / "training_canonical/summary.json", summary)
        fig,ax=plt.subplots(figsize=(7,3.7))
        ax.bar(list(summary),[v["AP_mean"] for v in summary.values()],yerr=[v["AP_sd"] for v in summary.values()],capsize=5,color="#087f8c")
        initial=load(OUT / "training_canonical/initial_42.json")["AP"]*100
        lines=[f"Checkpoint ban đầu đạt AP {initial:.2f}\\% trên cùng 100 ảnh test ở kích thước 320."]
        for method in ["weighted","diversity"]:
            delta=summary[method]["AP_mean"]-summary["random"]["AP_mean"]
            lines.append(f"{method.capitalize()} chênh {delta:+.2f} điểm phần trăm AP so với Random cùng 150 ảnh.")
        lines.append("Các chênh lệch là mô tả mẫu; ba seed không đủ để khẳng định khác biệt có ý nghĩa thống kê hoặc ưu thế trên các tập dữ liệu khác. Độ lệch chuẩn phản ánh biến thiên huấn luyện, không phải khoảng tin cậy do lấy mẫu ảnh test.")
        (GEN / "training_discussion.tex").write_text(" ".join(lines)+"\n",encoding="utf-8")
        ax.axhline(initial,color="#c45522",ls="--",label="Checkpoint trước fine-tune")
        ax.set_ylabel("AP@[0.50:0.95] (%)");ax.legend(fontsize=9)
        fig.tight_layout();fig.savefig(IMG / "mining_training.pdf");plt.close(fig)
    checks=OUT / "checks/summary.json"
    if checks.exists():
        val=load(checks)
        audit=load(OUT / "checks/audit.json")
        passed=val["run"]-len(val["failures"])-len(val["errors"])-len(val["skipped"])
        text=f"Kiểm thử revision mở rộng chạy {val['run']} ca: {passed} đạt, {len(val['skipped'])} bỏ qua, {len(val['failures'])} thất bại và {len(val['errors'])} lỗi. Audit trả mã {audit['returncode']}. Log đầy đủ nằm trong thư mục extension/checks; số liệu này tách biệt với 360 ca của revision nền tảng.\n"
        (GEN / "extension_checks.tex").write_text(text,encoding="utf-8")

if __name__ == "__main__": main()
