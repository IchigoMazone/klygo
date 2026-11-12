"""Validate new scheduling evidence and generate standalone report charts/tables."""
from extension_common import *
from build_extension_assets import table, IMG, GEN
import numpy as np
import matplotlib.pyplot as plt

SIX = OUT / "execution_six_20261107"
STUDY = OUT / "scheduling_20261107"
CONFIGS=[(0,0),(1,1),(2,1),(4,1),(2,2),(2,4)]

def parity(reference, predictions):
    assert len(reference)==len(predictions)==100
    mismatches=[]
    for index,(left,right) in enumerate(zip(reference,predictions)):
        a=np.array(left).reshape(-1,6);b=np.array(right).reshape(-1,6)
        if a.shape!=b.shape:
            mismatches.append(index);continue
        if not len(a):continue
        a=a[np.lexsort((a[:,0],-a[:,4],a[:,5]))];b=b[np.lexsort((b[:,0],-b[:,4],b[:,5]))]
        if not np.array_equal(a[:,5],b[:,5]) or not np.allclose(a[:,:4],b[:,:4],atol=.01,rtol=0) or not np.allclose(a[:,4],b[:,4],atol=.001,rtol=0):
            mismatches.append(index)
    return mismatches

if __name__=="__main__":
    summary={};discussion=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,model in zip(axes,["yolo","world"]):
        reference=load(SIX / f"{model}_raw.json")
        rows=[];values=[]
        for workers,prefetch in CONFIGS:
            path=(SIX / f"{model}_managed_stream.json") if (workers,prefetch)==(1,1) else STUDY / f"{model}_managed_stream_w{workers}_p{prefetch}.json"
            record=load(path)
            assert record["input_files"]==reference["input_files"]
            assert record["checkpoint_sha256"]==reference["checkpoint_sha256"]
            assert record["revision"]==REVISION
            assert record["torch"]==reference["torch"] and record["gpu"]==reference["gpu"]
            assert record["confidence"]==reference["confidence"]==.25
            med={key:float(np.median([r[key] for r in record["rows"]])) for key in record["rows"][0]}
            med.update(workers=workers,prefetch=prefetch,mismatching_frames=parity(reference["predictions"],record["predictions"]))
            values.append(med)
            runs=[r["total_s"] for r in record["rows"]]
            med["min_total_s"],med["max_total_s"]=min(runs),max(runs)
            rows.append([workers,prefetch,f'{med["total_s"]:.2f}',f'{min(runs):.2f}--{max(runs):.2f}',
                f'{med["fps"]:.2f}',f'{med["first_s"]*1000:.1f}',f'{med["peak_rss_mib"]:.0f}'])
        summary[model]=values
        table(f"scheduling_{model}.tex",["W","P","Giây","Min--max (s)","Ảnh/s","Đầu (ms)","RAM"],rows,"rrrrrrr")
        times=np.array([v["total_s"] for v in values])
        errors=np.array([[v["total_s"]-v["min_total_s"] for v in values],
                         [v["max_total_s"]-v["total_s"] for v in values]])
        ax.bar(range(6),times,yerr=errors,capsize=3,color=["#64748b","#087f8c","#087f8c","#087f8c","#d18b30","#d18b30"])
        ax.set_xticks(range(6),[f"({w},{p})" for w,p in CONFIGS])
        ax.set(xlabel="(workers, prefetch)",ylabel="Thời gian 100 ảnh (giây)",title="YOLOv8n" if model=="yolo" else "YOLO-World")
        best=values[int(times.argmin())]
        base=values[1]
        serial=values[0]
        name="YOLOv8n" if model=="yolo" else "YOLO-World"
        discussion.append(f"{name}: trung vị nhỏ nhất trong các cấu hình đã đo là ({best['workers']},{best['prefetch']}) với {best['total_s']:.2f} s; mặc định (1,1) là {base['total_s']:.2f} s và tuần tự (0,0) là {serial['total_s']:.2f} s. Đây là mô tả phiên đo, không khẳng định cấu hình tối ưu trên mọi workload.")
        mismatches=sum(len(v["mismatching_frames"]) for v in values)
        discussion.append(f"Sáu cấu hình {name} có {mismatches}/600 cặp frame vượt tiêu chí tương đương so với Raw, với cùng ngưỡng sai số như benchmark sáu mode.")
    fig.tight_layout();fig.savefig(IMG / "scheduling_comparison.pdf");plt.close(fig)
    save(STUDY / "summary.json",summary)
    (GEN / "scheduling_discussion.tex").write_text("\n\n".join(discussion)+"\n",encoding="utf-8")
