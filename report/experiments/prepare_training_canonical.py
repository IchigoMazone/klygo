"""Align independently exported YOLO class IDs to one fixed 80-class vocabulary."""
from extension_common import *
import shutil
import hashlib
import yaml
from collections import Counter

if __name__ == "__main__":
    categories=sorted(load(DATA / "instances_sample.json")["categories"],key=lambda x:x["id"])
    names={i:c["name"] for i,c in enumerate(categories)}
    canonical={v:k for k,v in names.items()}
    summary={}
    for method in ["full","random","weighted","diversity","val","test"]:
        source=OUT / "mining/datasets" / method
        dest=OUT / "mining/training_datasets" / method
        (dest / "labels").mkdir(parents=True,exist_ok=True)
        (dest / "images").mkdir(exist_ok=True)
        local=yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))["names"]
        before=Counter(); after=Counter()
        for label in sorted((source / "labels").glob("*.txt")):
            rows=[]
            for line in label.read_text().splitlines():
                if not line.strip(): continue
                parts=line.split()
                name=local[int(parts[0])]
                before[name]+=1
                parts[0]=str(canonical[name])
                rows.append(" ".join(parts))
                after[names[int(parts[0])]]+=1
            (dest / "labels" / label.name).write_text("\n".join(rows)+("\n" if rows else ""),encoding="utf-8")
        for image in (source / "images").glob("*"):
            target=dest / "images" / image.name
            if not target.exists(): shutil.copyfile(image,target)
        assert before==after
        (dest / "data.yaml").write_text(yaml.safe_dump({"names":names},allow_unicode=True),encoding="utf-8")
        summary[method]={"images":len(list((dest / "images").glob("*"))),"boxes":sum(after.values()),
                         "boxes_per_class":dict(after),"source_registry_size":len(local),"canonical_registry_size":80,
                         "name_counts_preserved":True}
    save(OUT / "mining/canonical_validation.json",summary)
    print(summary)
