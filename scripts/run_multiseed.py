import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--target",default="FD004"); p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4]); p.add_argument("--suite",default="baselines",choices=["baselines","ablation"]); a=p.parse_args()
    for seed in a.seeds: subprocess.run([sys.executable,"main.py","--suite",a.suite,"--target",a.target,"--seed",str(seed)],check=True)
    records=[]
    for seed in a.seeds:
        path=Path("outputs")/f"results_{a.target}_seed{seed}.json"
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    grouped={}
    for r in records:
        grouped.setdefault(r["experiment"],[]).append(r)
    summary=[]
    for name,items in grouped.items():
        row={"experiment":name,"n_seeds":len(items)}
        for metric in ["rmse","mae","r2","nasa_score"]:
            values=np.array([x[metric] for x in items],float); row[f"{metric}_mean"]=float(values.mean()); row[f"{metric}_std"]=float(values.std(ddof=1)) if len(values)>1 else 0.0
        summary.append(row)
    out=Path("outputs")/f"summary_{a.suite}_{a.target}.json"; out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
