"""Create tiny C-MAPSS-shaped files only for pipeline smoke testing."""
from pathlib import Path
import argparse, numpy as np

def rows(n_units, min_cycles, rng):
    result=[]
    for unit in range(1,n_units+1):
        total=min_cycles+int(rng.integers(0,15))
        for cycle in range(1,total+1):
            settings=rng.normal(0,0.1,3); degradation=cycle/total
            sensors=rng.normal(0,0.2,21)+degradation*np.linspace(0.1,1.0,21)
            result.append([unit,cycle,*settings,*sensors])
    return np.asarray(result)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="data"); a=p.parse_args(); root=Path(a.output); rng=np.random.default_rng(7)
    for d in ["FD001","FD002","FD003","FD004"]:
        folder=root/d; folder.mkdir(parents=True,exist_ok=True)
        np.savetxt(folder/f"train_{d}.txt",rows(12,55,rng),fmt="%.6f")
        np.savetxt(folder/f"test_{d}.txt",rows(6,50,rng),fmt="%.6f")
        np.savetxt(folder/f"RUL_{d}.txt",rng.integers(10,60,6),fmt="%d")
    print(f"Mock data written to {root.resolve()}")
if __name__=="__main__": main()

