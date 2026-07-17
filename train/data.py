import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from preprocess import load_domain, add_train_rul, add_test_rul, FeatureNormalizer, WindowDataset, make_windows

def split_units(df, val_fraction, seed):
    rng=np.random.default_rng(seed); units=np.array(sorted(df.unit.unique())); rng.shuffle(units)
    n=max(1,int(round(len(units)*val_fraction))); return units[n:],units[:n]

def loader_from_df(df, sensors, cfg, shuffle, last_only=False):
    x,y,u=make_windows(df,sensors,cfg["window_size"],cfg["window_stride"],last_only)
    return DataLoader(WindowDataset(x,y,u),batch_size=cfg["batch_size"],shuffle=shuffle,drop_last=False)

def prepare_experiment(cfg):
    domains=list(dict.fromkeys(cfg["source_domains"]+[cfg["target_domain"]])); raw={}
    for d in domains:
        tr,te,final=load_domain(cfg["data_dir"],d); raw[d]=(add_train_rul(tr,cfg["rul_cap"]),add_test_rul(te,final,cfg["rul_cap"]))
    sensors=cfg["sensor_columns"]
    source_fit=pd.concat([raw[d][0] for d in cfg["source_domains"]],ignore_index=True)
    norm=FeatureNormalizer(cfg["normalization"]).fit(source_fit,sensors)
    normalized={d:(norm.transform(tr,sensors),norm.transform(te,sensors)) for d,(tr,te) in raw.items()}
    task_loaders={}
    for d in cfg["source_domains"]:
        units,_=split_units(normalized[d][0],cfg["validation_fraction"],cfg["seed"])
        task_loaders[d]=loader_from_df(normalized[d][0].query("unit in @units"),sensors,cfg,True)
    target_train,target_test=normalized[cfg["target_domain"]]
    units=np.array(sorted(target_train.unit.unique())); rng=np.random.default_rng(cfg["seed"]); rng.shuffle(units)
    ratio=cfg.get("target_support_ratio")
    requested=max(2,int(round(len(units)*ratio))) if ratio is not None else cfg["target_support_units"]
    n=min(requested,len(units)); labeled_units=units[:n]
    if len(labeled_units)>1:
        nval=max(1,int(round(len(labeled_units)*cfg["validation_fraction"])))
        val_units=labeled_units[:nval]; support_units=labeled_units[nval:]
    else: support_units=val_units=labeled_units
    support=loader_from_df(target_train.query("unit in @support_units"),sensors,cfg,True)
    val=loader_from_df(target_train.query("unit in @val_units"),sensors,cfg,False)
    test=loader_from_df(target_test,sensors,cfg,False,last_only=True)
    return task_loaders,support,val,test,len(sensors),{"labeled_target_units":labeled_units.tolist(),"adaptation_units":support_units.tolist(),"validation_units":val_units.tolist()}
