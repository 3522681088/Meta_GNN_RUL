import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import yaml
from baselines import build_model
from meta_learning import TaskSampler, reptile_meta_step, adapt_target
from train import prepare_experiment, train_supervised, evaluate

META_MODELS={"meta_gnn","reptile_lstm"}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def resolve_device(value):
    if value=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

def run_one(name,cfg,loaders=None,tag=None):
    seed_everything(cfg["seed"]); device=resolve_device(cfg["device"])
    if loaders is None: loaders=prepare_experiment(cfg)
    tasks,support,val,test,sensor_num,split_info=loaders
    model=build_model(name,sensor_num,cfg).to(device)
    if name in META_MODELS:
        sampler=TaskSampler(tasks,cfg["tasks_per_meta_batch"])
        for epoch in range(cfg["meta_epochs"]):
            model=reptile_meta_step(model,sampler.sample(),cfg["inner_steps"],cfg["inner_lr"],cfg["outer_lr"],device,cfg.get("pair_aux_weight",0.0))
            if (epoch+1)%5==0: print(f"meta_epoch={epoch+1:03d}/{cfg['meta_epochs']}")
        model=adapt_target(model,support,cfg["adapt_epochs"],cfg["inner_lr"],device,cfg.get("pair_aux_weight",0.0))
    else:
        model=train_supervised(model,support,val,cfg,device)
    label=tag or name
    metrics=evaluate(model,test,device); metrics.update({"model":name,"experiment":label,"target_domain":cfg["target_domain"],"seed":cfg["seed"],"graph_method":cfg.get("graph_method","cosine")})
    out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"config":cfg,"split":split_info,"metrics":metrics},out/f"{label}_{cfg['target_domain']}_seed{cfg['seed']}.pt")
    print(json.dumps(metrics,ensure_ascii=False,indent=2)); return metrics

def main():
    p = argparse.ArgumentParser(
        description="Meta-GNN-RUL for NASA C-MAPSS"
    )

    p.add_argument(
        "--config",
        default="configs/default.yaml"
    )

    p.add_argument(
        "--model",
        default=None
    )

    p.add_argument(
        "--suite",
        choices=["single", "baselines", "ablation"],
        default="single"
    )

    p.add_argument(
        "--target",
        choices=["FD001", "FD002", "FD003", "FD004"]
    )

    p.add_argument(
        "--seed",
        type=int
    )

    p.add_argument(
        "--support-ratio",
        type=float,
        choices=[0.05, 0.1, 0.2, 0.8],
        help="Target labeled-data ratio, matching MetaFluAD's data-volume experiments"
    )

    # 解析PyCharm脚本形参
    args = p.parse_args()

    # 读取YAML配置文件
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 如果脚本形参指定目标域，就覆盖YAML中的设置
    if args.target:
        cfg["target_domain"] = args.target
        cfg["source_domains"] = [
            domain
            for domain in ["FD001", "FD002", "FD003", "FD004"]
            if domain != args.target
        ]

    # 如果指定随机种子，就覆盖YAML中的随机种子
    if args.seed is not None:
        cfg["seed"] = args.seed

    # 如果指定目标域数据比例，就覆盖YAML中的设置
    if args.support_ratio is not None:
        cfg["target_support_ratio"] = args.support_ratio

    # 加载和处理C-MAPSS数据
    loaders = prepare_experiment(cfg)

    # 选择需要运行的实验
    if args.suite == "baselines":
        specs = [
            (name, {}, name)
            for name in [
                "lstm",
                "cnn_lstm",
                "transformer",
                "gnn",
                "reptile_lstm",
                "meta_gnn",
            ]
        ]

    elif args.suite == "ablation":
        specs = [
            ("meta_gnn", {}, "full_cosine"),
            ("no_gat", {}, "without_gat"),
            ("no_attention", {}, "without_self_attention"),
            ("gnn", {}, "without_reptile"),
            (
                "meta_gnn",
                {"graph_method": "dtw"},
                "full_dtw",
            ),
        ]

    else:
        model_name = args.model or cfg["model"]
        specs = [
            (model_name, {}, model_name)
        ]

    # 运行实验
    results = []

    for name, overrides, tag in specs:
        run_cfg = {
            **cfg,
            **overrides,
        }

        result = run_one(
            name,
            run_cfg,
            loaders,
            tag,
        )

        results.append(result)

    # 保存全部实验结果
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    result_path = (
        out
        / f"results_{cfg['target_domain']}_seed{cfg['seed']}.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()