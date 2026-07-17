from meta_learning import TaskSampler,reptile_meta_step,adapt_target

def train_meta(model,task_loaders,target_support,cfg,device):
    sampler=TaskSampler(task_loaders,cfg["tasks_per_meta_batch"])
    for _ in range(cfg["meta_epochs"]):
        model=reptile_meta_step(model,sampler.sample(),cfg["inner_steps"],cfg["inner_lr"],cfg["outer_lr"],device,cfg.get("pair_aux_weight",0.0))
    return adapt_target(model,target_support,cfg["adapt_epochs"],cfg["inner_lr"],device,cfg.get("pair_aux_weight",0.0))
