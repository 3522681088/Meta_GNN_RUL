from copy import deepcopy
import torch
from torch import nn

def _loss(model,x,y,pair_aux_weight):
    if pair_aux_weight>0 and hasattr(model,"pairwise_predictor") and x.size(0)>1:
        pred,aux=model(x,return_attention=True); f=aux["features"]; mate=torch.roll(torch.arange(x.size(0),device=x.device),1)
        pair=model.pairwise_predictor(torch.cat([f,f[mate]],-1)).squeeze(-1)
        return nn.functional.mse_loss(pred,y)+pair_aux_weight*nn.functional.mse_loss(pair,torch.abs(y-y[mate]))
    return nn.functional.mse_loss(model(x),y)

def inner_adapt(model, loader, steps=5, lr=1e-3, device="cpu",pair_aux_weight=0.0):
    learner = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    iterator = iter(loader); learner.train()
    for _ in range(steps):
        try: x, y = next(iterator)
        except StopIteration: iterator = iter(loader); x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); loss = _loss(learner,x,y,pair_aux_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0); optimizer.step()
    return learner

def reptile_meta_step(meta_model, tasks, inner_steps, inner_lr, outer_lr, device,pair_aux_weight=0.0):
    adapted = [inner_adapt(meta_model, loader, inner_steps, inner_lr, device,pair_aux_weight) for _, loader in tasks]
    with torch.no_grad():
        for params in zip(meta_model.parameters(), *[m.parameters() for m in adapted]):
            meta_p, task_ps = params[0], params[1:]
            target = torch.stack([p.data for p in task_ps]).mean(0)
            meta_p.data.add_(outer_lr * (target - meta_p.data))
    return meta_model

def adapt_target(model, loader, epochs, lr, device,pair_aux_weight=0.0):
    learner = deepcopy(model).to(device); optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    learner.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad()
            loss = _loss(learner,x,y,pair_aux_weight); loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 5.0); optimizer.step()
    return learner
