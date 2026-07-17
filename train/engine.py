from copy import deepcopy
import numpy as np
import torch
from torch import nn
from evaluation.metrics import regression_metrics
from .losses import rul_training_loss

def predict(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)); ys.extend(y.numpy().tolist()); ps.extend(p.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(ps)

def evaluate(model, loader, device):
    y, p = predict(model, loader, device)
    return regression_metrics(y, p)

def train_supervised(model, train_loader, val_loader, cfg, device):
    model = model.to(device); optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_state, best_rmse, patience = None, float("inf"), 0
    for epoch in range(cfg["epochs"]):
        model.train(); losses=[]
        for x, y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); loss,_=rul_training_loss(model,x,y,cfg.get("pair_aux_weight",0.0))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); losses.append(loss.item())
        score=evaluate(model,val_loader,device)["rmse"]
        print(f"epoch={epoch+1:03d} train_mse={np.mean(losses):.4f} val_rmse={score:.4f}")
        if score < best_rmse:
            best_rmse=score; best_state=deepcopy(model.state_dict()); patience=0
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]: break
    if best_state is not None: model.load_state_dict(best_state)
    return model
