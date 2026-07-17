from .engine import train_supervised

def train_baseline(model,train_loader,val_loader,cfg,device):
    return train_supervised(model,train_loader,val_loader,cfg,device)
