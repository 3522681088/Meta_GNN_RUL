import torch
from torch.nn import functional as F

def rul_training_loss(model,x,y,pair_aux_weight=0.0):
    """RUL MSE plus optional pairwise degradation-distance MSE inspired by MetaFluAD Eq. (7)."""
    if pair_aux_weight>0 and hasattr(model,"pairwise_predictor") and x.size(0)>1:
        pred,aux=model(x,return_attention=True); features=aux["features"]
        mate=torch.roll(torch.arange(x.size(0),device=x.device),1)
        pair_pred=model.pairwise_predictor(torch.cat([features,features[mate]],dim=-1)).squeeze(-1)
        pair_target=torch.abs(y-y[mate])
        return F.mse_loss(pred,y)+pair_aux_weight*F.mse_loss(pair_pred,pair_target),pred
    pred=model(x); return F.mse_loss(pred,y),pred
