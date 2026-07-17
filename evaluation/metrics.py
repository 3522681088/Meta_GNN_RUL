import numpy as np

def regression_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float); p = np.asarray(y_pred, dtype=float)
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    # NASA scoring: late predictions (d>0) receive the steeper exponential penalty.
    nasa = float(np.sum(np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "nasa_score": nasa}

