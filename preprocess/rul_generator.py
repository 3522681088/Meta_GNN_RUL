import numpy as np

def add_train_rul(df, cap=125):
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=cap).astype("float32")
    return out

def add_test_rul(df, final_rul, cap=125):
    if final_rul is None:
        raise ValueError("RUL_FDxxx.txt is required for test evaluation")
    out = df.copy()
    units = np.sort(out["unit"].unique())
    if len(units) != len(final_rul):
        raise ValueError("Number of test engines and final RUL labels do not match")
    final = dict(zip(units, final_rul))
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = [min(cap, (max_cycle.iloc[i] - row.cycle) + final[row.unit]) for i, row in out.iterrows()]
    out["rul"] = out["rul"].astype("float32")
    return out

