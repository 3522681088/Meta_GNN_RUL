import numpy as np
import torch
from torch.utils.data import Dataset

def make_windows(df, sensors, window_size=50, stride=5, last_only=False):
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        x = g[sensors].to_numpy(np.float32); y = g["rul"].to_numpy(np.float32)
        if len(x) < window_size:
            pad = window_size - len(x)
            x = np.pad(x, ((pad, 0), (0, 0)), mode="edge")
            y = np.pad(y, (pad, 0), mode="edge")
        ends = [len(x)] if last_only else list(range(window_size, len(x) + 1, stride))
        if not ends or ends[-1] != len(x): ends.append(len(x))
        for end in ends:
            xs.append(x[end-window_size:end]); ys.append(y[end-1]); units.append(int(unit))
    return np.stack(xs), np.asarray(ys, np.float32), np.asarray(units)

class WindowDataset(Dataset):
    def __init__(self, x, y, units=None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.units = units
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

