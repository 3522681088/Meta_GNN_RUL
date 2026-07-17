from pathlib import Path
import pandas as pd

COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 26:
        raise ValueError(f"{path} has {df.shape[1]} columns; expected at least 26")
    df = df.iloc[:, :26]
    df.columns = COLUMNS
    return df

def load_domain(data_dir: str, domain: str):
    root = Path(data_dir)
    candidates = [root / domain, root]
    folder = next((p for p in candidates if (p / f"train_{domain}.txt").exists()), candidates[0])
    train = _read(folder / f"train_{domain}.txt")
    test = _read(folder / f"test_{domain}.txt")
    rul_path = folder / f"RUL_{domain}.txt"
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy() if rul_path.exists() else None
    return train, test, test_rul

