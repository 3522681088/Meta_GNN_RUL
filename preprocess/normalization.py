import numpy as np

class FeatureNormalizer:
    def __init__(self, method="zscore"):
        self.method = method

    def fit(self, df, columns):
        x = df[columns].to_numpy(dtype=np.float32)
        if self.method == "minmax":
            self.offset = x.min(0); self.scale = x.max(0) - self.offset
        else:
            self.offset = x.mean(0); self.scale = x.std(0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, df, columns):
        out = df.copy()

        # 先计算归一化结果
        normalized = (
            out[columns].to_numpy(dtype=np.float32)
            - self.offset
        ) / self.scale

        # 逐列替换，允许整数列安全转换为浮点数列
        for index, column in enumerate(columns):
            out[column] = normalized[:, index].astype(np.float32)

        return out
