"""Dane syntetyczne o schemacie jak Kaggle -- do testów i szybkiego smoke-testu.

python -m anomaly.synthetic data/raw/synthetic.csv
"""
import sys

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET, TIME


def make_synthetic(n: int = 6000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 28))
    y = (rng.random(n) < fraud_rate).astype(int)
    X[y == 1, :6] += rng.normal(2.5, 1.0, size=(int(y.sum()), 6))  # oszustwa przesunięte w kilku cechach
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.lognormal(3, 1, size=n)
    df[TIME] = np.sort(rng.uniform(0, 172_800, size=n))
    df[TARGET] = y
    return df[[TIME] + FEATURES + [TARGET]]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/synthetic.csv"
    make_synthetic(20000, 0.01).to_csv(out, index=False)
    print("zapisano", out)
