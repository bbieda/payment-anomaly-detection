"""Wczytywanie danych i podział czasowy (Kaggle: Credit Card Fraud Detection, ULB)."""
from pathlib import Path

import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
TIME = "Time"


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET, TIME]) - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn w zbiorze: {sorted(missing)}")
    return df


def temporal_split(df: pd.DataFrame, train: float = 0.7, val: float = 0.15):
    """Podział wg czasu (bez tasowania): train < val < test.

    Losowy podział zawyża wyniki, bo transakcje z tego samego okresu trafiają
    do train i test. Tutaj test zawsze jest „przyszłością” względem train.
    """
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("Niepoprawne proporcje podziału")
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]
