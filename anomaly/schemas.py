"""Schematy cech. Model jest związany ze schematem, na którym był trenowany.

kaggle_pca  -- V1..V28 (PCA) + Amount; działa tylko dla danych w tym formacie (zbiór Kaggle).
generic_tx  -- cechy wyprowadzane z surowej transakcji (kwota, czas); pozwala oceniać
               DOWOLNE transakcje (własny plik CSV, w przyszłości API banku).
Prawdziwa transakcja bankowa nie ma cech V1..V28, więc modele Kaggle nie mogą jej ocenić.
"""
import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .data import FEATURES


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    features: list


KAGGLE = FeatureSchema("kaggle_pca", FEATURES)
GENERIC = FeatureSchema("generic_tx", ["log_amount", "hour_sin", "hour_cos", "dow_sin", "dow_cos"])
SCHEMAS = {s.name: s for s in (KAGGLE, GENERIC)}


class SchemaMismatch(ValueError):
    pass


def generic_features(amount: float, ts: datetime) -> dict:
    h = ts.hour + ts.minute / 60
    d = ts.weekday()
    return {"log_amount": math.log1p(abs(amount)),
            "hour_sin": math.sin(2 * math.pi * h / 24), "hour_cos": math.cos(2 * math.pi * h / 24),
            "dow_sin": math.sin(2 * math.pi * d / 7), "dow_cos": math.cos(2 * math.pi * d / 7)}


def to_frame(schema: FeatureSchema, txs) -> pd.DataFrame:
    """txs: obiekty z polami amount, timestamp, features (opcjonalnie)."""
    rows = []
    for t in txs:
        if schema.name == "kaggle_pca":
            f = dict(t.features or {})
            f.setdefault("Amount", t.amount)
            missing = [c for c in schema.features if c not in f]
            if missing:
                raise SchemaMismatch(
                    f"Transakcje nie mają cech schematu '{schema.name}' (brak m.in. {missing[:3]}). "
                    "Wybierz model ze schematem 'generic_tx' lub dostarcz dane w formacie Kaggle.")
            rows.append({c: f[c] for c in schema.features})
        else:
            rows.append(generic_features(t.amount, t.timestamp))
    return pd.DataFrame(rows, columns=schema.features)
