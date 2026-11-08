"""Odtwarzanie zbioru testowego Kaggle jako strumienia.

fraud_rate > 0: tryb demo -- strumień wzbogacony o prawdziwe oszustwa z zbioru testowego
(naturalnie jest ich 0,12%, więc na żywo prawie nigdy nie byłoby alertu).
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from anomaly.data import FEATURES, TARGET, load_raw, temporal_split

from .base import Transaction, TransactionSource


class ReplaySource(TransactionSource):
    name = "replay"

    def __init__(self, data_path: str, fraud_rate: float = 0.0, seed: int = 0):
        df = temporal_split(load_raw(data_path))[2].reset_index(drop=True)
        self.fraud_rate = fraud_rate
        self._fraud = df[df[TARGET] == 1]
        self._normal = df[df[TARGET] == 0] if fraud_rate > 0 else df
        self._rng = np.random.default_rng(seed)
        self._i = self._n = 0
        self._t0 = datetime(2013, 9, 1, tzinfo=timezone.utc)

    def fetch(self, limit: int = 1) -> list[Transaction]:
        out = []
        for _ in range(limit):
            if self.fraud_rate > 0 and len(self._fraud) and self._rng.random() < self.fraud_rate:
                r = self._fraud.iloc[int(self._rng.integers(len(self._fraud)))]
            else:
                r = self._normal.iloc[self._i % len(self._normal)]
                self._i += 1
            out.append(Transaction(id=f"replay-{self._n}", timestamp=self._t0 + timedelta(seconds=float(r["Time"])),
                                   amount=float(r["Amount"]), features={f: float(r[f]) for f in FEATURES},
                                   label=int(r[TARGET])))
            self._n += 1
        return out
