"""Symulator transakcji (sandbox). Dla schematu kaggle_pca generuje dane o formacie Kaggle,
dla generic_tx -- „zwykłe” transakcje z rzadkimi anomaliami (duża kwota w nocy)."""
from datetime import datetime, timedelta, timezone

import numpy as np

from anomaly.data import FEATURES
from anomaly.synthetic import make_synthetic

from .base import Transaction, TransactionSource

MERCHANTS = ["Grocery", "Cafe", "Transport", "Online shop", "Pharmacy", "Restaurant"]


class SimulatedSource(TransactionSource):
    name = "simulated"

    def __init__(self, schema: str = "generic_tx", fraud_rate: float = 0.02, seed: int = 0):
        self.schema, self.fraud_rate, self.seed, self._i = schema, fraud_rate, seed, 0
        self._t0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)

    def fetch(self, limit: int = 1) -> list[Transaction]:
        rng = np.random.default_rng(self.seed + self._i)
        out = []
        if self.schema == "kaggle_pca":
            df = make_synthetic(limit, self.fraud_rate, seed=self.seed + self._i)
            for k, r in enumerate(df.itertuples(index=False)):
                d = r._asdict()
                out.append(Transaction(id=f"sim-{self._i + k}", timestamp=self._t0 + timedelta(seconds=float(d["Time"])),
                                       amount=float(d["Amount"]), features={f: float(d[f]) for f in FEATURES},
                                       label=int(d["Class"])))
        else:
            for k in range(limit):
                fraud = rng.random() < self.fraud_rate
                hour = int(rng.integers(0, 6)) if fraud else int(rng.integers(8, 22))
                amount = float(rng.lognormal(6.5, 0.5)) if fraud else float(rng.lognormal(3.2, 0.8))
                ts = self._t0 + timedelta(days=int(rng.integers(0, 60))) + timedelta(hours=hour - 8, minutes=int(rng.integers(0, 60)))
                out.append(Transaction(id=f"sim-{self._i + k}", timestamp=ts, amount=round(amount, 2),
                                       merchant=str(rng.choice(MERCHANTS)), label=int(fraud)))
        self._i += limit
        return out
