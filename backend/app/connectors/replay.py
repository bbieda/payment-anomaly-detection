"""Odtwarzanie zbioru testowego Kaggle jako strumienia (realistyczniejsze niż symulacja)."""
from datetime import datetime, timedelta, timezone

from anomaly.data import FEATURES, load_raw, temporal_split

from .base import Transaction, TransactionSource


class ReplaySource(TransactionSource):
    name = "replay"

    def __init__(self, data_path: str):
        self.df = temporal_split(load_raw(data_path))[2].reset_index(drop=True)
        self._i = 0
        self._t0 = datetime(2013, 9, 1, tzinfo=timezone.utc)

    def fetch(self, limit: int = 1) -> list[Transaction]:
        out = []
        for _ in range(limit):
            r = self.df.iloc[self._i % len(self.df)]
            out.append(Transaction(id=f"replay-{self._i}", timestamp=self._t0 + timedelta(seconds=float(r["Time"])),
                                   amount=float(r["Amount"]), features={f: float(r[f]) for f in FEATURES},
                                   label=int(r["Class"])))
            self._i += 1
        return out
