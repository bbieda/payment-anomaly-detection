"""Import własnych transakcji z CSV.

Format A (Kaggle): kolumny Time, V1..V28, Amount, [Class].
Format B (ogólny): kolumny timestamp, amount, opcjonalnie merchant, category, currency, label.
"""
import io
from datetime import datetime, timedelta, timezone

import pandas as pd

from anomaly.data import FEATURES

from .base import Transaction

MAX_ROWS = 50_000


def parse_csv(content: bytes) -> list[Transaction]:
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=MAX_ROWS)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Nie udało się wczytać CSV: {e}")
    cols = {c.lower(): c for c in df.columns}
    out = []
    if all(f in df.columns for f in FEATURES):
        t0 = datetime(2013, 9, 1, tzinfo=timezone.utc)
        for i, r in enumerate(df.to_dict("records")):
            out.append(Transaction(id=f"csv-{i}", timestamp=t0 + timedelta(seconds=float(r.get("Time", i))),
                                   amount=float(r["Amount"]), features={f: float(r[f]) for f in FEATURES},
                                   label=int(r["Class"]) if "Class" in r else None))
        return out
    if "amount" not in cols or not ({"timestamp", "date"} & set(cols)):
        raise ValueError("CSV musi zawierać kolumny 'timestamp' (lub 'date') i 'amount' albo komplet cech Kaggle (V1..V28, Amount).")
    ts_col = cols.get("timestamp") or cols["date"]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    bad = int(df[ts_col].isna().sum())
    if bad == len(df):
        raise ValueError("Nie udało się odczytać żadnej daty w kolumnie czasu.")
    lab = cols.get("label") or cols.get("is_fraud")
    for i, r in enumerate(df.to_dict("records")):
        if pd.isna(r[ts_col]):
            continue
        out.append(Transaction(id=f"csv-{i}", timestamp=r[ts_col].to_pydatetime(), amount=float(r[cols["amount"]]),
                               currency=str(r.get(cols.get("currency", ""), "PLN") or "PLN"),
                               merchant=(str(r[cols["merchant"]]) if "merchant" in cols else None),
                               category=(str(r[cols["category"]]) if "category" in cols else None),
                               label=int(r[lab]) if lab else None))
    return out
