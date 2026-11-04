"""Model „profilu użytkownika”: Isolation Forest uczony na własnej historii transakcji (schemat generic_tx).
Bez etykiet; próg = wysoki kwantyl wyników na danych treningowych."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import anomaly_score
from .schemas import GENERIC, to_frame


def train_profile(txs, out_dir="models", name="profile_isolation_forest", quantile=0.99, seed=42) -> dict:
    if len(txs) < 50:
        raise ValueError("Za mało transakcji do zbudowania profilu (minimum 50).")
    X = to_frame(GENERIC, txs)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", IsolationForest(n_estimators=200, random_state=seed))]).fit(X)
    thr = float(np.quantile(anomaly_score(pipe, X, "unsupervised"), quantile))
    meta = {"name": name, "kind": "unsupervised", "schema": GENERIC.name, "params": {"quantile": quantile},
            "seed": seed, "threshold": thr, "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": len(txs), "metrics": None, "fit_seconds": None, "predict_latency_ms": None}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / f"{name}.joblib")
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta
