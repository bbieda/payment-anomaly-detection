"""Jedno miejsce, w którym transakcje są oceniane i zapisywane -- używane przez /predict, sandbox i stream."""
import time

from anomaly.models import anomaly_score
from anomaly.schemas import SCHEMAS, SchemaMismatch, to_frame

from ..connectors.base import Transaction
from ..db import Prediction


def resolve_model(store, name: str | None) -> str:
    name = name or store.active
    if not name:
        raise LookupError("Brak wytrenowanych modeli (uruchom: python -m anomaly.train).")
    if name not in store.names():
        raise KeyError(name)
    return name


def score_transactions(store, name: str, txs: list[Transaction]) -> list[dict]:
    pipe, meta = store.get(name)
    schema = SCHEMAS[meta.get("schema", "kaggle_pca")]
    X = to_frame(schema, txs)                      # może rzucić SchemaMismatch
    t0 = time.perf_counter()
    scores = anomaly_score(pipe, X, meta["kind"])
    per_tx_ms = (time.perf_counter() - t0) * 1000 / max(len(txs), 1)
    thr = meta["threshold"]
    return [{"id": t.id, "timestamp": t.timestamp.isoformat(), "amount": t.amount, "merchant": t.merchant,
             "label": t.label, "model": name, "score": float(s), "threshold": thr,
             "is_anomaly": bool(s >= thr), "latency_ms": per_tx_ms} for t, s in zip(txs, scores)]


def save_results(db, source: str, results: list[dict]):
    from datetime import datetime
    db.add_all([Prediction(source=source, tx_id=r["id"], tx_time=datetime.fromisoformat(r["timestamp"]),
                           amount=r["amount"], merchant=r["merchant"], label=r["label"], model_name=r["model"],
                           score=r["score"], is_anomaly=r["is_anomaly"], latency_ms=r["latency_ms"])
                for r in results])
    db.commit()


def summarize(results: list[dict], max_rows: int = 500) -> dict:
    from anomaly.metrics import evaluate
    n_flag = sum(r["is_anomaly"] for r in results)
    out = {"model": results[0]["model"] if results else None, "n": len(results), "n_flagged": n_flag,
           "flag_rate": n_flag / len(results) if results else 0.0, "metrics": None,
           "results": sorted(results, key=lambda r: -r["score"])[:max_rows]}
    labels = [r["label"] for r in results]
    if results and all(l is not None for l in labels) and 0 < sum(labels) < len(labels):
        out["metrics"] = evaluate(labels, [r["score"] for r in results], results[0]["threshold"])
    return out


__all__ = ["SchemaMismatch", "resolve_model", "score_transactions", "save_results", "summarize"]
