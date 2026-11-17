import uuid
from datetime import datetime, timezone

from sqlalchemy import case, func
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, model_validator

from ..connectors.base import Transaction
from ..connectors.replay import ReplaySource
from ..connectors.simulated import SimulatedSource
from ..deps import get_db
from ..db import Prediction
from ..services.scoring import SchemaMismatch, resolve_model, save_results, score_transactions

router = APIRouter(prefix="/api", tags=["predict"])


class PredictRequest(BaseModel):
    amount: float | None = None
    timestamp: datetime | None = None
    merchant: str | None = None
    features: dict[str, float] | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _need_amount(self):
        if self.amount is None and not (self.features and "Amount" in self.features):
            raise ValueError("Podaj 'amount' albo 'features' zawierające 'Amount'.")
        return self


def to_http(e: Exception) -> HTTPException:
    if isinstance(e, SchemaMismatch):
        return HTTPException(422, str(e))
    if isinstance(e, KeyError):
        return HTTPException(404, f"Nieznany model: {e.args[0]}")
    return HTTPException(409, str(e))


@router.post("/predict")
def predict(req: PredictRequest, request: Request, db=Depends(get_db)):
    store = request.app.state.store
    try:
        name = resolve_model(store, req.model)
        tx = Transaction(id=str(uuid.uuid4()), timestamp=req.timestamp or datetime.now(timezone.utc),
                         amount=req.amount if req.amount is not None else req.features["Amount"],
                         merchant=req.merchant, features=req.features)
        res = score_transactions(store, name, [tx])
    except (SchemaMismatch, KeyError, LookupError) as e:
        raise to_http(e)
    save_results(db, "api", res)
    r = res[0]
    return {**r, "risk": "high" if r["is_anomaly"] else "low"}   # TODO: skala ryzyka z percentyli


@router.get("/transactions/recent")
def recent(limit: int = 50, source: str | None = None, db=Depends(get_db)):
    q = db.query(Prediction)
    if source:
        q = q.filter(Prediction.source == source)
    rows = q.order_by(Prediction.id.desc()).limit(min(limit, 500)).all()
    return [{"id": r.id, "tx_id": r.tx_id, "source": r.source, "created_at": r.created_at.isoformat(),
             "model": r.model_name, "score": r.score, "is_anomaly": r.is_anomaly, "amount": r.amount,
             "merchant": r.merchant, "label": r.label} for r in rows]


@router.get("/stream/next")
def stream_next(request: Request, source: str = "simulated", model: str | None = None,
                n: int = Query(1, ge=1, le=100), fraud_rate: float | None = Query(None, ge=0, le=0.5),
                db=Depends(get_db)):
    """Kolejne transakcje ze strumienia (do odpytywania przez dashboard co ~2 s)."""
    app, store = request.app, request.app.state.store
    try:
        name = resolve_model(store, model)
        schema = store.meta(name).get("schema", "kaggle_pca")
        key = (source, schema, fraud_rate)
        if key not in app.state.streams:
            if source == "simulated":
                app.state.streams[key] = SimulatedSource(schema=schema, fraud_rate=0.05 if fraud_rate is None else fraud_rate)
            elif source == "replay":
                if schema != "kaggle_pca":
                    raise LookupError("Źródło 'replay' wymaga modelu o schemacie kaggle_pca.")
                try:
                    app.state.streams[key] = ReplaySource(app.state.data_path, fraud_rate or 0.0)
                except FileNotFoundError:
                    raise LookupError("Brak pliku z danymi (DATA_PATH).")
            else:
                raise LookupError(f"Nieznane lub niedostępne źródło: {source}")
        res = score_transactions(store, name, app.state.streams[key].fetch(n))
    except (SchemaMismatch, KeyError, LookupError) as e:
        raise to_http(e)
    save_results(db, source, res)
    return res


@router.get("/stats")
def stats(db=Depends(get_db)):
    """Zbiorcze statystyki historii ocen (dla panelu)."""
    flagged = func.sum(case((Prediction.is_anomaly, 1), else_=0))
    total, anomalies, avg_lat = db.query(func.count(Prediction.id), flagged, func.avg(Prediction.latency_ms)).one()
    by_source = db.query(Prediction.source, func.count(Prediction.id), flagged).group_by(Prediction.source).all()
    by_model = db.query(Prediction.model_name, func.count(Prediction.id), flagged, func.avg(Prediction.latency_ms)) \
        .group_by(Prediction.model_name).all()
    return {"total": total or 0, "anomalies": int(anomalies or 0), "avg_latency_ms": avg_lat,
            "by_source": [{"source": s, "count": c, "anomalies": int(a or 0)} for s, c, a in by_source],
            "by_model": [{"model": m, "count": c, "anomalies": int(a or 0), "avg_latency_ms": l} for m, c, a, l in by_model]}


@router.delete("/transactions")
def clear_history(db=Depends(get_db)):
    """Czyści historię ocen (przydatne przed demonstracją). Prototyp: bez uwierzytelniania."""
    n = db.query(Prediction).delete()
    db.commit()
    return {"deleted": n}
