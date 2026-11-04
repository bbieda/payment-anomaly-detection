import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from anomaly.data import load_raw, temporal_split
from anomaly.models import REGISTRY
from anomaly.train import train_one

router = APIRouter(prefix="/api", tags=["models"])


class ActivateRequest(BaseModel):
    name: str


class RetrainRequest(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)
    normal_only: bool = False


@router.get("/models")
def list_models(request: Request):
    s = request.app.state.store
    return [{"name": n, "kind": s.meta(n)["kind"], "schema": s.meta(n).get("schema", "kaggle_pca"),
             "active": n == s.active, "threshold": s.meta(n)["threshold"], "params": s.meta(n)["params"],
             "trained_at": s.meta(n)["trained_at"], "metrics": s.meta(n).get("metrics")} for n in s.names()]


@router.get("/models/registry")
def registry():
    return [{"name": n, "kind": sp.kind, "defaults": sp.defaults} for n, sp in REGISTRY.items()]


@router.post("/models/activate")
def activate(req: ActivateRequest, request: Request):
    try:
        request.app.state.store.activate(req.name)
    except KeyError:
        raise HTTPException(404, f"Nieznany model: {req.name}")
    return {"active_model": req.name}


def _job(app, req: RetrainRequest):
    app.state.jobs[req.name] = "running"
    try:
        train_one(req.name, temporal_split(load_raw(app.state.data_path)), params=req.params,
                  normal_only=req.normal_only, out_dir=app.state.model_dir)
        app.state.store.refresh()
        app.state.jobs[req.name] = "done"
    except Exception as e:  # noqa: BLE001
        app.state.jobs[req.name] = f"error: {e}"


@router.post("/models/retrain", status_code=202)
def retrain(req: RetrainRequest, bg: BackgroundTasks, request: Request):
    if req.name not in REGISTRY:
        raise HTTPException(404, f"Nieznany model: {req.name}")
    if not os.path.exists(request.app.state.data_path):
        raise HTTPException(409, "Brak pliku z danymi (DATA_PATH)")
    if request.app.state.jobs.get(req.name) == "running":
        raise HTTPException(409, "Trening tego modelu już trwa")
    bg.add_task(_job, request.app, req)
    return {"status": "accepted", "model": req.name}


@router.get("/models/retrain/status")
def retrain_status(request: Request):
    return request.app.state.jobs


@router.get("/metrics")
def all_metrics(request: Request):
    s = request.app.state.store
    return {n: {"kind": s.meta(n)["kind"], **s.meta(n)["metrics"], "fit_seconds": s.meta(n)["fit_seconds"],
                "predict_latency_ms": s.meta(n)["predict_latency_ms"]}
            for n in s.names() if s.meta(n).get("metrics")}


@router.get("/metrics/{name}")
def model_metrics(name: str, request: Request):
    try:
        return request.app.state.store.meta(name)
    except KeyError:
        raise HTTPException(404, f"Nieznany model: {name}")
